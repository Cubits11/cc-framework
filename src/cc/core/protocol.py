# src/cc/core/protocol.py
"""
Two-world adaptive attack protocol implementation
Core orchestration for CC framework experiments

Author: Pranav Bhave
Date: 2025-08-27
Institution: Penn State University
"""
from __future__ import annotations
import hashlib
import time
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Protocol as TypingProtocol
from dataclasses import dataclass, asdict
from pathlib import Path
import warnings
from contextlib import contextmanager

# Import from cc package (absolute imports)
from cc.core.models import AttackResult, GuardrailSpec, WorldConfig
from cc.core.attackers import AttackStrategy, RandomInjectionAttacker
from cc.core.logging import ChainedJSONLLogger, audit_context
from cc.guardrails.base import Guardrail

# Import specific guardrails
try:
    from cc.guardrails.regex_filters import RegexFilter
    from cc.guardrails.keyword_blocker import KeywordBlocker
    from cc.guardrails.semantic_filter import SemanticFilter
except ImportError as e:
    warnings.warn(f"Some guardrails not available: {e}")
    RegexFilter = KeywordBlocker = SemanticFilter = None

@dataclass
class SessionMetadata:
    """Metadata for an attack session"""
    session_id: str
    world_bit: int
    start_time: float
    end_time: float
    turns: int
    final_success: bool
    attack_history: List[Dict[str, Any]]
    guardrails_triggered: List[str]
    utility_score: Optional[float] = None

@dataclass
class ExperimentMetadata:
    """Metadata for entire experiment"""
    experiment_id: str
    start_time: float
    end_time: float
    total_sessions: int
    config_hash: str
    git_commit: str
    environment: Dict[str, str]

class TwoWorldProtocol:
    """
    Two-world adaptive attack protocol for guardrail evaluation
    
    This implements the core experimental protocol where an adaptive adversary
    attempts to distinguish between two worlds (with/without protection) through
    interactive querying.
    """
    
    def __init__(self,
                 logger: ChainedJSONLLogger,
                 base_success_rate: float = 0.6,
                 episode_length: int = 10,
                 random_seed: int = 42,
                 cache_attacks: bool = True,
                 validate_inputs: bool = True):
        """
        Initialize two-world protocol
        
        Args:
            logger: Chained JSONL logger for audit trail
            base_success_rate: Success rate without guardrails
            episode_length: Maximum turns per attack episode
            random_seed: Random seed for reproducibility
            cache_attacks: Whether to cache attack patterns
            validate_inputs: Whether to validate all inputs
        """
        self.logger = logger
        self.base_success_rate = base_success_rate
        self.episode_length = episode_length
        self.rng = np.random.default_rng(random_seed)
        self.cache_attacks = cache_attacks
        self.validate_inputs = validate_inputs
        
        # Session tracking
        self.session_count = 0
        self.results: List[AttackResult] = []
        self.session_metadata: List[SessionMetadata] = []
        
        # Attack cache for efficiency
        self.attack_cache: Dict[str, Any] = {}
        
        # Performance tracking
        self.timing_stats = {
            'attack_generation': [],
            'guardrail_evaluation': [],
            'total_session': []
        }
    
    def build_guardrail_stack(self, specs: List[GuardrailSpec]) -> List[Guardrail]:
        """
        Build guardrail stack from specifications
        
        Args:
            specs: List of guardrail specifications
            
        Returns:
            List of instantiated guardrail objects
        """
        stack = []
        
        for spec in specs:
            try:
                guardrail = self._create_guardrail(spec)
                stack.append(guardrail)
                
                # Log guardrail creation
                self.logger.log({
                    'event': 'guardrail_created',
                    'name': spec.name,
                    'version': spec.version,
                    'params_hash': self._hash_dict(spec.params)
                })
            except Exception as e:
                self.logger.log({
                    'event': 'guardrail_creation_failed',
                    'name': spec.name,
                    'error': str(e)
                })
                raise
        
        return stack
    
    def _create_guardrail(self, spec: GuardrailSpec) -> Guardrail:
        """Create guardrail instance from specification"""
        # Map guardrail names to classes
        guardrail_map = {
            'regex': RegexFilter,
            'regex_filter': RegexFilter,
            'regex_filters': RegexFilter,
            'keyword': KeywordBlocker,
            'keyword_blocker': KeywordBlocker,
            'semantic': SemanticFilter,
            'semantic_filter': SemanticFilter
        }
        
        guardrail_class = guardrail_map.get(spec.name.lower())
        if guardrail_class is None:
            raise ValueError(f"Unknown guardrail type: {spec.name}")
        
        # Create instance with parameters
        return guardrail_class(**spec.params)
    
    def apply_guardrail_stack(self, 
                             stack: List[Guardrail], 
                             text: str) -> Tuple[bool, float, List[str]]:
        """
        Apply guardrail stack to text with detailed tracking
        
        Args:
            stack: List of guardrails to apply
            text: Input text to evaluate
            
        Returns:
            (is_blocked, max_score, triggered_guardrails)
        """
        if not stack:
            return False, 0.0, []
        
        max_score = 0.0
        is_blocked = False
        triggered = []
        
        for guardrail in stack:
            try:
                score = guardrail.score(text)
                max_score = max(max_score, score)
                
                if guardrail.blocks(text):
                    is_blocked = True
                    triggered.append(guardrail.__class__.__name__)
                    # Early stopping on first block (configurable)
                    break
                    
            except Exception as e:
                self.logger.log({
                    'event': 'guardrail_error',
                    'guardrail': guardrail.__class__.__name__,
                    'error': str(e)
                })
                # Continue with other guardrails
        
        return is_blocked, max_score, triggered
    
    def run_session(self,
                   attacker: AttackStrategy,
                   world_configs: Dict[int, WorldConfig],
                   session_id: Optional[str] = None,
                   collect_metadata: bool = True) -> AttackResult:
        """
        Run single two-world attack session
        
        Args:
            attacker: Attack strategy to use
            world_configs: Configuration for world 0 and world 1
            session_id: Optional session identifier
            collect_metadata: Whether to collect detailed metadata
            
        Returns:
            AttackResult with session outcome
        """
        if session_id is None:
            session_id = f"session_{self.session_count:06d}_{int(time.time())}"
        
        self.session_count += 1
        start_time = time.time()
        
        # Randomly select world with balanced sampling
        world_bit = self._select_world()
        world_config = world_configs[world_bit]
        
        # Build guardrail stack for selected world
        guardrail_stack = self.build_guardrail_stack(world_config.guardrail_stack)
        
        # Log session start
        with audit_context(self.logger, 'attack_session', session_id=session_id, world=world_bit):
            
            # Run attack episode
            history = []
            final_success = False
            triggered_guardrails = set()
            
            for turn in range(self.episode_length):
                turn_start = time.time()
                
                # Generate attack
                attack = attacker.generate_attack(history)
                prompt = attack['prompt']
                
                # Track generation time
                gen_time = time.time() - turn_start
                self.timing_stats['attack_generation'].append(gen_time)
                
                # Apply guardrails
                guard_start = time.time()
                is_blocked, block_score, triggered = self.apply_guardrail_stack(
                    guardrail_stack, prompt
                )
                guard_time = time.time() - guard_start
                self.timing_stats['guardrail_evaluation'].append(guard_time)
                
                triggered_guardrails.update(triggered)
                
                # Determine success
                if is_blocked:
                    success = False
                    response = "BLOCKED"
                else:
                    # Simulate underlying system response
                    success = self.rng.random() < world_config.baseline_success_rate
                    response = "SUCCESS" if success else "FAILURE"
                
                # Update history
                turn_result = {
                    'turn': turn,
                    'prompt': prompt if not self.validate_inputs else self._sanitize_prompt(prompt),
                    'response': response,
                    'success': success,
                    'blocked': is_blocked,
                    'block_score': block_score,
                    'triggered_guardrails': triggered
                }
                history.append(turn_result)
                
                # Update attacker strategy
                attacker.update_strategy(attack, turn_result)
                
                # Check for early success
                if success:
                    final_success = True
                    if turn < self.episode_length - 1:
                        # Early termination on success
                        break
            
            # Session timing
            end_time = time.time()
            session_duration = end_time - start_time
            self.timing_stats['total_session'].append(session_duration)
            
            # Create result
            result = AttackResult(
                world_bit=world_bit,
                success=final_success,
                attack_id=f"{session_id}_world{world_bit}",
                transcript_hash=self._hash_transcript(history),
                guardrails_applied=','.join(spec.name for spec in world_config.guardrail_stack),
                rng_seed=int(self.rng.bit_generator.random_raw() & 0xffffffff),
                timestamp=end_time,
                session_id=session_id,
                attack_strategy=type(attacker).__name__,
                utility_score=self._compute_utility_score(history)
            )
            
            # Store metadata if requested
            if collect_metadata:
                metadata = SessionMetadata(
                    session_id=session_id,
                    world_bit=world_bit,
                    start_time=start_time,
                    end_time=end_time,
                    turns=len(history),
                    final_success=final_success,
                    attack_history=history,
                    guardrails_triggered=list(triggered_guardrails),
                    utility_score=result.utility_score
                )
                self.session_metadata.append(metadata)
            
            # Log session summary
            self.logger.log({
                'event': 'session_complete',
                'session_id': session_id,
                'world_bit': world_bit,
                'success': final_success,
                'turns': len(history),
                'duration': session_duration,
                'guardrails_triggered': list(triggered_guardrails),
                'attack_strategy': type(attacker).__name__
            })
            
            self.results.append(result)
            return result
    
    def run_experiment(self,
                      attacker: AttackStrategy,
                      world_configs: Dict[int, WorldConfig],
                      n_sessions: int,
                      experiment_id: Optional[str] = None,
                      parallel: bool = False,
                      checkpoint_every: int = 100) -> List[AttackResult]:
        """
        Run full two-world experiment with checkpointing
        
        Args:
            attacker: Attack strategy to use
            world_configs: World configurations
            n_sessions: Number of sessions to run
            experiment_id: Experiment identifier
            parallel: Whether to run sessions in parallel (future feature)
            checkpoint_every: Save checkpoint every N sessions
            
        Returns:
            List of AttackResult objects
        """
        if experiment_id is None:
            experiment_id = f"exp_{int(time.time())}_{self._generate_id()}"
        
        exp_start = time.time()
        
        # Log experiment start
        self.logger.log({
            'event': 'experiment_start',
            'experiment_id': experiment_id,
            'n_sessions': n_sessions,
            'attacker': type(attacker).__name__,
            'world_configs': {
                k: {
                    'guardrails': [spec.name for spec in config.guardrail_stack],
                    'baseline_success': config.baseline_success_rate
                }
                for k, config in world_configs.items()
            },
            'episode_length': self.episode_length,
            'random_seed': int(self.rng.bit_generator.random_raw() & 0xffffffff)
        })
        
        session_results = []
        
        try:
            for i in range(n_sessions):
                # Progress reporting
                if i > 0 and i % 10 == 0:
                    elapsed = time.time() - exp_start
                    rate = i / elapsed
                    eta = (n_sessions - i) / rate
                    print(f"Session {i+1}/{n_sessions} | "
                          f"Rate: {rate:.1f} sessions/sec | "
                          f"ETA: {eta/60:.1f} min")
                
                # Run session
                session_id = f"{experiment_id}_s{i:06d}"
                result = self.run_session(attacker, world_configs, session_id)
                session_results.append(result)
                
                # Checkpointing
                if checkpoint_every and i > 0 and i % checkpoint_every == 0:
                    self._save_checkpoint(experiment_id, session_results)
                
                # Reset attacker periodically to prevent overfitting
                if i > 0 and i % 500 == 0:
                    attacker.reset()
                    self.logger.log({
                        'event': 'attacker_reset',
                        'session': i,
                        'reason': 'periodic_reset'
                    })
            
            # Final statistics
            exp_end = time.time()
            exp_duration = exp_end - exp_start
            
            # Compute summary statistics
            world_0_successes = sum(1 for r in session_results if r.world_bit == 0 and r.success)
            world_1_successes = sum(1 for r in session_results if r.world_bit == 1 and r.success)
            world_0_total = sum(1 for r in session_results if r.world_bit == 0)
            world_1_total = sum(1 for r in session_results if r.world_bit == 1)
            
            self.logger.log({
                'event': 'experiment_complete',
                'experiment_id': experiment_id,
                'total_sessions': len(session_results),
                'duration': exp_duration,
                'sessions_per_second': len(session_results) / exp_duration,
                'world_0_success_rate': world_0_successes / world_0_total if world_0_total > 0 else 0,
                'world_1_success_rate': world_1_successes / world_1_total if world_1_total > 0 else 0,
                'world_0_sessions': world_0_total,
                'world_1_sessions': world_1_total
            })
            
        except KeyboardInterrupt:
            self.logger.log({
                'event': 'experiment_interrupted',
                'experiment_id': experiment_id,
                'completed_sessions': len(session_results)
            })
            print(f"\nExperiment interrupted. Completed {len(session_results)} sessions.")
            
        except Exception as e:
            self.logger.log({
                'event': 'experiment_error',
                'experiment_id': experiment_id,
                'error': str(e),
                'completed_sessions': len(session_results)
            })
            raise
        
        finally:
            # Always save final results
            if session_results:
                self._save_checkpoint(experiment_id, session_results, final=True)
        
        return session_results
    
    def calibrate_guardrails(self,
                            guardrail_specs: List[GuardrailSpec],
                            benign_texts: List[str],
                            target_fpr: float = 0.05,
                            tolerance: float = 0.01) -> List[GuardrailSpec]:
        """
        Calibrate guardrails to achieve target false positive rate
        
        Args:
            guardrail_specs: Guardrail specifications to calibrate
            benign_texts: Benign texts for calibration
            target_fpr: Target false positive rate
            tolerance: Acceptable deviation from target
            
        Returns:
            Calibrated guardrail specifications
        """
        calibrated_specs = []
        
        for spec in guardrail_specs:
            guardrail = self._create_guardrail(spec)
            
            # Calibrate if the guardrail supports it
            if hasattr(guardrail, 'calibrate'):
                guardrail.calibrate(benign_texts, target_fpr)
                
                # Validate calibration
                false_positives = sum(1 for text in benign_texts if guardrail.blocks(text))
                actual_fpr = false_positives / len(benign_texts)
                
                if abs(actual_fpr - target_fpr) > tolerance:
                    warnings.warn(
                        f"Guardrail {spec.name} calibration off target: "
                        f"actual FPR={actual_fpr:.3f}, target={target_fpr:.3f}"
                    )
                
                # Update spec with calibration info
                spec.calibration_fpr_target = target_fpr
                spec.calibration_data_hash = self._hash_list(benign_texts)
                
                self.logger.log({
                    'event': 'guardrail_calibrated',
                    'guardrail': spec.name,
                    'target_fpr': target_fpr,
                    'actual_fpr': actual_fpr,
                    'n_benign': len(benign_texts)
                })
            
            calibrated_specs.append(spec)
        
        return calibrated_specs
    
    # Helper methods
    
    def _select_world(self) -> int:
        """Select world with balanced sampling"""
        # Count existing world distribution
        if self.results:
            world_0_count = sum(1 for r in self.results if r.world_bit == 0)
            world_1_count = len(self.results) - world_0_count
            
            # Bias toward undersampled world
            if world_0_count < world_1_count - 5:
                return 0
            elif world_1_count < world_0_count - 5:
                return 1
        
        # Random selection
        return int(self.rng.random() < 0.5)
    
    def _hash_transcript(self, history: List[Dict[str, Any]]) -> str:
        """Create hash of interaction transcript"""
        # Serialize history deterministically
        canonical = json.dumps(history, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()
    
    def _hash_dict(self, d: Dict[str, Any]) -> str:
        """Hash dictionary contents"""
        canonical = json.dumps(d, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]
    
    def _hash_list(self, items: List[Any]) -> str:
        """Hash list contents"""
        combined = ''.join(str(item) for item in items)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def _generate_id(self) -> str:
        """Generate unique identifier"""
        return hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]
    
    def _sanitize_prompt(self, prompt: str) -> str:
        """Sanitize prompt for safe storage"""
        # Remove potential injection attempts
        # This is a simplified version - real implementation would be more thorough
        return prompt[:1000]  # Truncate to reasonable length
    
    def _compute_utility_score(self, history: List[Dict[str, Any]]) -> float:
        """
        Compute utility score for a session
        
        Simple metric: fraction of non-blocked turns
        """
        if not history:
            return 0.0
        
        non_blocked = sum(1 for turn in history if not turn['blocked'])
        return non_blocked / len(history)
    
    def _save_checkpoint(self, experiment_id: str, results: List[AttackResult], final: bool = False):
        """Save checkpoint of experiment results"""
        checkpoint_dir = Path('checkpoints') / experiment_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if final:
            filename = checkpoint_dir / 'final_results.json'
        else:
            filename = checkpoint_dir / f'checkpoint_{len(results):06d}.json'
        
        # Convert results to serializable format
        data = {
            'experiment_id': experiment_id,
            'n_sessions': len(results),
            'timestamp': time.time(),
            'results': [r.to_dict() for r in results]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.log({
            'event': 'checkpoint_saved',
            'experiment_id': experiment_id,
            'filename': str(filename),
            'n_sessions': len(results),
            'final': final
        })
    
    def get_results(self) -> List[AttackResult]:
        """Get all accumulated results"""
        return self.results.copy()
    
    def get_metadata(self) -> List[SessionMetadata]:
        """Get session metadata"""
        return self.session_metadata.copy()
    
    def get_timing_stats(self) -> Dict[str, Any]:
        """Get timing statistics"""
        stats = {}
        for key, times in self.timing_stats.items():
            if times:
                stats[key] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'median': np.median(times),
                    'total': np.sum(times),
                    'count': len(times)
                }
        return stats
    
    def reset(self):
        """Reset protocol state for new experiment"""
        self.session_count = 0
        self.results.clear()
        self.session_metadata.clear()
        self.attack_cache.clear()
        for key in self.timing_stats:
            self.timing_stats[key].clear()

# Export key classes and functions
__all__ = [
    'TwoWorldProtocol',
    'SessionMetadata',
    'ExperimentMetadata'
]
