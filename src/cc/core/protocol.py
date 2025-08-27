# src/cc/core/protocol.py
"""Two-world adaptive attack protocol implementation"""
from __future__ import annotations
import hashlib
import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

from .models import AttackResult, GuardrailSpec, WorldConfig
from .attackers import AttackStrategy
from .logging import ChainedJSONLLogger
from ..guardrails.base import Guardrail
from ..guardrails.regex_filter import RegexFilter  
from ..guardrails.keyword_blocker import KeywordBlocker
from ..guardrails.semantic_filter import SemanticFilter

def hash_transcript(prompt: str, response: str, success: bool) -> str:
    """Create hash of interaction transcript"""
    transcript = f"{prompt}|{response}|{success}"
    return hashlib.sha256(transcript.encode()).hexdigest()

def build_guardrail_stack(specs: List[GuardrailSpec]) -> List[Guardrail]:
    """Build guardrail stack from specifications"""
    stack = []
    
    for spec in specs:
        if spec.name == "regex":
            guardrail = RegexFilter(**spec.params)
        elif spec.name == "keyword":  
            guardrail = KeywordBlocker(**spec.params)
        elif spec.name == "semantic":
            guardrail = SemanticFilter(**spec.params)
        else:
            raise ValueError(f"Unknown guardrail type: {spec.name}")
        
        stack.append(guardrail)
    
    return stack

def apply_guardrail_stack(stack: List[Guardrail], text: str) -> Tuple[bool, float]:
    """
    Apply guardrail stack to text
    Returns: (is_blocked, max_score)
    """
    if not stack:
        return False, 0.0
    
    max_score = 0.0
    is_blocked = False
    
    for guardrail in stack:
        score = guardrail.score(text)
        max_score = max(max_score, score)
        if guardrail.blocks(text):
            is_blocked = True
            break  # Early stopping on first block
    
    return is_blocked, max_score

class TwoWorldProtocol:
    """Two-world adaptive attack protocol"""
    
    def __init__(self,
                 logger: ChainedJSONLLogger,
                 base_success_rate: float = 0.6,
                 episode_length: int = 10,
                 random_seed: int = 42):
        
        self.logger = logger
        self.base_success_rate = base_success_rate
        self.episode_length = episode_length
        self.rng = np.random.default_rng(random_seed)
        
        # Session tracking
        self.session_count = 0
        self.results: List[AttackResult] = []
        
    def run_session(self,
                   attacker: AttackStrategy, 
                   world_configs: Dict[int, WorldConfig],
                   session_id: Optional[str] = None) -> AttackResult:
        """
        Run single two-world attack session
        
        Args:
            attacker: Attack strategy to use
            world_configs: Configuration for world 0 and world 1
            session_id: Optional session identifier
        """
        if session_id is None:
            session_id = f"session_{self.session_count:06d}"
        
        self.session_count += 1
        
        # Randomly select world
        world_bit = self.rng.integers(0, 2)
        world_config = world_configs[world_bit]
        
        # Build guardrail stack
        guardrail_stack = build_guardrail_stack(world_config.guardrail_stack)
        
        # Run attack episode
        history = []
        final_success = False
        
        for turn in range(self.episode_length):
            # Generate attack
            attack = attacker.generate_attack(history)
            prompt = attack["prompt"]
            
            # Apply guardrails
            is_blocked, block_score = apply_guardrail_stack(guardrail_stack, prompt)
            
            # Determine success (if not blocked)
            if is_blocked:
                success = False
                response = "BLOCKED"
            else:
                # Simulate underlying system response based on base success rate
                success = self.rng.random() < world_config.baseline_success_rate
                response = "SUCCESS" if success else "FAILURE"
            
            # Update history
            turn_result = {
                "turn": turn,
                "prompt": prompt,
                "response": response, 
                "success": success,
                "blocked": is_blocked,
                "block_score": block_score
            }
            history.append(turn_result)
            
            # Update attacker
            attacker.update_strategy(attack, turn_result)
            
            # Check for early success
            if success:
                final_success = True
                break
        
        # Create result
        result = AttackResult(
            world_bit=world_bit,
            success=final_success,
            attack_id=f"{session_id}_world{world_bit}",
            transcript_hash=hash_transcript(
                str(history), 
                "SUCCESS" if final_success else "FAILURE",
                final_success
            ),
            guardrails_applied=",".join(spec.name for spec in world_config.guardrail_stack),
            rng_seed=int(self.rng.bit_generator.random_raw() & 0xffffffff),
            timestamp=time.time(),
            session_id=session_id,
            attack_strategy=type(attacker).__name__
        )
        
        # Log session
        self.logger.log({
            "event": "attack_session",
            "session_id": session_id,
            "world_bit": world_bit,
            "success": final_success,
            "turns": len(history),
            "guardrails": result.guardrails_applied,
            "attack_strategy": result.attack_strategy,
            "history_summary": {
                "total_turns": len(history),
                "blocked_turns": sum(1 for h in history if h["blocked"]),
                "success_turns": sum(1 for h in history if h["success"])
            }
        })
        
        self.results.append(result)
        return result
    
    def run_experiment(self,
                      attacker: AttackStrategy,
                      world_configs: Dict[int, WorldConfig],
                      n_sessions: int,
                      experiment_id: str = "default") -> List[AttackResult]:
        """Run full two-world experiment"""
        
        self.logger.log({
            "event": "experiment_start",
            "experiment_id": experiment_id,
            "n_sessions": n_sessions,
            "attacker": type(attacker).__name__,
            "world_configs": {
                k: {
                    "guardrails": [spec.name for spec in config.guardrail_stack],
                    "baseline_success": config.baseline_success_rate
                }
                for k, config in world_configs.items()
            }
        })
        
        session_results = []
        
        try:
            for i in range(n_sessions):
                if i % 100 == 0:
                    print(f"Running session {i+1}/{n_sessions}")
                
                session_id = f"{experiment_id}_s{i:06d}"
                result = self.run_session(attacker, world_configs, session_id)
                session_results.append(result)
                
                # Reset attacker periodically to prevent overfitting
                if i > 0 and i % 500 == 0:
                    attacker.reset()
            
            self.logger.log({
                "event": "experiment_complete",
                "experiment_id": experiment_id,
                "total_sessions": len(session_results),
                "world_0_successes": sum(1 for r in session_results if r.world_bit == 0 and r.success),
                "world_1_successes": sum(1 for r in session_results if r.world_bit == 1 and r.success)
            })
            
        except Exception as e:
            self.logger.log({
                "event": "experiment_error", 
                "experiment_id": experiment_id,
                "error": str(e),
                "completed_sessions": len(session_results)
            })
            raise
        
        return session_results
    
    def get_results(self) -> List[AttackResult]:
        """Get all accumulated results"""
        return self.results.copy()