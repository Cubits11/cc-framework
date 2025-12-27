# experiments/correlation_cliff/guardrail_adapters.py
"""
Unified adapters for real guardrail APIs.
Each adapter implements: .check(text: str) -> bool
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import os

class GuardrailAdapter(ABC):
    """Base class for guardrail wrappers."""
    
    @abstractmethod
    def check(self, text: str) -> bool:
        """Returns True if guardrail triggers (blocks), False if passes."""
        pass
    
    def batch_check(self, texts: List[str]) -> List[bool]:
        """Batch interface (default: sequential calls)."""
        return [self.check(t) for t in texts]


class OpenAIModerationAdapter(GuardrailAdapter):
    """OpenAI Moderation API v2."""
    
    def __init__(self, api_key: str = None):
        import openai
        self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    
    def check(self, text: str) -> bool:
        response = self.client.moderations.create(input=text)
        return response.results[0].flagged  # True = blocked


class PerspectiveAPIAdapter(GuardrailAdapter):
    """Google Perspective API (toxicity)."""
    
    def __init__(self, api_key: str = None, threshold: float = 0.7):
        from googleapiclient import discovery
        self.api_key = api_key or os.getenv("PERSPECTIVE_API_KEY")
        self.client = discovery.build(
            "commentanalyzer", "v1alpha1",
            developerKey=self.api_key,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )
        self.threshold = threshold
    
    def check(self, text: str) -> bool:
        analyze_request = {
            'comment': {'text': text},
            'requestedAttributes': {'TOXICITY': {}}
        }
        response = self.client.comments().analyze(body=analyze_request).execute()
        score = response['attributeScores']['TOXICITY']['summaryScore']['value']
        return score >= self.threshold


class LlamaGuardAdapter(GuardrailAdapter):
    """Llama Guard 3 (local inference via transformers)."""
    
    def __init__(self, model_name: str = "meta-llama/Llama-Guard-3-8B", device: str = "cuda"):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
        )
    
    def check(self, text: str) -> bool:
        # Llama Guard outputs "safe" or "unsafe"
        prompt = f"[INST] {text} [/INST]"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=10)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return "unsafe" in response.lower()


# Usage example:
def evaluate_guardrail_pair(
    rail_A: GuardrailAdapter,
    rail_B: GuardrailAdapter,
    dataset: TwoWorldDataset,
) -> Dict[str, float]:
    """
    Evaluate two guardrails on a two-world dataset.
    Returns marginals: {pA0, pB0, pA1, pB1}
    """
    # World 0 (safe)
    A0 = rail_A.batch_check(dataset.world_0)
    B0 = rail_B.batch_check(dataset.world_0)
    pA0 = sum(A0) / len(A0)
    pB0 = sum(B0) / len(B0)
    
    # World 1 (unsafe)
    A1 = rail_A.batch_check(dataset.world_1)
    B1 = rail_B.batch_check(dataset.world_1)
    pA1 = sum(A1) / len(A1)
    pB1 = sum(B1) / len(B1)
    
    return {
        'pA0': pA0, 'pB0': pB0,
        'pA1': pA1, 'pB1': pB1,
        'n0': len(dataset.world_0),
        'n1': len(dataset.world_1),
    }