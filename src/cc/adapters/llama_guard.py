# src/cc/adapters/llama_guard.py
"""Adapter for Meta Purple Llama (Llama Guard)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from .base import Decision, GuardrailAdapter

LLAMA_GUARD_SYSTEM_PROMPT = (
    "You are a safety classifier. Given a user prompt and optional assistant "
    "response, classify the content as safe or unsafe and, if unsafe, provide "
    "a category."
)


def _default_prompt(prompt: str, response: Optional[str]) -> str:
    if response:
        return f"User: {prompt}\nAssistant: {response}"
    return f"User: {prompt}"


@dataclass
class LlamaGuardAdapter(GuardrailAdapter):
    """Llama Guard adapter using Transformers-style generation.

    Parameters
    ----------
    model_name:
        HuggingFace model id (e.g., meta-llama/Llama-Guard-3-8B).
    device_map:
        Device map for Transformers (e.g., "auto", "cpu", "cuda").
    max_new_tokens:
        Generation cap for classification tokens.
    temperature:
        Generation temperature; use 0.0 for deterministic decoding.
    threshold:
        Optional decision threshold when a probabilistic score is available.
    score_mode:
        "logprob" to compute a probability over {"safe","unsafe"} if possible.
    generator:
        Optional callable for dependency injection in tests. Should return
        (text, score, raw_payload).
    """

    model_name: str = "meta-llama/Llama-Guard-3-8B"
    device_map: str = "auto"
    max_new_tokens: int = 8
    temperature: float = 0.0
    threshold: float = 0.5
    score_mode: str = "logprob"
    generator: Optional[Callable[[str], Tuple[str, Optional[float], Dict[str, Any]]]] = None
    model: Any = None
    tokenizer: Any = None

    name: str = "llama_guard"
    version: str = "unknown"
    supports_input_check: bool = True
    supports_output_check: bool = True

    def __post_init__(self) -> None:
        if self.generator is not None:
            self.version = self.model_name
            return
        if self.model is None or self.tokenizer is None:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise ImportError(
                    "transformers is required for LlamaGuardAdapter; install it or pass a generator/model."
                ) from exc
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, device_map=self.device_map
            )
        self.version = self.model_name

    def _build_prompt(self, prompt: str, response: Optional[str]) -> str:
        content = _default_prompt(prompt, response)
        if self.tokenizer is None:
            return content
        messages = [
            {"role": "system", "content": LLAMA_GUARD_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        return self.tokenizer.apply_chat_template(  # type: ignore[no-any-return]
            messages, tokenize=False, add_generation_prompt=True
        )

    def _generate(self, prompt_text: str) -> Tuple[str, Optional[float], Dict[str, Any]]:
        if self.generator is not None:
            return self.generator(prompt_text)
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError("torch is required for LlamaGuardAdapter generation.") from exc

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            output_scores=(self.score_mode == "logprob"),
            return_dict_in_generate=True,
        )
        decoded = self.tokenizer.decode(
            outputs.sequences[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        )
        score = None
        if self.score_mode == "logprob" and outputs.scores:
            score = _score_from_logits(outputs.scores[0], self.tokenizer)
        return decoded.strip(), score, {"decoded": decoded}

    def check(self, prompt: str, response: Optional[str], metadata: Dict[str, Any]) -> Decision:
        prompt_text = self._build_prompt(prompt, response)
        generated, score, raw = self._generate(prompt_text)
        verdict, category = _parse_llama_guard_output(generated)
        if score is not None:
            verdict = "block" if score >= self.threshold else "allow"
        return Decision(
            verdict=verdict,
            category=category,
            score=score,
            rationale=None,
            raw={"text": generated, **raw, "metadata": metadata},
            adapter_name=self.name,
            adapter_version=self.version,
        )


def _score_from_logits(logits: Any, tokenizer: Any) -> Optional[float]:
    try:
        import torch
    except ImportError:  # pragma: no cover - dependency guard
        return None

    tokens = {"safe": tokenizer.encode("safe", add_special_tokens=False),
              "unsafe": tokenizer.encode("unsafe", add_special_tokens=False)}
    if any(len(v) != 1 for v in tokens.values()):
        return None
    safe_id = tokens["safe"][0]
    unsafe_id = tokens["unsafe"][0]
    probs = torch.softmax(logits, dim=-1)
    return float(probs[0, unsafe_id])


def _parse_llama_guard_output(text: str) -> Tuple[str, Optional[str]]:
    lowered = text.lower()
    if "unsafe" in lowered:
        category = None
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            try:
                unsafe_idx = next(i for i, ln in enumerate(lines) if "unsafe" in ln.lower())
                if unsafe_idx + 1 < len(lines):
                    category = lines[unsafe_idx + 1]
            except StopIteration:
                category = None
        return "block", category
    return "allow", None
