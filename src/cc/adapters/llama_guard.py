# src/cc/adapters/llama_guard.py
"""Adapter for Meta Purple Llama (Llama Guard)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .base import (
    Decision,
    GuardrailAdapter,
    build_audit_payload,
    error_summary_from_exception,
    fingerprint_payload,
    sanitize_metadata,
    sanitize_vendor_payload,
    summarize_value,
)

LLAMA_GUARD_SYSTEM_PROMPT = (
    "You are a safety classifier. Given a user prompt and optional assistant "
    "response, classify the content as safe or unsafe and, if unsafe, provide "
    "a category."
)


def _default_prompt(prompt: str, response: str | None) -> str:
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
    generator: Callable[[str], tuple[str, float | None, dict[str, Any]]] | None = None
    model: Any = None
    tokenizer: Any = None

    name: str = "llama_guard"
    version: str = "unknown"
    supports_input_check: bool = True
    supports_output_check: bool = True
    _config_fingerprint: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._config_fingerprint = fingerprint_payload(
            {
                "model_name": self.model_name,
                "temperature": self.temperature,
                "threshold": self.threshold,
                "score_mode": self.score_mode,
                "max_new_tokens": self.max_new_tokens,
                "device_map": self.device_map,
            },
            strict=False,
        )
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

    def _build_prompt(self, prompt: str, response: str | None) -> str:
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

    def _generate(self, prompt_text: str) -> tuple[str, float | None, dict[str, Any]]:
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

    def check(self, prompt: str, response: str | None, metadata: dict[str, Any]) -> Decision:
        started_at = time.time()
        try:
            prompt_text = self._build_prompt(prompt, response)
            generated, score, raw = self._generate(prompt_text)
            completed_at = time.time()
            verdict, category = _parse_llama_guard_output(generated)
            if score is not None:
                verdict = "block" if score >= self.threshold else "allow"
            rationale = "Llama Guard classification."
            error_summary = None
        except Exception as exc:  # fail-closed
            completed_at = time.time()
            generated, score, raw = "", None, {"error": type(exc).__name__}
            verdict, category, rationale = (
                "review",
                "adapter_error",
                f"Llama Guard errored: {type(exc).__name__}",
            )
            error_summary = error_summary_from_exception(exc, where="llama_guard.generate")
        parameters = {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "threshold": self.threshold,
            "score_mode": self.score_mode,
            "max_new_tokens": self.max_new_tokens,
            "device_map": self.device_map,
        }
        audit_payload = build_audit_payload(
            prompt=prompt,
            response=response,
            adapter_name=self.name,
            adapter_version=self.version,
            parameters=parameters,
            decision=verdict,
            category=category,
            rationale=rationale,
            started_at=started_at,
            completed_at=completed_at,
            vendor_request_id=None,
            config_fingerprint=self._config_fingerprint,
            metadata=metadata,
            error_summary=error_summary,
        )
        return Decision(
            verdict=verdict,
            category=category,
            score=score,
            rationale=rationale,
            raw={
                "generated_summary": summarize_value(generated),
                "raw": sanitize_vendor_payload(raw),
                "metadata": sanitize_metadata(metadata),
            },
            adapter_name=self.name,
            adapter_version=self.version,
            audit=audit_payload,
        )


def _score_from_logits(logits: Any, tokenizer: Any) -> float | None:
    try:
        import torch
    except ImportError:  # pragma: no cover - dependency guard
        return None

    tokens = {
        "safe": tokenizer.encode("safe", add_special_tokens=False),
        "unsafe": tokenizer.encode("unsafe", add_special_tokens=False),
    }
    if any(len(v) != 1 for v in tokens.values()):
        return None
    tokens["safe"][0]
    unsafe_id = tokens["unsafe"][0]
    probs = torch.softmax(logits, dim=-1)
    return float(probs[0, unsafe_id])


def _parse_llama_guard_output(text: str) -> tuple[str, str | None]:
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
