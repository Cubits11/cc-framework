# Guardrail API

The Guardrail API defines a minimal contract for components that inspect text
and decide whether it should be blocked.  The core of the framework interacts
only with this API so that guardrails can be swapped or composed transparently.

## Interface

```python
class GuardrailAPI(ABC):
    def evaluate(self, text: str) -> tuple[bool, float]:
        """Return (blocked, score) for the supplied text."""

    def calibrate(self, benign_texts: Sequence[str], target_fpr: float = 0.05) -> None:
        """Fit internal thresholds using benign examples."""
```

### `evaluate`
* **Parameters**: `text` – candidate text to assess.
* **Returns**: `(blocked, score)` where
  * `blocked` (`bool`) – `True` if the guardrail rejects the text.
  * `score` (`float`) – risk score in the range `[0, 1]` (higher means more
    risky).

### `calibrate`
* **Parameters**:
  * `benign_texts` – sequence of safe examples used to tune a threshold.
  * `target_fpr` – desired false-positive rate (default `0.05`).
* **Returns**: `None`.

## Adapter Example

```python
from cc.guardrails.keyword_blocker import KeywordBlocker
from cc.core.guardrail_api import GuardrailAdapter

guardrail = KeywordBlocker(["secret"])
api = GuardrailAdapter(guardrail)

# Calibrate on benign data
api.calibrate(["hello world", "nice day"], target_fpr=0.01)

# Evaluate new text
blocked, score = api.evaluate("contains secret token")
print(blocked, score)
```

This wrapper allows the rest of the system to interact with the guardrail
without needing to know its concrete type.