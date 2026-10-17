"""DeepSeek chat backend with a hard budget ledger, plus an offline mock.

The API key is read from the environment only.  It is never logged, never
written to a result file and never committed -- the repository ships a
``.env.example`` naming the variable, not its value.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"


class BudgetExceeded(RuntimeError):
    pass


class TruncatedResponse(RuntimeError):
    """The provider stopped at the output-token limit, so the JSON is incomplete."""



@dataclass
class Ledger:
    """Append-only record of every request, with a hard spend cap."""

    path: Path
    cap_usd: float = 5.0
    input_usd_per_mtok: float = 0.27
    output_usd_per_mtok: float = 1.10
    calls: list = field(default_factory=list)

    def spent(self) -> float:
        return sum(c["cost_usd"] for c in self.calls)

    def record(self, prompt_tokens: int, completion_tokens: int, tag: str) -> float:
        cost = (
            prompt_tokens / 1e6 * self.input_usd_per_mtok
            + completion_tokens / 1e6 * self.output_usd_per_mtok
        )
        self.calls.append(
            {
                "tag": tag,
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "cost_usd": cost,
                "ts": time.time(),
            }
        )
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"cap_usd": self.cap_usd, "spent_usd": self.spent(),
                                             "n_calls": len(self.calls), "calls": self.calls}, indent=2))
        if self.spent() > self.cap_usd:
            raise BudgetExceeded(f"spent {self.spent():.4f} USD > cap {self.cap_usd}")
        return cost


class DeepSeekBackend:
    """Minimal OpenAI-compatible chat client.  No automatic retry on failure."""

    kind = "deepseek"

    def __init__(self, ledger: Ledger, model: str = DEFAULT_MODEL,
                 base_url: str = DEFAULT_BASE_URL, api_key_env: str = "DEEPSEEK_API_KEY",
                 max_output_tokens: int = 8192, timeout: int = 180):
        key = os.environ.get(api_key_env, "").strip()
        if not key:
            raise RuntimeError(f"{api_key_env} is not set; no request was made")
        self._key = key
        self.ledger = ledger
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        import requests

        self._session = requests.Session()

    def chat(self, messages, tag: str = "", temperature: float = 0.0,
             response_format: dict | None = None) -> str:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_output_tokens,
            "stream": False,
        }
        if response_format:
            body["response_format"] = response_format
        response = self._session.post(
            f"{self.base_url}/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {self._key}"},
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(f"provider returned HTTP {response.status_code}; no retry attempted")
        payload = response.json()
        usage = payload.get("usage", {})
        self.ledger.record(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), tag)
        choice = payload["choices"][0]
        reason = choice.get("finish_reason")
        if reason == "length":
            raise TruncatedResponse(
                f"reply hit the {self.max_output_tokens}-token output limit and is incomplete")
        if reason != "stop":
            raise RuntimeError(f"unexpected finish_reason {reason}")
        return choice["message"]["content"]


class MockBackend:
    """Deterministic stand-in so the whole pipeline runs with zero network.

    It mimics the *shape* of an agent response (hypotheses + a selection drawn
    from the shortlist with an exploration tail); it is an engineering fixture
    and carries no scientific meaning.
    """

    kind = "mock"

    def __init__(self, *_, **__):
        self.ledger = Ledger(path=None, cap_usd=0.0)

    def chat(self, messages, tag: str = "", **__) -> str:
        payload = json.loads(messages[-1]["content"])
        if "constraint_updates" in (payload.get("required_output") or {}):
            # rule-setting protocol: return a deterministic, schema-valid rule update
            return json.dumps({
                "hypotheses": [{"statement": "Fixture: large side chains at site 40 help.",
                                "site": 40, "substitution": "W", "direction": "beneficial"}],
                "constraint_updates": [{"id": "C6_exploration_quota",
                                        "params": {"fraction": 0.25},
                                        "reason": "fixture: keep the exploration quota"}],
                "weights": {"prior": 1.0, "model": 1.0},
                "rationale": "deterministic fixture, no scientific content",
            })
        shortlist = [c["variant"] for c in payload["shortlist"]]
        k = payload["batch_size"]
        exploit = shortlist[: int(k * 0.75)]
        tail = shortlist[int(k * 0.75):][::-1][: k - len(exploit)]
        return json.dumps(
            {
                "hypotheses": [
                    {"statement": "Deterministic fixture: favour model-ranked candidates "
                                  "while keeping a diverse tail.", "site": 40, "direction": "unknown"}
                ],
                "selection": exploit + tail,
                "reasons": {v: "fixture selection" for v in exploit[:3]},
            }
        )
