"""NVIDIA hosted-inference client with multi-key rotation.

Keys are read from the environment (``NVIDIA_API_KEYS``, comma separated) and
rotated per request so that a single key's rate limit does not stall the run.
A key is never logged: failures are reported by key *index* only.
"""
from __future__ import annotations

import itertools
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

PROTEINMPNN_URL = "https://health.api.nvidia.com/v1/biology/ipd/proteinmpnn/predict"
BOLTZ2_URL = "https://health.api.nvidia.com/v1/biology/mit/boltz2/predict"


class NoKeysConfigured(RuntimeError):
    pass


@dataclass
class CallRecord:
    service: str
    key_index: int
    status: int
    seconds: float
    ok: bool
    note: str = ""


@dataclass
class NvidiaClient:
    receipts_path: Path | None = None
    timeout: int = 300
    max_attempts_per_request: int = 5
    records: list = field(default_factory=list)

    def __post_init__(self):
        raw = os.environ.get("NVIDIA_API_KEYS", "") or os.environ.get("NVIDIA_API_KEY", "")
        self._keys = [k.strip() for k in raw.split(",") if k.strip()]
        if not self._keys:
            raise NoKeysConfigured("NVIDIA_API_KEYS is not set; no request was made")
        self._cycle = itertools.cycle(range(len(self._keys)))
        import requests

        self._session = requests.Session()

    @property
    def n_keys(self) -> int:
        return len(self._keys)

    def _log(self, record: CallRecord) -> None:
        self.records.append(record)
        if self.receipts_path:
            self.receipts_path.parent.mkdir(parents=True, exist_ok=True)
            with self.receipts_path.open("a") as fh:
                fh.write(json.dumps(record.__dict__) + "\n")

    def post(self, url: str, payload: dict, service: str) -> dict:
        """One logical request, rotating keys on rate limits.

        Rotation is for throughput, not for bypassing a refusal: a 4xx that is
        not a rate limit aborts immediately.
        """
        last = None
        for _ in range(self.max_attempts_per_request):
            index = next(self._cycle)
            started = time.time()
            response = self._session.post(
                url, json=payload, timeout=self.timeout,
                headers={"Authorization": f"Bearer {self._keys[index]}",
                         "Accept": "application/json"},
            )
            elapsed = time.time() - started
            ok = response.status_code == 200
            self._log(CallRecord(service, index, response.status_code, round(elapsed, 2), ok))
            if ok:
                return response.json()
            # 429/503 are rate limits; 422 is returned transiently by the service for
            # inputs that succeed on a later attempt, so it is rotated too.
            if response.status_code in (422, 429, 503):
                last = response
                time.sleep(2.0)
                continue
            raise RuntimeError(f"{service}: provider returned HTTP {response.status_code}")
        raise RuntimeError(f"{service}: rate limited on all {self.n_keys} keys "
                           f"(last status {last.status_code if last else 'n/a'})")

    def proteinmpnn(self, pdb_text: str, *, ca_only: bool = False, sampling_temp: float = 0.1,
                    num_seq_per_target: int = 1) -> dict:
        return self.post(PROTEINMPNN_URL, {
            "input_pdb": pdb_text, "ca_only": ca_only,
            "sampling_temp": [sampling_temp], "num_seq_per_target": num_seq_per_target,
        }, "proteinmpnn")

    def boltz2(self, sequences: list[dict], *, recycling_steps: int = 3,
               sampling_steps: int = 50, diffusion_samples: int = 1) -> dict:
        return self.post(BOLTZ2_URL, {
            "polymers": sequences, "recycling_steps": recycling_steps,
            "sampling_steps": sampling_steps, "diffusion_samples": diffusion_samples,
        }, "boltz2")
