import os
import random
from typing import Sequence, List, Tuple, Optional
from pathlib import Path

import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DEFAULTS = {
    "dataset": "imdb",
    "n": 100,
    "test_size": 0.30,
    "seed": 42,
    "models": ["meta-llama/llama-3.1-8b-instruct:free"],
    "temperatures": [0.0, 0.3],
    "nshots": [0, 1, 2, 3, 5],
    "labels": ("positive", "negative"),
    "metrics": ["accuracy", "precision", "recall", "f1"],
    "out_dir": "img",
    "smooth": 0,  
}

__all__ = ["DEFAULTS"]

def clip(s: str, limit: int = 256) -> str:
   """
   Truncate to save tokens; adjust the limit as needed.
   """
   return s if len(s) <= limit else s[:limit]

class LLM:
    def __init__(
        self,
        labels: Sequence[str] = ("positive", "negative"),
        model: str = "llama-3.1-8b-instant",
        nshots: int = 0,
        temperature: float = 0.3,
        seed: int | None = None,
        max_tokens: int | None = None,
    ):
        """
        Parameters
        ---------

        -labels: list of allowed outputs
        -model: groq model to be used
        -nshots: how many examples to be included in the prompt
        -temperature: decodign params
        -seed: for deterministic example selection
        """
        self.labels = list(labels)
        self.model = model
        self.nshots = int(nshots)
        self.temperature  = float(temperature)
        self.seed = seed
        self.max_tokens = 4 if max_tokens is None else int(max_tokens)
    
        self.examples: List[Tuple[str, str]] = []  # (text, label)
 
        self._rng = random.Random(seed if seed is not None else 47)
        
        self.base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENROUTER_API_KEY in your .env")
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # Optional (recommended by OpenRouter): helps with telemetry/allow-list
            "HTTP-Referer": "https://github.com/LukeF2/senior-thesis",
            "X-Title": "senior-thesis nshot runner",
        }

        self.examples: List[Tuple[str, str]] = []  # (text, label)
        
        # usage tallies per LLM instance

        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.total_cost = 0.0 
        
     
    def fit(self, X, y):
        # selects the examples, up to noshots examples per label from (X, y)
        if self.nshots <= 0:
            self.examples = []
            return self
    
        by_label = {L: [] for L in self.labels}
        for text, lab in zip(X, y):
            if lab in by_label:
                by_label[lab].append(text)

        # Shuffle each label bucket deterministically
        for L in self.labels:
            self._rng.shuffle(by_label[L])

        # Take first nshots per label
        selected: List[Tuple[str, str]] = []
        for L in self.labels:
            for text in by_label[L][: self.nshots]:
                selected.append((text, L))

        # interleave deterministically
        self._rng.shuffle(selected)
        self.examples = selected
        return self
  
    def _build_messages(self, text: str) -> list[dict]:
        sys_msg = (
            "You are a strict sentiment classifier.\n"
            f"Allowed labels: {', '.join(self.labels)}.\n"
            "Reply with ONLY one of the allowed labels, nothing else."
        )
        msgs = [{"role": "system", "content": sys_msg}]

        for ex_text, ex_label in self.examples:
            msgs.append({"role": "user", "content": ex_text})
            msgs.append({"role": "assistant", "content": ex_label})
        msgs.append({"role": "user", "content": text})
        return msgs

 
    def infer(self, text: str) -> str:

        payload = {
           "model": self.model, 
           "temperature": self.temperature,
           "max_tokens": self.max_tokens,
           "messages": self._build_messages(text)
        }
        
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.HTTPError as e:
            # Return a deterministic fallback label on errors
            return self.labels[0]

        # --- Extract content ---
        content = (
            (data.get("choices") or [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

        # --- Track usage (tokens & cost if present) ---
        usage = data.get("usage", {}) or {}
        self.prompt_tokens += int(usage.get("prompt_tokens", 0))
        self.completion_tokens += int(usage.get("completion_tokens", 0))
        self.total_tokens += int(usage.get("total_tokens", 0))

        # OpenRouter sometimes provides cost in usage or meta; be defensive
        cost = (
            usage.get("total_cost")
            or usage.get("cost")
            or data.get("cost")
            or (data.get("meta", {}) or {}).get("api_total_cost")
        )
        if cost is not None:
            try:
                self.total_cost += float(cost)
            except Exception:
                pass

        # --- Normalize to allowed label ---
        low = content.lower()
        for L in self.labels:
            if low == L.lower() or L.lower() in low:
                return L
        return self.labels[0]
