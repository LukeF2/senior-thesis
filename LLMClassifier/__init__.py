import os
import random
from groq import Groq
from typing import Sequence, List, Tuple

DEFAULTS = {
    "dataset": "imdb",
    "n": 100,
    "test_size": 0.30,
    "seed": 42,
    "models": ["llama-3.1-8b-instant"],
    "temperatures": [0.0, 0.3],
    "nshots": [0, 1, 2, 3, 5],
    "labels": ("positive", "negative"),
    "metrics": ["accuracy", "precision", "recall", "f1"],
    "out_dir": "img",
    "smooth": 0,  # moving-average window; 0/1 = off
}

__all__ = ["DEFAULTS"]


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
        self.max_tokens = max_tokens
    
        self.examples: List[Tuple[str, str]] = []  # (text, label)
 
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self._rng = random.Random(seed if seed is not None else 47)
    
        
     
    def fit(self, X, y):
        # selects the examples, up to noshots examples per label from (X, y)
        if self.nshots <= 0:
            self.examples = []
            return self
    

        by_label: dict[str, List[int]] = {lab: [] for lab in self.labels}
        for i, lab in enumerate(y):
            if lab in by_label:
                by_label[lab].append(i)

        selected: List[Tuple[str, str]] = []
        for lab in self.labels:
            idxs = by_label.get(lab, [])
            idxs = idxs.copy()
            self._rng.shuffle(idxs)
            take = idxs[: self.nshots]
            for i in take:
                selected.append((X[i], lab))

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
        messages = self._build_messages(text)
        out = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        ).choices[0].message.content.strip()

        out_low = out.lower()
        for L in self.labels:
            if out_low == L.lower() or L.lower() in out_low:
                return L
        return self.labels[0]
