import os
import random
from groq import Groq


class LLM:
    def __init__(self, labels=("positive", "negative"), model="llama-3.1-8b-instant", nshots=0):
        """
        -labels: list of allowed outputs
        -model: groq model to be used
        -nshots: how many examples to be included in the prompt
        """
        self.labels = list(labels)
        self.model = model
        self.nshots = nshots
        self.examples = []
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
        
     
    def fit(self, X, y):
        # selects the examples, up to noshots examples per label from (X, y)
        if self.nshots <= 0:
            self.examples = []
            return self
        # collecting examples by label
        examples_label = {label: [] for label in self.labels}
        for text, label in zip(X, y):
            if label in examples_label:
                examples_label[label].append(text)
    
        random.seed(47)
        selected = []
        for label in self.labels:
            if len(examples_label[label]) >= self.nshots:
                samples = examples_label[label][:self.nshots]
            else:
                samples = examples_label[label]

            for text in samples:
                selected.append((text, label))
    
        self.examples = selected
        return self  
 
    def infer(self, text):
        """
        Predicts the label for a single text.
        """

        # 1. SYSTEM - instructions + output
        sys_msg = (
            "You are a strict sentiment classifier.\n"
            f"Allowed labels: {', '.join(self.labels)}.\n"
            "Reply with ONLY ONE WORD: Exactly ONE of the allowed labels."
        )
        
        # 2. Messages: system + user (zero shot?)
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": text},
        ]
        
        # 3. Call groq
        out = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
        ).choices[0].message.content.strip()
        
        # 4 Normalize to one of the allowed labels
        out_low = out.lower()
        for L in self.labels:
            if out_low == L.lower() or L.lower() in out_low:
                return L
        return labels[0]
    
