import os
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sentiment_infer import LLM

def load_imdb_subset(n=100, seed=42):
    ds = load_dataset("imdb")
    X = ds["train"]["text"][:n]
    y = ["positive" if lbl == 1 else "negative" for lbl in ds["train"]["label"][:n]]
    return X, y

def main():
    # 1) Data (100 samples)
    X, y = load_imdb_subset(n=100)

    # 2) Split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # 3) Model (no few-shot examples; nshots=0)
    clf = LLM(labels=("positive", "negative"), model="llama-3.1-8b-instant", nshots=0)
    clf.fit(X_tr, y_tr)   # still fine to call; it won’t include examples in the prompt

    # 4) Predict
    y_pred = [clf.infer(x) for x in X_te]

    # 5) Metrics: accuracy + 3 others
    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, pos_label="positive")
    rec  = recall_score(y_te, y_pred,  pos_label="positive")
    f1   = f1_score(y_te, y_pred,      pos_label="positive")

    print("\n--- IMDB (n=100) — Results ---")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1:        {f1:.3f}")

if __name__ == "__main__":
    # Ensure your Groq API key is available
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit("Please set GROQ_API_KEY in your environment.")
    main()
