import os
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import load_dataset, concatenate_datasets
from collections import Counter
from sentiment_infer import LLM



def load_imdb_subset(n=100):
    """
    Return a balanced subset of IMDB train: n//2 negatives + n//2 positives, shuffled.
    """
    ds = load_dataset("imdb", split="train", cache_dir="./.hf_cache")
    k = n // 2

    neg = ds.filter(lambda ex: ex["label"] == 0).select(range(k))
    pos = ds.filter(lambda ex: ex["label"] == 1).select(range(k))

    small = concatenate_datasets([neg, pos]).shuffle(seed=42)

    X = small["text"]
    y = ["positive" if int(lbl) == 1 else "negative" for lbl in small["label"]]
    return X, y

def main():
    # 1) Data (balanced 100)
    X, y = load_imdb_subset(n=100)

    # 2) Stratified split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    print("Train:", Counter(y_tr))
    print("Test: ", Counter(y_te))

    # 3) Model, few shot with 3 examples per label in prompt
    clf = LLM(labels=("positive", "negative"), model="llama-3.1-8b-instant", nshots=3)
    clf.fit(X_tr, y_tr)

    # 4) Predict
    y_pred = [clf.infer(x) for x in X_te]

    # 5) Metrics
    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, pos_label="positive", zero_division=0)
    rec  = recall_score(y_te, y_pred,  pos_label="positive", zero_division=0)
    f1   = f1_score(y_te, y_pred,      pos_label="positive", zero_division=0)

    print("\n--- IMDB (n=100) — Results ---")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1:        {f1:.3f}")


if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit("Please set GROQ_API_KEY in your environment.")
    main()
