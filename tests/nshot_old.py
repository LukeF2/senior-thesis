import os
import json
import argparse
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset, concatenate_datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# import from your package (you said everything lives in __init__.py)
from LLMClassifier import DEFAULTS, LLM

CONFIG = {
    "dataset": "imdb",          # later you can add "sst2", etc.
    "n": 100,                   # total examples (balanced)
    "test_size": 0.30,
    "seed": 42,                 # master seed for deterministic runs
    "models": ["llama-3.1-8b-instant"],
    "temperatures": [0.0, 0.3], # sweep temps
    "nshots": [0, 1, 2, 3, 5],  # x-axis
    "labels": ("positive", "negative"),
    "metrics": ["accuracy", "precision", "recall", "f1"],
    "out_dir": "img",
    "smooth": 0,                # moving-average window (0 = off)
}


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def git_short_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "no-git"

def ensure_outdir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def moving_average(x, k):
    if not k or k <= 1:
        return x
    out = []
    for i in range(len(x)):
        lo = max(0, i - (k - 1))
        out.append(sum(x[lo:i+1]) / len(x[lo:i+1]))
    return out

def load_imdb_subset(n=100, seed=42):
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

DATASETS = {
    'imdb': load_imdb_subset,
}

# 4 metrics

def compute_metrics(y_true, y_pred, positive_label="positive"):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0),
        "recall":    recall_score(y_true, y_pred,  pos_label=positive_label, zero_division=0),
        "f1":        f1_score(y_true, y_pred,      pos_label=positive_label, zero_division=0),
    }


# -----argparse------

def parse_args():
    d = DEFAULTS
    p = argparse.ArgumentParser(
	description="N-shot sweep and metric plots"
    )
    p.add_argument("--dataset", default=d["dataset"], choices=list(DATASET.keys()))
    p.add_argument("--n", type=int, default=d["n"], help="Total examples (balanced for IMDB).")
    p.add_argument("--test-size", type=float, default=d["test_size"], help="Test fraction.")
    p.add_argument("--seed", type=int, default=d["seed"], help="Master seed.")
    p.add_argument("--out-dir", default=d["out_dir"], help="Output folder for images/JSON.")
    p.add_argument("--model", dest="models", nargs="*", default=d["models"],
                   help="One or more models, e.g. --model llama-3.1-8b-instant llama-3.1-70b")
    p.add_argument("--temperature", nargs="*", type=float, default=d["temperatures"],
                   help="One or more temperatures, e.g. --temperature 0 0.3 0.7")
    p.add_argument("--nshots", nargs="*", type=int, default=d["nshots"],
                   help="One or more n-shot values, e.g. --nshots 0 1 2 3 5")

    p.add_argument("--metrics", nargs="*", default=d["metrics"],
                   choices=["accuracy", "precision", "recall", "f1"],
                   help="Which metrics to compute/plot.")
    p.add_argument("--labels", nargs="*", default=list(d["labels"]),
                   help="Ordered labels, e.g. --labels negative positive")

    p.add_argument("--smooth", type=int, default=d["smooth"],
                   help="Moving-average window size; 0/1 disables smoothing.")
    return p.parse_args()


def run_and_plot(args):
    if args.dataset not in DATASETS:
        raise SystemExit(f"Unsupported dataset '{args.dataset}'. Options: {list(DATASETS)}")
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit("Please set GROQ_API_KEY in your environment.")

    set_all_seeds(cfg["seed"])

    # Load + split
    X, y = DATASETS[cfg["dataset"]](n=cfg["n"], seed=cfg["seed"])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg["test_size"], random_state=cfg["seed"], stratify=y
    )
    print("Train:", Counter(y_tr))
    print("Test: ", Counter(y_te))

    # results[metric][(model, temp)] = list aligned with cfg["nshots"]
    results = {m: defaultdict(list) for m in cfg["metrics"]}

    # Sweep grid and collect metric series
    for model in cfg["models"]:
        for temp in cfg["temperatures"]:
            key = (model, temp)
            for nshot in cfg["nshots"]:
                # make each grid point deterministic
                set_all_seeds(cfg["seed"])

                clf = LLM(
                    labels=tuple(cfg["labels"]),
                    model=model,
                    nshots=int(nshot),
                    temperature=float(temp),
                    seed=cfg["seed"],
                )
                clf.fit(X_tr, y_tr)
                y_pred = [clf.infer(x) for x in X_te]

                m = compute_metrics(y_te, y_pred, positive_label=cfg["labels"][1])
                for metric_name in cfg["metrics"]:
                    results[metric_name][key].append(m[metric_name])

                # quick console log
                print(f"[{model}] T={temp} nshots={nshot} -> " +
                      " ".join(f"{mn}={results[mn][key][-1]:.3f}" for mn in cfg["metrics"]))

    # Plot & save (one output file per metric)
    ensure_outdir(cfg["out_dir"])
    gh = git_short_hash()

    common_params = {
        "dataset": cfg["dataset"],
        "models": cfg["models"],
        "temps": cfg["temperatures"],
        "nshots": cfg["nshots"],
        "seed": cfg["seed"],
        "n": cfg["n"],
        "test_size": cfg["test_size"],
    }

    for metric_name, series in results.items():
        plt.figure()
        for (model, temp), values in series.items():
            ys = moving_average(values, cfg["smooth"])
            label = f"{model}, T={temp}"
            plt.plot(cfg["nshots"], ys, marker="o", label=label)

        plt.xlabel("# shots (nshots)")
        plt.ylabel(metric_name.title())
        plt.title(f"{metric_name.title()} vs N-shots — {cfg['dataset']}")
        plt.legend()
        plt.grid(True, alpha=0.3)

        fname = (
            f"{metric_name}"
            f"__dataset={cfg['dataset']}"
            f"__models={'-'.join(cfg['models'])}"
            f"__temps={','.join(map(str, cfg['temperatures']))}"
            f"__nshots={','.join(map(str, cfg['nshots']))}"
            f"__seed={cfg['seed']}"
            f"__n={cfg['n']}"
            f"__hash={gh}.png"
        ).replace(" ", "")
        out_path = os.path.join(cfg["out_dir"], fname)
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"Wrote: {out_path}")

        # raw numbers as JSON for reproducibility
        json_path = out_path.replace(".png", ".json")
        serializable = {f"{k[0]}|T={k[1]}": v for k, v in series.items()}
        with open(json_path, "w") as f:
            json.dump({
                "metric": metric_name,
                "nshots": cfg["nshots"],
                "series": serializable,
                "params": common_params,
                "git_hash": gh,
            }, f, indent=2)
        print(f"Wrote: {json_path}")

def main():
    args = parse_args()          # get CLI args
    # make sure labels is a tuple for LLM
    if isinstance(args.labels, list):
        args.labels = tuple(args.labels)
    run_and_plot(args)

if __name__ == "__main__":
    run_and_plot()
