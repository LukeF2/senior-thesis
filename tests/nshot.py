#!/usr/bin/env python3

import csv
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

from LLMClassifier import DEFAULTS, LLM

import re

def sanitize_for_filename(s: str) -> str:
    """Replace unsafe filename characters with underscores."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)

def write_usage_row(csv_path, row):
   """
   Appends or creates a CSV w/ token and cost accounting per grid point.
   """
   os.makedirs(os.path.dirname(csv_path), exist_ok=True)
   file_exists = os.path.exists(csv_path)
   fieldnames = [
       "dataset", "n", "test_size", "seed",
       "model", "temperature", "nshots",
       "prompt_tokens", "completion_tokens", "total_tokens", "cost",
       "git_hash",
   ]
   with open(csv_path, "a", newline="") as f:
       w = csv.DictWriter(f, fieldnames=fieldnames)
       if not file_exists:
           w.writeheader()
       w.writerow(row)

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
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
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


# ---------------- datasets -----------------

def load_imdb_subset(n=100, seed=42):
    ds = load_dataset("imdb", split="train", cache_dir="./.hf_cache")
    k = n // 2
    neg = ds.filter(lambda ex: ex["label"] == 0).select(range(k))
    pos = ds.filter(lambda ex: ex["label"] == 1).select(range(k))
    small = concatenate_datasets([neg, pos]).shuffle(seed=seed)
    X = small["text"]
    y = ["positive" if int(lbl) == 1 else "negative" for lbl in small["label"]]
    return X, y

DATASETS = {
    "imdb": load_imdb_subset,
    # "sst2": load_sst2_subset,  # add later if you implement it
}


# ---------------- metrics ------------------

def compute_metrics(y_true, y_pred, positive_label="positive"):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0),
        "recall":    recall_score(y_true, y_pred,  pos_label=positive_label, zero_division=0),
        "f1":        f1_score(y_true, y_pred,      pos_label=positive_label, zero_division=0),
    }


# --------------- argparse ------------------

def parse_args():
    d = DEFAULTS
    p = argparse.ArgumentParser(description="N-shot sweep and metric plots")
    p.add_argument("--dataset", default=d["dataset"], choices=list(DATASETS.keys()))
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


# ----------------- run/plot ----------------

def run_and_plot(args):
    if args.dataset not in DATASETS:
        raise SystemExit(f"Unsupported dataset '{args.dataset}'. Options: {list(DATASETS)}")
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit("Please set GROQ_API_KEY in your environment.")

    # determinism
    set_all_seeds(args.seed)

    # data
    X, y = DATASETS[args.dataset](n=args.n, seed=args.seed)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    print("Train:", Counter(y_tr))
    print("Test: ", Counter(y_te))
    
    ensure_outdir(args.out_dir)
    png_dir  = os.path.join(args.out_dir, "png")
    json_dir = os.path.join(args.out_dir, "json")
    logs_dir = os.path.join(args.out_dir, "logs")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    gh = git_short_hash()

    # results[metric][(model, temp)] = list aligned with args.nshots
    results = {m: defaultdict(list) for m in args.metrics}

    # sweep grid
    for model in args.models:
        for temp in args.temperature:
            key = (model, temp)
            for nshot in args.nshots:
                set_all_seeds(args.seed)  # deterministic per grid point

                clf = LLM(
                    labels=tuple(args.labels),
                    model=model,
                    nshots=int(nshot),
                    temperature=float(temp),
                    seed=args.seed,
                )
                clf.fit(X_tr, y_tr)
                y_pred = [clf.infer(x) for x in X_te]

                m = compute_metrics(y_te, y_pred, positive_label=args.labels[-1])
                for metric_name in args.metrics:
                    results[metric_name][key].append(m[metric_name])

                print(f"[{model}] T={temp} nshots={nshot} -> " +
                      " ".join(f"{mn}={results[mn][key][-1]:.3f}" for mn in args.metrics))

# --- NEW: write usage/cost row per grid point ---
                usage_csv = os.path.join(args.out_dir, "token_costs.csv")
                write_usage_row(usage_csv, {
                        "dataset": args.dataset,
                        "n": args.n,
                        "test_size": args.test_size,
                        "seed": args.seed,
                        "model": model,
                        "temperature": temp,
                        "nshots": nshot,
                        "prompt_tokens": getattr(clf, "prompt_tokens", 0),
                        "completion_tokens": getattr(clf, "completion_tokens", 0),
                        "total_tokens": getattr(clf, "total_tokens", 0),
                        "cost": f"{getattr(clf, 'total_cost', 0.0):.6f}",
                        "git_hash": gh,
    })

    # plotting
    safe_models = "-".join(sanitize_for_filename(m) for m in args.models)
    safe_temps  = ",".join(map(str, args.temperature))
    safe_nshots = ",".join(map(str, args.nshots))
    
    for metric_name, series in results.items():
        plt.figure()
        for (model, temp), values in series.items():
            ys = moving_average(values, args.smooth)
            label = f"{model}, T={temp}"
            plt.plot(args.nshots, ys, marker="o", label=label)

        plt.xlabel("# shots (nshots)")
        plt.ylabel(metric_name.title())
        plt.title(f"{metric_name.title()} vs N-shots — {args.dataset}")
        plt.legend()
        plt.grid(True, alpha=0.3)


        base = (
            f"{metric_name}"
            f"__dataset={sanitize_for_filename(args.dataset)}"
            f"__models={safe_models}"
            f"__temps={safe_temps}"
            f"__nshots={safe_nshots}"
            f"__seed={args.seed}"
            f"__n={args.n}"
            f"__hash={gh}.png"
        ).replace(" ", "")

        # save PNG
        png_path = os.path.join(png_dir, base + ".png")
        os.makedirs(os.path.dirname(png_path), exist_ok=True)
        plt.savefig(png_path, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"Wrote: {png_path}")

        json_path = os.path.join(json_dir, base + ".json")
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w") as f:
            json.dump({
                "metric": metric_name,
                "nshots": args.nshots,
                "series": {f"{k[0]}|T={k[1]}": v for k, v in series.items()},
                "params": {
                    "dataset": args.dataset,
                    "models": args.models,
                    "temps": args.temperature,
                    "nshots": args.nshots,
                    "seed": args.seed,
                    "n": args.n,
                    "test_size": args.test_size,
                },
                "git_hash": gh,
            }, f, indent=2)
        print(f"Wrote: {json_path}")

def main():
    args = parse_args()
    # ensure tuple for LLM constructor
    if isinstance(args.labels, list):
        args.labels = tuple(args.labels)
    from datetime import datetime
    import sys, os
    

    log_dir = os.path.join(args.out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    run_log = f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    run_log_path = os.path.join(log_dir, run_log)
    sys.stdout = open(run_log_path, "w")
    sys.stderr = sys.stdout
    print(f"Logging to {run_log_path}\n")
    
    run_and_plot(args)


if __name__ == "__main__":
    main()

