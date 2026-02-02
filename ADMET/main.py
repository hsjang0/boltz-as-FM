import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch
from sklearn.model_selection import KFold
import yaml
from utils.data_utils import load_dataset
from utils.constants import ALLOWED_DATASET_NAMES, DEFAULT_TRAINING_PARAMS
from utils.training_utils import (
    build_result_blob,
    evaluate_multi_runs_on_test,
    hp_filename,
    train_with_selection_return_model_and_scalers,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config file.")
    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()
    return args


def _load_config_from_file(config_file: str) -> Dict:
    with open(config_file, "r", encoding="utf-8") as f:
        content = f.read()
    loaded = yaml.safe_load(content)
    return loaded


def aggregate_metrics(test_summaries: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    metrics_mean_std: Dict[str, Dict[str, float]] = {}
    if not test_summaries:
        return metrics_mean_std
    keys = test_summaries[0].keys()
    for k in keys:
        vals = [ts[k] for ts in test_summaries]
        metrics_mean_std[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return metrics_mean_std


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = _load_config_from_file(args.config)
    config.update(DEFAULT_TRAINING_PARAMS)
    if args.data_dir:
        config["data_dir"] = args.data_dir
    config.setdefault("results_dir", "results")


    if config["dataset_name"] is None:
        raise SystemExit("dataset_name must be set via config.")
    if config["dataset_name"] not in ALLOWED_DATASET_NAMES:
        allowed = ", ".join(sorted(ALLOWED_DATASET_NAMES))
        raise SystemExit(f"dataset_name '{config['dataset_name']}' is invalid. Choose from: {allowed}.")


    X_train, y_train, X_test, y_test = load_dataset(
        config["dataset_name"], config["log_y"], config["task"], config["data_dir"]
    )
    seeds = [int(s) for s in config["test_seeds"].split(",") if s.strip()]
    all_test_summaries: List[Dict[str, float]] = []
    last_seed_fold_best_vals: List[float] = []


    for seed in seeds:
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        multi_runs = []
        fold_best_vals: List[float] = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_va, y_va = X_train[val_idx], y_train[val_idx]
            model, x_scaler, y_scaler, fold_best_val = train_with_selection_return_model_and_scalers(
                X_tr, y_tr, X_va, y_va, config, device, seed, fold_idx
            )
            multi_runs.append((model, x_scaler, y_scaler))
            fold_best_vals.append(fold_best_val)
        last_seed_fold_best_vals = fold_best_vals

        test_summary = evaluate_multi_runs_on_test(multi_runs, X_test, y_test, config, device)
        all_test_summaries.append(test_summary)


    metrics_mean_std = aggregate_metrics(all_test_summaries)
    print("Validation metrics:", metrics_mean_std)

    out_dir = os.path.join(config["results_dir"], config["dataset_name"])
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, hp_filename(config))
    
    best_vals_mean_last_seed = float(np.mean(last_seed_fold_best_vals)) if last_seed_fold_best_vals else float("nan")
    result_blob = build_result_blob(config, all_test_summaries, metrics_mean_std, seeds, best_vals_mean_last_seed)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(result_blob, ensure_ascii=False, indent=2))
    print("\nResults saved: ", out_file)
    print(json.dumps(result_blob, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
