import os
import time
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from sklearn.preprocessing import StandardScaler

from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .data_utils import IdentityScaler
from .modeling import MLP, make_scheduler


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def hp_filename(config) -> str:
    def fmt(x):
        return str(x).replace(".", "p").replace(",", "_")
    return (
        f"lr{fmt(config['lr'])}_hid{config['hidden']}_layers{config['layers']}_ep{config['epochs']}.txt"
    )


@torch.no_grad()
def evaluate_val_metric(
    model: nn.Module,
    loader: DataLoader,
    task: str,
    metric: str,
    y_scaler: Optional[StandardScaler],
) -> Tuple[float, bool]:
    model.eval()

    if task == "classification":
        logits_all, y_all = [], []
        total_loss, n = 0.0, 0
        bce = nn.BCEWithLogitsLoss()

        for xb, yb in loader:
            logits = model(xb)
            loss = bce(logits, yb.float())
            bs = xb.size(0)
            total_loss += loss.item() * bs
            n += bs

            logits_all.append(logits.detach().cpu().numpy().ravel())
            y_all.append(yb.detach().cpu().numpy().astype(int).ravel())

        logits_all = np.concatenate(logits_all) if logits_all else np.array([])
        y_all = np.concatenate(y_all) if y_all else np.array([])

        if metric == "auroc":
            if len(np.unique(y_all)) < 2:
                return (total_loss / max(1, n)), False
            probs = 1.0 / (1.0 + np.exp(-logits_all))
            try:
                auroc = roc_auc_score(y_all, probs)
            except ValueError:
                return (total_loss / max(1, n)), False
            return float(auroc), True

        if metric == "aurpc":
            if len(np.unique(y_all)) < 2:
                return (total_loss / max(1, n)), False
            probs = 1.0 / (1.0 + np.exp(-logits_all))
            try:
                auprc = average_precision_score(y_all, probs)
            except ValueError:
                return (total_loss / max(1, n)), False
            return float(auprc), True

        return (total_loss / max(1, n)), False

    preds, trues = [], []
    for xb, yb_s in loader:
        y_pred_s = model(xb).detach().cpu().numpy().ravel()
        y_true_s = yb_s.detach().cpu().numpy().ravel()

        if y_scaler is not None:
            y_pred = y_scaler.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
            y_true = y_scaler.inverse_transform(y_true_s.reshape(-1, 1)).ravel()
        else:
            y_pred, y_true = y_pred_s, y_true_s

        preds.append(y_pred)
        trues.append(y_true)

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)

    if metric == "mae":
        return float(mean_absolute_error(y_true, y_pred)), False
    if metric == "spearman":
        s = spearmanr(y_true, y_pred).correlation
        s = -1.0 if np.isnan(s) else s
        return float(s), True

    mse = float(np.mean((y_true - y_pred) ** 2))
    return mse, False


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, task: str):
    model.train()
    criterion = nn.MSELoss() if task == "regression" else nn.BCEWithLogitsLoss()
    for xb, yb in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb.float() if task == "classification" else yb)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def predict_with_model(model: nn.Module, X_t: torch.Tensor, task: str) -> np.ndarray:
    model.eval()
    out = model(X_t)
    if task == "classification":
        out = torch.sigmoid(out)
    return out.detach().cpu().numpy().ravel()


def train_with_selection_return_model_and_scalers(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    config,
    device: str,
    seed: int,
    fold_idx: Optional[int] = None,
) -> Tuple[nn.Module, IdentityScaler, Optional[StandardScaler], float]:
    set_global_seed(seed)

    x_scaler = IdentityScaler()
    X_tr_s = x_scaler.transform(X_tr)
    X_va_s = x_scaler.transform(X_va)

    if config["task"] == "regression":
        y_tv = np.concatenate([y_tr, y_va]).reshape(-1, 1)
        y_scaler = StandardScaler().fit(y_tv)
        y_tr_s = y_scaler.transform(y_tr.reshape(-1, 1)).ravel()
        y_va_s = y_scaler.transform(y_va.reshape(-1, 1)).ravel()
    else:
        y_scaler = None
        y_tr_s = y_tr
        y_va_s = y_va

    X_tr_t = torch.tensor(X_tr_s, dtype=torch.float32, device=device)
    X_va_t = torch.tensor(X_va_s, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr_s, dtype=torch.float32, device=device)
    y_va_t = torch.tensor(y_va_s, dtype=torch.float32, device=device)

    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t), batch_size=config["batch_size"], shuffle=True, pin_memory=False
    )
    val_loader = DataLoader(
        TensorDataset(X_va_t, y_va_t), batch_size=config["batch_size"], shuffle=False, pin_memory=False
    )

    model = MLP(
        input_dim=X_tr_t.shape[1],
        hidden=config["hidden"],
        layers=config["layers"],
        dropout=config["dropout"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = make_scheduler(optimizer, total_epochs=config["epochs"], warmup=config["warm_up"])

    if config["task"] == "classification" and config["metric"] not in ("auroc", "aurpc"):
        print("[Info] classification: using BCE(val loss) for selection. Set --metric auroc to select by AUROC.")


    best_score: Optional[float] = None
    best_state = None
    higher_is_better: Optional[bool] = None
    no_improve = 0


    fold_display = fold_idx + 1 if fold_idx is not None else None
    epoch_desc = f"Seed {seed}" + (f" | Fold {fold_display}" if fold_display is not None else "") + " | Epochs"

    for epoch in tqdm(range(1, config["epochs"] + 1), desc=epoch_desc, unit="epoch"):
        train_one_epoch(model, train_loader, optimizer, config["task"])
        scheduler.step()

        cur_score, hib = evaluate_val_metric(
            model=model, loader=val_loader, task=config["task"], metric=config["metric"], y_scaler=y_scaler
        )
        if higher_is_better is None:
            higher_is_better = hib

        improved = (best_score is None) or (cur_score > best_score if higher_is_better else cur_score < best_score)
        if improved:
            best_score = cur_score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= config["patience"]:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    best_val_for_selection = -best_score if (higher_is_better is True) else float(best_score)
    return model, x_scaler, y_scaler, best_val_for_selection


def evaluate_multi_runs_on_test(
    multi_runs: List[Tuple[nn.Module, IdentityScaler, Optional[StandardScaler]]],
    X_test: np.ndarray,
    y_test: np.ndarray,
    config,
    device: str,
) -> Dict[str, float]:
    preds = []
    for (model, x_scaler, y_scaler) in multi_runs:
        X_te_s = x_scaler.transform(X_test)
        X_te_t = torch.tensor(X_te_s, dtype=torch.float32, device=device)
        pred = predict_with_model(model, X_te_t, config["task"])

        if config["task"] == "classification":
            preds.append(pred)
        else:
            pred_orig = y_scaler.inverse_transform(pred.reshape(-1, 1)).ravel() if y_scaler is not None else pred
            preds.append(pred_orig)

    if config["task"] == "classification":
        eps = 1e-7
        preds = np.stack(preds, axis=1)
        preds = np.clip(preds, eps, 1 - eps)
        preds = np.log(preds / (1 - preds))
        avg_pred = 1 / (1 + np.exp(-np.mean(preds, axis=1)))
        try:
            auroc = roc_auc_score(y_test.astype(int), avg_pred)
            aurpc = average_precision_score(y_test.astype(int), avg_pred)
        except ValueError:
            auroc, aurpc = float("nan"), float("nan")
        return {"auroc": float(auroc), "aurpc": float(aurpc)}

    avg_pred = np.mean(np.stack(preds, axis=1), axis=1)
    if config["log_y"]:
        y_test = np.exp(y_test)
        avg_pred = np.exp(avg_pred)

    sp = spearmanr(y_test, avg_pred).correlation
    sp = 0.0 if np.isnan(sp) else float(sp)
    mae = float(mean_absolute_error(y_test, avg_pred))
    return {"spearman": sp, "mae": mae}


def build_result_blob(
    config,
    all_test_summaries: List[Dict[str, float]],
    metrics_mean_std: Dict[str, Dict[str, float]],
    seeds: List[int],
    best_vals_mean_last_seed: float,
) -> Dict:
    return {
        "dataset": config["dataset_name"],
        "task": config["task"],
        "metric": config["metric"],
        "hyperparameters": {
            "lr": config["lr"],
            "hidden": config["hidden"],
            "layers": config["layers"],
            "epochs": config["epochs"],
        },
        "test_summary_each_seed": all_test_summaries,
        "test_summary_mean_std": metrics_mean_std,
        "seeds": seeds,
        "best_vals_selection_scale_mean_last_seed": best_vals_mean_last_seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
