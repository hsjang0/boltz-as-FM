import os
from typing import Tuple

import numpy as np


class IdentityScaler:
    """Lightweight no-op scaler used when input features are already normalized."""

    def fit(self, x):
        return self

    def transform(self, x):
        return x

    def inverse_transform(self, x):
        return x


def _safe_load_npz(dataset_name: str, data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_path = os.path.join(data_dir, f"{dataset_name}_train.npz")
    test_path = os.path.join(data_dir, f"{dataset_name}_test.npz")
    try:
        train_set = np.load(train_path)
        test_set = np.load(test_path)
    except FileNotFoundError as exc:
        raise SystemExit("Missing Dataset") from exc

    X_train = train_set["x"]
    y_train = train_set["y"][:, :1].reshape(-1)
    X_test = test_set["x"]
    y_test = test_set["y"][:, :1].reshape(-1)
    return X_train, y_train, X_test, y_test



def load_dataset(
    dataset_name: str,
    log_y: bool,
    task: str,
    data_dir: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, y_train, X_test, y_test = _safe_load_npz(dataset_name, data_dir)

    if log_y and task == "regression":
        y_train = np.log(y_train + 1e-3)
        y_test = np.log(y_test + 1e-3)

    if task == "classification":
        y_train = y_train.astype(np.int32)
        y_test = y_test.astype(np.int32)

    return X_train, y_train, X_test, y_test
