import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import BasePredictionWriter
from torch import Tensor

from boltz.data.types import Record


class BoltzWriter(BasePredictionWriter):
    """Custom writer for predictions."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        output_format: Literal["pdb", "mmcif"] = "mmcif",
        boltz2: bool = False,
        write_embeddings: bool = False,
    ) -> None:
        """Initialize the writer.

        Parameters
        ----------
        output_dir : str
            The directory to save the predictions.

        """
        super().__init__(write_interval="batch")
        if output_format not in ["pdb", "mmcif"]:
            msg = f"Invalid output format: {output_format}"
            raise ValueError(msg)

        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_format = output_format
        self.failed = 0
        self.boltz2 = boltz2
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.write_embeddings = write_embeddings

    def write_on_batch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
        prediction: dict[str, Tensor],
        batch_indices: list[int],  # noqa: ARG002
        batch: dict[str, Tensor],
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int,  # noqa: ARG002
    ) -> None:
        """Write the predictions to disk."""
        if prediction["exception"]:
            self.failed += 1
            return

        if not self.write_embeddings:
            return

        if "s" not in prediction or "z" not in prediction:
            msg = "Predictions are missing required embeddings 's' and 'z'."
            raise KeyError(msg)

        records: list[Record] = batch["record"]
        s_pred = prediction["s"]
        z_pred = prediction["z"]

        s_np = s_pred.detach().cpu().numpy() if torch.is_tensor(s_pred) else np.asarray(s_pred)
        z_np = z_pred.detach().cpu().numpy() if torch.is_tensor(z_pred) else np.asarray(z_pred)

        batch_size = len(records)
        if s_np.shape[0] == batch_size and z_np.shape[0] == batch_size:
            s_iter = s_np
            z_iter = z_np
        elif batch_size == 1:
            s_iter = [s_np]
            z_iter = [z_np]
        else:
            msg = (
                "Shape mismatch between embeddings and records: "
                f"s has shape {s_np.shape}, z has shape {z_np.shape}, "
                f"records: {batch_size}"
            )
            raise ValueError(msg)

        for record, s_item, z_item in zip(records, s_iter, z_iter):
            struct_dir = self.output_dir / record.id
            struct_dir.mkdir(exist_ok=True)
            path = struct_dir / f"embeddings_{record.id}.npz"
            np.savez_compressed(path, s=s_item, z=z_item)

    def on_predict_epoch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
    ) -> None:
        """Print the number of failed examples."""
        # Print number of failed examples
        print(f"Number of failed examples: {self.failed}")  # noqa: T201






class BoltzAffinityWriter(BasePredictionWriter):
    """Custom writer for predictions."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
    ) -> None:
        """Initialize the writer.

        Parameters
        ----------
        output_dir : str
            The directory to save the predictions.

        """
        super().__init__(write_interval="batch")
        self.failed = 0
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_on_batch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
        prediction: dict[str, Tensor],
        batch_indices: list[int],  # noqa: ARG002
        batch: dict[str, Tensor],
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int,  # noqa: ARG002
    ) -> None:
        """Write the predictions to disk."""
        if prediction["exception"]:
            self.failed += 1
            return
        # Dump affinity summary
        affinity_summary = {}
        pred_affinity_value = prediction["affinity_pred_value"]
        pred_affinity_probability = prediction["affinity_probability_binary"]
        affinity_summary = {
            "affinity_pred_value": pred_affinity_value.item(),
            "affinity_probability_binary": pred_affinity_probability.item(),
        }
        if "affinity_pred_value1" in prediction:
            pred_affinity_value1 = prediction["affinity_pred_value1"]
            pred_affinity_probability1 = prediction["affinity_probability_binary1"]
            pred_affinity_value2 = prediction["affinity_pred_value2"]
            pred_affinity_probability2 = prediction["affinity_probability_binary2"]
            affinity_summary["affinity_pred_value1"] = pred_affinity_value1.item()
            affinity_summary["affinity_probability_binary1"] = (
                pred_affinity_probability1.item()
            )
            affinity_summary["affinity_pred_value2"] = pred_affinity_value2.item()
            affinity_summary["affinity_probability_binary2"] = (
                pred_affinity_probability2.item()
            )

        # Save the affinity summary
        struct_dir = self.output_dir / batch["record"][0].id
        struct_dir.mkdir(exist_ok=True)
        path = struct_dir / f"affinity_{batch['record'][0].id}.json"

        with path.open("w") as f:
            f.write(json.dumps(affinity_summary, indent=4))

    def on_predict_epoch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
    ) -> None:
        """Print the number of failed examples."""
        # Print number of failed examples
        print(f"Number of failed examples: {self.failed}")  # noqa: T201
