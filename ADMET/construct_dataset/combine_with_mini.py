import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from minimol import Minimol
from rdkit import Chem
from tdc.single_pred import ADME, Tox

featuriser = Minimol()

DATASET_SPECS = {
    "Caco2": ("ADME", "Caco2_Wang"),
    "Bioavailability": ("ADME", "Bioavailability_Ma"),
    "Lipophilicity": ("ADME", "Lipophilicity_AstraZeneca"),
    "Solubility": ("ADME", "Solubility_AqSolDB"),
    "HIA": ("ADME", "HIA_Hou"),
    "Pgp": ("ADME", "Pgp_Broccatelli"),
    "BBB": ("ADME", "BBB_Martins"),
    "PPBR": ("ADME", "PPBR_AZ"),
    "VDss": ("ADME", "VDss_Lombardo"),
    "Half": ("ADME", "Half_Life_Obach"),
    "Clearance_Hepatocyte": ("ADME", "Clearance_Hepatocyte_AZ"),
    "Clearance_Microsome": ("ADME", "Clearance_Microsome_AZ"),
    "LD50": ("Tox", "LD50_Zhu"),
    "hERG": ("Tox", "hERG"),
    "Ames": ("Tox", "AMES"),
    "DILI": ("Tox", "DILI"),
    "CYP2C9_Substrate_CarbonMangels": ("ADME", "CYP2C9_Substrate_CarbonMangels"),
    "CYP2D6_Substrate_CarbonMangels": ("ADME", "CYP2D6_Substrate_CarbonMangels"),
    "CYP3A4_Substrate_CarbonMangels": ("ADME", "CYP3A4_Substrate_CarbonMangels"),
    "CYP2D6_Veith": ("ADME", "CYP2D6_Veith"),
    "CYP3A4_Veith": ("ADME", "CYP3A4_Veith"),
    "CYP2C9_Veith": ("ADME", "CYP2C9_Veith"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_SPECS.keys()),
        choices=sorted(DATASET_SPECS.keys()),
        help="Datasets to process.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        help="Directory containing *_K1_cum_[train|test]_df.npz files.",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        help="Directory containing *_K1_cum_[train|test]_df.npz files.",
    )
    return parser.parse_args()


def clean_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol)


def load_dataset(dataset: str):
    dataset_type, dataset_name = DATASET_SPECS[dataset]
    loader = ADME if dataset_type == "ADME" else Tox
    return loader(name=dataset_name)

def stack_embeddings(smiles_series: pd.Series) -> np.ndarray:
    embeddings = featuriser([clean_smiles(smi) for smi in smiles_series.tolist()])
    return np.concatenate([np.asarray(embedding).reshape(1, -1) for embedding in embeddings], axis=0)


def main():
    args = parse_args()
    dataset_dir = args.dataset_dir
    save_dir = args.save_dir

    for dataset in args.datasets:
        data = load_dataset(dataset)
        splits = data.get_split()
        train_df = pd.concat([splits["train"], splits["valid"]], ignore_index=True)
        test_df = splits["test"]

        train_np = np.load(dataset_dir / f"{dataset}_train.npz")
        test_np = np.load(dataset_dir / f"{dataset}_test.npz")

        train_idx = train_np["y"][:, 1].astype(int)
        test_idx = test_np["y"][:, 1].astype(int)

        train_embeddings = stack_embeddings(train_df["Drug"].iloc[train_idx])
        test_embeddings = stack_embeddings(test_df["Drug"].iloc[test_idx])

        train_update = np.concatenate([train_np["x"], train_embeddings], axis=1)
        test_update = np.concatenate([test_np["x"], test_embeddings], axis=1)

        save_dir.mkdir(parents=True, exist_ok=True)
        np.savez(save_dir / f"{dataset}_train.npz", x=train_update, y=train_np["y"])
        np.savez(save_dir / f"{dataset}_test.npz", x=test_update, y=test_np["y"])
        print(f"\nDataset saved: {save_dir}/{dataset}_train.npz,{dataset}_test.npz")

if __name__ == "__main__":
    main()
