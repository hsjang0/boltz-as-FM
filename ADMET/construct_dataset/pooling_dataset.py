import os
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from rdkit import Chem

import numpy as np
import pandas as pd
import torch
import yaml
from tdc.single_pred import ADME, Tox

import sys
from utils import (
    pool_embedding, 
    pool_embedding_topk_norm
)

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "8")
torch.set_num_threads(4)          # intra-op
torch.set_num_interop_threads(1)  # inter-op

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


def get_smiles_from_yaml(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
        sequences = data.get("sequences", [])
        for item in sequences:
            if "ligand" in item:
                ligand_data = item["ligand"]
                if "smiles" in ligand_data:
                    return ligand_data["smiles"]
        return None
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None


def build_df(data, mode, dataset_name, emb_dir):
    rows = []
    drugs = data["Drug"].values
    targets = data["Y"].values

    for i, (smiles, value) in enumerate(zip(drugs, targets)):
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            continue
        
        file_idx = i + 1
        yaml_path = emb_dir / f"{dataset_name}_yaml_files/{mode}_{file_idx}.yaml"
        smiles_string = get_smiles_from_yaml(yaml_path)
        embedding_path = emb_dir / f"boltz_results_{dataset_name}_yaml_files/predictions/{mode}_{file_idx}/embeddings_{mode}_{file_idx}.npz"
        embedding = np.load(embedding_path)
        pooled = pool_embedding_all_std(embedding, edge_index)
        rows.append((pooled, (value, i)))

    features = [item for item, _ in rows]
    labels = [item for _, item in rows]
    return np.stack(features, axis=0), np.array(labels)


def load_dataset(dataset: str):
    dataset_type, dataset_name = DATASET_SPECS[dataset]
    loader = ADME if dataset_type == "ADME" else Tox
    return loader(name=dataset_name)


def process_dataset(dataset, output_dir, emb_dir, pool_type):
    output_dir = Path(output_dir)
    emb_dir = Path(emb_dir)
    data = load_dataset(dataset)
    split = data.get_split()
    train_df = pd.concat([split["train"], split["valid"]], ignore_index=True)
    train_x, train_y = build_df(train_df, "train", dataset, emb_dir, pool_type)
    test_x, test_y = build_df(split["test"], "test", dataset, emb_dir, pool_type)

    np.savez(output_dir / f"{dataset}_train.npz", x=train_x, y=train_y)
    np.savez(output_dir / f"{dataset}_test.npz", x=test_x, y=test_y)
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True, type=Path, help="Directory to save pooled datasets.")
    parser.add_argument("--emb_dir", required=True, type=Path, help="Base directory containing embeddings and YAML files.")
    args = parser.parse_args()
    output_dir = args.output_dir
    emb_dir = args.emb_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor() as executor:
        list(executor.map(process_dataset, DATASET_SPECS, repeat(output_dir), repeat(emb_dir)))


if __name__ == "__main__":
    main()
