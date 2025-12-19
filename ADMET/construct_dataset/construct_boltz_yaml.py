import os
from pathlib import Path
import yaml
from tdc.single_pred import ADME, Tox
import pandas as pd
import argparse
from tqdm import tqdm

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
def load_dataset(dataset: str):
    dataset_type, dataset_name = DATASET_SPECS[dataset]
    loader = ADME if dataset_type == "ADME" else Tox
    return loader(name=dataset_name)

parser = argparse.ArgumentParser(description="Construct YAML files for TDC datasets.")
parser.add_argument("--save_dir", required=True, help="Directory to save generated YAML files.")
args = parser.parse_args()


for dataset in DATASET_SPECS:
    data = load_dataset(dataset)
    splits = data.get_split()
    train_split = pd.concat([splits["train"], splits["valid"]], ignore_index=True)
    test_split = splits["test"]
    output_dir = Path(f"{args.save_dir}/{dataset}_yaml_files")
    output_dir.mkdir(parents=True, exist_ok=True)

    def write_yaml_files(df, prefix):
        for i, row in enumerate(tqdm(df.itertuples(index=False), total=len(df), desc=prefix), start=1):
            smi = getattr(row, "Drug")
            yaml_content = {
                "version": 1,
                "sequences": [
                    {"protein": {"id": "A", "sequence": "X", "msa": "empty"}},
                    {"ligand": {"id": "B", "smiles": smi}},
                ]
            }
            file_path = output_dir / f"{prefix}_{i}.yaml"
            with open(file_path, "w") as f:
                yaml.dump(yaml_content, f, sort_keys=False)


    write_yaml_files(train_split, "train")
    write_yaml_files(test_split, "test")
    print(f"Done. Files saved in {output_dir.resolve()}")
