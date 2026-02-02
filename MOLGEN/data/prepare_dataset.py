import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def sanitize_smiles(smiles: str) -> str:
    return smiles.replace("'", "").replace("\n", "").replace(" ", "")


def chunk_smiles(smiles_list, chunk_size):
    for chunk_start in range(0, len(smiles_list), chunk_size):
        yield chunk_start, smiles_list[chunk_start: chunk_start + chunk_size]


def main():
    parser = argparse.ArgumentParser(description="Generate Boltz YAML configs and path map.")
    parser.add_argument(
        "--csv-path",
        default="https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv",
        help="URL or local path to the SMILES CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/zinc_boltz_embedding",
        help="Base directory for YAML outputs and path mapping roots.",
    )
    parser.add_argument(
        "--yaml-prefix",
        default="mol_boltz_configs",
        help="Prefix for YAML directories inside output-dir.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10_000,
        help="Number of SMILES per chunk directory.",
    )
    parser.add_argument(
        "--map-path",
        default="data/path_to_dir.json",
        help="Path to write SMILES->embedding file map (JSON).",
    )
    parser.add_argument(
        "--protein-sequence",
        default="X",
        help="Protein sequence to embed in generated YAML files.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    map_path = Path(args.map_path).expanduser().resolve()

    df = pd.read_csv(args.csv_path)
    smiles_list = (
        df["smiles"].dropna().astype(str).drop_duplicates().tolist()
    )
    print(f"Total {len(smiles_list)} SMILES")

    output_dir.mkdir(parents=True, exist_ok=True)

    path_to_dir = {}
    global_idx = 1

    for chunk_start, chunk in chunk_smiles(smiles_list, args.chunk_size):
        dir_idx = chunk_start // args.chunk_size + 1

        yaml_dir = output_dir / f"{args.yaml_prefix}_{dir_idx}"
        yaml_dir.mkdir(parents=True, exist_ok=True)

        for smi in chunk:
            clean_smiles = sanitize_smiles(smi)
            """
            yaml_content = {
                "version": 1,
                "sequences": [
                    {"protein": {"id": "A", "sequence": args.protein_sequence, "msa": "empty"}},
                    {"ligand": {"id": "B", "smiles": clean_smiles}},
                ],
                "properties": [
                    {"affinity": {"binder": "B"}}
                ],
            }

            yaml_path = yaml_dir / f"mol_{global_idx}.yaml"
            with open(yaml_path, "w") as f:
                yaml.dump(yaml_content, f, sort_keys=False)
            """
            embedding_path = (
                output_dir
                / f"boltz_results_{args.yaml_prefix}_{dir_idx}"
                / "predictions"
                / f"mol_{global_idx}"
                / f"embeddings_mol_{global_idx}.npz"
            )
            path_to_dir[clean_smiles] = str(embedding_path)
            global_idx += 1

        print(f"Saved {len(chunk)} YAMLs -> {yaml_dir}")

    with open(map_path, "w") as f:
        json.dump(path_to_dir, f)
    print(f"Saved path map -> {map_path}")


if __name__ == "__main__":
    main()
