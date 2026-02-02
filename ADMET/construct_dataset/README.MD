# Boltz ADMET Dataset Construction

Obtain Boltz representations for ADMET datasets and pool them into a final feature set. We obtain pair representations from the 1/4, 2/4, 3/4, and final layers, concatenate them, and apply statistical pooling over entries, diagonals, and bond entries.

Our code includes a modified version of Boltz2 located in `../../_env/boltz_for_ADMET`.

## Prerequisites
- Install boltz_for_ADMET for GPU support: `cd ../../_env/boltz_for_ADMET && pip install -e .[cuda]` (already installed inside the provided Apptainer image).

## Steps
1. Create embeddings config for Boltz: `python construct_boltz_yaml.py --save_dir <directory_name>` (`<directory_name>` is where embeddings will be written).
2. Produce Boltz embeddings: `./get_emb.sh <directory_name>`.
3. Pool embeddings into the dataset: `python pooling_dataset.py --output_dir ../datasets/boltz_rep --emb_dir <directory_name>`.
