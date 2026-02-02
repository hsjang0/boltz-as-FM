# Molecular Generation Benchmark

This is the code to reproduce the molecular generation experiments. Our code is based on GruM (https://github.com/harryjo97/GruM)

## Data

You should construct Boltz2 embeddings to run the experiments

1. Create embeddings config for Boltz: `python data/prepare_dataset.py --output-dir <directory_name>` (`<directory_name>` is where embeddings will be written).
2. Produce Boltz embeddings: `./data/get_emb.sh <directory_name>`.
3. Create training datasets for generative GruM: `python data/preprocess.py`.
4. Run `data/preprocess_for_init_flags.py` and `data/preprocess_for_nspdk.py` to prepare evaluation datasets.

## Running an experiment

Train a model:

```bash
python main.py --type train --config configs/REPA_BOLTZ.yaml
```

Generate and evaluate molecules from a checkpoint:

```bash
python main.py --type sample --config configs/REPA_BOLTZ.yaml --ckpt <checkpoint_name>
```
