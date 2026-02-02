# Molecular Generation Benchmark

This is the code to reproduce the molecular generation experiments. Our code is based on GruM (https://github.com/harryjo97/GruM)

## Data

You should construct datasets with Boltz2 representations to run the experiments.

1. Create data configs for Boltz: `python data/prepare_dataset.py --output-dir <directory_name>` (`<directory_name>` is where embeddings will be written).
2. Produce Boltz representations: `./data/get_emb.sh <directory_name> <num_gpus>`.
3. Create ZINC250k dataset with Boltz representations for GruM: `python data/preprocess.py`.
4. Run `python data/preprocess_for_init_flags.py` and `python data/preprocess_for_nspdk.py` to prepare evaluation datasets.

## Running an experiment

Train a model:

```bash
python main.py --type train --config REPA_BOLTZ
```

Generate and evaluate molecules from a checkpoint:

```bash
python main.py --type sample --config REPA_BOLTZ --ckpt <checkpoint_name>
```
