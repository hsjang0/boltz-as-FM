# ADMET Property Prediction Benchmark

![Overview of the Boltz workflow](../_assets/admet_performance.png)


This is the code to reproduce the TDC ADMET property prediction experiments. This trains lightweight MLP heads on Boltz representations for each dataset across five seeds.

Our code includes a modified version of Boltz2 located in `../../_env/boltz_for_ADMET`.

## Data
- We store datasets under `datasets/boltz_rep/<dataset_name>_{train,test}.npz`.
- The details of dataset construction are described in `construct_dataset/README.md`.
- Pre-extracted representations are stored as `.npz` like `Ames_train.npz` and `Ames_test.npz`, each containing `x` (features) and `y` (labels).
- To rebuild representations, run the scripts in `construct_dataset/`.

## Running an experiment
```bash
python main.py --config configs/Ames/probing_boltz.yaml 
```

Results are stored under `results/<dataset_name>/`.

## Configs
- YAML files in `configs/<dataset_name>/` define dataset, task (e.g., `classification`), metric (e.g., `auroc`), data path, and core hyperparameters (`lr`, `hidden`, `layers`, `epochs`).

