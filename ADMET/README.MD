# ADMET Property Prediction Benchmark

This is the code to reproduce the TDC ADMET property prediction experiments. This trains lightweight MLP heads on Boltz representations for each dataset across five seeds.


## Data
- Download TDC ADMET benchmark datasets with Boltz representations from THIS LINK.
- Store datasets under `datasets/<boltz_rep|boltz_rep_w_mini>/<dataset_name>_{train,test}.npz`.
- Pre-extracted embeddings should be `.npz` pairs like `Ames_train.npz` and `Ames_test.npz`, each containing `x` (features) and `y` (labels).
- To rebuild embeddings, run the scripts in `construct_dataset/`.

## Environment

Use the Apptainer image distributed at <www.LINK 이건 내가 알아서 넣을게> to run experiments. To run natively instead of Apptainer, install Python deps with `pip install -r requirements.txt`.

## Running an experiment
```bash
python main.py --config configs/Ames/probing_boltz.yaml 
```

Results are stored under `results/<dataset_name>/`.

## Configs
- YAML files in `configs/<dataset_name>/` define dataset, task (e.g., `classification`), metric (e.g., `auroc`), data path, and core hyperparameters (`lr`, `hidden`, `layers`, `epochs`).

## Notes
- This is on-going research.
