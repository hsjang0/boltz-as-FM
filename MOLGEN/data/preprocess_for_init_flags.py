from time import time
import pickle
import json
import pandas as pd
import os
import sys
sys.path.insert(0, os.getcwd())
import argparse
from utils.mol_utils import mols_to_nx, smiles_to_mols
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "8")



parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', type=str, default='zinc250k', choices=['zinc250k'])
args = parser.parse_args()

dataset = args.dataset
start_time = time()

with open(f'data/valid_idx_{dataset.lower()}.json') as f:
    test_idx = json.load(f)

if dataset == 'zinc250k':
    col = 'smiles'
else:
    raise ValueError(f"[ERROR] Unexpected value data_name={dataset}")

smiles = pd.read_csv(f'data/{dataset.lower()}.csv')[col]
train_idx = [i for i in range(len(smiles)) if i not in test_idx]
smiles = [smiles.iloc[i] for i in train_idx]
nx_graphs = mols_to_nx(smiles_to_mols(smiles))
print(f'Converted the molecules into {len(nx_graphs)} graphs')

with open(f'data/{dataset.lower()}_train_nx.pkl', 'wb') as f:
    pickle.dump(nx_graphs, f)

print(f'Total {time() - start_time:.2f} sec elapsed')
