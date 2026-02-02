### Original code from MoFlow (under MIT License) https://github.com/calvin-zcx/moflow
import os
import sys
sys.path.insert(0, os.getcwd())
import pandas as pd
import argparse
import time
from data.parser import DataFrameParser
from utils.numpytupledataset import NumpyTupleDataset
from utils.smile_to_graph import GGNNPreprocessor


parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', type=str, default='zinc250k', choices=['zinc250k'])
parser.add_argument('--boltz-path-dict', type=str, default='data/path_to_dir.json')
args = parser.parse_args()

start_time = time.time()
data_name = args.dataset

if data_name == 'zinc250k':
    max_atoms = 38
    path = 'data/zinc250k.csv'
    smiles_col = 'smiles'
    label_idx = 1
else:
    raise ValueError(f"[ERROR] Unexpected value data_name={data_name}")

preprocessor = GGNNPreprocessor(out_size=max_atoms, kekulize=True)

print(f'Preprocessing {data_name} data')
df = pd.read_csv(path, index_col=0)
labels = df.keys().tolist()[label_idx:]
parser = DataFrameParser(preprocessor, boltz_path_dict=args.boltz_path_dict, labels=labels, smiles_col=smiles_col)
result = parser.parse(df, return_smiles=True)

dataset = result['dataset']
smiles = result['smiles']

NumpyTupleDataset.save(f'data/{data_name.lower()}_kekulized.npz', dataset)
print('Total time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
