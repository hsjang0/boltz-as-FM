import os
from time import time
import numpy as np
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
import json
from threading import Lock
import hashlib
import joblib
import scipy.sparse as sps




def _load_npz_dict(path):
    # Materialize all arrays so they can be shared across forked workers.
    with np.load(path, allow_pickle=False) as npz:
        return {k: npz[k] for k in npz}


def _emb_paths(emb, fm_option):
    if fm_option == "Boltz":
        return {"boltz": emb}
    raise ValueError(f"Unsupported fm_option {fm_option}")


def _preload_embeddings(mols, fm_option):
    # Intentionally left minimal: disk reads happen per worker with mmap to keep memory low.
    return {}


### Code adapted from GraphEBM
def load_mol(filepath):
    print(f'Loading file {filepath}')
    if not os.path.exists(filepath):
        raise ValueError(f'Invalid filepath {filepath} for dataset')
    load_data = np.load(filepath)
    result = []
    i = 0
    while True:
        key = f'arr_{i}'
        if key in load_data.keys():
            result.append(load_data[key])
            i += 1
        else:
            break
    return list(map(lambda x, a, emb: (x, a, emb), result[0], result[1], result[2]))


class MolDataset(Dataset):
    def __init__(self, mols, transform):
        self.mols = mols
        self.transform = transform

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, idx):
        return self.transform(self.mols[idx])


def get_transform_fn(dataset, adj_scale, fm_option='Boltz', preloaded_cache=None):
    fm_key = fm_option
    device = torch.device("cuda")
    zinc_atomic_nums = torch.tensor([6, 7, 8, 9, 15, 16, 17, 35, 53, 0], dtype=torch.long)
    zinc_one_hot_size = zinc_atomic_nums.numel()
    zinc_lookup = torch.full((int(zinc_atomic_nums.max().item()) + 1,), -1, dtype=torch.long)
    zinc_lookup[zinc_atomic_nums] = torch.arange(zinc_one_hot_size, dtype=torch.long)
    bond_map = torch.tensor([1.0, 2.0, 3.0, 0.0], dtype=torch.float32)

    def load_cached(path, loader):
        return loader()

    def transform(data):
        x, adj, emb = data
        x_int = torch.as_tensor(x, dtype=torch.long)
        indices = zinc_lookup[x_int]
        invalid = indices < 0
        if invalid.any():
            invalid_atoms = x_int[invalid].detach().cpu().numpy()
            raise ValueError(f'Unsupported atomic number in input: {invalid_atoms}')

        x_one_hot = torch.zeros((38, zinc_one_hot_size), dtype=torch.float32)
        x_one_hot.scatter_(1, indices.view(-1, 1), 1.0)
        x = x_one_hot[:, :-1]

        adj_first3 = torch.as_tensor(adj[:3], dtype=torch.float32)
        adj_all = torch.empty((4, 38, 38), dtype=torch.float32)
        adj_all[:3] = adj_first3
        adj_all[3] = 1.0 - adj_first3.sum(dim=0)

        adj_idx = adj_all.argmax(dim=0)
        adj = bond_map[adj_idx] / adj_scale
        
        if fm_option == "Boltz":
            emb_path = _emb_paths(emb, fm_option)["boltz"]
            boltz_np = _load_npz_dict(emb_path)
            boltz_adj = torch.cat([torch.as_tensor(arr) for arr in boltz_np["z"]], dim=-1)
            boltz_s = torch.cat([torch.as_tensor(arr) for arr in boltz_np["s"]], dim=-1)
            N = boltz_adj.shape[0]

            emb_adj = torch.zeros((38, 38, boltz_adj.shape[-1]), dtype=boltz_adj.dtype)
            emb_adj[:N, :N, :] = boltz_adj
            emb2_arr = torch.zeros((38, boltz_s.shape[-1]), dtype=boltz_s.dtype)
            emb2_arr[:N, :boltz_s.shape[-1]] = boltz_s

            emb = emb_adj.float()
            emb2 = emb2_arr.float()
        return x, adj, emb, emb2
    return transform


class _CudaPrefetcher:
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device=device)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        first = True
        for batch in self.loader:
            with torch.cuda.stream(self.stream):
                batch_on_device = []
                for item in batch:
                    if torch.is_tensor(item):
                        batch_on_device.append(
                            item.to(self.device, non_blocking=True)
                        )
                    else:
                        batch_on_device.append(item)

            if not first:
                cur_stream = torch.cuda.current_stream(self.device)
                cur_stream.wait_stream(self.stream)

                for t in prepared:
                    if torch.is_tensor(t):
                        t.record_stream(cur_stream)

                yield prepared

            prepared = batch_on_device
            first = False

        if not first:
            cur_stream = torch.cuda.current_stream(self.device)
            cur_stream.wait_stream(self.stream)
            for t in prepared:
                if torch.is_tensor(t):
                    t.record_stream(cur_stream)
            yield prepared


def _maybe_prefetch(loader, device):
    if device.type != "cuda":
        return loader
    return _CudaPrefetcher(loader, device)


def dataloader(config, get_graph_list=False):
    start_time = time()
    
    mols = load_mol(os.path.join(config.data.dir, f'{config.data.data.lower()}_kekulized.npz'))

    with open(os.path.join(config.data.dir, f'valid_idx_{config.data.data.lower()}.json')) as f:
        test_idx = json.load(f)
    
    train_idx = [i for i in range(len(mols)) if i not in test_idx]
    print(f'Number of training mols: {len(train_idx)} | Number of test mols: {len(test_idx)}')

    train_mols = [mols[i] for i in train_idx]
    test_mols = [mols[i] for i in test_idx if i < len(mols)]

    fm_option = config.model.FM
    preloaded_cache = None
    transform_fn = get_transform_fn(config.data.data, config.model.adj_scale, fm_option, preloaded_cache=preloaded_cache)
    train_dataset = MolDataset(train_mols, transform_fn)
    test_dataset = MolDataset(test_mols, transform_fn)

    train_dataloader = DataLoader(train_dataset, 
                                  prefetch_factor=3,
                                  num_workers=12,
                                  batch_size=config.data.batch_size, 
                                  shuffle=True,
                                  pin_memory=True,
                                  persistent_workers=True)
    test_dataloader = DataLoader(test_dataset, 
                                  prefetch_factor=3,
                                  num_workers=12,
                                  batch_size=config.data.batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  persistent_workers=True)

    print(f'{time() - start_time:.2f} sec elapsed for data loading')
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train_loader_wrapped = _maybe_prefetch(train_dataloader, device)
    test_loader_wrapped = _maybe_prefetch(test_dataloader, device)
    return train_loader_wrapped, test_loader_wrapped
