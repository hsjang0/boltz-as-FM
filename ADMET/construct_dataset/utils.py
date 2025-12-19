import numpy as np
from chembl_structure_pipeline.exclude_flag import exclude_flag
from chembl_structure_pipeline.standardizer import standardize_mol
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops
from rdkit.Chem.MolStandardize import rdMolStandardize


def smiles_to_edge_index(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol = AllChem.AddHs(mol)
    canonical_order = AllChem.CanonicalRankAtoms(mol)
    mol = rdmolops.RenumberAtoms(mol, canonical_order)
    rdkit2node = {}

    atom_idx = 0
    for e, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() == 1:
            continue
        rdkit2node[e] = atom_idx
        atom_idx += 1

    src, dst = [], []
    for bond in mol.GetBonds():
        i_raw, j_raw = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i_raw in rdkit2node and j_raw in rdkit2node:
            i, j = rdkit2node[i_raw], rdkit2node[j_raw]
            src += [i, j]
            dst += [j, i]

    edge_index = np.asarray([src, dst], dtype=np.int32)
    return edge_index, atom_idx


def k_hop_masks(adj, k_max=1):
    n = adj.shape[0]
    dist = np.full((n, n), np.inf)
    np.fill_diagonal(dist, 0)
    remaining = np.isinf(dist)
    reach = adj.copy()
    k = 1
    while k <= k_max and remaining.any():
        newly = reach & remaining
        dist[newly] = k
        remaining[newly] = False
        reach = (reach @ adj) > 0
        k += 1
    masks = [dist == r for r in range(0, k_max + 1)]
    return dist, masks


def pool_embedding(embedding, edge_index):
    z_full = np.concatenate([embedding["z"][-1], embedding["z"][-2], embedding["z"][-3], embedding["z"][-4]], axis=-1)
    s_full = np.concatenate([embedding["s"][-1], embedding["s"][-2], embedding["s"][-3], embedding["s"][-4]], axis=-1)
    z = (z_full + np.transpose(z_full, (1, 0, 2))) / 2
    adjacency = np.zeros(z.shape[:2], bool)
    adjacency[edge_index[0], edge_index[1]] = True
    np.fill_diagonal(adjacency, False)

    _, masks = k_hop_masks(adjacency, k_max=1)
    rings_mean = []
    rings_std = []
    for mask in masks:
        vals = z[mask]
        if vals.size:
            rings_mean.append(vals.mean(0))
            rings_std.append(vals.std(0))
        else:
            rings_mean.append(np.zeros(z.shape[-1]))
            rings_std.append(np.zeros(z.shape[-1]))

    rings_mean = np.stack(rings_mean).reshape(-1)
    rings_std = np.stack(rings_std).reshape(-1)
    all_pooled = z_full.mean(axis=(0, 1))
    all_pooled_std = z_full.reshape(-1, z_full.shape[-1]).std(axis=0)
    return np.concatenate([rings_mean, rings_std, all_pooled, all_pooled_std], axis=0)

