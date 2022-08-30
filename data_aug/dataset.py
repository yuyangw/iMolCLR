import os
import csv
import math
import time
import signal
import random
import numpy as np
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms

from torch_scatter import scatter
from torch_geometric.data import Data, Batch

import networkx as nx
from networkx.algorithms.components import node_connected_component

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from rdkit.Chem.BRICS import BRICSDecompose, FindBRICSBonds, BreakBRICSBonds


ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [
    BT.SINGLE, 
    BT.DOUBLE, 
    BT.TRIPLE, 
    BT.AROMATIC
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]


class TimeoutError(Exception):
    pass


class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def read_smiles(data_path):
    smiles_data = []
    with open(data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            smiles = row[-1]
            smiles_data.append(smiles)
    return smiles_data


def get_fragment_indices(mol):
    bonds = mol.GetBonds()
    edges = []
    for bond in bonds:
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    molGraph = nx.Graph(edges)

    BRICS_bonds = list(FindBRICSBonds(mol))
    break_bonds = [b[0] for b in BRICS_bonds]
    break_atoms = [b[0][0] for b in BRICS_bonds] + [b[0][1] for b in BRICS_bonds]
    molGraph.remove_edges_from(break_bonds)

    indices = []
    for atom in break_atoms:
        n = node_connected_component(molGraph, atom)
        if len(n) > 3 and n not in indices:
            indices.append(n)
    indices = set(map(tuple, indices))
    return indices


def get_fragments(mol):
    try:
        with timeout(seconds=2):

            ref_indices = get_fragment_indices(mol)

            frags = list(BRICSDecompose(mol, returnMols=True))
            mol2 = BreakBRICSBonds(mol)

            extra_indices = []
            for i, atom in enumerate(mol2.GetAtoms()):
                if atom.GetAtomicNum() == 0:
                    extra_indices.append(i)
            extra_indices = set(extra_indices)

            frag_mols = []
            frag_indices = []
            for frag in frags:
                indices = mol2.GetSubstructMatches(frag)
                # if len(indices) >= 1:
                #     idx = indices[0]
                #     idx = set(idx) - extra_indices
                #     if len(idx) > 3:
                #         frag_mols.append(frag)
                #         frag_indices.append(idx)
                if len(indices) == 1:
                    idx = indices[0]
                    idx = set(idx) - extra_indices
                    if len(idx) > 3:
                        frag_mols.append(frag)
                        frag_indices.append(idx)
                else:
                    for idx in indices:
                        idx = set(idx) - extra_indices
                        if len(idx) > 3:
                            for ref_idx in ref_indices:
                                if (tuple(idx) == ref_idx) and (idx not in frag_indices):
                                    frag_mols.append(frag)
                                    frag_indices.append(idx)

            return frag_mols, frag_indices
    
    except:
        print('timeout!')
        return [], [set()]


class MoleculeDataset(Dataset):
    def __init__(self, smiles_data):
        super(Dataset, self).__init__()
        self.smiles_data = smiles_data

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.smiles_data[index])
        # mol = Chem.AddHs(mol)

        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        type_idx = []
        chirality_idx = []
        atomic_number = []

        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        x = torch.cat([x1, x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)

        # random mask a subgraph of the molecule
        num_mask_nodes = max([1, math.floor(0.25*N)])
        num_mask_edges = max([0, math.floor(0.25*M)])
        mask_nodes_i = random.sample(list(range(N)), num_mask_nodes)
        mask_nodes_j = random.sample(list(range(N)), num_mask_nodes)
        mask_edges_i_single = random.sample(list(range(M)), num_mask_edges)
        mask_edges_j_single = random.sample(list(range(M)), num_mask_edges)
        mask_edges_i = [2*i for i in mask_edges_i_single] + [2*i+1 for i in mask_edges_i_single]
        mask_edges_j = [2*i for i in mask_edges_j_single] + [2*i+1 for i in mask_edges_j_single]

        x_i = deepcopy(x)
        for atom_idx in mask_nodes_i:
            x_i[atom_idx,:] = torch.tensor([len(ATOM_LIST), 0])
        edge_index_i = torch.zeros((2, 2*(M-num_mask_edges)), dtype=torch.long)
        edge_attr_i = torch.zeros((2*(M-num_mask_edges), 2), dtype=torch.long)
        count = 0
        for bond_idx in range(2*M):
            if bond_idx not in mask_edges_i:
                edge_index_i[:,count] = edge_index[:,bond_idx]
                edge_attr_i[count,:] = edge_attr[bond_idx,:]
                count += 1
        data_i = Data(x=x_i, edge_index=edge_index_i, edge_attr=edge_attr_i)

        x_j = deepcopy(x)
        for atom_idx in mask_nodes_j:
            x_j[atom_idx,:] = torch.tensor([len(ATOM_LIST), 0])
        edge_index_j = torch.zeros((2, 2*(M-num_mask_edges)), dtype=torch.long)
        edge_attr_j = torch.zeros((2*(M-num_mask_edges), 2), dtype=torch.long)
        count = 0
        for bond_idx in range(2*M):
            if bond_idx not in mask_edges_j:
                edge_index_j[:,count] = edge_index[:,bond_idx]
                edge_attr_j[count,:] = edge_attr[bond_idx,:]
                count += 1
        data_j = Data(x=x_j, edge_index=edge_index_j, edge_attr=edge_attr_j)

        frag_mols, frag_indices = get_fragments(mol)
        
        return data_i, data_j, mol, N, frag_mols, frag_indices
    
    def __len__(self):
        return len(self.smiles_data)


def collate_fn(batch):
    gis, gjs, mols, atom_nums, frag_mols, frag_indices = zip(*batch)

    frag_mols = [j for i in frag_mols for j in i]

    # gis = Batch().from_data_list(gis)
    # gjs = Batch().from_data_list(gjs)
    gis = Batch.from_data_list(gis)
    gjs = Batch.from_data_list(gjs)

    gis.motif_batch = torch.zeros(gis.x.size(0), dtype=torch.long)
    gjs.motif_batch = torch.zeros(gjs.x.size(0), dtype=torch.long)

    curr_indicator = 1
    curr_num = 0
    for N, indices in zip(atom_nums, frag_indices):
        for idx in indices:
            curr_idx = np.array(list(idx)) + curr_num
            gis.motif_batch[curr_idx] = curr_indicator
            gjs.motif_batch[curr_idx] = curr_indicator
            curr_indicator += 1
        curr_num += N

    return gis, gjs, mols, frag_mols


class MoleculeDatasetWrapper(object):
    def __init__(self, batch_size, num_workers, valid_size, data_path):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size

    def get_data_loaders(self):
        smiles_data = read_smiles(self.data_path)

        num_train = len(smiles_data)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        train_smiles = [smiles_data[i] for i in train_idx]
        valid_smiles = [smiles_data[i] for i in valid_idx]
        del smiles_data
        print(len(train_smiles), len(valid_smiles))

        train_dataset = MoleculeDataset(train_smiles)
        valid_dataset = MoleculeDataset(valid_smiles)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, collate_fn=collate_fn,
            num_workers=self.num_workers, drop_last=True, shuffle=True
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=self.batch_size, collate_fn=collate_fn,
            num_workers=self.num_workers, drop_last=True
        )

        return train_loader, valid_loader