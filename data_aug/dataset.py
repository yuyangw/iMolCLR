import os
import csv
import math
import time
import random
import networkx as nx
import numpy as np
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms

from torch_scatter import scatter
from torch_geometric.data import Data, Batch

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem


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


def read_smiles(data_path):
    smiles_data = []
    with open(data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            smiles = row[-1]
            smiles_data.append(smiles)
            # mol = Chem.MolFromSmiles(smiles)
            # if mol != None:
            #     smiles_data.append(smiles)
    return smiles_data


def removeSubgraph(Graph, center, percent=0.2):
    assert percent <= 1
    G = Graph.copy()
    num = int(np.floor(len(G.nodes)*percent))
    removed = []
    temp = [center]
    
    while len(removed) < num:
        neighbors = []
        for n in temp:
            neighbors.extend([i for i in G.neighbors(n) if i not in temp])      
        for n in temp:
            if len(removed) < num:
                G.remove_node(n)
                removed.append(n)
            else:
                break
        temp = list(set(neighbors))
    return G, removed


class MoleculeDataset(Dataset):
    def __init__(self, smiles_data):
        super(Dataset, self).__init__()
        self.smiles_data = smiles_data

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.smiles_data[index])
        mol = Chem.AddHs(mol)

        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        type_idx = []
        chirality_idx = []
        atomic_number = []
        atoms = mol.GetAtoms()
        bonds = mol.GetBonds()
        # Sample 2 different centers to start for i and j
        start_i, start_j = random.sample(list(range(N)), 2)

        # Construct the original molecular graph from edges (bonds)
        edges = []
        for bond in bonds:
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        molGraph = nx.Graph(edges)
        
        # Get the graph for i and j after removing subgraphs
        percent_i, percent_j = 0.25, 0.25
        G_i, removed_i = removeSubgraph(molGraph, start_i, percent_i)
        G_j, removed_j = removeSubgraph(molGraph, start_j, percent_j)
        
        for atom in atoms:
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        x = torch.cat([x1, x2], dim=-1)
        # x shape (N, 2) [type, chirality]

        # Mask the atoms in the removed list
        x_i = deepcopy(x)
        for atom_idx in removed_i:
            # Change atom type to 118, and chirality to 0
            x_i[atom_idx,:] = torch.tensor([len(ATOM_LIST), 0])
        x_j = deepcopy(x)
        for atom_idx in removed_j:
            # Change atom type to 118, and chirality to 0
            x_j[atom_idx,:] = torch.tensor([len(ATOM_LIST), 0])

        # Only consider bond still exist after removing subgraph
        row_i, col_i, row_j, col_j = [], [], [], []
        edge_feat_i, edge_feat_j = [], []
        G_i_edges = list(G_i.edges)
        G_j_edges = list(G_j.edges)
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            feature = [
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ]
            if (start, end) in G_i_edges:
                row_i += [start, end]
                col_i += [end, start]
                edge_feat_i.append(feature)
                edge_feat_i.append(feature)
            if (start, end) in G_j_edges:
                row_j += [start, end]
                col_j += [end, start]
                edge_feat_j.append(feature)
                edge_feat_j.append(feature)

        edge_index_i = torch.tensor([row_i, col_i], dtype=torch.long)
        edge_attr_i = torch.tensor(np.array(edge_feat_i), dtype=torch.long)
        edge_index_j = torch.tensor([row_j, col_j], dtype=torch.long)
        edge_attr_j = torch.tensor(np.array(edge_feat_j), dtype=torch.long)
        
        data_i = Data(x=x_i, edge_index=edge_index_i, edge_attr=edge_attr_i)
        data_j = Data(x=x_j, edge_index=edge_index_j, edge_attr=edge_attr_j)
        
        return data_i, data_j, mol

    def __len__(self):
        return len(self.smiles_data)


def collate_fn(batch):
    gis, gjs, mols = zip(*batch)

    gis = Batch().from_data_list(gis)
    gjs = Batch().from_data_list(gjs)

    return gis, gjs, mols

