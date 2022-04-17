import os
import csv
import random
import numpy as np
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms

from torch_scatter import scatter
from torch_geometric.data import Data, Dataset, DataLoader

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]


def _generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def generate_scaffolds(dataset, log_every_n=1000):
    scaffolds = {}
    data_len = len(dataset)

    print("About to generate scaffolds")
    for ind, smiles in enumerate(dataset.smiles_data):
        if ind % log_every_n == 0:
            print("Generating scaffold %d/%d" % (ind, data_len))
        scaffold = _generate_scaffold(smiles)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)

    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    return scaffold_sets


def scaffold_split(dataset, valid_size, test_size, seed=None, log_every_n=1000):
    train_size = 1.0 - valid_size - test_size
    scaffold_sets = generate_scaffolds(dataset)

    train_cutoff = train_size * len(dataset)
    valid_cutoff = (train_size + valid_size) * len(dataset)
    train_inds: List[int] = []
    valid_inds: List[int] = []
    test_inds: List[int] = []

    print("About to sort in scaffold sets")
    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set
    
    print('train: {}, valid: {}, test: {}'.format(
        len(train_inds), len(valid_inds), len(test_inds)))
    return train_inds, valid_inds, test_inds


def read_smiles(data_path, target, task):
    smiles_data, labels = [], []
    with open(data_path) as csv_file:
        # csv_reader = csv.reader(csv_file, delimiter=',')
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i != 0:
                # smiles = row[3]
                smiles = row['smiles']
                label = row[target]
                mol = Chem.MolFromSmiles(smiles)
                if mol != None and label != '':
                    smiles_data.append(smiles)
                    if task == 'classification':
                        labels.append(int(label))
                    elif task == 'regression':
                        labels.append(float(label))
                    else:
                        ValueError('task must be either regression or classification')
    print('Number of data:', len(smiles_data))
    return smiles_data, labels


class MolTestDataset(Dataset):
    def __init__(self, data_path, target='p_np', task='classification'):
        super(Dataset, self).__init__()
        self.smiles_data, self.labels = read_smiles(data_path, target, task)
        self.task = task

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.smiles_data[index])
        mol = Chem.AddHs(mol)

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
            # edge_type += 2 * [MOL_BONDS[bond.GetBondType()]]
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
        if self.task == 'classification':
            y = torch.tensor(self.labels[index], dtype=torch.long).view(1,-1)
        elif self.task == 'regression':
            y = torch.tensor(self.labels[index], dtype=torch.float).view(1,-1)
        data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)

        return data


    def __len__(self):
        return len(self.smiles_data)


class MolTestDatasetWrapper(object):
    def __init__(self, batch_size, num_workers, valid_size, test_size, data_path, target, task):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.test_size = test_size
        self.target = target
        self.task = task

    def get_data_loaders(self):
        train_dataset = MolTestDataset(data_path=self.data_path, target=self.target, task=self.task)
        train_loader, valid_loader, test_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader, test_loader

    def get_train_validation_data_loaders(self, train_dataset):
        train_idx, valid_idx, test_idx = scaffold_split(train_dataset, self.valid_size, self.test_size)

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=False, shuffle=False)

        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=False)
                                
        test_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=test_sampler,
                                  num_workers=self.num_workers, drop_last=False)

        return train_loader, valid_loader, test_loader


# import os
# import csv
# import math
# import time
# import random
# import networkx as nx
# import numpy as np
# from copy import deepcopy

# import torch
# import torch.nn.functional as F
# import torchvision.transforms as transforms

# from torch_scatter import scatter
# from torch_geometric.data import Data, Dataset, DataLoader

# import rdkit
# from rdkit import Chem
# from rdkit.Chem.rdchem import HybridizationType as HT
# from rdkit.Chem.rdchem import BondType as BT
# from rdkit.Chem.rdchem import BondStereo
# from rdkit.Chem import AllChem
# from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
# from rdkit import RDLogger
# RDLogger.DisableLog('rdApp.*')

# # node feature lists
# ATOM_LIST = list(range(0,119))
# CHIRALITY_LIST = [
#     Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
#     Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
#     Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
#     Chem.rdchem.ChiralType.CHI_OTHER
# ]
# CHARGE_LIST = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
# HYBRIDIZATION_LIST = [
#     HT.S, HT.SP, HT.SP2, HT.SP3, HT.SP3D,
#     HT.SP3D2, HT.UNSPECIFIED
# ]
# NUM_H_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8]
# VALENCE_LIST = [0, 1, 2, 3, 4, 5, 6]
# DEGREE_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# # edge feature lists
# BOND_LIST = [0, BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
# BONDDIR_LIST = [
#     Chem.rdchem.BondDir.NONE,
#     Chem.rdchem.BondDir.ENDUPRIGHT,
#     Chem.rdchem.BondDir.ENDDOWNRIGHT
# ]
# STEREO_LIST = [
#     BondStereo.STEREONONE, BondStereo.STEREOANY, 
#     BondStereo.STEREOZ, BondStereo.STEREOE,
#     BondStereo.STEREOCIS, BondStereo.STEREOTRANS
# ]


# def _generate_scaffold(smiles, include_chirality=False):
#     mol = Chem.MolFromSmiles(smiles)
#     scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
#     return scaffold


# def generate_scaffolds(smiles_data, log_every_n=1000):
#     scaffolds = {}
#     data_len = len(smiles_data)
#     print(data_len)

#     print("About to generate scaffolds")
#     for ind, smiles in enumerate(smiles_data):
#         if ind % log_every_n == 0:
#             print("Generating scaffold %d/%d" % (ind, data_len))
#         scaffold = _generate_scaffold(smiles)
#         if scaffold not in scaffolds:
#             scaffolds[scaffold] = [ind]
#         else:
#             scaffolds[scaffold].append(ind)

#     # Sort from largest to smallest scaffold sets
#     scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
#     scaffold_sets = [
#         scaffold_set for (scaffold, scaffold_set) in sorted(
#             scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
#     ]
#     # print(scaffold_sets)
#     return scaffold_sets


# def scaffold_split(smiles_data, valid_size, test_size, seed=None, log_every_n=1000):
#     train_size = 1.0 - valid_size - test_size
#     scaffold_sets = generate_scaffolds(smiles_data)

#     train_cutoff = train_size * len(smiles_data)
#     valid_cutoff = (train_size + valid_size) * len(smiles_data)
#     train_inds: List[int] = []
#     valid_inds: List[int] = []
#     test_inds: List[int] = []

#     print("About to sort in scaffold sets")
#     for scaffold_set in scaffold_sets:
#         if len(train_inds) + len(scaffold_set) > train_cutoff:
#             if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
#                 test_inds += scaffold_set
#             else:
#                 valid_inds += scaffold_set
#         else:
#             train_inds += scaffold_set
#     return train_inds, valid_inds, test_inds


# def read_smiles(data_path, target, task):
#     smiles_data, labels = [], []
#     with open(data_path) as csv_file:
#         # csv_reader = csv.reader(csv_file, delimiter=',')
#         csv_reader = csv.DictReader(csv_file, delimiter=',')
#         for i, row in enumerate(csv_reader):
#             if i != 0:
#                 # smiles = row[3]
#                 smiles = row['smiles']
#                 label = row[target]
#                 mol = Chem.MolFromSmiles(smiles)
#                 if mol != None and label != '':
#                     smiles_data.append(smiles)
#                     if task == 'classification':
#                         labels.append(int(label))
#                     elif task == 'regression':
#                         labels.append(float(label))
#                     else:
#                         ValueError('task must be either regression or classification')
#     print(len(smiles_data))
#     return smiles_data, labels


# class MolTestDataset(Dataset):
#     def __init__(self, smiles_data, labels, task):
#         super(Dataset, self).__init__()
#         # self.smiles_data, self.labels = read_smiles(data_path, target, task)
#         # self.task = task
#         self.smiles_data = smiles_data
#         self.labels = labels
#         self.task = task
    
#     def __getitem__(self, index):
#         mol = Chem.MolFromSmiles(self.smiles_data[index])
#         # mol = Chem.AddHs(mol)

#         N = mol.GetNumAtoms()
#         M = mol.GetNumBonds()

#         atomic = []
#         degree, charge, hybrization = [], [], []
#         aromatic, num_hs, chirality = [], [], []
#         atoms = mol.GetAtoms()
#         bonds = mol.GetBonds()
        
#         for atom in atoms:
#             atomic.append(ATOM_LIST.index(atom.GetAtomicNum()))
#             degree.append(DEGREE_LIST.index(atom.GetDegree()))
#             charge.append(CHARGE_LIST.index(atom.GetFormalCharge()))
#             hybrization.append(HYBRIDIZATION_LIST.index(atom.GetHybridization()))
#             aromatic.append(1 if atom.GetIsAromatic() else 0)
#             num_hs.append(NUM_H_LIST.index(atom.GetTotalNumHs()))
#             chirality.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
        
#         atomic = F.one_hot(torch.tensor(atomic, dtype=torch.long), num_classes=len(ATOM_LIST))
#         degree = F.one_hot(torch.tensor(degree, dtype=torch.long), num_classes=len(DEGREE_LIST))
#         charge = F.one_hot(torch.tensor(charge, dtype=torch.long), num_classes=len(CHARGE_LIST))
#         hybrization = F.one_hot(torch.tensor(hybrization, dtype=torch.long), num_classes=len(HYBRIDIZATION_LIST))
#         aromatic = F.one_hot(torch.tensor(aromatic, dtype=torch.long), num_classes=2)
#         num_hs = F.one_hot(torch.tensor(num_hs, dtype=torch.long), num_classes=len(NUM_H_LIST))
#         chirality = F.one_hot(torch.tensor(chirality, dtype=torch.long), num_classes=len(CHIRALITY_LIST))
#         x = torch.cat([atomic, degree, charge, hybrization, aromatic, num_hs, chirality], dim=-1).type(torch.FloatTensor)
#         node_feat_dim = x.shape[1]

#         # Only consider bond still exist after removing subgraph
#         row_i, col_i = [], []
#         bond_i, bond_dir_i, stereo_i = [], [], []
#         for bond in mol.GetBonds():
#             start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
#             row_i += [start, end]
#             col_i += [end, start]

#             b = BOND_LIST.index(bond.GetBondType())
#             bd = BONDDIR_LIST.index(bond.GetBondDir())
#             s = STEREO_LIST.index(bond.GetStereo())
#             bond_i.append(b)
#             bond_i.append(b)
#             bond_dir_i.append(bd)
#             bond_dir_i.append(bd)
#             stereo_i.append(s)
#             stereo_i.append(s)

#         edge_index_i = torch.tensor([row_i, col_i], dtype=torch.long)

#         bond_i = F.one_hot(torch.tensor(bond_i, dtype=torch.long), num_classes=len(BOND_LIST))
#         bond_dir_i = F.one_hot(torch.tensor(bond_dir_i, dtype=torch.long), num_classes=len(BONDDIR_LIST))
#         stereo_i = F.one_hot(torch.tensor(stereo_i, dtype=torch.long), num_classes=len(STEREO_LIST))
#         edge_attr_i = torch.cat([bond_i, bond_dir_i, stereo_i], dim=-1).type(torch.FloatTensor)

#         if self.task == 'classification':
#             y = torch.tensor(self.labels[index], dtype=torch.long).view(1,-1)
#         elif self.task == 'regression':
#             y = torch.tensor(self.labels[index], dtype=torch.float).view(1,-1)
#         data = Data(x=x, y=y, edge_index=edge_index_i, edge_attr=edge_attr_i)
        
#         return data

#     def __len__(self):
#         return len(self.smiles_data)


# class MolTestDatasetWrapper(object):
#     def __init__(self, batch_size, num_workers, valid_size, test_size, data_path, target, task):
#         super(object, self).__init__()
#         self.data_path = data_path
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.valid_size = valid_size
#         self.test_size = test_size
#         self.target = target
#         self.task = task
#         self.smiles_data, self.labels = read_smiles(data_path, target, task)
#         self.smiles_data = np.asarray(self.smiles_data)
#         self.labels = np.asarray(self.labels)

#     def get_data_loaders(self):
#         train_idx, valid_idx, test_idx = scaffold_split(self.smiles_data, self.valid_size, self.test_size)

#         # define dataset
#         train_set = MolTestDataset(self.smiles_data[train_idx], self.labels[train_idx], task=self.task)
#         valid_set = MolTestDataset(self.smiles_data[valid_idx], self.labels[valid_idx], task=self.task)
#         test_set = MolTestDataset(self.smiles_data[test_idx], self.labels[test_idx], task=self.task)

#         train_loader = DataLoader(
#             train_set, batch_size=self.batch_size, 
#             num_workers=self.num_workers, drop_last=True, shuffle=True
#         )
#         valid_loader = DataLoader(
#             valid_set, batch_size=self.batch_size, 
#             num_workers=self.num_workers, drop_last=False
#         )
#         test_loader = DataLoader(
#             test_set, batch_size=self.batch_size, 
#             num_workers=self.num_workers, drop_last=False
#         )

#         return train_loader, valid_loader, test_loader
