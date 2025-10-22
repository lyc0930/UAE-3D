import copy
from itertools import product
from typing import Any, List

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import ChiralType

import torch
import torch.nn.functional as F
# from torch_scatter import scatter
from torch_geometric.data import Dataset, Data, DataLoader
from torch_geometric.utils.smiles import x_map, e_map

from data_provider.utils import get_dataset_info

dihedral_pattern = Chem.MolFromSmarts('[*]~[*]~[*]~[*]')
chirality = {ChiralType.CHI_TETRAHEDRAL_CW: -1.,
             ChiralType.CHI_TETRAHEDRAL_CCW: 1.,
             ChiralType.CHI_UNSPECIFIED: 0,
             ChiralType.CHI_OTHER: 0}

bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}  # ! BT.UNSPECIFIED: 0 -> without edge, so remind to cat after

def one_k_encoding(value, choices):
    """
    Creates a one-hot encoding with an extra category for uncommon values.
    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
             If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def from_rdmol(mol: Any) -> 'Data':
    r"""Converts a :class:`rdkit.Chem.Mol` instance to a
    :class:`torch_geometric.data.Data` instance.
    Modified from :func:`torch_geometric.utils.smiles.from_rdmol`.

    Args:
        mol (rdkit.Chem.Mol): The :class:`rdkit` molecule.
    """

    x_map['hybridization'] = [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP2D', # ! not included in torch_geometric.utils.smiles.e_map
        'SP3D',
        'SP3D2',
        'OTHER',
    ]
    e_map['stereo'] += [
        'STEREOATROPCW',
        'STEREOATROPCCW',
    ] # ! not included in torch_geometric.utils.smiles.e_map

    assert isinstance(mol, Chem.Mol)

    xs: List[List[int]] = []
    for atom in mol.GetAtoms():
        row: List[int] = []
        row.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
        row.append(x_map['chirality'].index(str(atom.GetChiralTag())))
        row.append(x_map['degree'].index(atom.GetTotalDegree()))
        row.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
        row.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
        row.append(x_map['num_radical_electrons'].index(
            atom.GetNumRadicalElectrons()))
        row.append(x_map['hybridization'].index(str(atom.GetHybridization())))
        row.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
        row.append(x_map['is_in_ring'].index(atom.IsInRing()))
        xs.append(row)

    x = torch.tensor(xs, dtype=torch.long).view(-1, 9)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = []
        e.append(e_map['bond_type'].index(str(bond.GetBondType())))
        e.append(e_map['stereo'].index(str(bond.GetStereo())))
        e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def featurize_mol(mol, dataset='qm9_with_h'):
    """
    Part of the featurisation code taken from GeoMol https://github.com/PattanaikL/GeoMol
    Returns:
        x:  node features
        z: atomic numbers of the nodes (the symbol one hot is included in x)
        edge_index: [2, E] tensor of node indices forming edges
        edge_attr: edge features
    """
    dataset_info = get_dataset_info(dataset)
    atom_encoder = dataset_info['atom_encoder']

    N = mol.GetNumAtoms()
    atom_type_idx = []
    atomic_number = []
    atom_features = []
    chiral_tag = []
    ring = mol.GetRingInfo()
    for i, atom in enumerate(mol.GetAtoms()):
        atom_type_idx.append(atom_encoder[atom.GetSymbol()])
        chiral_tag.append(chirality[atom.GetChiralTag()])
        atomic_number.append(atom.GetAtomicNum())
        atom_features.extend([atom.GetAtomicNum(),
                              1 if atom.GetIsAromatic() else 0])
        atom_features.extend(one_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]))
        atom_features.extend(one_k_encoding(atom.GetHybridization(), [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2]))
        atom_features.extend(one_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]))
        atom_features.extend(one_k_encoding(atom.GetFormalCharge(), [-1, 0, 1]))
        atom_features.extend([int(ring.IsAtomInRingOfSize(i, 3)),
                              int(ring.IsAtomInRingOfSize(i, 4)),
                              int(ring.IsAtomInRingOfSize(i, 5)),
                              int(ring.IsAtomInRingOfSize(i, 6)),
                              int(ring.IsAtomInRingOfSize(i, 7)),
                              int(ring.IsAtomInRingOfSize(i, 8))])
        atom_features.extend(one_k_encoding(int(ring.NumAtomRings(i)), [0, 1, 2, 3]))

    z = torch.tensor(atomic_number, dtype=torch.long)

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bonds[bond.GetBondType()]]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=len(bonds)).to(torch.float)

    x1 = F.one_hot(torch.tensor(atom_type_idx), num_classes=len(atom_encoder))
    x2 = torch.tensor(atom_features).view(N, -1)
    x = torch.cat([x1.to(torch.float), x2], dim=-1)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, z=z)

def featurize_mol_from_smiles(smiles, dataset='qm9_with_h'):
    # filter fragments
    if '.' in smiles:
        return None, None

    # filter mols rdkit can't intrinsically handle
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mol = Chem.AddHs(mol)
    else:
        return None, None

    # N = mol.GetNumAtoms()

    # # filter out mols model can't make predictions for
    # if not mol.HasSubstructMatch(dihedral_pattern):
    #     return None, None
    # if N < 4:
    #     return None, None

    data = featurize_mol(mol, dataset)
    data.name = smiles
    return mol, data

def set_rdmol_positions(rdkit_mol, pos, removeHs=True, add_conformer=False):
    rdkit_mol = copy.deepcopy(rdkit_mol)
    if removeHs:
        rdkit_mol = Chem.RemoveHs(rdkit_mol)
    if add_conformer:
        num_atoms = rdkit_mol.GetNumAtoms()
        assert num_atoms == pos.shape[0], f'{num_atoms} != {pos.shape[0]}, {pos.shape=}'
        conf = Chem.Conformer(num_atoms)
        for i in range(num_atoms):
            conf.SetAtomPosition(i, pos[i].tolist())
        rdkit_mol.AddConformer(conf, assignId=True)
    elif rdkit_mol.GetNumConformers() == 0:
        num_atoms = rdkit_mol.GetNumAtoms()
        assert num_atoms == pos.shape[0], f'{num_atoms} != {pos.shape[0]}, {pos.shape=}'
        conf = Chem.Conformer(num_atoms)
        for i in range(num_atoms):
            conf.SetAtomPosition(i, pos[i].tolist())
        rdkit_mol.AddConformer(conf, assignId=True)
    else:
        assert rdkit_mol.GetConformer(0).GetPositions().shape[0] == pos.shape[0], print(rdkit_mol.GetConformer(0).GetPositions().shape, pos.shape)
        for i in range(pos.shape[0]):
            rdkit_mol.GetConformer(0).SetAtomPosition(i, pos[i].tolist())
    return rdkit_mol

def get_full_edge(x, edge_index, edge_attr):
    num_atom = x.shape[0]
    indices = list(product(range(num_atom), range(num_atom)))
    indices.sort(key=lambda x: (x[0], x[1]))
    full_edge_index = torch.LongTensor(indices).t() # [2, N_edge]
    full_edge_attr = torch.zeros((num_atom, num_atom, edge_attr.size(1)), dtype=edge_attr.dtype) # [num_atom, num_atom, edge_nf]
    full_edge_attr[edge_index[0], edge_index[1]] = edge_attr
    full_edge_attr = torch.cat([full_edge_attr, torch.eye(num_atom, dtype=full_edge_attr.dtype).unsqueeze(-1)], dim=-1) # [num_atom, num_atom, edge_nf + 1], add self-loop edge_attr
    full_edge_attr = full_edge_attr.reshape(-1, full_edge_attr.size(-1)) # [N_edge, edge_nf + 1]
    return full_edge_index, full_edge_attr
