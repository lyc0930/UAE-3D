import os
import os.path as osp
from typing import Callable, List, Optional

import numpy as np
from rdkit.Chem import rdMolTransforms
import torch
from torch_geometric.data.makedirs import makedirs
from tqdm import tqdm
import copy
from functools import lru_cache
from data_provider.utils import featurize_mol, get_full_edge

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.datasets import QM9
from openbabel import openbabel, pybel




HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])

atomrefs = {
    6: [0., 0., 0., 0., 0.],
    7: [
        -13.61312172, -1029.86312267, -1485.30251237, -2042.61123593,
        -2713.48485589
    ],
    8: [
        -13.5745904, -1029.82456413, -1485.26398105, -2042.5727046,
        -2713.44632457
    ],
    9: [
        -13.54887564, -1029.79887659, -1485.2382935, -2042.54701705,
        -2713.42063702
    ],
    10: [
        -13.90303183, -1030.25891228, -1485.71166277, -2043.01812778,
        -2713.88796536
    ],
    11: [0., 0., 0., 0., 0.],
}

Cv_atomref = [2.981, 2.981, 2.981, 2.981, 2.981]


def files_exist(files: List[str]) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])


def construct_mol(atoms, coordinates, title=None):
    mol = openbabel.OBMol()
    for atom, (x, y, z) in zip(atoms, coordinates):
        ob_atom = mol.NewAtom()
        ob_atom.SetAtomicNum(atom)
        ob_atom.SetVector(x, y, z)
    mol.ConnectTheDots()
    mol.PerceiveBondOrders()
    if title:
        mol.SetTitle(title)
    return mol


class QM9(InMemoryDataset):
    r"""The QM9 dataset from the `"MoleculeNet: A Benchmark for Molecular
    Machine Learning" <https://arxiv.org/abs/1703.00564>`_ paper, consisting of
    about 130,000 molecules with 19 regression targets.
    Each molecule includes complete spatial information for the single low
    energy conformation of the atoms in the molecule.
    In addition, we provide the atom features from the `"Neural Message
    Passing for Quantum Chemistry" <https://arxiv.org/abs/1704.01212>`_ paper.

    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | Target | Property                         | Description                                                                       | Unit                                        |
    +========+==================================+===================================================================================+=============================================+
    | 0      | :math:`\mu`                      | Dipole moment                                                                     | :math:`\textrm{D}`                          |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 1      | :math:`\alpha`                   | Isotropic polarizability                                                          | :math:`{a_0}^3`                             |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 2      | :math:`\epsilon_{\textrm{HOMO}}` | Highest occupied molecular orbital energy                                         | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 3      | :math:`\epsilon_{\textrm{LUMO}}` | Lowest unoccupied molecular orbital energy                                        | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 4      | :math:`\Delta \epsilon`          | Gap between :math:`\epsilon_{\textrm{HOMO}}` and :math:`\epsilon_{\textrm{LUMO}}` | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 5      | :math:`\langle R^2 \rangle`      | Electronic spatial extent                                                         | :math:`{a_0}^2`                             |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 6      | :math:`\textrm{ZPVE}`            | Zero point vibrational energy                                                     | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 7      | :math:`U_0`                      | Internal energy at 0K                                                             | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 8      | :math:`U`                        | Internal energy at 298.15K                                                        | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 9      | :math:`H`                        | Enthalpy at 298.15K                                                               | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 10     | :math:`G`                        | Free energy at 298.15K                                                            | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 11     | :math:`c_{\textrm{v}}`           | Heat capacity at 298.15K                                                          | :math:`\frac{\textrm{cal}}{\textrm{mol K}}` |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 12     | :math:`U_0^{\textrm{ATOM}}`      | Atomization energy at 0K                                                          | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 13     | :math:`U^{\textrm{ATOM}}`        | Atomization energy at 298.15K                                                     | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 14     | :math:`H^{\textrm{ATOM}}`        | Atomization enthalpy at 298.15K                                                   | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 15     | :math:`G^{\textrm{ATOM}}`        | Atomization free energy at 298.15K                                                | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 16     | :math:`A`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 17     | :math:`B`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 18     | :math:`C`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)

    Stats:
        .. list-table::
            :widths: 10 10 10 10 10
            :header-rows: 1

            * - #graphs
              - #nodes
              - #edges
              - #features
              - #tasks
            * - 130,831
              - ~18.0
              - ~37.3
              - 11
              - 19
    """  # noqa: E501

    raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
               'molnet_publish/qm9.zip')
    raw_url2 = 'https://ndownloader.figshare.com/files/3195404'

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None, addHs=False):
        self.addHs = addHs # note, this parameter on applies on selfies but not rdmols
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def atomref(self, target) -> Optional[torch.Tensor]:
        if target in atomrefs:
            out = torch.zeros(100)
            out[torch.tensor([1, 6, 7, 8, 9])] = torch.tensor(atomrefs[target])
            return out.view(-1, 1)
        return None

    @property
    def raw_file_names(self) -> List[str]:
        return ['gdb9.sdf', 'gdb9.sdf.csv', 'uncharacterized.txt']

    @property
    def processed_file_names(self) -> str:
        return ['data_qm9.pt', 'babel_mol.sdf', 'id2id.txt']

    def _download(self):
        if files_exist(self.processed_paths):
            return

        if files_exist(self.raw_paths):  # pragma: no cover
            return

        makedirs(self.raw_dir)
        self.download()

    def download(self):
        file_path = download_url(self.raw_url, self.raw_dir)
        extract_zip(file_path, self.raw_dir)
        os.unlink(file_path)

        file_path = download_url(self.raw_url2, self.raw_dir)
        os.rename(osp.join(self.raw_dir, '3195404'),
                  osp.join(self.raw_dir, 'uncharacterized.txt'))


    def process(self):
        try:
            import rdkit
            from rdkit import Chem, RDLogger
            from rdkit.Chem.rdchem import BondType as BT
            from rdkit.Chem.rdchem import HybridizationType
            RDLogger.DisableLog('rdApp.*')

        except ImportError:
            raise ImportError("Please install 'rdkit' to alternatively process the raw data.")

        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        bonds = {BT.SINGLE: 1, BT.DOUBLE: 2, BT.TRIPLE: 3, BT.AROMATIC: 4}  # 0 -> without edge
        charge_dict = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}

        with open(self.raw_paths[1], 'r') as f:
            target = f.read().split('\n')[1:-1]
            target = [[float(x) for x in line.split(',')[1:20]]
                      for line in target]
            target = torch.tensor(target, dtype=torch.float)
            target = torch.cat([target[:, 3:], target[:, :3]], dim=-1)
            target = target * conversion.view(1, -1)

        with open(self.raw_paths[2], 'r') as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

        if not osp.exists(self.processed_paths[1]) or not osp.exists(self.processed_paths[2]):
            babel_mol_list = []
            name_list = []
            suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False, sanitize=False)

            id2id = {}
            new_id = 0
            split_id = 0
            for qm9_id, mol in enumerate(tqdm(suppl)):
                if qm9_id in skip:
                    continue
                try:
                    Chem.SanitizeMol(mol)
                except:
                    split_id += 1
                    continue
                id2id[new_id] = (qm9_id, split_id)
                name = mol.GetProp('_Name')
                babel_mol = construct_mol([atom.GetAtomicNum() for atom in mol.GetAtoms()], mol.GetConformer().GetPositions(), name)
                babel_mol_list.append(babel_mol)
                name_list.append(name)

                new_id += 1
                split_id += 1

            output = pybel.Outputfile("sdf", self.processed_paths[1], overwrite=True)
            for mol in babel_mol_list:
                mol = pybel.Molecule(mol)
                # Write each molecule to the SDF file
                output.write(mol)
            output.close()
            with open(self.processed_paths[2], 'w') as f:
                for k, (v1, v2) in id2id.items():
                    f.write(f'{k} {v1} {v2}\n')

        with open(self.processed_paths[2], 'r') as f:
            id2id = {}
            for line in f:
                k, v1, v2 = line.split()
                id2id[int(k)] = (int(v1), int(v2))

        suppl = Chem.SDMolSupplier(self.processed_paths[1], removeHs=False, sanitize=False)
        data_list = []

        for new_id, mol in enumerate(tqdm(suppl)):
            try:
                Chem.SanitizeMol(mol)
            except:
                continue

            name = mol.GetProp('_Name')
            N = mol.GetNumAtoms()
            data = featurize_mol(mol, dataset='qm9_with_h')
            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)

            type_idx = []
            charges = []
            formal_charges = []

            for atom in mol.GetAtoms():
                atom_str = atom.GetSymbol()
                type_idx.append(types[atom_str])
                charges.append(charge_dict[atom_str])
                formal_charges.append(atom.GetFormalCharge())

            qm9_id, split_id = id2id[new_id]
            y = target[qm9_id].unsqueeze(0)
            data['idx'] = qm9_id
            data['split_id'] = split_id
            data['rdmol'] = copy.deepcopy(mol)
            data['gdb_id'] = name
            data['y'] = y
            data['num_atom'] = N
            data['pos'] = pos
            data['charge']=torch.tensor(charges)
            data['formal_charge']=torch.tensor(formal_charges)

            if self.pre_filter is not None and not self.pre_filter(data):
                assert False
                continue
            if self.pre_transform is not None:
                assert False
                data = self.pre_transform(data)
            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])

    def get_idx_split(self):
        # load split idx for train, val, test
        split_path = osp.join(self.processed_dir, 'split_dict_qm9.pt')
        if osp.exists(split_path):
            # print('Loading existing split data.')
            return torch.load(split_path)

        data_num = len(self.indices())
        # assert data_num == 130831
        # data_num = 127924
        train_num = 100000
        test_num = int(0.1 * data_num)
        valid_num = data_num - (train_num + test_num)

        # Generate random permutation
        np.random.seed(0)
        data_perm = np.random.permutation(data_num)
        train, valid, test, extra = np.split(
            data_perm, [train_num, train_num + valid_num, train_num + valid_num + test_num])

        split_id = self._data.split_id.tolist()
        split_id2new_id = {}
        for i, idx in enumerate(split_id):
            split_id2new_id[idx] = i

        train = train.tolist()
        valid = valid.tolist()
        test = test.tolist()
        train = [split_id2new_id[x] for x in train if x in split_id2new_id]
        valid = [split_id2new_id[x] for x in valid if x in split_id2new_id]
        test = [split_id2new_id[x] for x in test if x in split_id2new_id]
        train = np.array(train)
        valid = np.array(valid)
        test = np.array(test)

        train = np.array(self.indices())[train]
        valid = np.array(self.indices())[valid]
        test = np.array(self.indices())[test]

        splits = {'train': train, 'valid': valid, 'test': test}
        torch.save(splits, split_path)
        return splits

    def get_cond_idx_split(self):
        # load conditional generation split idx for first train, second train, val, test
        split_path = osp.join(self.processed_dir, 'split_dict_cond_qm9.pt')
        splits = torch.load(split_path)

        split_id = self._data.split_id.tolist()
        split_id2new_id = {}
        for i, idx in enumerate(split_id):
            split_id2new_id[idx] = i

        for split in ['first_train', 'second_train', 'valid', 'test']:
            splits[split] = [split_id2new_id[x] for x in splits[split] if x in split_id2new_id]
            splits[split] = np.array(splits[split])

        return splits

    def compute_property_mean_mad(self, prop2idx):
        prop_values = []

        prop_ids = torch.tensor(list(prop2idx.values()))
        for idx in range(len(self.indices())):
            data = self.get(self.indices()[idx])
            tars = []
            for prop_id in prop_ids:
                if prop_id == 11:
                    tars.append(self.sub_Cv_thermo(data).reshape(1))
                else:
                    tars.append(data.y[0][prop_id].reshape(1))
            tars = torch.cat(tars)
            prop_values.append(tars)
        prop_values = torch.stack(prop_values, dim=0)
        mean = torch.mean(prop_values, dim=0, keepdim=True)
        ma = torch.abs(prop_values - mean)
        mad = torch.mean(ma, dim=0)

        prop_norm = {}
        for tmp_i, key in enumerate(prop2idx.keys()):
            prop_norm[key] = {
                'mean': mean[0, tmp_i].item(),
                'mad': mad[tmp_i].item()
            }
        return prop_norm

    # add the property of Cv thermo
    @staticmethod
    def sub_Cv_thermo(data):
        atom_types = data.atom_type
        atom_counts = torch.bincount(atom_types, minlength=len(Cv_atomref))
        property = data.y[0, 11] - torch.sum((atom_counts * torch.tensor(Cv_atomref)))
        return property


class QM9Dataset(QM9):
    def download(self):
        super(QM9Dataset, self).download()

    def process(self):
        super(QM9Dataset, self).process()

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        data = self.get(self.indices()[idx])
        data = data if self.transform is None else self.transform(data)

        data['pos'] -= data['pos'].mean(dim=0, keepdim=True)

        # rdmol = copy.deepcopy(data['rdmol'])
        # rdmol.RemoveAllConformers()
        # data['rdmol'] = set_rdmol_positions(rdmol, data['pos'], removeHs=False)
        # assert data['rdmol'].GetNumConformers() == 1

        # if self.use_full_edge:
        full_edge_index, full_edge_attr = get_full_edge(data.x, data.edge_index, data.edge_attr)
        data['edge_index'] = full_edge_index
        data['edge_attr'] = full_edge_attr
        return data



