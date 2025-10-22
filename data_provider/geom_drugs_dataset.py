import copy
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
from torch_geometric.data import Dataset, InMemoryDataset
from torch_geometric.data.in_memory_dataset import nested_iter
from torch_geometric.data.separate import separate
from tqdm import tqdm

from data_provider.utils import featurize_mol, get_full_edge, set_rdmol_positions


class GEOMDrugsDataset(Dataset):
    def __init__(self, root, addHs=False):
        super().__init__()
        self.root = Path(root)
        self.addHs = addHs
        if (processed_file := self.root / 'processed' / self.processed_file_names[0]).exists():
            self.data, self.slices = torch.load(processed_file)
        else:
            processed_file.parent.mkdir(parents=True, exist_ok=True)
            self.data, self.slices = self.reprocess()

    def len(self):
        if self.slices is None:
            return 1
        for _, value in nested_iter(self.slices):
            return len(value) - 1
        return 0

    def get(self, idx: int):
        # TODO (matthias) Avoid unnecessary copy here.
        if not hasattr(self, '_data_list') or self._data_list is None:
            self._data_list = self.len() * [None]
        elif self._data_list[idx] is not None:
            return self._data_list[idx].clone()

        data = separate(
            cls=self.data.__class__,
            batch=self.data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False,
        )

        self._data_list[idx] = data.clone()
        return data.clone()

    def __getitem__(self, idx):
        idx = self.indices()[idx]
        data = self.get(idx)
        data['idx'] = idx

        data['rdmol'] = copy.deepcopy(data['rdmol'])
        assert data['rdmol'].GetNumConformers() == 1

        full_edge_index, full_edge_attr = get_full_edge(data.x, data.edge_index, data.edge_attr)
        data['edge_index'] = full_edge_index
        data['edge_attr'] = full_edge_attr
        return data

    def get_idx_split(self):
        # load split idx for train, val, test
        split_path = self.root / 'preprocessed' / 'split_dict_geom_drug_1.pt'
        # print('Loading existing split data.')
        return torch.load(split_path)

    @property
    def preprocessed_file_names(self):
        return ['data_geom_drug_1.pt', 'split_dict_geom_drug_1.pt']

    @property
    def processed_file_names(self):
        return ['processed_data.pt']

    def reprocess(self):
        preprocessed_dataset = _GEOMDrugsDataset(self.root)
        data_list = []
        for data in tqdm(preprocessed_dataset.data):
            rdmol = data['rdmol']
            data = featurize_mol(rdmol, dataset='geom_with_h_1')
            pos = rdmol.GetConformers()[0].GetPositions()
            pos = pos - pos.mean(axis=0, keepdims=True)
            rdmol.RemoveAllConformers()
            rdmol = set_rdmol_positions(rdmol, pos, removeHs=False)
            assert rdmol.GetNumConformers() == 1
            data['rdmol'] = rdmol
            data['pos'] = torch.from_numpy(pos)
            data_list.append(data)

        data, slices = InMemoryDataset.collate(data_list)
        torch.save((data, slices), self.root / 'processed' / self.processed_file_names[0])

        return data, slices



class _GEOMDrugsDataset(Dataset):
    def __init__(self, root: str, transform: Optional[Callable] = None):
        super().__init__()
        self.root = Path(root)
        self.data = torch.load(self.root / 'preprocessed' / 'data_geom_drug_1.pt')
        self.transform = transform

    def __getitem__(self, idx):
        data = copy.copy(self.data[self.indices()[idx]])
        data = data if self.transform is None else self.transform(data)
        data.idx = idx

        return data

    def len(self):
        return len(self.data)

    def get_idx_split(self):
        # load split idx for train, val, test
        split_path = self.root / 'preprocessed' / 'split_dict_geom_drug_1.pt'
        if split_path.exists():
            # print('Loading existing split data.')
            return torch.load(split_path)

        data_num = len(self.indices())
        print(f"Total number of data: {data_num}")
        # assert data_num == 304294

        valid_proportion = 0.1
        test_proportion = 0.1

        valid_num = int(valid_proportion * data_num)
        test_num = int(test_proportion * data_num)
        train_num = data_num - (valid_num + test_num)

        # Generate random permutation
        np.random.seed(0)
        data_perm = np.random.permutation(data_num)
        train, valid, test, _ = np.split(
        data_perm, [train_num, train_num + valid_num, train_num + valid_num + test_num])

        train = np.array(self.indices())[train]
        valid = np.array(self.indices())[valid]
        test = np.array(self.indices())[test]

        splits = {'train': train, 'valid': valid, 'test': test}
        torch.save(splits, split_path)
        return splits
