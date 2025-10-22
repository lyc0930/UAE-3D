from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader

from data_provider.geom_drugs_dataset import GEOMDrugsDataset
from data_provider.utils import DataCollater, SimpleDataset, SimpleCollater, get_dataset_info, get_node_dist, datamodule_setup_evaluator

class GEOMDrugsVAEDataModule(L.LightningDataModule):
    def __init__(
        self,
        root: str = 'data/GEOMDrugs',
        num_workers: int = 0,
        batch_size: int = 256,
        aug_rotation: bool = True,
        aug_translation: bool = True,
        aug_translation_scale: float = 0.1,
    ):
        super().__init__()
        self.root = Path(root)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.aug_rotation = aug_rotation
        self.aug_translation = aug_translation
        self.aug_translation_scale = aug_translation_scale

        self.dataset_info = get_dataset_info('geom_with_h_1')
        dataset = GEOMDrugsDataset(root=root, addHs=False)

        self.nodes_dist = get_node_dist(self.dataset_info)

        splits = dataset.get_idx_split()
        self._train_idx = splits['train']
        self._valid_idx = splits['valid']
        self._test_idx = splits['test']

        self.dataset = dataset

        self.train_dataset = dataset.index_select(self._train_idx)
        self.valid_dataset = dataset.index_select(self._valid_idx)
        self.test_dataset = dataset.index_select(self._test_idx)

        rdmols = dataset.data.rdmol
        self.train_rdmols = [rdmols[i] for i in self._train_idx]
        # self.valid_rdmols = [rdmols[i] for i in self._valid_idx]
        self.test_rdmols = [rdmols[i] for i in self._test_idx]

        # self.n_atom_types = len(set([atom.GetSymbol() for rdmol in rdmols for atom in rdmol.GetAtoms()]))
        self.n_atom_types = 16 # ! actually 13
        # self.n_bond_types = len(set([bond.GetBondType() for rdmol in rdmols for bond in rdmol.GetBonds()]))
        self.n_bond_types = 4

        # max_atoms = max([rdmol.GetNumAtoms() for rdmol in rdmols]) + 2
        max_atoms = 181 + 2
        self.max_atoms = max_atoms

        # position_std = torch.std(torch.cat([rdmol.GetConformers()[0].GetPositions() for rdmol in rdmols], dim=0)).item()
        # train 2.386332925400705
        # valid 2.3831518739523503
        # test  2.385923923561268
        # all   2.3859748144389545
        position_std = 2.3859
        self.position_std = position_std

    def setup_evaluator(self):
        datamodule_setup_evaluator(self)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.collate_fn,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            collate_fn=self.collate_fn
        )

    def collate_fn(self, batch):
        return DataCollater(
            aug_rotation=self.aug_rotation,
            aug_translation=self.aug_translation,
            aug_translation_scale=self.aug_translation_scale,
            position_std=self.position_std
        )(batch)

class GEOMDrugsLDMDataModule(GEOMDrugsVAEDataModule):
    def __init__(self,
        root: str = 'data/GEOMDrugs',
        num_workers: int = 0,
        batch_size: int = 256,
        aug_rotation: bool = True,
        aug_translation: bool = False,
        aug_translation_scale: float = 0.1,
        condition_property=None,
        num_samples=10000
    ):
        super().__init__(root, num_workers, batch_size, aug_rotation, aug_translation, aug_translation_scale)
        self.test_dataset = SimpleDataset(num_samples)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            collate_fn=SimpleCollater()
        )
