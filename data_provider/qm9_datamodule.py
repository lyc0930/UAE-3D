from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader

from data_provider.utils import DataCollater, SimpleDataset, SimpleCollater, get_dataset_info, get_node_dist, datamodule_setup_evaluator, DistributionProperty, ConditionTransform
from data_provider.qm9_dataset import QM9Dataset

class QM9VAEDataModule(L.LightningDataModule):
    def __init__(
        self,
        root: str = 'data/QM9',
        num_workers: int = 0,
        batch_size: int = 256,
        aug_rotation: bool = True,
        aug_translation: bool = True,
        aug_translation_scale: float = 0.1,
        condition_property=None
    ):
        super().__init__()
        self.root = Path(root)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.aug_rotation = aug_rotation
        self.aug_translation = aug_translation
        self.aug_translation_scale = aug_translation_scale


        if condition_property is None:
            self.conditional = False
            dataset_info = get_dataset_info('qm9_with_h')

            dataset = QM9Dataset(root=root)
            splits = dataset.get_idx_split()
            self._train_idx = splits['train']
            self._valid_idx = splits['valid']
            self._test_idx = splits['test']
        else:
            self.conditional = True
            dataset_info = get_dataset_info('qm9_second_half')

            prop2idx = dataset_info['prop2idx']
            if condition_property in prop2idx:
                property = condition_property
                self.condition_property = property
                self.transform = ConditionTransform(dataset_info['atom_encoder'].values(), prop2idx[property])

                prop2idx_sub = {
                    property: prop2idx[property]
                }
            elif '&' in condition_property:
                properties = condition_property.split('&')
                if len(properties) != 2 or any(property not in prop2idx for property in properties):
                    raise NotImplementedError(f"{condition_property} is not supported")
                self.condition_property = properties
                self.transform = ConditionTransform(dataset_info['atom_encoder'].values(), prop2idx[properties[0]], prop2idx[properties[1]])

                prop2idx_sub = {
                    properties[0]: prop2idx[properties[0]],
                    properties[1]: prop2idx[properties[1]]
                }
            else:
                raise NotImplementedError(f"{condition_property} is not supported")

            dataset = QM9Dataset(root=root, transform=self.transform)

            splits = dataset.get_cond_idx_split()
            first_train_idx = splits['first_train']
            second_train_idx = splits['second_train']
            self._train_idx = second_train_idx
            self._valid_idx = splits['valid']
            self._test_idx = splits['test']

            self.prop_norms = dataset.index_select(self._valid_idx).compute_property_mean_mad(prop2idx_sub)
            self.prop_dist = DistributionProperty(dataset.index_select(self._train_idx), prop2idx_sub, normalizer=self.prop_norms)

        self.dataset_info = dataset_info
        self.dataset = dataset

        self.nodes_dist = get_node_dist(dataset_info)

        self.train_dataset = dataset.index_select(self._train_idx)
        self.valid_dataset = dataset.index_select(self._valid_idx)
        self.test_dataset = dataset.index_select(self._test_idx)

        rdmols = dataset._data.rdmol
        self.train_rdmols = [rdmols[i] for i in self._train_idx]
        # self.valid_rdmols = [rdmols[i] for i in self._valid_idx]
        self.test_rdmols = [rdmols[i] for i in self._test_idx]

        # self.n_atom_types = len(set([atom.GetSymbol() for rdmol in rdmols for atom in rdmol.GetAtoms()]))
        self.n_atom_types = 5
        # self.n_bond_types = len(set([bond.GetBondType() for rdmol in rdmols for bond in rdmol.GetBonds()]))
        self.n_bond_types = 4

        # max_atoms = int(dataset._data.num_atom.max()) + 2
        max_atoms = 31 + 2
        self.max_atoms = max_atoms

        # position_std = torch.std(torch.cat([rdmol.GetConformers()[0].GetPositions() for rdmol in rdmols], dim=0)).item()
        # train 1.3988368511199951
        # valid 1.400497555732727
        # test  1.4015264511108398
        # all   1.3981486558914185
        position_std = 1.3981
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
            position_std=self.position_std,
            conditional=self.conditional,
            prop_norm=self.prop_norms if self.conditional else None
        )(batch)

class QM9LDMDataModule(QM9VAEDataModule):
    def __init__(self,
        root: str = 'data/QM9',
        num_workers: int = 0,
        batch_size: int = 256,
        aug_rotation: bool = True,
        aug_translation: bool = False,
        aug_translation_scale: float = 0.1,
        condition_property=None,
        num_samples=10000
    ):
        super().__init__(root, num_workers, batch_size, aug_rotation, aug_translation, aug_translation_scale, condition_property)
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
