import copy
from scipy.spatial.transform import Rotation
import torch
from torch_geometric.data import Batch, Data

from evaluation.jodo import get_edm_metric, get_2D_edm_metric, get_sub_geometry_metric, get_moses_metrics, get_fcd_metric

def add_datamodule_specific_args(parent_parser):
    parser = parent_parser.add_argument_group("Data Module")
    parser.add_argument('--dataset', type=str, required=True, choices=['qm9', 'drugs'])
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--aug_rotation', action='store_true', default=True)
    parser.add_argument('--not_aug_rotation', action='store_false', dest='aug_rotation')
    parser.add_argument('--aug_translation', action='store_true', default=True)
    parser.add_argument('--not_aug_translation', action='store_false', dest='aug_translation')
    parser.add_argument('--aug_translation_scale', type=float, default=0.1)

class DataCollater():
    def __init__(self, aug_rotation=False, aug_translation=False, aug_translation_scale=0.01, position_std=1.3981, conditional=False, prop_norm=None):
        self.aug_rotation = aug_rotation
        self.aug_translation = aug_translation
        self.aug_translation_scale = aug_translation_scale
        self.position_std = position_std
        self.conditional = conditional
        self.prop_norm = prop_norm

    def augmentation(self, data):
        bs = len(data['ptr']) - 1
        dtype = torch.float
        if self.aug_rotation:
            rot_aug = Rotation.random(bs)
            rot_aug = rot_aug[data.batch.numpy()]
            data['pos'] = torch.from_numpy(rot_aug.apply(data['pos'].numpy())).to(dtype)

        if self.aug_translation:
            trans_aug = self.aug_translation_scale * torch.randn(bs, 3, dtype=dtype)
            data['pos'] = data['pos'] + trans_aug[data.batch]

        return data

    def __call__(self, data_list):
        ## graph batch
        data_batch = Batch.from_data_list(data_list)
        # position = (position - position.mean(dim=0, keepdim=True))
        data_batch.pos = data_batch.pos / self.position_std
        data_batch = self.augmentation(data_batch)
        data_batch.x = data_batch.x.to(torch.float)
        data_batch.edge_attr = data_batch.edge_attr.to(torch.float)

        if self.conditional:
            context = copy.deepcopy(data_batch.property)
            for i, key in enumerate(self.prop_norm.keys()):
                context[:, i] = (context[:, i] - self.prop_norm[key]['mean']) / self.prop_norm[key]['mad']
            data_batch.context = context

        return data_batch

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=10000):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        data = Data(
            idx=idx,
        )
        return data

class SimpleCollater(object):
    def __init__(self):
        pass

    def __call__(self, data_list):
        return Batch.from_data_list(data_list)

def datamodule_setup_evaluator(datamodule):
    datamodule.evaluate_3D_edm = get_edm_metric(datamodule.dataset_info, datamodule.train_rdmols)
    datamodule.evaluate_sub_geometry = get_sub_geometry_metric(datamodule.test_rdmols, datamodule.dataset_info, datamodule.root / 'processed')
    datamodule.evaluate_2D_edm = get_2D_edm_metric(datamodule.dataset_info, datamodule.train_rdmols)
    datamodule.evaluate_moses = get_moses_metrics(datamodule.test_rdmols, n_jobs=1)

    # datamodule.evaluate_fcd = get_fcd_metric(datamodule.test_rdmols, n_jobs=1)