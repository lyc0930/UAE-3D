import torch
from torch_geometric.utils import to_dense_batch, to_dense_adj
from model.utils import get_align_pos, remove_mean


def atom_prediction_accuracy(atom_logits, atom_labels, batch):
    assert len(atom_logits.size()) == 2  # [B * N, n_atom_types]
    logits, atom_mask = to_dense_batch(atom_logits, batch)
    labels, _ = to_dense_batch(atom_labels, batch)

    predictions = logits.argmax(dim=-1)  # [B, N]
    groundtruth = labels.argmax(dim=-1)  # [B, N]

    correct = (predictions == groundtruth) & atom_mask  # [B, N]
    accuracy = correct.sum() / atom_mask.sum()

    return accuracy

def bond_prediction_accuracy(bond_logits, bond_labels):
    logits = bond_logits
    labels = bond_labels

    predictions = logits.argmax(dim=-1)  # [B, N, N, n_bond_types + 1]
    groundtruth = labels.argmax(dim=-1)  # [B, N, N, n_bond_types + 1]

    correct = (predictions == groundtruth) # [B, N, N, n_bond_types + 1]
    accuracy = correct.sum() / correct.numel()

    return accuracy

def coordinate_prediction_rmsd(coordinates_prediction, coordinates_groundtruth, batch):
    assert len(coordinates_prediction.size()) == 2 # [B * N, 3]
    coordinates_prediction = remove_mean(coordinates_prediction, batch)
    coordinates_groundtruth = remove_mean(coordinates_groundtruth, batch)
    pos_pred, atom_mask = to_dense_batch(coordinates_prediction, batch)
    pos_gt, _ = to_dense_batch(coordinates_groundtruth, batch)

    align_pos_gt = get_align_pos(pos_pred, pos_gt)
    atom_num = atom_mask.sum(dim=-1)
    rmsd = torch.sqrt(torch.sum((pos_pred - align_pos_gt) ** 2, dim=-1).sum(dim=-1) / atom_num).mean()
    return rmsd


