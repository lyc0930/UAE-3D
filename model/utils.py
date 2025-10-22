import torch
from torch_scatter import scatter
from torch_geometric.utils import to_dense_batch

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def coord2dist(x, edge_index):
    # coordinates to distance
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)
    return radial

def remove_mean_with_mask(x, node_mask, return_mean=False):
    # masked_max_abs_value = (x * (1 - node_mask)).abs().sum().item()
    # assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    if return_mean:
        return x, mean
    return x

def remove_mean(pos, batch):
    mean_pos = scatter(pos, batch, dim=0, reduce='mean') # shape = [B, 3]
    pos = pos - mean_pos[batch]
    return pos

def modulate(x, shift, scale):
    # return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    return x * (1 + scale) + shift

def get_precision(precision):
    if precision in {'16', '16-mixed'}:
        return torch.float16
    elif precision in {'bf16', 'bf16-mixed'}:
        return torch.bfloat16
    elif precision in {'32', 'fp32', '32-true'}:
        return torch.float32
    else:
        print(precision)
        raise NotImplementedError

@torch.no_grad()
def kabsch_batch(coords_pred, coords_tar):
    '''
    coords_pred: [batch_size, num_nodes, 3]
    coords_tar: [batch_size, num_nodes, 3]
    '''
    """Batch version of Kabsch algorithm."""
    A = torch.einsum("...ki, ...kj -> ...ij", coords_pred, coords_tar)
    A = A.to(torch.float32)
    U, S, Vt = torch.linalg.svd(A)
    sign_detA = torch.sign(torch.det(A))  # [batch_size]
    corr_mat_diag = torch.ones((A.size(0), U.size(-1)), device=A.device)  # [batch_size, 3]
    corr_mat_diag[:, -1] = sign_detA  # [batch_size, 3]
    corr_mat = torch.diag_embed(corr_mat_diag)  # [batch_size, 3, 3]
    rotation = torch.einsum("...ij, ...jk, ...kl -> ...il", U, corr_mat, Vt)  # [batch_size, 3, 3]
    return rotation

@torch.no_grad()
def get_align_pos(pos_t, pos_0):
    '''
    pos_t: [batch_size, num_nodes, 3]
    '''
    rotations = kabsch_batch(pos_t, pos_0)  # [batch_size, 3, 3]
    align_pos_0 = torch.einsum("...ki, ...ji -> ...jk", rotations, pos_0)
    return align_pos_0

@torch.no_grad()
def get_align_noise(pos_t, pos_0, pos_pred, alpha_t, sigma_t, batch_mask=None, translation_correction=False, align_prediction=False):
    if translation_correction:
        # center the coordinates with mask
        batch_mask = batch_mask.unsqueeze(-1)
        pos_0_centered, _ = remove_mean_with_mask(pos_0, batch_mask, return_mean=True)
        if align_prediction:
            pos_pred_centered, pos_pred_mean = remove_mean_with_mask(pos_pred, batch_mask, return_mean=True) # shape = [batch_size, num_nodes, 3], [batch_size, 1, 3]
            rotations = kabsch_batch(pos_pred_centered, pos_0_centered)  # [batch_size, 3, 3]
            align_pos_0 = torch.einsum("...ki, ...ji -> ...jk", rotations, pos_0_centered) + pos_pred_mean
        else:
            pos_t_centered, pos_t_mean = remove_mean_with_mask(pos_t, batch_mask, return_mean=True) # shape = [batch_size, num_nodes, 3], [batch_size, 1, 3]
            rotations = kabsch_batch(pos_t_centered, pos_0_centered)  # [batch_size, 3, 3]
            align_pos_0 = torch.einsum("...ki, ...ji -> ...jk", rotations, pos_0_centered) + pos_t_mean
        aligned_noise = (pos_t - alpha_t * align_pos_0) / sigma_t
        return aligned_noise
    else:
        if align_prediction:
            rotations = kabsch_batch(pos_pred, pos_0)  # [batch_size, 3, 3]
        else:
            rotations = kabsch_batch(pos_t, pos_0)  # [batch_size, 3, 3]
        align_pos_0 = torch.einsum("...ki, ...ji -> ...jk", rotations, pos_0)
        aligned_noise = (pos_t - alpha_t * align_pos_0) / sigma_t
        return aligned_noise

def get_pos_loss(pos_pred, pos_gt, batch, centering=False, align_prediction=True):
    '''
    coordinate_predict: [batch_size, num_nodes, 3]
    coordinate_target: [batch_size, num_nodes, 3]
    '''
    if centering:
        pos_pred = remove_mean(pos_pred, batch)
        pos_gt = remove_mean(pos_gt, batch)

    if not align_prediction:
        return torch.mean(torch.square(pos_pred - pos_gt))

    N = pos_pred.size(0)
    pos_pred, _ = to_dense_batch(pos_pred, batch)
    pos_gt, _ = to_dense_batch(pos_gt, batch)
    align_pos_gt = get_align_pos(pos_pred, pos_gt) # shape = [batch_size, max_num_nodes, 3]
    pos_loss = torch.square(pos_pred - align_pos_gt) # shape = [batch_size, max_num_nodes, 3]
    pos_loss = torch.mean(pos_loss, dim=-1) # shape = [batch_size, max_num_nodes]
    pos_loss = torch.sum(pos_loss, dim=-1) # shape = [batch_size]
    pos_loss = pos_loss.sum() / N
    return pos_loss


def get_dist_loss(pos_pred, pos_gt, edge_index):
    pred_dist = torch.norm(pos_pred[edge_index[0]] - pos_pred[edge_index[1]], dim=-1) # shape = [B*N*N]
    gt_dist = torch.norm(pos_gt[edge_index[0]] - pos_gt[edge_index[1]], dim=-1) # shape = [B*N*N]
    return torch.mean(torch.square(pred_dist - gt_dist))
