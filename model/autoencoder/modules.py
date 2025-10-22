import math
from typing import Tuple, Optional

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.typing import Adj, OptTensor

from model.utils import coord2dist, remove_mean_with_mask, remove_mean, modulate


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)

class GaussianLayer(nn.Module):
    """Gaussian basis function layer for 3D distance features"""
    def __init__(self, K, *args, **kwargs):
        super().__init__()
        self.K = K - 1
        self.means = nn.Embedding(1, self.K)
        self.stds = nn.Embedding(1, self.K)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)

    def forward(self, x, *args, **kwargs):
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return torch.cat([x, gaussian(x, mean, std).type_as(self.means.weight)], dim=-1)


class TransLayerOptim(MessagePassing):
    """The version for involving the edge feature. Multiply Msg. Without FFN and norm."""

    _alpha: OptTensor

    def __init__(self, x_channels: int, out_channels: int,
                 heads: int = 1, dropout: float = 0., edge_dim: Optional[int] = None,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(TransLayerOptim, self).__init__(node_dim=0, **kwargs)

        self.x_channels = x_channels
        self.in_channels = in_channels = x_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim

        self.lin_q = nn.Linear(in_channels + edge_dim, heads * out_channels, bias=bias)
        self.lin_kv = nn.Linear(in_channels + edge_dim, heads * out_channels * 2, bias=bias)
        self.proj = nn.Linear(heads * out_channels, heads * out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_q.reset_parameters()
        self.lin_kv.reset_parameters()
        self.proj.reset_parameters()


    def forward(self, x: OptTensor,
                edge_index: Adj,
                edge_attr: OptTensor = None
                ) -> Tensor:
        """"""
        x_feat = x

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out_x = self.propagate(edge_index, x_feat=x_feat, edge_attr=edge_attr)

        out_x = out_x.view(-1, self.heads * self.out_channels)

        out_x = self.proj(out_x)
        return out_x

    def message(self, x_feat_i: Tensor, x_feat_j: Tensor,
                edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tuple[Tensor, Tensor]:
        query_ij = self.lin_q(torch.cat([x_feat_i, edge_attr], dim=-1)).view(-1, self.heads, self.out_channels)
        edge_key_ij, edge_value_ij = self.lin_kv(torch.cat([x_feat_j, edge_attr], dim=-1)).view(-1, self.heads, 2, self.out_channels).unbind(dim=2) # shape [N * N, heads, out_channels]

        alpha_ij = (query_ij * edge_key_ij).sum(dim=-1) / math.sqrt(self.out_channels) # shape [N * N, heads]
        alpha_ij = softmax(alpha_ij, index, ptr, size_i)
        alpha_ij = F.dropout(alpha_ij, p=self.dropout, training=self.training)

        # node feature message
        msg = edge_value_ij * alpha_ij.view(-1, self.heads, 1) # shape [N * N, heads, out_channels]
        return msg

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

class DMTBlock(nn.Module):
    """Equivariant block based on graph relational transformer layer, without extra heads."""

    def __init__(self, node_dim, edge_dim, time_dim, num_heads,
                 cond_time=True, mlp_ratio=4, act=nn.GELU, dropout=0.0, pair_update=True):
        super().__init__()
        self.dropout = dropout
        self.act = act()
        self.cond_time = cond_time
        self.pair_update = pair_update

        if not self.pair_update:
            self.edge_emb = nn.Sequential(
                nn.Linear(edge_dim, edge_dim * 2),
                nn.GELU(),
                nn.Linear(edge_dim * 2, edge_dim),
                nn.LayerNorm(edge_dim),
            )

        self.attn_mpnn = TransLayerOptim(node_dim, node_dim // num_heads, num_heads, edge_dim=edge_dim, dropout=dropout)

        # Feed forward block -> node.
        self.ff_linear1 = nn.Linear(node_dim, node_dim * mlp_ratio)
        self.ff_linear2 = nn.Linear(node_dim * mlp_ratio, node_dim)

        if pair_update:
            self.node2edge_lin = nn.Linear(node_dim * 2 + edge_dim, edge_dim)
            # Feed forward block -> edge.
            self.ff_linear3 = nn.Linear(edge_dim, edge_dim * mlp_ratio)
            self.ff_linear4 = nn.Linear(edge_dim * mlp_ratio, edge_dim)

        # equivariant edge update layer
        if self.cond_time:
            self.node_time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_dim, node_dim * 6)
            )
            # Normalization for MPNN
            self.norm1_node = nn.LayerNorm(node_dim, elementwise_affine=False, eps=1e-6)
            self.norm2_node = nn.LayerNorm(node_dim, elementwise_affine=False, eps=1e-6)

            if self.pair_update:
                self.edge_time_mlp = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(time_dim, edge_dim * 6)
                )
                self.norm1_edge = nn.LayerNorm(edge_dim, elementwise_affine=False, eps=1e-6)
                self.norm2_edge = nn.LayerNorm(edge_dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1_node = nn.LayerNorm(node_dim, elementwise_affine=True, eps=1e-6)
            self.norm2_node = nn.LayerNorm(node_dim, elementwise_affine=True, eps=1e-6)
            if self.pair_update:
                self.norm1_edge = nn.LayerNorm(edge_dim, elementwise_affine=True, eps=1e-6)
                self.norm2_edge = nn.LayerNorm(edge_dim, elementwise_affine=True, eps=1e-6)

    def _ff_block_node(self, x):
        x = F.dropout(self.act(self.ff_linear1(x)), p=self.dropout, training=self.training)
        return F.dropout(self.ff_linear2(x), p=self.dropout, training=self.training)

    def _ff_block_edge(self, x):
        x = F.dropout(self.act(self.ff_linear3(x)), p=self.dropout, training=self.training)
        return F.dropout(self.ff_linear4(x), p=self.dropout, training=self.training)

    def forward(self, h, edge_attr, edge_index, node_time_emb=None, edge_time_emb=None):
        """
        A more optimized version of forward_old using torch.compile
        Params:
            h: [B*N, hid_dim]
            edge_attr: [N_edge, edge_hid_dim]
            edge_index: [2, N_edge]
        """
        h_in_node = h
        h_in_edge = edge_attr

        if self.cond_time:
            ## prepare node features
            node_shift_msa, node_scale_msa, node_gate_msa, node_shift_mlp, node_scale_mlp, node_gate_mlp = \
                self.node_time_mlp(node_time_emb).chunk(6, dim=1)
            h = modulate(self.norm1_node(h), node_shift_msa, node_scale_msa)

            ## prepare edge features
            if self.pair_update:
                edge_shift_msa, edge_scale_msa, edge_gate_msa, edge_shift_mlp, edge_scale_mlp, edge_gate_mlp = \
                    self.edge_time_mlp(edge_time_emb).chunk(6, dim=1)
                edge_attr = modulate(self.norm1_edge(edge_attr), edge_shift_msa, edge_scale_msa)
            else:
                edge_attr = self.edge_emb(edge_attr)

            # apply transformer-based message passing, update node features and edge features (FFN + norm)
            h_node = self.attn_mpnn(h, edge_index, edge_attr)

            ## update node features
            h_out = self.node_update(h_in_node, h_node, node_gate_msa, node_shift_mlp, node_scale_mlp, node_gate_mlp)

            ## update edge features
            if self.pair_update:
                # h_edge = torch.cat([h_node[edge_index[0]], h_node[edge_index[1]]], dim=-1)
                h_edge = h_node[edge_index.transpose(0, 1)].flatten(1, 2) # shape [N_edge, 2 * edge_hid_dim]
                h_edge = torch.cat([h_edge, h_in_edge], dim=-1)
                h_edge_out = self.edge_update(h_in_edge, h_edge, edge_gate_msa, edge_shift_mlp, edge_scale_mlp, edge_gate_mlp)
            else:
                h_edge_out = h_in_edge
        else:
            ## prepare node features
            h = self.norm1_node(h)

            ## prepare edge features
            if self.pair_update:
                edge_attr = self.norm1_edge(edge_attr)
            else:
                edge_attr = self.edge_emb(edge_attr)

            # apply transformer-based message passing, update node features and edge features (FFN + norm)
            h_node = self.attn_mpnn(h, edge_index, edge_attr)

            ## update node features
            h_out = self.node_update(h_in_node, h_node)

            ## update edge features
            if self.pair_update:
                # h_edge = h_node[edge_index[0]] + h_node[edge_index[1]]
                h_edge = h_node[edge_index.transpose(0, 1)].flatten(1, 2) # shape [N_edge, 2 * edge_hid_dim]
                h_edge = torch.cat([h_edge, h_in_edge], dim=-1)
                h_edge_out = self.edge_update(h_in_edge, h_edge)
            else:
                h_edge_out = h_in_edge
        return h_out, h_edge_out

    def node_update(self, h_in_node, h_node, node_gate_msa=None, node_shift_mlp=None, node_scale_mlp=None, node_gate_mlp=None):
        h_node = h_in_node + node_gate_msa * h_node if self.cond_time else h_in_node + h_node
        _h_node = modulate(self.norm2_node(h_node), node_shift_mlp, node_scale_mlp) if self.cond_time else \
                self.norm2_node(h_node)
        h_out = h_node + node_gate_mlp * self._ff_block_node(_h_node) if self.cond_time else \
                h_node + self._ff_block_node(_h_node)
        return h_out

    def edge_update(self, h_in_edge, h_edge, edge_gate_msa=None, edge_shift_mlp=None, edge_scale_mlp=None, edge_gate_mlp=None):
        h_edge = self.node2edge_lin(h_edge)
        h_edge = h_in_edge + edge_gate_msa * h_edge if self.cond_time else h_in_edge + h_edge
        _h_edge = modulate(self.norm2_edge(h_edge), edge_shift_mlp, edge_scale_mlp) if self.cond_time else \
                self.norm2_edge(h_edge)
        h_edge_out = h_edge + edge_gate_mlp * self._ff_block_edge(_h_edge) if self.cond_time else \
                    h_edge + self._ff_block_edge(_h_edge)
        return h_edge_out
