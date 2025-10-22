import torch
from torch import nn
from torch_geometric.utils import to_dense_batch

from model.autoencoder.modules import DMTBlock, GaussianLayer
from model.utils import remove_mean, get_precision, get_align_pos, get_align_noise, AttrDict, get_pos_loss, get_dist_loss

class Encoder(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, latent_dim, n_heads, n_blocks, dropout):
        super().__init__()

        self.node_embedding = nn.Sequential(
            nn.Linear(node_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_dim + hidden_dim // 4 + 1, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )

        # distance GBF embedding
        self.distance_gbf = GaussianLayer(hidden_dim // 4)

        # transformer blocks
        self.encoder_blocks = nn.ModuleList([
            DMTBlock(
                hidden_dim,
                edge_dim=hidden_dim // 4,
                time_dim=hidden_dim,
                num_heads=n_heads,
                cond_time=False,
                dropout=dropout,
                pair_update=False
            ) for _ in range(n_blocks)
        ])

        # self.out_norm = nn.LayerNorm(hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.log_var_layer = nn.Linear(hidden_dim, latent_dim)

    def forward(self, node_feature, edge_index, edge_feature, position):
        # Node
        node_feature = torch.cat([node_feature, position], dim=-1)
        h = self.node_embedding(node_feature)

        # Edge
        distance = torch.sum((position[edge_index[0]] - position[edge_index[1]]) ** 2, dim=-1, keepdim=True)
        distance_embedding = self.distance_gbf(distance)
        edge_h = self.edge_embedding(torch.cat([edge_feature, distance_embedding], dim=-1))

        # transformer blocks
        for block in self.encoder_blocks:
            h, _ = block(h, edge_h, edge_index)

        # h = self.out_norm(h)
        z_mean = self.mean_layer(h)
        z_log_var = self.log_var_layer(h)

        return z_mean, z_log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_heads, n_blocks, dropout, n_atom_types, n_bond_types, use_trans_encoder=False):
        super().__init__()

        self.input_projection = nn.Linear(latent_dim, hidden_dim)

        self.n_atom_types = n_atom_types
        self.n_bond_types = n_bond_types

        self.use_trans_encoder = use_trans_encoder
        if use_trans_encoder:
            self.decoder_blocks = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=n_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                ) for _ in range(n_blocks)
            ])
        else:
            self.decoder_blocks = nn.ModuleList([
                nn.TransformerDecoderLayer(
                    d_model=hidden_dim,
                    nhead=n_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                ) for _ in range(n_blocks)
            ])

        self.atom_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_atom_types)
        )
        self.bond_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_bond_types + 2) # 0: BT.UNSPECIFIED, -1: self-loop
        )
        self.coordinate_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3, bias=False)
        )

    def forward(self, z, padding_mask=None):
        z = self.input_projection(z) # [B, N, latent_dim] -> [B, N, hidden_dim]
        h = z # [B, N, hidden_dim]

        for block in self.decoder_blocks:
            if self.use_trans_encoder:
                h = block(h, src_key_padding_mask=padding_mask)
            else:
                h = block(h, z, tgt_key_padding_mask=padding_mask, memory_key_padding_mask=padding_mask) # [B, N, hidden_dim]

        # transform data back to sparse batch
        unpadding_mask = ~padding_mask # [B, N]
        pair = (h.unsqueeze(1) + h.unsqueeze(2))[unpadding_mask.unsqueeze(1) & unpadding_mask.unsqueeze(2)] # [B * N * N, hidden_dim]
        h = h[unpadding_mask] # [B * N, hidden_dim]

        atom_logits = self.atom_head(h)  # [B * N, n_atom_types]
        bond_logits = self.bond_head(pair)  # [B * N * N, n_bond_types + 2]
        coordinates = self.coordinate_head(h)      # [B * N, 3]
        return atom_logits, bond_logits, coordinates

class UnifiedAutoEncoder(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.encoder = Encoder(
            node_dim=args.node_dim,
            edge_dim=args.edge_dim,
            hidden_dim=args.encoder_hidden_dim,
            latent_dim=args.latent_dim,
            n_heads=args.encoder_n_heads,
            n_blocks=args.encoder_blocks,
            dropout=args.dropout,
        )

        self.decoder = Decoder(
            latent_dim=args.latent_dim,
            hidden_dim=args.decoder_hidden_dim,
            n_heads=args.decoder_n_heads,
            n_blocks=args.decoder_blocks,
            dropout=args.dropout,
            n_atom_types=args.n_atom_types,
            n_bond_types=args.n_bond_types
        )
        self.args = args
        self.loss_weights = {
            'atom': args.atom_loss_weight,
            'bond': args.bond_loss_weight,
            'coordinate': args.coordinate_loss_weight,
            'distance': args.dist_loss_weight,
            'bond_distance': args.bond_dist_loss_weight,
            'KLD': args.kld_weight
        }

        self.atom_criterion = nn.CrossEntropyLoss()
        self.bond_criterion = nn.CrossEntropyLoss()
        self.coordinate_criterion = get_pos_loss

    def forward(self, batch):
        node_feature, edge_index, edge_feature, position = batch.x, batch.edge_index, batch.edge_attr, batch.pos

        # Encode
        z_mean, z_log_var = self.encoder(node_feature, edge_index, edge_feature, position)

        # Sample latent variable
        std = torch.exp(0.5 * z_log_var)  # reparameterize variance
        eps = torch.randn_like(std)
        z = z_mean + eps * std

        # Decode
        atom_logits, bond_logits, coordinates = self.decode(z, batch=batch.batch)

        # Reconstruction loss
        n_atom_types = self.decoder.n_atom_types
        n_bond_types = self.decoder.n_bond_types
        atom_loss = self.atom_criterion(atom_logits, batch.x[:, :n_atom_types])

        # Bond loss
        bond_types = batch.edge_attr[:, :n_bond_types+1] # the +1 is for self-loop
        bond_types = torch.cat([(bond_types == 0).all(dim=-1, keepdim=True).float(), bond_types], dim=-1) # BT.UNSPECIFIED
        bond_loss = self.bond_criterion(bond_logits, bond_types)

        # Coordinate loss
        coordinate_loss = self.coordinate_criterion(
            pos_pred=coordinates,
            pos_gt=batch.pos,
            batch=batch.batch,
            centering=self.args.center_prediction,
            align_prediction=self.args.align_prediction
        )

        # KL divergence loss
        kl_loss = (-0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=1)).mean()

        loss = (
            self.loss_weights['atom'] * atom_loss +
            self.loss_weights['bond'] * bond_loss +
            self.loss_weights['coordinate'] * coordinate_loss +
            self.loss_weights['KLD'] * kl_loss
        )

        dist_loss = 0
        if self.loss_weights['distance'] > 0:
            dist_loss = get_dist_loss(coordinates, batch.pos, batch.edge_index)
            loss += self.loss_weights['distance'] * dist_loss

        bond_dist_loss = 0
        if self.loss_weights['bond_distance'] > 0:
            bond_mask = (batch.edge_attr[:, :n_bond_types] == 1).any(dim=-1)
            bond_dist_loss = get_dist_loss(coordinates, batch.pos, batch.edge_index[:, bond_mask])
            loss += self.loss_weights['bond_distance'] * bond_dist_loss

        loss_dict = {
            'atom_loss': atom_loss,
            'bond_loss': bond_loss,
            'coordinate_loss': coordinate_loss,
            'distance_loss': dist_loss,
            'bond_distance_loss': bond_dist_loss,
            'KLD': kl_loss,
            'loss': loss
        }

        return loss_dict

    def encode(self, batch):
        node_feature, edge_index, edge_feature, position = batch.x, batch.edge_index, batch.edge_attr, batch.pos

        z_mean, z_log_var = self.encoder(node_feature, edge_index, edge_feature, position)

        # Sample latent variable
        std = torch.exp(0.5 * z_log_var)  # reparameterize variance
        eps = torch.randn_like(std)
        z = z_mean + eps * std

        return z

    def decode(self, z, batch=None, padding_mask=None):
        if len(z.shape) == 3: # [B, N, hidden_dim]
            assert padding_mask is not None
            atom_logits, bond_logits, coordinates = self.decoder(z, padding_mask=padding_mask)

            unpadding_mask = ~padding_mask
            batch_size, max_num_nodes = padding_mask.shape

            atom_logits_full = torch.zeros((batch_size, max_num_nodes, self.decoder.n_atom_types), dtype=atom_logits.dtype, device=atom_logits.device)
            atom_logits_full.masked_scatter_(unpadding_mask.unsqueeze(-1), atom_logits) # [B, max_N, n_atom_types]

            bond_logits_full = torch.zeros((batch_size, max_num_nodes, max_num_nodes, self.decoder.n_bond_types + 2), dtype=bond_logits.dtype, device=bond_logits.device)
            bond_mask = unpadding_mask.unsqueeze(1) & unpadding_mask.unsqueeze(2)  # [B, N, N]
            bond_logits_full.masked_scatter_(bond_mask.unsqueeze(-1), bond_logits) # [B, max_N, max_N, n_bond_types + 2]

            coordinates_full = torch.zeros((batch_size, max_num_nodes, 3), dtype=coordinates.dtype, device=coordinates.device)
            coordinates_full.masked_scatter_(unpadding_mask.unsqueeze(-1), coordinates) # [B, max_N, 3]

            return atom_logits_full, bond_logits_full, coordinates_full
        elif len(z.shape) == 2: # [B * N, hidden_dim]
            assert batch is not None
            z, unpadding_mask = to_dense_batch(z, batch) # [B, N, hidden_dim]
            padding_mask = ~unpadding_mask # [B, N]
            return self.decoder(z, padding_mask=padding_mask) # [B * N, n_atom_types], [B * N * N, n_bond_types + 2], [B * N, 3]
        else:
            raise ValueError("Invalid shape of z")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Multimodal Autoencoder")
        parser.add_argument('--encoder_hidden_dim', type=int, default=64)
        parser.add_argument('--encoder_n_heads', type=int, default=8)
        parser.add_argument('--encoder_blocks', type=int, default=6)
        parser.add_argument("--latent_dim", type=int, default=16)
        parser.add_argument('--decoder_hidden_dim', type=int, default=64)
        parser.add_argument('--decoder_n_heads', type=int, default=8)
        parser.add_argument('--decoder_blocks', type=int, default=4)
        parser.add_argument('--dropout', type=float, default=0.1)


        loss_weights = parent_parser.add_argument_group("VAE loss weights")
        loss_weights.add_argument('--atom_loss_weight', type=float, default=1.0)
        loss_weights.add_argument('--bond_loss_weight', type=float, default=1.0)
        loss_weights.add_argument('--center_prediction', action='store_true', default=False)
        loss_weights.add_argument('--not_center_prediction', action='store_false', dest='center_prediction')
        loss_weights.add_argument('--align_prediction', action='store_true', default=False)
        loss_weights.add_argument('--not_align_prediction', action='store_false', dest='align_prediction')
        loss_weights.add_argument('--coordinate_loss_weight', type=float, default=1.0)
        loss_weights.add_argument('--dist_loss_weight', type=float, default=1.0)
        loss_weights.add_argument('--bond_dist_loss_weight', type=float, default=10.0)
        loss_weights.add_argument('--kld_weight', type=float, default=1e-8)
