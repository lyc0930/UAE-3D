import torch
from torch import nn
from model.diffusion.modules import LearnedSinusodialPosEmb, DiTBlock, FinalLayer


class DiffusionTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()

        latent_dim = args.latent_dim
        self.hidden_dim = args.diffusion_hidden_dim
        self.time_dim = args.diffusion_hidden_dim

        # noise level conditioning embedding
        learned_dim = 16
        sinu_pos_emb = LearnedSinusodialPosEmb(learned_dim)
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(learned_dim + 1, self.time_dim),
            nn.GELU(),
            nn.Linear(self.time_dim, self.time_dim)
        )

        # Conditional embedding
        if args.condition_property is None:
            self.conditional = False
        else:
            self.conditional = True
            self.condition_mlp = nn.Sequential(
                nn.Linear(1, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            )
            n_conditions = args.condition_property.count('&') + 1
            # classifier-free guiance
            self.cfg_drop = args.cfg_drop
            if args.cfg_drop > 0:
                self.mask_token = nn.Parameter(torch.zeros(1, n_conditions, self.hidden_dim))
                nn.init.normal_(self.mask_token, std=0.02)
            self.condition_linear = nn.Linear(n_conditions * self.hidden_dim, self.time_dim)

        self.use_dit = args.use_dit
        if self.use_dit:
            self.enhance_dit = args.enhance_dit
            if args.enhance_dit:
                self.input_linear = nn.Sequential(
                    nn.Linear(latent_dim, self.hidden_dim * 2),
                    nn.GELU(),
                    nn.Linear(self.hidden_dim * 2, self.hidden_dim)
                )
            else:
                self.input_linear = nn.Linear(latent_dim, self.hidden_dim, bias=False)

            self.blocks = nn.ModuleList([
                DiTBlock(
                    hidden_size=self.hidden_dim,
                    num_heads=args.diffusion_n_heads,
                    mlp_ratio=args.diffusion_mlp_ratio,
                    attn_drop=args.diffusion_dropout,
                    proj_drop=args.diffusion_dropout
                ) for _ in range(args.diffusion_n_layers)
            ])

            self.final_layer = FinalLayer(self.hidden_dim, latent_dim, enhance_dit=args.enhance_dit)

        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=args.diffusion_n_heads,
                dim_feedforward=self.hidden_dim * 4,
                dropout=args.diffusion_dropout,
                batch_first=True,
                norm_first=True
            )

            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=args.diffusion_n_layers
            )

            self.out_proj = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim * 2, bias=False),
                nn.Tanh(),
                nn.Linear(self.hidden_dim * 2, latent_dim, bias=False)
            )

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # # Initialize timestep embedding MLP:
        # nn.init.normal_(self.time_mlp[0].weight, std=0.02)
        # nn.init.normal_(self.time_mlp[2].weight, std=0.02)

        if self.use_dit:
            # Zero-out adaLN modulation layers in DiT blocks:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

            # Zero-out output layers:
            if self.enhance_dit: # final layer is a Sequential, linear + SiLU + linear
                nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
                nn.init.constant_(self.final_layer.linear[0].weight, 0)
                nn.init.constant_(self.final_layer.linear[0].bias, 0)
                nn.init.constant_(self.final_layer.linear[2].weight, 0)
                nn.init.constant_(self.final_layer.linear[2].bias, 0)
            else:
                nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
                nn.init.constant_(self.final_layer.linear.weight, 0)
                nn.init.constant_(self.final_layer.linear.bias, 0)

    def get_extended_attention_mask(self, dtype, attention_mask: torch.Tensor) -> torch.Tensor:
        assert attention_mask.dim() == 2 # [B, max_N]
        extended_attention_mask = attention_mask[:, None, None, :] # [B, 1, 1, max_N]
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask

    def forward(self, z, time, context=None, key_padding_mask=None):
        """
        :param z: [B, max_N, latent_dim]
        :param time: [B, max_N]
        :param context: [B, n_conditions]
        :param key_padding_mask: [B, max_N]
        :returns: predict_noise: [B, max_N, hidden_dim]
        """
        t = self.time_mlp(time)  # [B, max_N, time_dim]

        if self.conditional:
            if context is not None:
                c = context.unsqueeze(-1) # [B, n_conditions, 1]
                c = self.condition_mlp(c) # [B, n_conditions, hidden_dim]
                if self.cfg_drop > 0:
                    drop = torch.bernoulli(torch.ones(c.size(0), c.size(1)) * self.cfg_drop).long()  # [B, n_conditions]
                    c[drop] = self.mask_token.to(dtype=c.dtype)
            elif context is None and not self.training:
                c = self.mask_token.expand(t.size(0), -1, -1) # [B, n_conditions, hidden_dim]
            else:
                raise ValueError("context is None during training")
            c = c.reshape(t.size(0), -1) # [B, n_conditions * hidden_dim]
            c = self.condition_linear(c) # [B, time_dim]
            c = c.unsqueeze(1).expand(-1, t.size(1), -1) # [B, max_N, time_dim]
            t += c

        if self.use_dit:
            h_t = self.input_linear(z)  # [B, max_N, hidden_dim]
            attention_mask = ~key_padding_mask  # [B, max_N]
            extended_attention_mask = self.get_extended_attention_mask(h_t.dtype, attention_mask)  # [B, 1, 1, max_N]
            for block in self.blocks:
                h_t = block(h_t, t, extended_attention_mask)
            # h_t = h_t * attention_mask[:, :, None]
            predict_noise = self.final_layer(h_t, t, attention_mask)
        else:
            z_t = self.input_linear(z) + t
            h_t = self.transformer(z_t, src_key_padding_mask=key_padding_mask)  # (B, N, hidden_dim)

            predict_noise = self.out_proj(h_t)  # (B, N, hidden_dim)

        return predict_noise

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Diffusion Transformer")
        parser.add_argument('--time_dim', type=int, default=None)
        parser.add_argument('--diffusion_hidden_dim', type=int, default=512)
        parser.add_argument('--diffusion_n_heads', type=int, default=8)
        parser.add_argument('--diffusion_n_layers', type=int, default=8)
        parser.add_argument("--diffusion_mlp_ratio", type=float, default=4.0)
        parser.add_argument('--diffusion_dropout', type=float, default=0.0)
        parser.add_argument('--use_dit', action='store_true', default=True)
        parser.add_argument('--not_use_dit', action='store_false', dest='use_dit')
        parser.add_argument('--enhance_dit', action='store_true', default=False)
        return parent_parser
