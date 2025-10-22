from pathlib import Path

import lightning as L
import pickle
import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.utils import to_dense_batch
from tqdm import tqdm

from model.utils import remove_mean, get_precision, get_align_pos, get_align_noise, AttrDict
from model.diffusion.diffusion_scheduler import NoiseScheduleVP
from model.diffusion.diffusion_transformer import DiffusionTransformer as Diffusion
from model.autoencoder.unified_autoencoder import UnifiedAutoEncoder
from model.property_prediction import EGNN
from training_utils import disabled_train


class LatentDiffusion(L.LightningModule):
    def __init__(self, args, datamodule=None):
        super().__init__()
        if isinstance(args, dict):
            args = AttrDict(**args)

        self.args = args
        self.cfg_weight = args.cfg_weight

        self.noise_scheduler = NoiseScheduleVP(
            schedule=args.noise_scheduler,
            continuous_beta_0=args.continuous_beta_0,
            continuous_beta_1=args.continuous_beta_1,
            discrete_mode=args.discrete_schedule
        )

        self.num_sampling_timesteps = self.args.sampling_timesteps
        self.noise_temperature = self.args.noise_temperature

        self.max_nodes = datamodule.max_atoms
        self.node_distribution = datamodule.nodes_dist

        self.vae_model = self.init_vae(args)

        if args.latent_dist_pth is not None:
            latent_distribution = torch.load(args.latent_dist_pth)
            latent_mean = latent_distribution['latent_mean']
            latent_std = latent_distribution['latent_std']
            if args.latent_whiten == 'isotropic':
                assert latent_mean.shape == latent_std.shape == (1,)
            elif args.latent_whiten == 'anisotropic':
                assert latent_mean.shape == latent_std.shape == (args.latent_dim,)
        elif datamodule is not None:
            latent_dist_path = Path(args.vae_ckpt).parent / 'latent_distribution.pth'
            if not latent_dist_path.exists():
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    latent_mean, latent_std = self.compute_latent_distribution(datamodule, args.latent_whiten)
                    torch.save({'latent_mean': latent_mean, 'latent_std': latent_std}, latent_dist_path)
                    print(f"Saved latent distribution to {latent_dist_path}")
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
            latent_distribution = torch.load(latent_dist_path)
            latent_mean = latent_distribution['latent_mean']
            latent_std = latent_distribution['latent_std']
        else:
            raise NotImplementedError("Please provide a datamodule to calculate latent distribution or provide latent_dist_pth directly")

        self.register_buffer('latent_mean', latent_mean, persistent=True)
        self.register_buffer('latent_std', latent_std, persistent=True)

        self.diffusion_model = self.init_diffusion(args)

        if args.condition_property is None:
            self.property = None
        else:
            assert datamodule is not None
            self.property_normalizations = datamodule.prop_norms
            self.property_distribution = datamodule.prop_dist
            allowed_properties = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'Cv']
            property_output_norms = {'mu': 1., 'alpha': 1, 'homo': 1000., 'lumo': 1000., 'gap': 1000, 'Cv': 1.}

            if args.condition_property in allowed_properties:
                self.property = [args.condition_property]
            elif '&' in args.condition_property:
                self.property = args.condition_property.split('&')
                if len(self.property) != 2 or any(property not in allowed_properties for property in self.property):
                    raise NotImplementedError(f"{args.condition_property} is not supported")
            else:
                raise NotImplementedError(f"{args.condition_property} is not supported")

            self.property_mean = [self.property_normalizations[property]['mean'] for property in self.property]
            self.property_mad = [self.property_normalizations[property]['mad'] for property in self.property]
            self.property_prediction_model = ModuleList([self.init_property_prediction(property).to(self.device) for property in self.property])
            self.property_output_norm = [property_output_norms[property] for property in self.property]

    @classmethod
    def init_vae(cls, args):
        vae_model = UnifiedAutoEncoder(args)

        assert args.vae_ckpt is not None

        vae_ckpt = torch.load(args.vae_ckpt, map_location='cpu')
        vae_ckpt = {'.'.join(k.split('.')[1:]): v for k, v in vae_ckpt['state_dict'].items()}
        loading_info = vae_model.load_state_dict(vae_ckpt, strict=False)
        print(loading_info)

        vae_model.eval()
        for param in vae_model.parameters():
            param.requires_grad = False
        vae_model.train = disabled_train
        return vae_model

    def compute_latent_distribution(self, datamodule, latent_whiten='isotropic'):
        self.vae_model = self.vae_model.to('cuda:0')
        all_z = []
        for batch_data in tqdm(datamodule.train_dataloader(), desc="Encoding: ", leave=False):
            batch_data = batch_data.to('cuda:0')
            encoded_z = self.vae_model.encode(batch_data)
            all_z.append(encoded_z)
        all_z = torch.cat(all_z, dim=0)
        if latent_whiten == 'isotropic':
            mean = all_z.mean()
            std = all_z.std()
        elif latent_whiten == 'anisotropic':
            mean = all_z.mean(dim=0, keepdim=True)
            std = all_z.std(dim=0, keepdim=True)
        else:
            raise NotImplementedError(f"Unknown latent_whiten_type: {latent_whiten}")
        self.vae_model = self.vae_model.to('cpu')
        del all_z
        return mean, std

    @classmethod
    def init_diffusion(cls, args):
        diffusion_model = Diffusion(args)

        if args.diffusion_ckpt is not None:
            diffusion_ckpt = torch.load(args.diffusion_ckpt, map_location='cpu')
            diffusion_ckpt = {'.'.join(k.split('.')[1:]): v for k, v in diffusion_ckpt['state_dict'].items()}
            loading_info = diffusion_model.load_state_dict(diffusion_ckpt, strict=False)
            print(loading_info)

        if args.diffusion_tune == 'full':
            for param in diffusion_model.parameters():
                param.requires_grad = True
        elif args.diffusion_tune == 'freeze':
            for param in diffusion_model.parameters():
                param.requires_grad = False
        elif args.diffusion_tune == 'lora':
            from peft import get_peft_model, LoraConfig
            lora_config = LoraConfig(r=args.lora_r,
                                     lora_alpha=args.lora_alpha,
                                     lora_dropout=args.lora_dropout,
                                     target_modules=["proj", "ff_linear1", "ff_linear2", "ff_linear3", "ff_linear4", "node2edge_lin"],
                                     modules_to_save=["projector"])
            diffusion_model = get_peft_model(diffusion_model, lora_config)
            diffusion_model.print_trainable_parameters()
        else:
            raise NotImplementedError()

        return diffusion_model

    @classmethod
    def init_property_prediction(cls, condition_property):
        property_path = Path(f"data/QM9/property_classifier/evaluate_{condition_property}")
        classifier_path = property_path / "best_checkpoint.npy"
        args_classifier_path = property_path / "args.pickle"
        with open(args_classifier_path, 'rb') as f:
            args_classifier = pickle.load(f)
        classifier = EGNN(in_node_nf=5, in_edge_nf=0, hidden_nf=args_classifier.nf, device='cpu', n_layers=args_classifier.n_layers, coords_weight=1.0, attention=args_classifier.attention, node_attr=args_classifier.node_attr)
        classifier_state_dict = torch.load(classifier_path, map_location=torch.device('cpu'))
        classifier.load_state_dict(classifier_state_dict)
        for param in classifier.parameters():
            param.requires_grad = False
        classifier.eval()
        return classifier

    def forward(self, batch):
        assert not self.vae_model.training
        with torch.no_grad():
            z_flat = (self.vae_model.encode(batch) - self.latent_mean) / self.latent_std  # (B * N, latent_dim)
            z, mask = to_dense_batch(z_flat, batch.batch)  # (B, N, latent_dim), (B, N)
            z = z.detach()

        noise = torch.randn_like(z)  # (B, N, latent_dim)
        # timesteps = torch.rand(z.shape[0], device=z.device)  # (B,)
        timesteps = (torch.rand(1, device=z.device) + torch.linspace(0, 1, z.shape[0], device=z.device)) % 1  # (B,)

        noisy_z = self.add_noise(z, noise, timesteps)  # (B, N, latent_dim)

        t_emb = timesteps.unsqueeze(1).expand(-1, z.shape[1])  # (B, N)

        if hasattr(batch, 'context'):
            context = batch.context
        else:
            context = None

        predict_noise = self.diffusion_model(noisy_z, t_emb, context, key_padding_mask=~mask)  # (B, N, latent_dim)
        loss = F.mse_loss(predict_noise[mask], noise[mask])

        return loss

    def add_noise(self, z, noise, t):
        '''Add noise to input at timestep t'''
        t_eps = 1e-5
        t = t * (1. - t_eps) + t_eps

        alpha_t, sigma_t = self.noise_scheduler.marginal_prob(t)
        noisy_z = (alpha_t[:, None, None] * z + sigma_t[:, None, None] * noise)

        return noisy_z

    @torch.no_grad()
    def sample(self, batch_size, timesteps=None, temperature=None):
        device = self.device
        num_nodes = self.node_distribution.sample(batch_size).to(device)

        if self.property is not None:
            context = self.property_distribution.sample_batch(num_nodes).to(device)
        else:
            context = None

        z = torch.zeros((batch_size, self.max_nodes, self.args.latent_dim), device=device) # [batch_size, max_N, latent_dim]
        unpadding_mask = torch.zeros((batch_size, self.max_nodes), dtype=torch.bool, device=device) # [batch_size, max_N]

        for i in range(batch_size):
            n_i = num_nodes[i].item()
            z[i, :n_i] = torch.randn((n_i, self.args.latent_dim), device=device)
            unpadding_mask[i, :n_i] = True

        sample_timesteps = self.num_sampling_timesteps if timesteps is None else timesteps
        noise_temperature = self.noise_temperature if temperature is None else temperature
        epsilon = 1e-3

        t_array = torch.linspace(self.noise_scheduler.T, epsilon, sample_timesteps, device=device) # (T,)
        s_array = torch.cat([t_array[1:], torch.zeros(1, device=device)]) # (T,)

        for i in range(sample_timesteps):
            t = t_array[i]
            s = s_array[i]

            alpha_t, sigma_t = self.noise_scheduler.marginal_prob(t)
            alpha_s, sigma_s = self.noise_scheduler.marginal_prob(s)

            alpha_t_given_s = alpha_t / alpha_s
            sigma2_t_given_s = sigma_t**2 - alpha_t_given_s**2 * sigma_s**2
            sigma_t_given_s = torch.sqrt(sigma2_t_given_s)
            sigma = sigma_t_given_s * sigma_s / sigma_t  # * eta

            t_emb = t.expand(batch_size) # [batch_size,]
            t_emb = t_emb.unsqueeze(1).expand(-1, self.max_nodes) # [batch_size, max_N]

            predict_noise = self.diffusion_model(z, t_emb, context, key_padding_mask=~unpadding_mask)

            if self.cfg_weight > 0: # Classifier-free guidance
                predict_noise_cfg = self.diffusion_model(z, t_emb, context=None, key_padding_mask=~unpadding_mask)
                predict_noise = (1 + self.cfg_weight) * predict_noise - self.cfg_weight * predict_noise_cfg

            predict_z = (z - predict_noise * sigma_t) / alpha_t

            z_mean = predict_z * alpha_s + predict_noise * torch.sqrt(1 - alpha_s**2 - sigma**2)
            z = z_mean + sigma * torch.randn_like(z) * noise_temperature

        z_mean = z_mean * self.latent_std + self.latent_mean
        atom_logits, bond_logits, coordinates = self.vae_model.decode(z_mean, padding_mask=~unpadding_mask) # [batch_size, max_N, n_atom_types], [batch_size, max_N, max_N, n_bond_types + 2], [batch_size, max_N, 3]

        if self.property is not None:
            mae_dict = self.calculate_property_MAE(context, atom_logits, bond_logits, coordinates, unpadding_mask)
            return atom_logits, bond_logits, coordinates, unpadding_mask, mae_dict
        else:
            return atom_logits, bond_logits, coordinates, unpadding_mask, {}

    def calculate_property_MAE(self, context, atom_logits, bond_logits, coordinates, unpadding_mask):
        assert self.property is not None

        # (atom_logits, bond_logits, coordinates, unpadding_mask) -> (one_hot, pos, full_edges, node_mask, edge_mask)
        batch_size, max_nodes, n_atom_types = atom_logits.shape
        atom_types = atom_logits.argmax(dim=-1)
        one_hot = F.one_hot(atom_types, n_atom_types).float()
        one_hot = one_hot.reshape(-1, n_atom_types)  # [batch_size * max_N, n_atom_types]
        positions = coordinates.reshape(-1, 3)  # [batch_size * max_N, 3]

        edges_dic = {}
        def get_adj_matrix(n_nodes, batch_size, device):
            if n_nodes in edges_dic:
                edges_dic_b = edges_dic[n_nodes]
                if batch_size in edges_dic_b:
                    return edges_dic_b[batch_size]
                else:
                    # get edges for a single sample
                    rows, cols = [], []
                    for batch_idx in range(batch_size):
                        for i in range(n_nodes):
                            for j in range(n_nodes):
                                rows.append(i + batch_idx * n_nodes)
                                cols.append(j + batch_idx * n_nodes)
            else:
                edges_dic[n_nodes] = {}
                return get_adj_matrix(n_nodes, batch_size, device)

            edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
            return edges

        full_edges = get_adj_matrix(max_nodes, batch_size, coordinates.device)

        node_mask = unpadding_mask.reshape(-1, 1)  # [batch_size * max_N, 1]

        edge_mask = unpadding_mask.unsqueeze(1) * unpadding_mask.unsqueeze(2) # [batch_size, max_N, max_N]
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0).to(edge_mask.device)
        edge_mask &= diag_mask # [batch_size, max_N, max_N]
        edge_mask = edge_mask.view(-1, 1) # [batch_size * max_N * max_N, 1]

        mae_dict = {}
        for i, property in enumerate(self.property):
            prediction = self.property_prediction_model[i](h0=one_hot, x=positions, edges=full_edges, edge_attr=None, node_mask=node_mask, edge_mask=edge_mask, n_nodes=max_nodes)  # Predict property using the classifier
            prediction = prediction * self.property_mad[i] + self.property_mean[i]  # Rescale the predictions
            target = context[:, i].clone().squeeze(-1)  # [batch_size]
            target = target * self.property_mad[i] + self.property_mean[i]
            mae_dict[f"MAE_{property}"] =  F.l1_loss(prediction, target, reduction='mean') * self.property_output_norm[i]

        return mae_dict

    @staticmethod
    def add_model_specific_args(parent_parser):
        UnifiedAutoEncoder.add_model_specific_args(parent_parser)
        Diffusion.add_model_specific_args(parent_parser)

        parser = parent_parser.add_argument_group("Latent Diffusion")
        parser.add_argument("--vae_ckpt", type=str, required=True)
        parser.add_argument("--diffusion_ckpt", type=str, default=None)
        parser.add_argument('--diffusion_tune', type=str, default="full")
        parser.add_argument('--latent_dist_pth', type=str, default=None)
        parser.add_argument('--latent_whiten', type=str, choices=['isotropic', 'anisotropic'], default='isotropic')
        parser.add_argument('--sampling_timesteps', type=int, default=100)
        parser.add_argument('--noise_temperature', type=float, default=1.0)
        parser.add_argument('--condition_property', type=str, default=None, choices=['mu', 'alpha', 'homo', 'lumo', 'gap', 'Cv'])
        parser.add_argument('--cfg_drop', type=float, default=0.1)
        parser.add_argument('--cfg_weight', type=float, default=0.5)

        parser = parent_parser.add_argument_group("Noise Scheduler")
        parser.add_argument('--noise_scheduler', type=str, default='cosine')
        parser.add_argument('--continuous_beta_0', type=float, default=0.1)
        parser.add_argument('--continuous_beta_1', type=float, default=20)
        parser.add_argument('--discrete_schedule', action='store_true', default=False)


        return parent_parser
