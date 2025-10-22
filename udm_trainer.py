import argparse
import copy
from pathlib import Path

import lightning as L
from lightning.pytorch.loggers import CSVLogger
import torch
import torch.distributed as dist
from torch_geometric.utils import to_dense_batch, to_dense_adj

from data_provider.utils import add_datamodule_specific_args
from evaluation import add_evaluation_specific_args, molecule_evaluate
from model.latent_diffusion import LatentDiffusion
from training_utils import add_training_specific_args, custom_callbacks, device_cast, print_args, suppress_warning, LinearWarmupLRScheduler, LinearWarmupCosineLRScheduler

torch.set_float32_matmul_precision('high') # can be medium (bfloat16), high (tensorfloat32), highest (float32)

class LatentDiffusionTrainer(L.LightningModule):
    def __init__(self, model, args, position_std):
        super().__init__()
        self.model = model
        self.args = args
        self.position_std = position_std
        self.test_molecule_list = []

        self.save_hyperparameters(args)

    def training_step(self, batch, batch_idx):
        if self.lr_scheduler:
            self.lr_scheduler.step(self.trainer.global_step)

        assert not self.model.vae_model.training

        loss = self.model(batch)
        self.log('train/loss', float(loss), sync_dist=True, batch_size=self.args.batch_size)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss = self.model(batch)
        self.log('valid/loss', float(loss), sync_dist=True, batch_size=self.args.batch_size)
        return loss

    @torch.no_grad()
    def on_test_epoch_start(self):
        self.test_molecule_list = []

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        batch_size = len(batch)

        atom_logits, bond_logits, coordinates, unpadding_mask, mae_dict = self.model.sample(batch_size=batch_size) # [batch_size, max_N, n_atom_types], [batch_size, max_N, max_N, n_bond_types + 2], [batch_size, max_N, 3], [batch_size, max_N]

        # Reconstruct molecules for evaluation
        # batch_size, max_num_nodes = unpadding_mask.shape

        atom_types = atom_logits.argmax(dim=-1) # [batch_size, max_num_nodes]
        bond_types = bond_logits[..., :-1].argmax(dim=-1) # [batch_size, max_num_nodes, max_num_nodes], exclude self-loop
        positions = coordinates * self.position_std

        for i, idx in enumerate(batch.idx):
            atom_type = atom_types[i][unpadding_mask[i]].long().cpu() # [num_nodes]
            bond_type = bond_types[i][unpadding_mask[i]][:, unpadding_mask[i]].long().cpu() # [num_nodes, num_nodes]
            position = positions[i][unpadding_mask[i]].float().cpu() # [num_nodes, 3]
            formal_charge = torch.zeros_like(atom_type) # [num_nodes]
            self.test_molecule_list.append((idx, (position, atom_type, bond_type, formal_charge)))

        for key, value in mae_dict.items():
            self.log(f'test/{key}', value)

    @torch.no_grad()
    @torch.autocast('cuda', dtype=torch.bfloat16)
    def on_test_epoch_end(self):
        if self.trainer.sanity_checking:
            return

        if len(self.test_molecule_list) <= 0:
            print("WARNING: No molecules generated")
            return

        if dist.is_initialized():
            gather_box = [None for _ in range(self.trainer.world_size)]
            dist.all_gather_object(gather_box, self.test_molecule_list)
        else:
            gather_box = [self.test_molecule_list]

        molecule_list = {idx: molecule_data for data in gather_box for idx, molecule_data in data}
        molecule_list = list(molecule_list.values())

        self.test_molecule_list = []

        if not self.trainer.is_global_zero:
            return

        evaluation_dict, reconstructed_rdmols_3D, reconstructed_rdmols_2D = molecule_evaluate(self.trainer.datamodule, molecule_list, self.args.evaluate_3D, self.args.evaluate_2D, self.args.evaluate_moses, self.args.evaluate_align)

        for key, value in evaluation_dict.items():
            self.log(f'test/{key}', value)

        if self.args.use_wandb:
            log_dir = Path(self.loggers[1].log_dir)
        else:
            log_dir = Path(self.logger.log_dir)
        torch.save(molecule_list, log_dir / f'generations_{self.current_epoch}.pt')

        if self.args.evaluate_3D:
            torch.save(reconstructed_rdmols_3D, log_dir / f'reconstructed_rdmols_3D_{self.current_epoch}.pt')
        if self.args.evaluate_2D:
            torch.save(reconstructed_rdmols_2D, log_dir / f'reconstructed_rdmols_2D_{self.current_epoch}.pt')

    def configure_optimizers(self):
        self.trainer.fit_loop.setup_data()

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )

        if self.args.lr_scheduler == 'constant':
            self.lr_scheduler = None
        elif self.args.lr_scheduler == 'linear_warmup':
            self.lr_scheduler = LinearWarmupLRScheduler(optimizer, self.args.learning_rate)
        elif self.args.lr_scheduler == 'linear_warmup_cosine':
            max_iters = self.args.max_epochs * len(self.trainer.train_dataloader) // self.args.accumulate_grad_batches
            assert max_iters > self.args.warmup_steps

            self.lr_scheduler = LinearWarmupCosineLRScheduler(optimizer, max_iters, self.args.min_lr, self.args.learning_rate, self.args.warmup_steps, self.args.warmup_lr) # ! warmup_lr actually not used
        else:
            raise NotImplementedError()

        return optimizer

def main(args):
    if args.dataset.lower().find('qm9') >= 0:
        from data_provider.qm9_datamodule import QM9LDMDataModule as DataModule
        args.node_dim = 44 + 3  # atom_features + position
        args.edge_dim = 4  # n_bond_types
    elif args.dataset.lower().find('drugs') >= 0:
        from data_provider.geom_drugs_datamodule import GEOMDrugsLDMDataModule as DataModule
        args.node_dim = 55 + 3  # atom_features + position
        args.edge_dim = 4  # n_bond_types
    else:
        raise NotImplementedError

    print("Loading data module...", end="")
    dm = DataModule(args.root, args.num_workers, args.batch_size, args.aug_rotation, args.aug_translation, args.aug_translation_scale, condition_property=args.condition_property, num_samples=args.num_samples)
    dm.setup_evaluator()
    print("Done!")

    args.n_atom_types = dm.n_atom_types
    args.n_bond_types = dm.n_bond_types

    model = LatentDiffusion(args, dm)

    # TODO: resume training & load ema_state_dict
    trainer_model = LatentDiffusionTrainer(model, args, dm.position_std)
    trainer_model = torch.compile(trainer_model, dynamic=True, fullgraph=False, disable=args.disable_compile)

    if args.use_wandb:
        from lightning.pytorch.loggers import WandbLogger
        wandb_logger = WandbLogger(project="Latent Diffusion", name=args.filename)
        csv_logger = CSVLogger(save_dir=f'./all_checkpoints/{args.filename}/')
        logger = [wandb_logger, csv_logger]
    else:
        logger = CSVLogger(save_dir=f'./all_checkpoints/{args.filename}/')

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=device_cast(args.devices),
        precision=args.precision,
        logger=logger,
        callbacks=custom_callbacks(args),
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        detect_anomaly=args.detect_anomaly,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val
    )

    if args.test_only:
        ckpt_path = f"./all_checkpoints/{args.filename}/last.ckpt" if args.ckpt_path is None else args.ckpt_path
        trainer.test(trainer_model, datamodule=dm, ckpt_path=ckpt_path)
    else:
        trainer.fit(trainer_model, datamodule=dm, ckpt_path=args.ckpt_path)
        for evaluate_suffix in ['3D', '2D', 'moses', 'align']:
            trainer_model.args.__setattr__(f'evaluate_{evaluate_suffix}', True) # set all evaluation flags to True
        trainer.test(trainer_model, datamodule=dm)


if __name__ == '__main__':
    suppress_warning()

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default='latent_diffusion_experiment')
    parser.add_argument('--test_only', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    add_training_specific_args(parser)
    add_datamodule_specific_args(parser)

    optimization =parser.add_argument_group("Optimization")
    optimization.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer learning rate or initial learning rate for non-constant scheduler')
    optimization.add_argument('--weight_decay', type=float, default=0.05, help='optimizer weight decay')
    optimization.add_argument('--min_lr', type=float, default=1e-5, help='optimizer min learning rate')
    optimization.add_argument('--warmup_lr', type=float, default=1e-6, help='optimizer warmup learning rate') # ! actually not used
    optimization.add_argument('--warmup_steps', type=int, default=1000, help='optimizer warmup steps')
    optimization.add_argument('--lr_scheduler', type=str, default='linear_warmup_cosine', help='type of learning rate scheduler')

    LatentDiffusion.add_model_specific_args(parser)

    evaluation = add_evaluation_specific_args(parser)
    evaluation.add_argument('--num_samples', type=int, default=10000)


    args = parser.parse_args()
    print_args(parser, args)

    main(args)
