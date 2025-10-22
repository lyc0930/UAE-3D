from copy import deepcopy
import math
from typing import Optional, Union, Dict, Any

import torch
import torch.distributed as dist
import lightning as L
from pytorch_lightning.utilities import rank_zero_only
from tqdm import tqdm
from overrides import overrides

class LinearWarmupLRScheduler:
    def __init__(self, optimizer, lr, **kwargs):
        self.optimizer = optimizer
        self.lr = lr

    def get_lr(self, it):
        # 1) linear warmup for warmup_steps steps
        if it < self.warmup_iters:
            return self.lr * it / self.warmup_iters

        return self.lr

    def step(self, cur_step):
        lr = self.get_lr(cur_step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

class LinearWarmupCosineLRScheduler:
    def __init__(
        self,
        optimizer,
        max_iters,
        min_lr,
        init_lr,
        warmup_iters=0,
        warmup_start_lr=-1,
        **kwargs
    ):
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.min_lr = min_lr
        self.init_lr = init_lr
        self.warmup_iters = warmup_iters
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr # ! Not used
        self.lr_decay_iters = max_iters

    def get_lr(self, it):
        # 1) linear warmup for warmup_steps steps
        if it < self.warmup_iters:
            return self.init_lr * it / self.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.lr_decay_iters:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.min_lr + coeff * (self.init_lr - self.min_lr)

    def step(self, cur_step):
        lr = self.get_lr(cur_step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

class PeriodicTestCallback(L.Callback):
    def __init__(self, test_every_n_epochs):
        super().__init__()
        self.test_every_n_epochs = test_every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.test_every_n_epochs == 0:
            self._run_test(trainer, pl_module)

    def _run_test(self, trainer, pl_module):
        training = pl_module.training

        pl_module.eval()

        test_loader = trainer.datamodule.test_dataloader()
        device = pl_module.device

        with torch.no_grad():
            pl_module.on_test_epoch_start()
            for batch_idx, batch in tqdm(enumerate(test_loader), desc="Testing: ", total=len(test_loader), leave=False):
                batch = batch.to(device)
                pl_module.test_step(batch, batch_idx)

            pl_module.on_test_epoch_end()

        pl_module.train(training)


def custom_callbacks(args):
    callbacks = []
    callbacks.append(L.pytorch.callbacks.ModelCheckpoint(
        dirpath=f"all_checkpoints/{args.filename}/",
        filename='{epoch:02d}',
        every_n_epochs=args.save_every_n_epochs,
        save_top_k=-1,
        save_on_train_epoch_end=True,
        save_last='link',
    ))

    callbacks.append(L.pytorch.callbacks.ModelCheckpoint(
        dirpath=f"all_checkpoints/{args.filename}/",
        filename='best_validation_epoch={epoch:02d}',
        monitor='valid/loss',
        mode='min',
        auto_insert_metric_name=False,
        every_n_epochs=args.cache_epoch,
        save_top_k=1,
        save_on_train_epoch_end=True,
    ))

    if args.test_every_n_epochs is not None:
        callbacks.append(PeriodicTestCallback(args.test_every_n_epochs))

    return callbacks

def disabled_train(self, mode: bool = True):
    """Overwrite model.train with this function to make sure train/eval mode does not change anymore."""
    return self

def suppress_warning():
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

def device_cast(args_device: str):
    """
    Number of devices to train on (int), which devices to train on (list or str), or "auto".
    """
    try:
        if args_device == 'auto':
            devices = 'auto'
        elif args_device.startswith('[') and args_device.endswith(']'):
            devices = eval(args_device)
            assert isinstance(devices, list)
            assert all(isinstance(device, int) for device in devices)
        else:
            devices = int(args_device)
    except Exception as e:
        raise NotImplementedError(f"devices should be a integer, a list (of integer), or 'auto', got {args_device}") from e

    return devices

def add_training_specific_args(parser):
    trainer_group = parser.add_argument_group("Trainer")
    trainer_group.add_argument('--max_epochs', type=int, default=2000)
    trainer_group.add_argument('--accelerator', type=str, default='gpu')
    trainer_group.add_argument('--devices', type=str, default='auto')
    trainer_group.add_argument('--precision', type=str, default='16-mixed')

    trainer_group.add_argument('--save_every_n_epochs', type=int, default=20)
    trainer_group.add_argument('--cache_epoch', type=int, default=5)
    trainer_group.add_argument('--check_val_every_n_epoch', type=int, default=5)
    trainer_group.add_argument('--test_every_n_epochs', type=int, default=200)

    trainer_group.add_argument('--disable_compile', action='store_true', default=False)
    trainer_group.add_argument('--detect_anomaly', action='store_true', default=False)
    trainer_group.add_argument('--accumulate_grad_batches', type=int, default=1)
    trainer_group.add_argument('--gradient_clip_val', type=float, default=1.0)
    trainer_group.add_argument('--ckpt_path', type=str, default=None)

    return trainer_group

def print_args(parser, args):
    LINE_LENGTH = 52
    print("=" * LINE_LENGTH)
    for group in parser._action_groups:
        if group.title not in ("positional arguments", "optional arguments"):
            title = group.title
            padding_length = LINE_LENGTH - len(title)
            left_padding = padding_length // 2
            right_padding = padding_length - left_padding

            print(f"{'-' * left_padding}{title}{'-' * right_padding}")

        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        for arg_name, arg_value in group_dict.items():
            if arg_name == "help":
                continue
            print(f"{arg_name.rjust(25)}: {arg_value}")

        # print("-" * LINE_LENGTH)

    print("=" * LINE_LENGTH)
