"""Train a VAE model."""
import argparse
import math
from math import sqrt
import logging
from pathlib import Path
from typing import Callable, Tuple


# torch

import torch
from torch.nn import Module
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

# vision imports

from torchvision import transforms as T
from torchvision.datasets import FakeData, ImageFolder
from torchvision.utils import make_grid

# Preload Lightning here to avoid conflict with XLA
import pytorch_lightning

# wandb

import wandb

# dalle classes and utils

from dalle_pytorch import distributed_utils
from dalle_pytorch.distributed_backends import (
    DeepSpeedBackend, DistributedBackend, HorovodBackend, XLABackend)
from dalle_pytorch import DiscreteVAE

# argument parsing

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

data_group = parser.add_mutually_exclusive_group(required=True)
data_group.add_argument(
    '--image_folder', type=str,
    help='path to your folder of images for learning the discrete VAE and its codebook')
data_group.add_argument(
    '--fake_data', action='store_true',
    help='use synthetically generated data instead of --image_folder')

parser.add_argument(
    '--image_size', type=int, required=False, default=128,
    help='image size')
parser.add_argument(
    '--wandb_mode', default='online', choices=('online', 'offline', 'disabled'),
    help='W&B mode')

parser = distributed_utils.wrap_arg_parser(parser)

train_group = parser.add_argument_group('Training settings')
train_group.add_argument(
    '--epochs', type=int, default=20,
    help='number of epochs')
train_group.add_argument(
    '--batch_size', type=int, default=8,
    help='batch size')
train_group.add_argument(
    '--learning_rate', type=float, default=1e-3, help='learning rate')
train_group.add_argument(
    '--lr_decay_rate', type=float, default=0.98,
    help='learning rate decay')
train_group.add_argument(
    '--starting_temp', type=float, default=1.,
    help='starting temperature')
train_group.add_argument(
    '--temp_min', type=float, default=0.5,
    help='minimum temperature to anneal to')
train_group.add_argument(
    '--anneal_rate', type=float, default=1e-6,
    help='temperature annealing rate')
train_group.add_argument(
    '--num_images_save', type=int, default=4,
    help='number of images to save')

model_group = parser.add_argument_group('Model settings')
model_group.add_argument(
    '--num_tokens', type=int, default=8192,
    help='number of image tokens')
model_group.add_argument(
    '--num_layers', type=int, default=3,
    help='number of layers (should be 3 or above)')
model_group.add_argument(
    '--num_resnet_blocks', type=int, default=2,
    help='number of residual net blocks')
model_group.add_argument(
    '--smooth_l1_loss', dest='smooth_l1_loss',
    action='store_true')
model_group.add_argument(
    '--emb_dim', type=int, default=512,
    help='embedding dimension')
model_group.add_argument(
    '--hidden_dim', type=int, default=256,
    help='hidden dimension')
model_group.add_argument(
    '--kl_loss_weight', type=float, default=0.,
    help='KL loss weight')

args = parser.parse_args()


def init_backend() -> DistributedBackend:
    """Initialize and return the distributed backend."""
    distr_backend = distributed_utils.set_backend_from_args(args)
    distr_backend.initialize()
    distr_backend.check_batch_size(args.batch_size)

    return distr_backend


def init_data(distr_backend: DistributedBackend) -> DataLoader:
    """Initialize dataloader with the appropriate sampler."""
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize(args.image_size),
        T.CenterCrop(args.image_size),
        T.ToTensor()
    ])

    if args.fake_data:
        dataset = FakeData(
            size=2 * args.batch_size * distr_backend.get_world_size(),
            image_size=(3, args.image_size, args.image_size),
            transform=transform)
    else:
        dataset = ImageFolder(args.image_path, transform)

    assert len(dataset) > 0, 'folder does not contain any images'
    if distr_backend.is_root_worker():
        print(f'{len(dataset)} images found for training')

    if isinstance(distr_backend, HorovodBackend) \
            or isinstance(distr_backend, XLABackend):
        data_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=distr_backend.get_world_size(),
            rank=distr_backend.get_rank())
    else:
        data_sampler = None

    return DataLoader(
        dataset,
        args.batch_size,
        shuffle=not data_sampler,
        sampler=data_sampler)


def init_vae(distr_backend: DistributedBackend) -> Module:
    """Initialize the VAE model."""
    vae = DiscreteVAE(
        image_size=args.image_size,
        num_layers=args.num_layers,
        num_tokens=args.num_tokens,
        codebook_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        num_resnet_blocks=args.num_resnet_blocks,
        smooth_l1_loss=args.smooth_l1_loss,
        kl_div_loss_weight=args.kl_loss_weight)

    if not isinstance(distr_backend, DeepSpeedBackend) \
            and not isinstance(distr_backend, XLABackend):
        vae = vae.cuda()

    return vae


def distribute(
        distr_backend: DistributedBackend,
        vae: Module,
        opt: torch.optim.Optimizer,
        dataloader: DataLoader,
        sched: torch.optim.lr_scheduler._LRScheduler) -> Tuple[
            Module, torch.optim.Optimizer, DataLoader,
            torch.optim.lr_scheduler._LRScheduler]:
    """Wrap the model and other components via the distributed backend."""
    if isinstance(distr_backend, DeepSpeedBackend):
        (distr_vae, distr_opt, distr_dl, distr_sched) = distr_backend.distribute(
            args=args,
            model=vae,
            optimizer=opt,
            model_parameters=vae.parameters(),
            training_data=dataloader.dataset,
            lr_scheduler=None,
            config_params={'train_batch_size': args.batch_size},
        )

    else:
        (distr_vae, distr_opt, distr_dl, distr_sched) = distr_backend.distribute(
            args=args,
            model=vae,
            optimizer=opt,
            model_parameters=vae.parameters(),
            training_data=dataloader,
            lr_scheduler=sched,
            config_params={'train_batch_size': args.batch_size},
        )

    return distr_vae, distr_opt, distr_dl, distr_sched


def log_artifact_factory(distr_backend: DistributedBackend) -> Callable:
    """Return a function for wandb artifact logging."""
    if not distr_backend.is_root_worker:
        return lambda x: None

    model_config = {
        'num_tokens': args.num_tokens,
        'smooth_l1_loss': args.smooth_l1_loss,
        'num_resnet_blocks': args.num_resnet_blocks,
        'kl_loss_weight': args.kl_loss_weight
    }
    run = wandb.init(
        project='dalle_train_vae',
        job_type='train_model',
        config=model_config,
        mode=args.wandb_mode)

    def log_artifact(file_name: str):
        model_artifact = wandb.Artifact(
            'trained-vae',
            type='model',
            metadata=model_config)
        model_artifact.add_file(file_name)
        run.log_artifact(model_artifact)

    return log_artifact


def save_model_factory(
        vae: Module,
        distr_vae: Module,
        distr_backend: DistributedBackend) -> Callable:
    """Return a function for saving the model checkpoint."""
    def save_model(path: str):
        save_obj = {
            'hparams': {
                'image_size': args.image_size,
                'num_layers': args.num_layers,
                'num_tokens': args.num_tokens,
                'codebook_dim': args.emb_dim,
                'hidden_dim': args.hidden_dim,
                'num_resnet_blocks': args.num_resnet_blocks,
            },
        }

        if isinstance(distr_backend, DeepSpeedBackend):
            cp_path = Path(path)
            path_sans_extension = cp_path.parent / cp_path.stem
            cp_dir = str(path_sans_extension) + '-ds-cp'

            distr_vae.save_checkpoint(cp_dir, client_state=save_obj)
            # We do not return so we do get a "normal" checkpoint to refer to.

        if not distr_backend.is_root_worker():
            return

        save_obj = {
            **save_obj,
            'weights': vae.state_dict()
        }

        torch.save(save_obj, path)

    return save_model


def generate_reconstruction_logs(
        vae: Module, images: torch.Tensor, recons: torch.Tensor) -> dict:
    """Generate image reconstructions for wandb logging."""
    k = args.num_images_save

    with torch.no_grad():
        codes = vae.get_codebook_indices(images[:k])
        hard_recons = vae.decode(codes)

    images, recons = map(lambda t: t[:k], (images, recons))
    images, recons, hard_recons, codes = map(
        lambda t: t.detach().cpu(),
        (images, recons, hard_recons, codes))
    images, recons, hard_recons = map(
        lambda t: make_grid(t.float(),
                            nrow=int(sqrt(k)),
                            normalize=True,
                            range=(-1, 1)),
        (images, recons, hard_recons))

    return {
        'sample images':        wandb.Image(images, caption = 'original images'),
        'reconstructions':      wandb.Image(recons, caption = 'reconstructions'),
        'hard reconstructions': wandb.Image(hard_recons, caption = 'hard reconstructions'),
        'codebook_indices':     wandb.Histogram(codes),
    }


def main():
    """Initialize and execute training."""

    # initialize

    distr_backend = init_backend()
    dataloader = init_data(distr_backend)
    vae = init_vae(distr_backend)

    # optimizer

    opt = Adam(vae.parameters(), lr=args.learning_rate)
    sched = ExponentialLR(optimizer=opt, gamma=args.lr_decay_rate)

    # distribute

    distr_vae, distr_opt, distr_dl, distr_sched = distribute(
        distr_backend, vae, opt, dataloader, sched)

    using_deepspeed_sched = False
    # Prefer scheduler in `deepspeed_config`.
    if distr_sched is None:
        distr_sched = sched
    elif isinstance(distr_backend, DeepSpeedBackend):
        # We are using a DeepSpeed LR scheduler and want to let DeepSpeed
        # handle its scheduling.
        using_deepspeed_sched = True

    # prepare logging and saving

    log_artifact = log_artifact_factory(distr_backend)
    save_model = save_model_factory(vae, distr_vae, distr_backend)

    # starting temperature

    global_step = 0
    temp = args.starting_temp

    # train loop

    for epoch in range(args.epochs):
        for i, (images, _) in enumerate(distr_dl):

            # forward & backward

            if not isinstance(distr_backend, XLABackend):
                images = images.cuda()

            loss, recons = distr_vae(
                images,
                return_loss = True,
                return_recons = True,
                temp = temp
            )

            if isinstance(distr_backend, DeepSpeedBackend):
                # Gradients are automatically zeroed after the step
                distr_vae.backward(loss)
                distr_vae.step()
            else:
                distr_opt.zero_grad()
                loss.backward()

                if isinstance(distr_backend, XLABackend):
                    distr_backend.backend_module.optimizer_step(distr_opt)
                else:
                    distr_opt.step()

            # save checkpoint

            if i % 100 == 0:
                save_model('./vae.pt')

            # log step

            # Collective loss, averaged
            avg_loss = distr_backend.average_all(loss).item()

            if distr_backend.is_root_worker():
                logs = {}

                if i % 100 == 0:
                    wandb.save('./vae.pt')

                    logs['temperature'] = temp
                    logs.update(generate_reconstruction_logs(vae, images, recons))

                if i % 10 == 0:
                    logs = {
                        **logs,
                        'epoch': epoch,
                        'iter': i,
                        'loss': avg_loss,
                        'lr': distr_sched.get_last_lr()[0],
                    }

                    print(epoch, i, f'lr - {logs["lr"]:6f} loss - {logs["loss"]}')

                wandb.log(logs)

            # step optimization

            if i % 100 == 0:

                # temperature anneal
                temp = max(temp * math.exp(-args.anneal_rate * global_step),
                           args.temp_min)

                # lr decay: do not advance schedulers from `deepspeed_config`.
                if not using_deepspeed_sched:
                    distr_sched.step()

            global_step += 1

        # save trained model to wandb as an artifact every epoch's end
        log_artifact('vae.pt')

    # save final vae and cleanup
    if distr_backend.is_root_worker():
        save_model('./vae-final.pt')
        wandb.save('./vae-final.pt')

        log_artifact('vae-final.pt')

        wandb.finish()


def _mp_fn(index, *args):
    """A XLA multiprocessing wrapper for main().

    Explicitly logs exceptions in child processes."""
    try:
        main()
    except Exception:
        logging.exception('Exception within child process')
        print('dd')


if __name__ == '__main__':
    if 'tpu_cores' in args and args.tpu_cores > 0:
        import torch_xla.distributed.xla_multiprocessing as xmp
        xmp.spawn(_mp_fn, nprocs=args.tpu_cores)
    else:
        main()
