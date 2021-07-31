import argparse
import math
from math import sqrt
import os
from pathlib import Path

# torch

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

# vision imports

from torchvision import transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData, ImageFolder
from torchvision.utils import make_grid

import pytorch_lightning  # Preload Lightning here to avoid conflict with XLA

# dalle classes and utils

from dalle_pytorch import distributed_utils
from dalle_pytorch.distributed_utils import (
    DeepSpeedBackend, HorovodBackend, XLABackend)
from dalle_pytorch import DiscreteVAE

def main(argv):

    # argument parsing

    parser = argparse.ArgumentParser()

    data_group = parser.add_mutually_exclusive_group(required=True)

    data_group.add_argument(
        '--image_folder',
        help='path to your folder of images for learning the discrete VAE and '
             'its codebook')

    data_group.add_argument(
        '--fake_data',
        action='store_true',
        help='use synthetically generated data instead of --image_folder')

    parser.add_argument(
        '--image_size',
        type=int,
        default=128,
        help='image size')

    parser.add_argument(
        '--wandb_mode', default='online',
        choices = ('online', 'offline', 'disabled'),
        help = 'W&B mode')

    parser.add_argument('--num_workers', type = int, default = 0,
                        help = 'number of DataLoader worker processes')
    
    parser.add_argument('--model_dir', default='.',
                        help='the directory to save the trained model')

    parser = distributed_utils.wrap_arg_parser(parser)


    train_group = parser.add_argument_group('Training settings')

    train_group.add_argument('--epochs', type = int, default = 20, help = 'number of epochs')

    train_group.add_argument('--batch_size', type = int, default = 8, help = 'batch size')

    train_group.add_argument('--learning_rate', type = float, default = 1e-3, help = 'learning rate')

    train_group.add_argument('--lr_decay_rate', type = float, default = 0.98, help = 'learning rate decay')

    train_group.add_argument('--starting_temp', type = float, default = 1., help = 'starting temperature')

    train_group.add_argument('--temp_min', type = float, default = 0.5, help = 'minimum temperature to anneal to')

    train_group.add_argument('--anneal_rate', type = float, default = 1e-6, help = 'temperature annealing rate')

    train_group.add_argument('--num_images_save', type = int, default = 4, help = 'number of images to save')

    model_group = parser.add_argument_group('Model settings')

    model_group.add_argument('--num_tokens', type = int, default = 8192, help = 'number of image tokens')

    model_group.add_argument('--num_layers', type = int, default = 3, help = 'number of layers (should be 3 or above)')

    model_group.add_argument('--num_resnet_blocks', type = int, default = 2, help = 'number of residual net blocks')

    model_group.add_argument('--smooth_l1_loss', dest = 'smooth_l1_loss', action = 'store_true')

    model_group.add_argument('--emb_dim', type = int, default = 512, help = 'embedding dimension')

    model_group.add_argument('--hidden_dim', type = int, default = 256, help = 'hidden dimension')

    model_group.add_argument('--kl_loss_weight', type = float, default = 0., help = 'KL loss weight')

    args = parser.parse_args(argv)

    # constants

    IMAGE_SIZE = args.image_size
    IMAGE_PATH = args.image_folder

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    LR_DECAY_RATE = args.lr_decay_rate

    NUM_TOKENS = args.num_tokens
    NUM_LAYERS = args.num_layers
    NUM_RESNET_BLOCKS = args.num_resnet_blocks
    SMOOTH_L1_LOSS = args.smooth_l1_loss
    EMB_DIM = args.emb_dim
    HIDDEN_DIM = args.hidden_dim
    KL_LOSS_WEIGHT = args.kl_loss_weight

    STARTING_TEMP = args.starting_temp
    TEMP_MIN = args.temp_min
    ANNEAL_RATE = args.anneal_rate

    NUM_IMAGES_SAVE = args.num_images_save

    # initialize distributed backend

    distr_backend = distributed_utils.set_backend_from_args(args)
    distr_backend.initialize()

    # data

    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize(IMAGE_SIZE),
        T.CenterCrop(IMAGE_SIZE),
        T.ToTensor()
    ])

    if args.fake_data:
        ds = FakeData(
            size=2 * BATCH_SIZE * distr_backend.get_world_size(),
            image_size=(3, IMAGE_SIZE, IMAGE_SIZE),
            transform=transform)

    else:
        ds = ImageFolder(IMAGE_PATH, transform)

    if isinstance(distr_backend, HorovodBackend) \
            or isinstance(distr_backend, XLABackend):
        data_sampler = torch.utils.data.distributed.DistributedSampler(
            ds, num_replicas=distr_backend.get_world_size(),
            rank=distr_backend.get_rank())
    else:
        data_sampler = None

    dl = DataLoader(ds, BATCH_SIZE, shuffle=not data_sampler,
                    sampler=data_sampler, num_workers=args.num_workers)

    vae_params = dict(
        image_size = IMAGE_SIZE,
        num_layers = NUM_LAYERS,
        num_tokens = NUM_TOKENS,
        codebook_dim = EMB_DIM,
        hidden_dim   = HIDDEN_DIM,
        num_resnet_blocks = NUM_RESNET_BLOCKS
    )

    vae = DiscreteVAE(
        **vae_params,
        smooth_l1_loss = SMOOTH_L1_LOSS,
        kl_div_loss_weight = KL_LOSS_WEIGHT
    )
    if not isinstance(distr_backend, DeepSpeedBackend) \
            and not isinstance(distr_backend, XLABackend):
        vae = vae.cuda()


    assert len(ds) > 0, 'folder does not contain any images'
    if distr_backend.is_root_worker():
        print(f'{len(ds)} images found for training')

    # optimizer

    opt = Adam(vae.parameters(), lr = LEARNING_RATE)
    sched = ExponentialLR(optimizer = opt, gamma = LR_DECAY_RATE)


    if distr_backend.is_root_worker():
        # weights & biases experiment tracking

        import wandb

        model_config = dict(
            num_tokens = NUM_TOKENS,
            smooth_l1_loss = SMOOTH_L1_LOSS,
            num_resnet_blocks = NUM_RESNET_BLOCKS,
            kl_loss_weight = KL_LOSS_WEIGHT
        )

        run = wandb.init(
            project = 'dalle_train_vae',
            job_type = 'train_model',
            config = model_config,
            mode = args.wandb_mode,
        )

    # distribute

    distr_backend.check_batch_size(BATCH_SIZE)
    deepspeed_config = {'train_batch_size': BATCH_SIZE}

    if isinstance(distr_backend, DeepSpeedBackend):
        training_data = ds
        lr_scheduler = None
    else:
        training_data = dl
        lr_scheduler = sched

    (distr_vae, distr_opt, distr_dl, distr_sched) = distr_backend.distribute(
        args=args,
        model=vae,
        optimizer=opt,
        model_parameters=vae.parameters(),
        training_data=training_data,
        lr_scheduler=lr_scheduler,
        config_params=deepspeed_config,
    )

    using_deepspeed_sched = False
    # Prefer scheduler in `deepspeed_config`.
    if distr_sched is None:
        distr_sched = sched
    elif isinstance(distr_backend, DeepSpeedBackend):
        # We are using a DeepSpeed LR scheduler and want to let DeepSpeed
        # handle its scheduling.
        using_deepspeed_sched = True

    def save_model(path):
        save_obj = {
            'hparams': vae_params,
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

    # starting temperature

    global_step = 0
    temp = STARTING_TEMP

    for epoch in range(EPOCHS):
        for i, (images, _) in enumerate(distr_dl):
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

            logs = {}

            if i % 100 == 0:
                if distr_backend.is_root_worker():
                    k = NUM_IMAGES_SAVE

                    with torch.no_grad():
                        codes = vae.get_codebook_indices(images[:k])
                        hard_recons = vae.decode(codes)

                    images, recons = map(lambda t: t[:k], (images, recons))
                    images, recons, hard_recons, codes = map(lambda t: t.detach().cpu(), (images, recons, hard_recons, codes))
                    images, recons, hard_recons = map(lambda t: make_grid(t.float(), nrow = int(sqrt(k)), normalize = True, range = (-1, 1)), (images, recons, hard_recons))

                    logs = {
                        **logs,
                        'sample images':        wandb.Image(images, caption = 'original images'),
                        'reconstructions':      wandb.Image(recons, caption = 'reconstructions'),
                        'hard reconstructions': wandb.Image(hard_recons, caption = 'hard reconstructions'),
                        'codebook_indices':     wandb.Histogram(codes),
                        'temperature':          temp
                    }

                    wandb.save('./vae.pt')
                save_model(os.path.join(args.model_dir, 'vae.pt'))

                # temperature anneal

                temp = max(temp * math.exp(-ANNEAL_RATE * global_step), TEMP_MIN)

                # lr decay

                # Do not advance schedulers from `deepspeed_config`.
                if not using_deepspeed_sched:
                    distr_sched.step()

            if i % 10 == 0:
                # Collective loss, averaged
                avg_loss = distr_backend.average_all(loss).item()

                if distr_backend.is_root_worker():
                    lr = distr_sched.get_last_lr()[0]
                    print(epoch, i, f'lr - {lr:6f} loss - {avg_loss}')

                    logs = {
                        **logs,
                        'epoch': epoch,
                        'iter': i,
                        'loss': avg_loss,
                        'lr': lr
                    }

            if distr_backend.is_root_worker():
                wandb.log(logs)
            global_step += 1

        if distr_backend.is_root_worker():
            # save trained model to wandb as an artifact every epoch's end

            model_artifact = wandb.Artifact('trained-vae', type = 'model', metadata = dict(model_config))
            model_artifact.add_file(os.path.join(args.model_dir, 'vae.pt'))
            run.log_artifact(model_artifact)

    if distr_backend.is_root_worker():
        # save final vae and cleanup

        save_model(os.path.join(args.model_dir, 'vae-final.pt'))

        wandb.save('vae-final.pt')

        model_artifact = wandb.Artifact('trained-vae', type = 'model', metadata = dict(model_config))
        model_artifact.add_file(os.path.join(args.model_dir, 'vae-final.pt'))
        run.log_artifact(model_artifact)

        wandb.finish()


def _mp_fn(index, *args):
    main(*args)


if __name__ == '__main__':
    pre_spawn_parser = argparse.ArgumentParser()
    pre_spawn_parser.add_argument(
        "--tpu_cores", type=int, default=0, choices=[0, 1, 8]
    )
    pre_spawn_flags, argv = pre_spawn_parser.parse_known_args()

    if pre_spawn_flags.tpu_cores > 0:
        import torch_xla.distributed.xla_multiprocessing as xmp
        xmp.spawn(_mp_fn, args=(argv,), nprocs=pre_spawn_flags.tpu_cores)
    else:
        main(argv)
