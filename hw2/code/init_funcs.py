import os
import random

import numpy as np
import torch
import utils
import wandb
from clearml import Task
from data import get_dataloader
from torchvision.datasets import CelebA
# from data_copy import get_dataloader
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from omegaconf import OmegaConf
from torch import distributed as dist
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817


# def create_task(cfg, rank):
#     if rank == 0:
#         wandb.init(
#             project=cfg.project_name,
#             name=cfg.exp_name,
#             config=OmegaConf.to_container(cfg, resolve=True),
#         )
#         return  wandb.run

def create_task(cfg, rank):
    if rank == 0:
        task = Task.init(
            project_name=cfg.project_name,
            task_name=f'{cfg.exp_name}',
            auto_connect_frameworks=True,
        )
        dict_cfg = OmegaConf.to_container(cfg)
        task.connect(dict_cfg)
        return task




def gather_and_init(config):  # noqa: WPS210
    init_seeds(config)
    # rank, model = init_ddp_model(config)

    rank, generator = init_model(config, type='generator')
    rank, discriminator = init_model(config, type='discriminator')

    use_amp, scaler = init_mixed_precision(config)
    outdir = create_save_folder(config, rank)
    task = create_task(config, rank)
    # task = None
    transform = utils.get_transform(config, is_train=True)

    celeba_dataset = CelebA(root='./data/celeba', split='train', target_type='attr', transform=transform, download=True)
    train_loader = DataLoader(celeba_dataset, batch_size=config.dataset.batch_size, shuffle=True)


    
    num_batches = len(train_loader)
    criterion, optimizer_G, optimizer_D, scheduler = utils.get_training_parameters(config, generator, discriminator, num_batches)
    mixer = utils.get_mixer(config)

    train_options = {
        'generator': generator,
        'discriminator':discriminator,
        'train_loader': train_loader,
        'loss': criterion,
        'optimizer_G': optimizer_G,
        'optimizer_D': optimizer_D,
        'use_amp': use_amp,
        'scaler': scaler,
        'rank': rank,
        'scheduler': scheduler,
        'outdir': outdir,
        'mixer': mixer,
    }

    return train_options, task
    # return model, train_options



def create_save_folder(config, rank):
    outdir = os.path.join(config.outdir, config.exp_name)
    if (rank == 0) and (not os.path.exists(outdir)):
        os.makedirs(outdir)
    return outdir


def init_seeds(config):
    seed = config.dataset.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True


def init_mixed_precision(config):
    use_amp = False
    if hasattr(config.train, 'use_amp'):
        if config.train.use_amp:
            use_amp = True

    if use_amp:
        scaler = GradScaler()
    else:
        scaler = None

    return use_amp, scaler


def init_model(config, type='generator'):
    dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    net = utils.load_model(config, type=type, is_train=True).cuda()
    #sync_bn_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net, rank)
    ddp_net = DDP(net, device_ids=[rank])

    return rank, ddp_net