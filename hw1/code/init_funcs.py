import os
import random

import numpy as np
import torch
import utils
from clearml import Task
from data import get_dataloader

from omegaconf import OmegaConf
from torch import distributed as dist
from torch.cuda.amp import GradScaler


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


def get_task_id_by_name(task_name):
    tasks = Task.get_tasks(filters={'name': task_name})
    if tasks:
        return tasks[0].id
    else:
        return None


def get_existing_task(task_name):
    task_id = get_task_id_by_name(task_name)
    task = Task.get_task(task_id)
    return task


def gather_and_init(config):  # noqa: WPS210
    init_seeds(config)
    rank, model = init_ddp_model(config)
    use_amp, scaler = init_mixed_precision(config)
    outdir = create_save_folder(config, rank)
    task = create_task(config, rank)

    train_loader = get_dataloader(config, split_type='train', task=task, rank=rank)
    num_batches = len(train_loader)
    optimizer, scheduler = utils.get_training_parameters(config, model, num_batches)

    train_options = {
        'train_loader': train_loader,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'use_amp': use_amp,
        'scaler': scaler,
        'rank': rank,
        'outdir': outdir,
    }

    return model, train_options, task


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


def init_ddp_model(config):
    dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()

    torch.cuda.set_device(rank)

    net = utils.load_model(config, is_train=True).cuda()
    # ddp_net = DDP(net, device_ids=[rank])

    return rank, net
