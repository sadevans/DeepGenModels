import os
from collections import OrderedDict

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from rich.markdown import Markdown
from rich.tree import Tree
from torch.hub import load_state_dict_from_url
from torchinfo import summary
from torchvision.models._api import WeightsEnum
from torchvision.transforms import v2

from rich.console import Console
from rich.tree import Tree
from rich.markdown import Markdown
from omegaconf import DictConfig
from typing import Union


def get_state_dict(self, *args, **kwargs):
    kwargs.pop('check_hash')
    return load_state_dict_from_url(self.url, *args, **kwargs)


def load_model(config, is_train=False):
    WeightsEnum.get_state_dict = get_state_dict

    model = instantiate(config.model)

    if is_train:
        if hasattr(config.model, 'checkpoint'):
            model.load_state_dict(load_checkpoint(config, model, is_train=True), strict=False)
    else:
        model.load_state_dict(load_checkpoint(config, model), strict=False)

    return model


def save_checkpoint(model, options, epoch):
    filename = f'model_{epoch:04d}.pth'
    directory = options['outdir']
    filename = os.path.join(directory, filename)
    weights = model.state_dict()
    state = OrderedDict([
        ('state_dict', weights),
        ('optimizer', options['optimizer'].state_dict()),
        ('epoch', epoch),
    ])

    torch.save(state, filename)


def get_scheduler(config, optimizer, num_batches=None):
    if config.train.lr_schedule.name == 'cosine':
        T_max = config.train.n_epoch * num_batches  # noqa: N806
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    elif config.train.lr_schedule.name == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.train.lr_schedule.step_size,
            gamma=config.train.lr_schedule.gamma,
        )
    else:
        raise KeyError(f'Unknown type of lr schedule: {config.train.lr_schedule}')
    return scheduler


def get_training_parameters(cfg, net, num_batches=None):
    optimizer = instantiate(cfg.train.optimizer, params=net.parameters())
    scheduler = get_scheduler(cfg, optimizer, num_batches)
    # return criterion, optimizer, scheduler
    return optimizer, scheduler



def get_mixer(config):
    targets = config.dataset.targets_column
    no_probs = True
    for target in targets:
        if 'prob' in target:
            no_probs=False
    if no_probs:
        if 'mixer' in config:
            mixers = []
            for _, conf in config.mixer.items():
                mixers.append(instantiate(conf, num_classes=config.model.num_classes))
            return v2.RandomChoice(mixers)
    else:
        print("NB. We have probs as targers and can't use any mixers!")


def load_checkpoint(config, model, is_train=False):
    checkpoint_name = config.model.checkpoint.name if is_train else f'model_{config.train.n_epoch:04d}.pth'
    checkpoint_path = checkpoint_name if os.path.exists(checkpoint_name) else os.path.join(config.outdir, config.exp_name, checkpoint_name)
    print(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cuda')['state_dict']

    new_state_dict = OrderedDict()
    for key, weight in checkpoint.items():
        name = key.replace('module.', '').replace('_orig_mod.', '')
        if is_train:
            if 'classifier' not in name or config.model.checkpoint.last_layer:
                new_state_dict[name] = weight
            else:
                print(f"Excluding key: {name} (not loading classifier weights)")
        else:
            new_state_dict[name] = weight

    model_state_dict = model.state_dict()
    missing_keys = set(model_state_dict.keys()) - set(new_state_dict.keys())
    if missing_keys:
        print(f"Warning: Missing keys in state_dict: {missing_keys}")

    return new_state_dict


def format_timedelta(delta):
    seconds = int(delta.total_seconds())

    secs_in_a_day = 86400
    secs_in_a_hour = 3600
    secs_in_a_min = 60

    days, seconds = divmod(seconds, secs_in_a_day)
    hours, seconds = divmod(seconds, secs_in_a_hour)
    minutes, seconds = divmod(seconds, secs_in_a_min)

    time_fmt = f'{hours:02d}:{minutes:02d}:{seconds:02d}'

    if days > 0:
        suffix = 's' if days > 1 else ''
        return f'{days} day{suffix} {time_fmt}'

    return time_fmt


def print_model(config, model, console):
    columns = ('input_size', 'output_size', 'num_params', 'kernel_size', 'mult_adds')
    image_size = (3, config.dataset.w, config.dataset.h)

    console.print(
        str(
            summary(
                model,
                image_size,
                batch_dim=0,
                col_names=columns,
                depth=8,
                verbose=0,
            ),
        ),
    )


def print_config(config: DictConfig, console: Console):
    """Prints the configuration in a structured way using Rich."""

    fields = ('exp_name', 'outdir', 'logdir', 'model', 'train', 'dataset', 'augmentation', 'mixer')
    console.print(Markdown('# ----------------- CONFIG ---------------\n'))

    for field in fields:
        if field in config:
            field_content = config[field]  # Use dictionary access, it's cleaner

            if isinstance(field_content, str):
                console.print(f'[bold red]{field}[/]: [magenta]{field_content}[/]')
            elif isinstance(field_content, (DictConfig, dict)):  # Handle both DictConfig and regular dict
                tree = Tree(f'[bold red]{str(field)}[/]')
                for sub_field, config_section in field_content.items():  # Iterate directly over items
                    _add_to_tree(tree, sub_field, config_section)
                console.print(tree)
            else:  # Handle other top-level types directly (if any)
                console.print(f'[bold red]{field}[/]: [magenta]{str(field_content)}[/]') # handle other type

    console.print(Markdown('# ----------------- END ---------------\n'))


def _add_to_tree(tree: Tree, key: str, value: Union[DictConfig, dict, list, float, int, str]):
    """Recursively adds configuration items to the Rich Tree."""

    if isinstance(value, (DictConfig, dict)):
        branch = tree.add(f'[bold]{str(key)}[/]')
        for sub_key, sub_value in value.items():
            _add_to_tree(branch, sub_key, sub_value)
    elif isinstance(value, list):
        tree.add(f'[bold]{key}[/]: [green]{str(value)}[/]')
    elif isinstance(value, float):
        tree.add(f'[bold]{key}[/]: [bold cyan]{str(value)}[/]')
    elif isinstance(value, int): #added int
        tree.add(f'[bold]{key}[/]: [bold cyan]{str(value)}[/]')
    else:  # Catch-all for strings and other types
        tree.add(f'[bold]{key}[/]: [magenta]{str(value)}[/]')
