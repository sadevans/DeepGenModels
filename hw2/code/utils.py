import os
from collections import OrderedDict

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from rich.markdown import Markdown
from rich.tree import Tree
from torch.hub import load_state_dict_from_url
from torchinfo import summary
from torchvision.models._api import WeightsEnum  # noqa, в torch 2.1 проблемы с чекпоинтом были
from torchvision.transforms import v2


def get_transform(config, is_train=False):
    if is_train and ('augmentation' in config):
        augs = []
        for aug_name, aug_item in config.augmentation.items():
            if aug_name == 'to_dtype':
                aug_object = instantiate(aug_item, dtype=torch.float32)
            else:
                aug_object = instantiate(aug_item, _convert_='all')
            augs.append(aug_object)

        return v2.Compose(augs)
    else:
        transform = v2.Compose([
            v2.PILToTensor(),
            v2.Resize((config.dataset.h, config.dataset.w)),
            v2.ToDtype(dtype=torch.float32, scale=True),
            v2.Normalize(mean=[0.5], std=[0.5]),
        ])

    return transform


def get_state_dict(self, *args, **kwargs):
    kwargs.pop('check_hash')
    return load_state_dict_from_url(self.url, *args, **kwargs)


def load_model(config, type='generator', is_train=False):
    # num_classes = config.model.num_classes
    WeightsEnum.get_state_dict = get_state_dict

    if type == 'generator':
        model = instantiate(config.generator.arch)
    elif type == 'discriminator':
        model = instantiate(config.discriminator.arch)

    # last_module_name = list(model.named_modules())[-1][0].split('.')[0]
    # if last_module_name == 'fc':
    #     model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    # elif last_module_name == 'classifier':
    #     if isinstance(model.classifier, torch.nn.Sequential):
    #         model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    #     else:
    #         model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)

    # elif last_module_name == 'head':
    #     model.head.fc = torch.nn.Linear(model.head.fc.in_features, num_classes)
    # elif last_module_name == 'heads':
    #     dropout = model.heads[0][0].p
    #     in_features = model.heads[0][-1].in_features
    #     model.heads = torch.nn.ModuleList([
    #         torch.nn.Sequential(
    #             torch.nn.Dropout(p=dropout),
    #             torch.nn.Linear(in_features, num_classes_i),
    #         )
    #         for num_classes_i in num_classes
    #     ])
    # else:
    #     raise KeyError(f'Model is not supported: {config.model.arch}')

    if is_train:
        if type == 'generator' and hasattr(config.generator, 'checkpoint'):
            model.load_state_dict(load_checkpoint(config, model, is_train=True), strict=False)
        elif type == 'discriminator' and hasattr(config.discriminator, 'checkpoint'):
            model.load_state_dict(load_checkpoint(config, model, is_train=True), strict=False)
    else:
        model.load_state_dict(load_checkpoint(config, model), strict=False)

    return model


def save_checkpoint(model, options, epoch, type='generator'):
    filename = f'{type}_{epoch:04d}.pth'
    directory = options['outdir']
    filename = os.path.join(directory, filename)
    weights = model.state_dict()
    if type == 'generator':
        state = OrderedDict([
            ('state_dict', weights),
            ('optimizer', options['optimizer_G'].state_dict()),
            ('scheduler', options['scheduler_G'].state_dict()),
            ('epoch', epoch),
        ])
    elif type == 'discriminator':
        state = OrderedDict([
            ('state_dict', weights),
            ('optimizer', options['optimizer_D'].state_dict()),
            ('scheduler', options['scheduler_D'].state_dict()),
            ('epoch', epoch),
        ])

    torch.save(state, filename)


def get_scheduler(config, optimizer, num_batches=None, type='generator'):
    scheduler = None
    if type == 'generator':
        if hasattr(config.train, 'lr_schedule_G') and config.train.lr_schedule_G.name == 'cosine':
            T_max = config.train.n_epoch * num_batches  # noqa: N806
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        elif hasattr(config.train, 'lr_schedule_G') and config.train.lr_schedule_G.name == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.train.lr_schedule_G.step_size,
                gamma=config.train.lr_schedule_G.gamma,
            )
        # else:
        #     raise KeyError(f'Unknown type of lr schedule: {config.train.lr_schedule_G}')
    elif type == 'discriminator':
        if hasattr(config.train, 'lr_schedule_D') and config.train.lr_schedule_D.name == 'cosine':
            T_max = config.train.n_epoch * num_batches  # noqa: N806
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        elif hasattr(config.train, 'lr_schedule_D') and config.train.lr_schedule_D.name == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.train.lr_schedule_D.step_size,
                gamma=config.train.lr_schedule_D.gamma,
            )
        # else:
        #     raise KeyError(f'Unknown type of lr schedule')
    return scheduler


def get_training_parameters(cfg, netG, netD, num_batches=None):
    criterion = instantiate(cfg.train.loss, _convert_='all')
    optimizer_G = instantiate(cfg.train.optimizer_G, params=netG.parameters())
    optimizer_D = instantiate(cfg.train.optimizer_D, params=netD.parameters())

    scheduler_G = get_scheduler(cfg, optimizer_G, num_batches, type='generator')
    scheduler_D = get_scheduler(cfg, optimizer_D, num_batches, type='discriminator')

    return criterion, optimizer_G, optimizer_D, scheduler_G, scheduler_D


def load_checkpoint(config, model, type='generator', is_train=False):
    if type == 'generator':
        cfg_model = config.generator
    elif type == 'discriminator':
        cfg_model = config.discriminator
    checkpoint_name = cfg_model.checkpoint.name if is_train else f'model_{config.train.n_epoch:04d}.pth'
    checkpoint_path = checkpoint_name if os.path.exists(checkpoint_name) else os.path.join(config.outdir, config.exp_name, checkpoint_name)
    
    print('checkpoint_name: ', checkpoint_name)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cuda')['state_dict']

    new_state_dict = OrderedDict()
    for key, weight in checkpoint.items():
        name = key.replace('module.', '').replace('_orig_mod.', '')
        if is_train:
            if 'classifier' not in name or cfg_model.checkpoint.last_layer:
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


# Я сама себя ненавижу за эту функцию
def print_config(config, console):  # noqa
    fields = ('exp_name', 'outdir', 'logdir', 'model', 'train', 'dataset', 'augmentation', 'mixer')
    console.print(Markdown('# ----------------- CONFIG ---------------\n'))
    for field in fields:
        if field in config:
            field_content = config.get(field)
            if isinstance(field_content, str):
                console.print(f'[bold red]{field}[/]: [magenta]{field_content}[/]')
            else:
                tree = Tree(f'[bold red]{str(field)}[/]')
                for sub_field in field_content.keys():
                    config_section = field_content.get(sub_field)
                    if isinstance(config_section, DictConfig):
                        branch = tree.add(f'[bold]{str(sub_field)}[/]')  # noqa
                        for last_param in config_section.keys():  # noqa
                            sub_section = config_section.get(last_param)  # noqa
                            if isinstance(sub_section, DictConfig):  # noqa
                                sub_branch = branch.add(f'[bold]{str(last_param)}[/]')  # noqa
                                for last_last_param in sub_section.keys():  # noqa
                                    sub_sub_section = sub_section.get(last_last_param)  # noqa
                                    sub_branch.add(f'[bold]{last_last_param}[/]: [magenta]{str(sub_sub_section)}[/]')  # noqa
                            elif isinstance(sub_section, float):  # noqa
                                branch.add(f'[bold]{last_param}[/]: [bold cyan]{str(sub_section)}[/]')  # noqa
                            elif isinstance(sub_section, list):  # noqa
                                branch.add(f'[bold]{last_param}[/]: [green]{str(sub_section)}[/]')  # noqa
                            else:
                                branch.add(f'[bold]{last_param}[/]: [magenta]{str(sub_section)}[/]')  # noqa
                    elif isinstance(config_section, (float, int)):
                        tree.add(f'[bold]{sub_field}[/]: [bold cyan]{str(config_section)}[/]')  # noqa
                    elif isinstance(sub_section, list):
                        tree.add(f'[bold]{sub_field}[/]: [green]{str(config_section)}[/]')  # noqa
                    else:
                        tree.add(f'[bold]{sub_field}[/]: [magenta]{str(config_section)}[/]')  # noqa

                console.print(tree)
    console.print(Markdown('# ----------------- END ---------------\n'))