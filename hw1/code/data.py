import os
import torch
from hydra.utils import instantiate


def get_train_loader(config, task=None, rank=0):
    train_lists = config.dataset.train_lists

    dataset = instantiate(
            config.dataset.dataset_class,
            config.dataset.root,
            os.path.join(config.dataset.prefix_lists, train_lists[0]), # select the first file
            targets_column=config.dataset.targets_column,
        )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.dataset.batch_size,
        shuffle=True,
        num_workers=config.dataset.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    return data_loader


def get_val_loader(config, task=None, rank=0):
    val_lists = config.dataset.val_lists

    dataset = instantiate(
        config.dataset.dataset_class,
        config.dataset.root,
        os.path.join(config.dataset.prefix_lists, val_lists[0]),  # select the first file
        targets_column=config.dataset.targets_column,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.dataset.batch_size,
        shuffle=True,
        num_workers=config.dataset.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return data_loader  # noqa: WPS331


def get_test_loader(config, data_list, is_inference=True):
    
    data_list = os.path.join(config.dataset.prefix_lists, data_list)
    test_dataset = instantiate(
        config.dataset.dataset_class,
        config.dataset.root,
        data_list,
        targets_column=config.dataset.targets_column,
        is_inference=is_inference,
    )

    data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        drop_last=False,
        pin_memory=True,
    )
    return data_loader


def get_dataloader(config, split_type, task=None, rank=0):
    if split_type == 'train':
        data_loader = get_train_loader(config, task, rank)
    elif split_type == 'val':
        data_loader = get_val_loader(config, task, rank)
    else:
        raise KeyError('Subset unexpected type:', split_type)

    return data_loader