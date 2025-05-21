import numpy as np
import pandas as pd
import torchvision
from matplotlib import pyplot as plt


def vis_batch(batch, title):

    batch_grid = torchvision.utils.make_grid(batch, nrow=4)
    batch_grid = batch_grid.permute(1, 2, 0).numpy() * 0.5 + 0.5
    # print("Data type:", batch_grid.dtype)
    # print("Min value in image:", np.min(batch_grid))
    # print("Max value in image:", np.max(batch_grid))
    plt.imshow(batch_grid)
    plt.title(title)
    plt.axis('off')
    plt.show()


def logging_images(batch, step, num=16):
    title = f'Generator pics: step {step}'
    # vis_batch(batch[:num], title)
    batch = batch[:num]
    batch_grid = torchvision.utils.make_grid(batch, nrow=4)
    batch_grid = batch_grid.permute(1, 2, 0).numpy() * 0.5 + 0.5

    plt.imshow(batch_grid)
    plt.title(title)
    plt.axis('off')
    plt.show()


def logging_lr(logger, value_to_log, step, type='generator'):
    logger.report_scalar(
        title=f'Learning Rate {type}',
        series='lr',
        iteration=step,
        value=value_to_log,
    )


def logging(logger, metrics, value_to_log, step, is_train=True, per_epoch=False, type='Generator'):
    if 'accuracy' in metrics:
        title = f'Accuracy {type}'
    elif 'loss' in metrics:
        title = f'Running Loss {type}'
    else:
        title = ''

    if is_train:
        title = ' '.join([title, 'over Epochs']) if per_epoch else ' '.join([title, 'over Steps'])
    else:
        title = ' '.join([title, 'over Epochs'])

    logger.report_scalar(
            title=title,
            series=metrics,
            iteration=step,
            value=value_to_log,
        )