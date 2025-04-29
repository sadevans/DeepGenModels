import torchvision
from matplotlib import pyplot as plt


def vis_batch(batch, title, num=16):
    batch = batch[:num]
    batch_grid = torchvision.utils.make_grid(batch, nrow=4)
    # batch_grid = batch_grid.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5
    batch_grid = batch_grid.permute(1, 2, 0).cpu().numpy()


    plt.imshow(batch_grid)
    plt.axis('off')
    plt.suptitle(title)
    plt.show()


def logging_images(original_batch, reconstructed_batch, is_train=True, num=16):
    split = 'Train' if is_train else 'Val'
    title = f'{split} Batch Visualization'

    vis_batch(original_batch, title + ': original', num=num)
    vis_batch(reconstructed_batch, title + ': reconstructed', num=num)


def logging_lr(logger, value_to_log, step):
    logger.report_scalar(
        title='Learning Rate',
        series='lr',
        iteration=step,
        value=value_to_log,
    )


def logging(logger, metrics, value_to_log, step, is_train=True, per_epoch=False):
    if 'TPR' in metrics:
        title = 'TPR'
    elif 'TNR' in metrics:
        title = 'TNR'
    elif 'loss' in metrics:
        title = 'Running Loss'
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
