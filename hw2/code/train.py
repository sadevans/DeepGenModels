import warnings
from datetime import datetime as dtm

import hydra
import numpy as np
import torch
import utils
from init_funcs import gather_and_init
from rich.console import Console
from torch import distributed as dist
from trainer import Trainer

warnings.filterwarnings('ignore')


@hydra.main(version_base=None)
def main(cfg):
    console = Console(record=True)
    # device = 
    train_options, task = gather_and_init(cfg)
    trainer = Trainer(cfg, train_options, task, console)
    best_loss = np.inf
    start_train = dtm.now()

    if train_options['rank'] == 0:
        utils.print_config(cfg, console)
        console.print('We are ready to start training! :rocket:', style='bold red')

    # main loop
    for epoch in range(1, cfg.train.n_epoch + 1):
        if train_options['rank'] == 0:
            console.print(f'[bold yellow]Epoch: [/]{epoch} / {cfg.train.n_epoch}')

        trainer.run_epoch(epoch)

    delta_train = dtm.now() - start_train
    if train_options['rank'] == 0:
        console.print(f'Total elapsed training time: {utils.format_timedelta(delta_train)}', style='bold')
        console.print('Congrats, the whole experiment is finished! :tada:', style='bold red')
        task.close()

    dist.destroy_process_group()



if __name__ == '__main__':
    main()