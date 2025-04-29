import sys
from datetime import datetime as dtm

import torch
import torch.nn as nn
import torch.nn.functional as F
from logger import logging, logging_images, logging_lr


class Trainer:
    def __init__(self, model, cfg, train_options, task, console):
        self.model = model
        self.cfg = cfg
        self.options = train_options
        self.len_train = len(self.options['train_loader'])
        self.lr = cfg.train.optimizer.lr
        self.epoch = 0
        self.step = 0
        self.console = console

        self.memory_items = F.normalize(
            torch.rand((cfg.model.memory_size, cfg.model.key_dim), dtype=torch.float),
            dim=1
        ).cuda()

        if self.options['rank'] == 0:
            self.logger = task.get_logger()
            logging_lr(self.logger, value_to_log=self.lr, step=0)

    def run_epoch(self, epoch):
        self.model.train()
        self.epoch = epoch
        start_epoch = dtm.now()
        torch.cuda.synchronize()

        total_loss = 0
        total_pixel_loss = 0
        total_compact_loss = 0
        total_separate_loss = 0

        for step, batch in enumerate(self.options['train_loader']):
            self.step += 1
            start_batch = dtm.now()

            batch = batch.cuda().to(self.options['rank'], memory_format=torch.contiguous_format, non_blocking=True)
            
            outputs, _, _, self.memory_items, _, _, separateness_loss, compactness_loss = self.model(
                batch, self.memory_items, train=True
            )

            loss_pixel = torch.nn.functional.mse_loss(outputs, batch, reduction='mean')
            loss = loss_pixel + \
                   self.cfg.train.loss_compact * compactness_loss + \
                   self.cfg.train.loss_separate * separateness_loss

            self._do_optim_step(loss)

            total_loss += loss.item()
            total_pixel_loss += loss_pixel.item()
            total_compact_loss += compactness_loss.item()
            total_separate_loss += separateness_loss.item()

            if self.cfg.train.lr_schedule.name == 'cosine':
                self.options['scheduler'].step()
                self.lr = self.options['scheduler'].get_last_lr()[0]

            delta_batch = dtm.now() - start_batch

            if step == 0:
                logging_images(batch, outputs, is_train=True)

            if step % self.cfg.train.freq_vis == 0 and step != 0:
                if self.options['rank'] == 0:
                    self.console.log(
                        f'Epoch: {self.epoch} / {self.cfg.train.n_epoch}, '
                        f'batch:[{step} / {self.len_train}], '
                        f'Total Loss: {total_loss/(step+1):.4f}, '
                        f'Pixel: {total_pixel_loss/(step+1):.4f}, '
                        f'Compact: {total_compact_loss/(step+1):.4f}, '
                        f'Separate: {total_separate_loss/(step+1):.4f}, '
                        f'time: {delta_batch.total_seconds():.3f}s'
                    )
                    sys.stdout.flush()

                    logging(self.logger, 'Total Loss', total_loss/(step+1), step=self.step)
                    logging(self.logger, 'Pixel Loss', total_pixel_loss/(step+1), step=self.step)
                    logging(self.logger, 'Compact Loss', total_compact_loss/(step+1), step=self.step)
                    logging(self.logger, 'Separate Loss', total_separate_loss/(step+1), step=self.step)
                    logging_lr(self.logger, self.lr, step=self.step)

        torch.cuda.synchronize()
        delta_epoch = dtm.now() - start_epoch

        if self.options['rank'] == 0:
            self.console.log(
                f'Epoch {epoch} results:\n'
                f'Mean Total Loss: {total_loss/self.len_train:.4f}\n'
                f'Mean Pixel Loss: {total_pixel_loss/self.len_train:.4f}\n'
                f'Mean Compact Loss: {total_compact_loss/self.len_train:.4f}\n'
                f'Mean Separate Loss: {total_separate_loss/self.len_train:.4f}\n'
                f'Time: {delta_epoch.total_seconds():.2f}s'
            )
            self.console.print(f'Train process of epoch {epoch} is done', style='bold red')

            logging(self.logger, 'Epoch Total Loss', total_loss/self.len_train, step=self.epoch, per_epoch=True)
            logging(self.logger, 'Epoch Pixel Loss', total_pixel_loss/self.len_train, step=self.epoch, per_epoch=True)
            logging(self.logger, 'Epoch Compact Loss', total_compact_loss/self.len_train, step=self.epoch, per_epoch=True)
            logging(self.logger, 'Epoch Separate Loss', total_separate_loss/self.len_train, step=self.epoch, per_epoch=True)

    def _do_optim_step(self, loss):
        self.options['optimizer'].zero_grad()
        loss.backward()
        self.options['optimizer'].step()


def find_mse_threshold(model, proliv_loader):
        model.eval()
        mse_values = []
        with torch.no_grad():
            for images in proliv_loader:
                outputs, mu, sigma = model(images)
                mse =  nn.functional.mse_loss(outputs, images, reduction='none')
                mse_values.append(mse.mean(dim=(1, 2, 3)))
        mse_values = torch.cat(mse_values)
        threshold = mse_values.mean() + mse_values.std()
        return threshold


def classify_images(model, test_loader, threshold):
    model.eval()
    true_positive, true_negative = 0, 0
    total_positive, total_negative = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            mse = nn.functional.mse_loss(outputs, images, reduction='none')
            mse_values = mse.mean(dim=(1, 2, 3))

            predicted_labels = (mse_values > threshold).float()

            true_positive += ((predicted_labels == 1) & (labels == 1)).sum().item()
            true_negative += ((predicted_labels == 0) & (labels == 0)).sum().item()
            total_positive += (labels == 1).sum().item()
            total_negative += (labels == 0).sum().item()

    tpr = true_positive / total_positive
    tnr = true_negative / total_negative
    return tpr, tnr
