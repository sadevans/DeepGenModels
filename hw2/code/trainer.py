import torch
import torchvision.utils as vutils
from datetime import datetime as dtm
import os
import utils
from logger import logging_images, logging_lr, logging

class Trainer:
    def __init__(self, cfg, options, task, console):
        self.options = options
        # self.device = options['device']
        self.generator = options['generator']
        self.discriminator = options['discriminator']
        
        self.optimizer_G = options['optimizer_G']
        self.optimizer_D = options['optimizer_D']
        self.criterion = options['criterion']
        
        self.cfg = cfg
        self.z_dim = cfg.generator.z_dim
        self.save_dir = cfg.train.save_dir
        self.log_interval = cfg.train.freq_vis
        self.sample_interval = cfg.train.sample_interval
        
        self.epoch = 0
        self.step = 0
        self.console = console
        self.task = task
        
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "checkpoints"), exist_ok=True)
        
        self.fixed_noise = torch.randn(64, self.z_dim).cuda().to(self.options['rank'], memory_format=torch.contiguous_format, non_blocking=True)
        
        if options['rank'] == 0:
            self.logger = task.get_logger()
            self._log_metrics(0, 0, 0)
            self._log_lr()

    def _log_metrics(self, d_loss, g_loss, real_score):
        """Логирование метрик в ClearML"""
        logging(self.logger, metrics='loss', value_to_log=d_loss, step=self.step, is_train=True, type='Discriminator')
        logging(self.logger, metrics='loss', value_to_log=g_loss, step=self.step, is_train=True, type='Generator')
        logging(self.logger, metrics='real score', value_to_log=real_score, step=self.step, is_train=True, type='Real')

    def _log_lr(self):
        """Логирование learning rate"""
        logging_lr(self.logger, self.optimizer_G.param_groups[0]['lr'], self.step, type='Generator')
        logging_lr(self.logger, self.optimizer_D.param_groups[0]['lr'], self.step, type='Discriminator')


    def _save_checkpoint(self):
        """Сохранение чекпоинта"""
        if self.epoch % self.cfg.train.save_interval == 0:
            utils.save_checkpoint(self.generator, self.options, self.epoch,  type='generator')
            utils.save_checkpoint(self.discriminator, self.options, self.epoch,  type='discriminator')


    def _generate_samples(self):
        """Генерация и сохранение примеров"""
        with torch.no_grad():
            fake_images = self.generator(self.fixed_noise).detach().cpu()
        
        img_grid = vutils.make_grid(fake_images, padding=2, normalize=True)
        # np_img = img_grid.permute(1, 2, 0).numpy()
        
        logging_images(
            fake_images, 
            self.step, 
        )
        
        if self.step % self.sample_interval == 0:
            vutils.save_image(
                img_grid,
                os.path.join(self.save_dir, f"fake_samples_step_{self.step}.png")
            )

    def train_step(self, real_imgs):
        """Один шаг обучения"""
        real_imgs = real_imgs.cuda().to(self.options['rank'], memory_format=torch.contiguous_format, non_blocking=True)
        batch_size = real_imgs.size(0)
        
        real_labels = torch.ones(batch_size, 1).cuda().to(self.options['rank'], memory_format=torch.contiguous_format, non_blocking=True)
        fake_labels = torch.zeros(batch_size, 1).cuda().to(self.options['rank'], memory_format=torch.contiguous_format, non_blocking=True)

        self.optimizer_D.zero_grad()
        
        real_outputs = self.discriminator(real_imgs)
        d_loss_real = self.criterion(real_outputs, real_labels)
        d_loss_real.backward()
        
        noise = torch.randn(batch_size, self.z_dim, 4, 4).cuda().to(self.options['rank'], memory_format=torch.contiguous_format, non_blocking=True)
        fake_imgs = self.generator(noise)
        fake_outputs = self.discriminator(fake_imgs.detach())
        d_loss_fake = self.criterion(fake_outputs, fake_labels)
        d_loss_fake.backward()
        
        d_loss = d_loss_real + d_loss_fake
        # d_loss.backward()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()

        outputs = self.discriminator(fake_imgs)
        g_loss = self.criterion(outputs, real_labels)
        
        g_loss.backward()
        self.optimizer_G.step()

        return {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item(),
            'real_score': real_outputs.mean().item(),
            'fake_imgs': fake_imgs.detach()
        }

    def run_epoch(self, epoch):
        """Запуск одной эпохи обучения"""
        self.epoch = epoch
        self.generator.train()
        self.discriminator.train()
        
        start_time = dtm.now()
        
        for step, (real_imgs, labels) in enumerate(self.options['train_loader']):
            self.step += 1
            
            losses = self.train_step(real_imgs)

            if self.step % self.log_interval == 0 and self.options['rank'] == 0:
                self._log_metrics(
                    losses['d_loss'], 
                    losses['g_loss'], 
                    losses['real_score']
                )
                self._log_lr()
                
                self.console.print(
                    f"[Epoch {epoch}/{self.cfg.train.epochs}] "
                    f"[Step {step}/{len(self.options['train_loader'])}] "
                    f"D_loss: {losses['d_loss']:.4f} "
                    f"G_loss: {losses['g_loss']:.4f} "
                    f"D(x): {losses['real_score']:.4f}"
                )

            if self.step % self.sample_interval == 0:
                self._generate_samples()
            
        self._save_checkpoint()
        
        epoch_time = (dtm.now() - start_time).total_seconds()
        self.console.print(f"Epoch {epoch} completed in {epoch_time:.2f} seconds")