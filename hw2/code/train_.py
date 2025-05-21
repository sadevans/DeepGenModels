import os
import torch
from torch.utils.data import DataLoader
from data.celeba_dataset import CelebADataset
from models.gan import GAN
from clearml import Task
import matplotlib.pyplot as plt
import numpy as np

# Инициализация ClearML
task = Task.init(project_name='GAN_CelebA', task_name='CSPUp_GAN')

# Параметры
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
z_dim = 100
batch_size = 64
epochs = 200
lr = 0.0002
data_dir = "data/celeba"

# Загрузка данных
dataset = CelebADataset(data_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Инициализация GAN
gan = GAN(device, z_dim)

# Логирование параметров
task.connect({
    "z_dim": z_dim,
    "batch_size": batch_size,
    "epochs": epochs,
    "learning_rate": lr
})

# Обучение
for epoch in range(epochs):
    for i, real_imgs in enumerate(dataloader):
        real_imgs = real_imgs.to(device)
        
        d_loss, g_loss, fake_imgs = gan.train_step(real_imgs)
        
        # Логирование
        iteration = epoch * len(dataloader) + i
        task.get_logger().report_scalar("Loss", "Discriminator", value=d_loss, iteration=iteration)
        task.get_logger().report_scalar("Loss", "Generator", value=g_loss, iteration=iteration)
        
        if i % 100 == 0:
            print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] "
                  f"[D loss: {d_loss:.4f}] [G loss: {g_loss:.4f}]")
            
            # Сохранение примеров изображений
            if epoch % 5 == 0 and i == 0:
                fig, axs = plt.subplots(1, 5, figsize=(15, 3))
                for j in range(5):
                    img = fake_imgs[j].cpu().permute(1, 2, 0).numpy()
                    img = (img * 0.5) + 0.5  # Денормализация
                    axs[j].imshow(img)
                    axs[j].axis('off')
                plt.tight_layout()
                
                # Логирование изображений в ClearML
                task.get_logger().report_matplotlib_figure(
                    title="Generated Images", series=f"epoch_{epoch}", iteration=iteration, figure=plt)
                plt.close()
    
    # Сохранение моделей
    if epoch % 10 == 0:
        torch.save(gan.generator.state_dict(), f"generator_epoch_{epoch}.pth")
        torch.save(gan.discriminator.state_dict(), f"discriminator_epoch_{epoch}.pth")

print("Training complete!")