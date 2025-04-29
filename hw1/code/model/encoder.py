import torch
import torch.nn as nn


class FlattenLayer(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Encoder(nn.Module):

    def __init__(self, height, width, channel, ksize, z_dim):
        super(Encoder, self).__init__()

        self.height, self.width, self.channel = height, width, channel
        self.ksize, self.z_dim = ksize, z_dim

        self.en_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.channel,
                out_channels=16,
                kernel_size=self.ksize,
                stride=1,
                padding=self.ksize//2,
            ),
            nn.ELU(),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=self.ksize,
                stride=1,
                padding=self.ksize//2,
            ),
            nn.ELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=self.ksize,
                stride=1,
                padding=self.ksize//2,
            ),
            nn.ELU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=self.ksize,
                stride=1,
                padding=self.ksize//2,
            ),
            nn.ELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=self.ksize,
                stride=1,
                padding=self.ksize//2,
            ),
            nn.ELU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=self.ksize,
                stride=1,
                padding=self.ksize//2,
            ),
            nn.ELU(),
        )

        self.en_linear = nn.Sequential(
            FlattenLayer(),
            nn.Linear((self.height // 4) * (self.width // 4) * 64, 512),
            # nn.Linear(9216, 512),
            nn.ELU(),
            nn.Linear(512, self.z_dim*2),
        )

    def split_z(self, z):
        z_mu = z[:, :self.z_dim]
        z_sigma = z[:, self.z_dim:]

        return z_mu, z_sigma

    def sample_z(self, mu, sigma):

        epsilon = torch.randn_like(mu)
        sample = mu + (sigma * epsilon)

        return sample

    def forward(self, input):

        convout = self.en_conv(input)
        z_params = self.en_linear(convout)
        z_mu, z_sigma = self.split_z(z=z_params)
        z_enc = self.sample_z(mu=z_mu, sigma=z_sigma)

        return z_enc, z_mu, z_sigma