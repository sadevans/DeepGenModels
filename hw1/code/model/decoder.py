import torch.nn as nn


class Decoder(nn.Module):

    def __init__(self, height, width, channel, ksize, z_dim):
        super(Decoder, self).__init__()


        self.height, self.width, self.channel = height, width, channel
        self.ksize, self.z_dim = ksize, z_dim

        self.de_dense = nn.Sequential(
            nn.Linear(self.z_dim, 512),
            nn.ELU(),
            nn.Linear(512, (self.height // 4) * (self.width // 4) * 64),
            nn.ELU(),
        )

        self.de_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
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

            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=self.ksize+1,
                stride=2,
                padding=1,
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

            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=16,
                kernel_size=self.ksize+1,
                stride=2,
                padding=1,
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

            nn.Conv2d(
                in_channels=16,
                out_channels=self.channel,
                kernel_size=self.ksize,
                stride=1,
                padding=self.ksize//2,
            ),
            nn.Sigmoid(),
        )

    def forward(self, input):
        out = self.de_dense(input)
        out_res = out.view(
            out.size(0),
            64,
            (self.height//4),
            (self.height//4),
        )
        x_hat = self.de_conv(out_res)

        return x_hat