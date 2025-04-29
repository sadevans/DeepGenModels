import torch
import torch.nn as nn
import torch.optim as optim
from model.encoder import Encoder
from model.decoder import Decoder


class CVAE(nn.Module):
    def __init__(self, height, width, channel, ksize, z_dim, learning_rate=1e-3):
        super(CVAE, self).__init__()

        self.height, self.width, self.channel = height, width, channel
        self.ksize, self.z_dim, self.learning_rate = ksize, z_dim, learning_rate

        self.encoder = Encoder(
            height=self.height,
            width=self.width,
            channel=self.channel,
            ksize=self.ksize,
            z_dim=self.z_dim,
        )

        self.decoder = Decoder(
            height=self.height,
            width=self.width,
            channel=self.channel,
            ksize=self.ksize,
            z_dim=self.z_dim,
        )

        

        self.models = [self.encoder, self.decoder]

        # for idx_m, model in enumerate(self.models):
        #     if(self.device.type == 'cuda'):
        #         self.models[idx_m] = nn.DataParallel(self.models[idx_m], list(range(self.models[idx_m].ngpu)))


        self.num_params = 0
        for idx_m, model in enumerate(self.models):
            for p in model.parameters():
                self.num_params += p.numel()
            print(model)
        print("The number of parameters: %d" %(self.num_params))

        self.params = list(self.encoder.parameters()) + list(self.decoder.parameters())


    def forward(self, x):
        z_enc, z_mu, z_sigma = self.encoder(x)
        x_hat = self.decoder(z_enc)
        return x_hat, z_mu, z_sigma
