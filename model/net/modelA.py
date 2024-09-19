import torch
import torch.nn as nn

from .layers import (HeadLayer, Conv1dLayer, FConv1dLayer,
                     ResBlockV1, LSTMLayer, re_parameterize, z_sample)


class Encoder(nn.Module):
    def __init__(self, in_channels, z_dim, negative_slope=0.2):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([HeadLayer(in_channels=in_channels,
                                               out_channels=16,
                                               negative_slope=negative_slope)])

        in_features = [16, 16, 24, 32]
        out_features = [16, 24, 32, 32]
        n_blocks = [2, 2, 2, 2]

        for in_chan, out_chan, n_block in zip(in_features, out_features, n_blocks):
            self.layers.append(nn.Sequential(Conv1dLayer(in_chan, out_chan, 3, 2, bias=False),
                                             nn.BatchNorm1d(out_chan),
                                             nn.LeakyReLU(negative_slope)))
            for _ in range(n_block):
                self.layers.append(ResBlockV1(out_chan, out_chan, 3, 1, negative_slope))

        self.layers.append(nn.Sequential(nn.Flatten(1),
                                         nn.Linear(256, 32),
                                         nn.BatchNorm1d(32),
                                         nn.LeakyReLU(negative_slope)))

        self.mu = nn.Linear(32, z_dim)
        self.log_var = nn.Linear(32, z_dim)

    def forward(self, x):
        # x: (N, 1, L)
        for m in self.layers:
            x = m(x)

        mu = self.mu(x)
        log_var = self.log_var(x)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, z_dim, negative_slope=0.2, last_lstm=True):
        super(Decoder, self).__init__()
        # (N, 256) to (N, 32, 8)
        self.fc = nn.Sequential(nn.Linear(z_dim, 256),
                                nn.BatchNorm1d(256),
                                nn.LeakyReLU(negative_slope))

        in_features = [32, 32, 24, 16, 16]
        out_features = [32, 24, 16, 16, 8]
        n_blocks = [2, 2, 2, 2, 2]

        self.layers = nn.ModuleList()

        for in_chan, out_chan, n_block in zip(in_features, out_features, n_blocks):
            self.layers.append(nn.Sequential(FConv1dLayer(in_chan, out_chan, 3, 2, bias=False),
                                             nn.BatchNorm1d(out_chan),
                                             nn.LeakyReLU(negative_slope)))
            for _ in range(n_block):
                self.layers.append(ResBlockV1(out_chan, out_chan, 3, 1, negative_slope))

        self.layers.append(nn.Sequential(Conv1dLayer(out_features[-1], out_features[-1], 3, 1, bias=False),
                                         nn.BatchNorm1d(out_features[-1]),
                                         nn.LeakyReLU(negative_slope)))
        if last_lstm:
            self.tail = LSTMLayer(out_features[-1], 1, 2)
        else:
            # self.tail = Conv1dLayer(out_features[-1], 1, 1, 1, bias=True)
            self.tail = nn.Sequential(Conv1dLayer(out_features[-1], out_features[-1] // 2, 5, 1, bias=True),
                                      nn.BatchNorm1d(out_features[-1] // 2),
                                      nn.LeakyReLU(negative_slope),
                                      Conv1dLayer(out_features[-1] // 2, 1, 3, 1, bias=True))

        self.last_lstm = last_lstm

    def forward(self, x):
        """

        :param x: (N, z_dims)
        :return: (N, 1, L)
        """
        x = self.fc(x)

        n_batch, nf = x.shape
        x = x.view(n_batch, 32, 8)

        for m in self.layers:
            x = m(x)

        if self.last_lstm:
            x = torch.permute(x, (2, 0, 1))
            x = self.tail(x)
            x = torch.permute(x, (1, 2, 0))
        else:
            x = self.tail(x)
        return x




class VAEEG(nn.Module):
    def __init__(self, in_channels, z_dim, negative_slope=0.2, decoder_last_lstm=True):
        super(VAEEG, self).__init__()
        self.encoder = Encoder(in_channels=in_channels, z_dim=z_dim, negative_slope=negative_slope)
        self.decoder = Decoder(z_dim=z_dim, negative_slope=negative_slope, last_lstm=decoder_last_lstm)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = re_parameterize(mu, log_var)
        xbar = self.decoder(z)

        return mu, log_var, xbar
