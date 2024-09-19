import torch
import torch.nn as nn
from torch.autograd import Variable

class Conv1dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilation=1, bias=True):
        super(Conv1dLayer, self).__init__()

        total_p = kernel_size + (kernel_size - 1) * (dilation - 1) - 1
        left_p = total_p // 2
        right_p = total_p - left_p

        self.conv = nn.Sequential(nn.ConstantPad1d((left_p, right_p), 0),
                                  nn.Conv1d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride, dilation=dilation,
                                            bias=bias))

    def forward(self, x):
        return self.conv(x)


class FConv1dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilation=1, bias=True):
        super(FConv1dLayer, self).__init__()

        p = (dilation * (kernel_size - 1)) // 2
        op = stride - 1

        self.fconv = nn.ConvTranspose1d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=p,
                                        output_padding=op,
                                        dilation=dilation, bias=bias)

    def forward(self, x):
        return self.fconv(x)


class LSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 dropout=0.0, bidirectional=False):
        super(LSTMLayer, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout, bidirectional=bidirectional)

    def forward(self, x):
        outputs, _ = self.lstm(x)
        return outputs


class GRULayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 dropout=0.0, bidirectional=False):
        super(GRULayer, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                          dropout=dropout, bidirectional=bidirectional)

    def forward(self, x):
        outputs, _ = self.gru(x)
        return outputs


class HeadLayer(nn.Module):
    """
    Multiple paths to process input data. Four paths with kernel size 5, 7, 9, 11, respectively.
    Each path has one convolution layer.
    """

    def __init__(self, in_channels, out_channels, negative_slope=0.2):
        super(HeadLayer, self).__init__()

        if out_channels % 4 != 0:
            raise ValueError("out_channels must be divisible by 4, but got: %d" % out_channels)

        unit = out_channels // 4

        self.conv1 = nn.Sequential(Conv1dLayer(in_channels=in_channels, out_channels=unit,
                                               kernel_size=11, stride=1, bias=False),
                                   nn.BatchNorm1d(unit),
                                   nn.LeakyReLU(negative_slope))

        self.conv2 = nn.Sequential(Conv1dLayer(in_channels=in_channels, out_channels=unit,
                                               kernel_size=9, stride=1, bias=False),
                                   nn.BatchNorm1d(unit),
                                   nn.LeakyReLU(negative_slope))

        self.conv3 = nn.Sequential(Conv1dLayer(in_channels=in_channels, out_channels=unit,
                                               kernel_size=7, stride=1, bias=False),
                                   nn.BatchNorm1d(unit),
                                   nn.LeakyReLU(negative_slope))

        self.conv4 = nn.Sequential(Conv1dLayer(in_channels=in_channels, out_channels=unit,
                                               kernel_size=5, stride=1, bias=False),
                                   nn.BatchNorm1d(unit),
                                   nn.LeakyReLU(negative_slope))

        self.conv5 = nn.Sequential(Conv1dLayer(in_channels=out_channels, out_channels=out_channels,
                                               kernel_size=3, stride=2, bias=False),
                                   nn.BatchNorm1d(out_channels),
                                   nn.LeakyReLU(negative_slope))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        out = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.conv5(out)
        return out


class ResBlockV1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, negative_slope=0.2):
        super(ResBlockV1, self).__init__()

        if stride == 1 and in_channels == out_channels:
            self.projection = None
        else:
            self.projection = nn.Sequential(Conv1dLayer(in_channels, out_channels, 1, stride, bias=False),
                                            nn.BatchNorm1d(out_channels))

        self.conv1 = nn.Sequential(Conv1dLayer(in_channels, out_channels, kernel_size, stride, bias=False),
                                   nn.BatchNorm1d(out_channels),
                                   nn.LeakyReLU(negative_slope))

        self.conv2 = nn.Sequential(Conv1dLayer(out_channels, out_channels, kernel_size, 1, bias=False),
                                   nn.BatchNorm1d(out_channels))

        self.act = nn.LeakyReLU(negative_slope)

    def forward(self, x):
        if self.projection:
            res = self.projection(x)
        else:
            res = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = out + res
        out = self.act(out)
        return out


def re_parameterize(mu, log_var):
    """
    Re-parameterize trick to sample from N(mu, var) from N(0,1).

    :param mu: (Tensor) Mean of the latent Gaussian [N, z_dims]
    :param log_var: (Tensor) Standard deviation of the latent Gaussian [N, z_dims]
    :return: (Tensor) [N, z_dims]
    """
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


def z_sample(z):
    zp = Variable(torch.randn_like(z))
    return zp
