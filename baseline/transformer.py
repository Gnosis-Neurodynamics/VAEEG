import torch 
import torch.nn as nn 


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.ModuleList):
            for m_ in m:
                initialize_weights(m_)
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.1)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, 0, 0.1)
            if m.bias is not None:
                nn.init.zeros_(m.bias.data)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, (nn.LSTM, nn.LSTMCell)):
            for name, parma in m.named_parameters():
                if name.find('weight') != -1 :
                    nn.init.xavier_normal_(parma)
                elif name.find('bias_ih') != -1:
                    nn.init.constant_(parma, 2)
    return model




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
    


class HeadLayer(nn.Module):
    """
    Multiple paths to process input data. Four paths with kernel size 5, 7, 9, respectively.
    Each path has one convolution layer.
    """

    def __init__(self, in_channels, out_channels, negative_slope=0.2):
        super(HeadLayer, self).__init__()

        if out_channels % 3 != 0:
            raise ValueError("out_channels must be divisible by 3, but got: %d" % out_channels)

        unit = out_channels // 3

        self.conv1 = nn.Sequential(Conv1dLayer(in_channels=in_channels, out_channels=unit,
                                               kernel_size=9, stride=1, bias=False),
                                   nn.BatchNorm1d(unit),
                                   nn.LeakyReLU(negative_slope))

        self.conv2 = nn.Sequential(Conv1dLayer(in_channels=in_channels, out_channels=unit,
                                               kernel_size=7, stride=1, bias=False),
                                   nn.BatchNorm1d(unit),
                                   nn.LeakyReLU(negative_slope))

        self.conv3 = nn.Sequential(Conv1dLayer(in_channels=in_channels, out_channels=unit,
                                               kernel_size=5, stride=1, bias=False),
                                   nn.BatchNorm1d(unit),
                                   nn.LeakyReLU(negative_slope))

        self.conv4 = nn.Sequential(Conv1dLayer(in_channels=out_channels, out_channels=out_channels,
                                               kernel_size=3, stride=1, bias=False),
                                   nn.BatchNorm1d(out_channels),
                                   nn.LeakyReLU(negative_slope))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        out = self.conv4(out)
        return out
    


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout=0):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        device = x.device
        x = x + self.pe.requires_grad_(False)
        return self.dropout(x).to(device)



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


class Decoder(nn.Module):
    def __init__(self, z_dim, negative_slope=0.2):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(nn.Linear(z_dim, 256 * 4),
                                nn.BatchNorm1d(256 * 4),
                                nn.LeakyReLU(negative_slope))

        in_features = [16, 16]
        out_features = [16, 8]
        n_blocks = [2, 2]

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
       
        self.tail = nn.Sequential(Conv1dLayer(out_features[-1], out_features[-1] // 2, 5, 1, bias=True),
                                      nn.BatchNorm1d(out_features[-1] // 2),
                                      nn.LeakyReLU(negative_slope),
                                      Conv1dLayer(out_features[-1] // 2, 1, 3, 1, bias=True))


    def forward(self, x):
        """
        :param x: (N, z_dims)
        :return: (N, 1, L)
        """
        x = self.fc(x)
        n_batch, nf = x.shape
        x = x.view(n_batch, 16, 16 * 4)

        for m in self.layers:
            x = m(x)
        x = self.tail(x).flatten(1,2)
        return x


class EEGTransformer(nn.Module):
    def __init__(self, token=12, nhead=3, num_layers=2):
        super(EEGTransformer, self).__init__()
        self.embedding = HeadLayer(1, token)
        self.pos_embeding = PositionalEncoding(token, 256)
        self.transformerencoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(token, nhead=nhead, 
                                                                                   dim_feedforward=64, dropout=0.1, 
                                                                                   activation="relu"), num_layers=num_layers)
        self.linearencoder = nn.Linear(256 * token, 50)
        self.ResDecoder = Decoder(50)
        self.token = token

    def forward(self, src):
        #### encoder 
        src = src.unsqueeze(1)
        src = self.embedding(src)
        src = src.transpose(1,2)
        src = self.pos_embeding(src)
        src = self.transformerencoder(src.transpose(1, 0))
        src = torch.flatten(src.transpose(1, 0), start_dim=1, end_dim=2)
        
        zim = self.linearencoder(src)
        ##### decoder
        res = self.ResDecoder(zim)
        return zim, res

