import torch 
import torch.nn as nn
import os 
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd


SAVE_FILE = "./SubAnalyze_SeizureDetection/Model/Models"


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


class Dataset_predata(Dataset):
    def __init__(self, mode, ds):
        self.data = np.load(f"./SubAnalyze_SeizureDetection/Model/dataset/{mode}_{ds}.npz", allow_pickle=True)
        self.data_2 = np.load(f"./SubAnalyze_SeizureDetection/Model/dataset/{mode}_eval.npz", allow_pickle=True)
        self.label = self.data["label"][:, np.newaxis]
        self.data = self.data["data"]
        self.label_2 = self.data_2["label"][:, np.newaxis]
        self.data_2 = self.data_2["data"]

        
        self.data = np.concatenate([self.data, self.data_2], axis=0)
        self.label = np.concatenate([self.label, self.label_2], axis=0)

        print(f"{mode}_{ds}", sum(self.label == 0) , sum(self.label == 1))

    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]




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
    


class HeadLayer(nn.Module):
    """
    Multiple paths to process input data. Four paths with kernel size 5, 7, 9, respectively.
    Each path has one convolution layer.
    """

    def __init__(self, in_channels, out_channels, negative_slope=0.2):
        super(HeadLayer, self).__init__()

        if out_channels % 4 != 0:
            raise ValueError("out_channels must be divisible by 4, but got: %d" % out_channels)

        unit = out_channels // 4

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

        self.conv4 = nn.Sequential(Conv1dLayer(in_channels=in_channels, out_channels=unit,
                                               kernel_size=3, stride=1, bias=False),
                                   nn.BatchNorm1d(unit),
                                   nn.LeakyReLU(negative_slope))
        

        self.conv5 = nn.Sequential(Conv1dLayer(in_channels=out_channels, out_channels=out_channels,
                                               kernel_size=3, stride=1, bias=False),
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


class SeizureDetect_Model(nn.Module):
    def __init__(self, target_channels=8):
        super(SeizureDetect_Model, self).__init__()
        self.flatten0 = nn.Flatten(start_dim=-2, end_dim=-1)
        self.headlayer = HeadLayer(19, target_channels)
        # reshape (8, 50) than transpose (10, 8, 50) than flatten 
        self.flatten1 = nn.Flatten(start_dim=-2, end_dim=-1)

        self.lstm0 = nn.LSTM(50 * target_channels, 256, num_layers=2, batch_first=True,  bidirectional=False, dropout=0.1)
        self.lstm1 = nn.LSTM(256, 16, num_layers=1, batch_first=True,  bidirectional=False)

        self.linear = nn.Sequential(nn.Flatten(start_dim=-2, end_dim=-1), nn.Linear(10 * 16, 32), nn.ReLU(), nn.Dropout(p=0.1),
                                    nn.Linear(32, 1), nn.Sigmoid())
        
    def forward(self, x):
        x = self.flatten0(x)
        x = self.headlayer(x)
        x = torch.split(x, 50, dim=-1)
        x = torch.stack(x, dim=1)
        x = self.flatten1(x)
        x, _ = self.lstm0(x)
        x, _ = self.lstm1(x)
        res = self.linear(x)
        return res    




def bce_Loss(input, target, weight=None, device='gpu'):
    if weight:
        weight_ = torch.ones(target.size()).to(device)
        weight_[target==0] = weight
        return nn.BCELoss(weight=weight_)(input, target)
    else:
        return nn.BCELoss()(input, target)



def save_model(model, file):
    torch.save(model.state_dict(), file)


def load_model(model_frame, file):
    model_frame.load_state_dict(torch.load(file))
    return model_frame


def get_metrics(predict, label, class_num=2):
    predict = predict.round()
    metric_cm = metrics.confusion_matrix(label, predict, labels=list(range(class_num)))
    metric = metric_cm / metric_cm.sum(axis=1, keepdims=True)
    acc0, acc1 = np.diagonal(metric)
    acc_total = metrics.accuracy_score(label, predict)
    return metric_cm, np.round(np.array([acc0, acc1, acc_total]), 3)



def train(mode, n_epoch=100, n_batch=5000, lr=0.01, device="cuda", weight=0.15):
    record_path = os.path.join(SAVE_FILE, f"{mode}_record.csv")
    if os.path.exists(record_path):
        record_pd = pd.read_csv(record_path)
        start_epoch = record_pd.epoch.values[-1]
        model = load_model(SeizureDetect_Model(), os.path.join(SAVE_FILE, mode, f"seizuremodel_{start_epoch}.pth")).to(device)
        record_pd = [record_pd]
        print(f"model exists at epoch {start_epoch}")

    else:
        model = SeizureDetect_Model().to(device)
        model.apply(initialize_weights)
        start_epoch = 0
        record_pd = []

    train_loader = Dataset_predata(mode, ds="train")
    eval_loader = Dataset_predata(mode, ds="dev")
    
    trainloader = DataLoader(train_loader, batch_size=n_batch, shuffle=True)
    evalloader = DataLoader(eval_loader, batch_size=n_batch, shuffle=True)

    

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for e in range(start_epoch, n_epoch, 1):
        model.train()
        train_lres, train_yres, loss_res = [], [], []
        
        for train, labels in trainloader:
            
            y = model(train.float().to(device))
            loss = bce_Loss(y, labels.float().to(device), weight=weight, device=device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_lres.append(labels)
            train_yres.append(y)
            loss_res.append(loss.item())

        train_lres = torch.cat(train_lres, dim=0)
        train_yres = torch.cat(train_yres, dim=0)

        _, accs = get_metrics(y.to("cpu").detach().numpy(), labels.to("cpu").detach().numpy(), class_num=2)
        loss_mean = torch.tensor(loss_res).mean()

        

        model.eval()
        valid_lres, valid_yres = [], []

        for valid, vlabels in evalloader:
            vy = model(valid.float().to(device))

            valid_yres.append(vy)
            valid_lres.append(vlabels)
        
        valid_lres = torch.cat(valid_lres, dim=0)
        valid_yres = torch.cat(valid_yres, dim=0)

        vcon, vaccs = get_metrics(valid_yres.to("cpu").detach().numpy(), valid_lres.to("cpu").detach().numpy(), class_num=2)

        record = pd.DataFrame({"epoch":[e], "loss":[loss_mean], "acc_0":[accs[0]], "acc_1":[accs[1]], "acc":[accs[2]],
                                "vacc_0":[vaccs[0]], "vacc_1":[vaccs[1]], "vacc":[vaccs[2]]})
        record_pd.append(record)

        print(f"epoch {e} train loss: {loss_mean}, train acc:{accs}, test acc {vaccs}")

        torch.save(model.state_dict(), os.path.join(SAVE_FILE, mode, f"seizuremodel_{e}.pth"))
        np.save(f"./SubAnalyze_SeizureDetection/Model/Test_metric/{mode}_{e}.npy", vcon)
        record_pd_ = pd.concat(record_pd, axis=0)
        record_pd_.to_csv(record_path, index=False)

    torch.save(model.state_dict(), os.path.join(SAVE_FILE, mode, f"seizuremodel_last.pth"))
    record_pd = pd.concat(record_pd, axis=0)


if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="vae")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--weight", type=float, default=0.09)
    opts = parser.parse_args()


    train(n_batch=5000, n_epoch=opts.epoch, mode=opts.mode, lr=opts.lr, weight=opts.weight)

