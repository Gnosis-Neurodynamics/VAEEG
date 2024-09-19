import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn 

from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DataParallel
import warnings, os
from copy import deepcopy

from sklearn import metrics

vision = "v1" 
chunks_dict = {'train':26, 'test':9, 'valid':2}
SAVE_FILE = f"./SubLearning_SleepStage/Model_train/Model_{vision.upper()}"


class LSTM_model(nn.Module):
    def __init__(self, dim=200):
        super(LSTM_model, self).__init__()
        # self.laynorm = nn.LayerNorm((30, 4, 50))
        self.flatten = nn.Flatten(start_dim=-2, end_dim=-1)
        self.lstm_0 = nn.LSTM(dim, 512, num_layers=2, batch_first=True,  bidirectional=False, dropout=0.2)
        self.layernorm_0 = nn.LayerNorm((30, 512))
        self.lstm_1 = nn.LSTM(512, 256, num_layers=2, batch_first=True, bidirectional=False, dropout=0.2)
        self.layernorm_1 = nn.LayerNorm((30, 256))
        self.lstm_2 = nn.LSTM(256, 128, num_layers=2, batch_first=True,  bidirectional=False, dropout=0.3)
        self.layernorm_2 = nn.LayerNorm((30, 128))
        self.lstm_3 = nn.LSTM(128, 32, num_layers=2, batch_first=True,  bidirectional=False, dropout=0.1)
        self.layernorm_3 = nn.LayerNorm((30, 32))
        self.linear = nn.Sequential(nn.Flatten(start_dim=-2, end_dim=-1), nn.Linear(32 * 30, 512), nn.ReLU(), nn.Dropout(p=0.3),
                                    nn.Linear(512, 64), nn.ReLU(), nn.Dropout(p=0.1),
                                    nn.Linear(64, 5))
    
    def forward(self, data):
        # data = self.laynorm(data)
        data = self.flatten(data)
        data, _ = self.lstm_0(data)
        data = self.layernorm_0(data)
        data, _ = self.lstm_1(data)
        data = self.layernorm_1(data)
        data, _ = self.lstm_2(data)
        data = self.layernorm_2(data)
        data, _ = self.lstm_3(data)
        data = self.layernorm_3(data)
        res = self.linear(data)
        return res


    

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




class SleepStage_Model(nn.Module):
    def __init__(self, target_channels=4):
        super(SleepStage_Model, self).__init__()
        self.flatten = nn.Flatten(start_dim=-2, end_dim=-1)
        self.headlayer = HeadLayer(4, target_channels)
        self.lstm = LSTM_model(200)

    def forward(self, x):
        x = self.flatten(x)
        x = self.headlayer(x)
        x = torch.split(x, 50, dim=-1)
        x = torch.stack(x, dim=1)
        x = self.lstm(x)

        return x
    
def get_metrics(predict, label, class_num):
    metric_cm = metrics.confusion_matrix(label, predict, labels=list(range(class_num)))
    metric = metric_cm / metric_cm.sum(axis=1, keepdims=True)
    acc0, acc1, acc2, acc3, acc4 = np.diagonal(metric)
    return metric_cm, np.round(np.array([acc0, acc1, acc2, acc3, acc4]), 3)



class DataSet_V1(Dataset):
    def __init__(self, picked='train', pick_chunk=0, mode="vae"):
        self.dataset = picked
        self.base = f".L/SubLearning_SleepStage/ChunkData/{mode}"
        self.data = np.load(f"{self.base}/{self.dataset}_{pick_chunk}.npz", allow_pickle=True)
        self.labels = self.data['label']
        self.data = self.data['data']

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        return data, label # , id[0]


class DataSet_V2(Dataset):
    def __init__(self, picked='train', mode="vae"):
        info = pd.read_csv("./SubLearning_SleepStage/ReCord/dataset.csv")
        self.files = info[info.DataSet == picked].FileID.values
        self.base = f"./SubLearning_SleepStage/NHC_recdata/{mode}_rec"

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = np.load(os.path.join(self.base, self.files[idx]+".npz"), allow_pickle=True)
        labels = data['labels']
        data = data['cliped_rec']
        return data, labels


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
            nn.init.constant_(m.weight, 0.1)
            nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, (nn.LSTM, nn.LSTMCell)):
            for name, parma in m.named_parameters():
                if name.find('weight') != -1 :
                    nn.init.xavier_normal_(parma)
                elif name.find('bias_ih') != -1:
                    nn.init.constant_(parma, 1)
    return model



def save_model(model, file):
    torch.save(model.state_dict(), file)


def load_model(model_frame, file):
    model_frame.load_state_dict(torch.load(file))
    return model_frame


def train(n_epoch=100, lr=0.01, n_batch=2048, class_weight=[1,1,1,1,1], device='cuda',  mode="vae"):
    record_path = f"./SubLearning_SleepStage/ReCord/Train_record/{mode}_train_record_{vision}.csv"
    save_file = os.path.join(SAVE_FILE, f"{mode}")
    if os.path.exists(record_path):
        Record = pd.read_csv(record_path)
        epoch_start = Record["epoch"].values[-1]
        Model = load_model(SleepStage_Model(), os.path.join(save_file, f"sleepstageLSTM_{epoch_start}.pth")).to(device)
        Record = [Record]
    else:
        Record = []
        Model = SleepStage_Model().to(device)
        epoch_start = -1
        if not os.path.exists(save_file):
            os.mkdir(save_file)
    
    
    # Model = DataParallel(Model, device_ids=[0,1,2])
    loss_fun = nn.CrossEntropyLoss(weight=torch.tensor(class_weight, device=device).float(),
                                           reduction="mean")
    
    optimizer = torch.optim.Adam(Model.parameters(), lr=lr)

    for e in range(epoch_start+1, n_epoch, 1):  
        Model.train().to(device)
        train_p, train_y, tloss_mean = [], [], []
        
        for b_id in range(chunks_dict['train']):
            print("load train dataset %d"%(b_id + 1))
            train_Dataset = DataSet_V1(picked='train', pick_chunk=b_id + 1, mode=mode)
            trainloader = DataLoader(train_Dataset, shuffle=True, batch_size=n_batch)
            print("finish load chunk data %d"%(b_id + 1))

            for idx, (data, labels) in enumerate(trainloader):
                data, labels = data.float().to(device), labels.float().to(device)
                y = Model(data)
                prediction = torch.argmax(y, dim=1)
                loss = loss_fun(y, labels.long())

                train_p.append(prediction.to("cpu"))
                train_y.append(labels.to("cpu"))
                tloss_mean.append(loss.item())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            prediction = torch.cat(train_p, dim=0)
            labels = torch.cat(train_y, dim=0)

            acc = (prediction == labels).to(torch.float32).mean()
            tloss_mean_ = torch.tensor(tloss_mean).mean()
            _, [acc0, acc1, acc2, acc3, acc4] = get_metrics(prediction, labels, class_num=5)
            print(f'epoch: {e} data_chunk: {b_id + 1} tloss: {tloss_mean_:.3f} W: {acc0:.3f}  N1: {acc1:.3f} N2: {acc2:.3f} N3: {acc3:.3f} R: {acc4:.3f} total: {acc:.3f}')

        prediction = torch.cat(train_p, dim=0)
        labels = torch.cat(train_y, dim=0)

        acc = (prediction == labels).to(torch.float32).mean()
        tloss_mean = torch.tensor(tloss_mean).mean()
        

        Model.eval()
        valid_p, valid_y, vloss_mean = [], [], []
        for b_id in range(chunks_dict['valid']):
            print("load valid dataset %d"%(b_id + 1))
            valid_Dataset = DataSet_V1(picked='valid', pick_chunk=b_id + 1, mode=mode)
            validloader = DataLoader(valid_Dataset, shuffle=True, batch_size=n_batch)
            print("finish load chunk data %d"%(b_id + 1))

            for idx, (valid_x, valid_l) in enumerate(validloader):
                valid_x, valid_l = valid_x.float().to(device), valid_l.float().to(device)
                valid_logit = Model(valid_x)
                valid_loss = loss_fun(valid_logit, valid_l.long())
                
                prediction = torch.argmax(valid_logit, dim=1)
                valid_p.append(prediction.to("cpu"))
                valid_y.append(valid_l.to("cpu"))
                vloss_mean.append(valid_loss.item())

        prediction = torch.cat(valid_p, dim=0)
        valid_y = torch.cat(valid_y, dim=0)

        valid_acc = (prediction == valid_y).to(torch.float32).mean()
        vloss_mean = torch.tensor(vloss_mean).mean()
        _, [vacc0, vacc1, vacc2, vacc3, vacc4] = get_metrics(prediction, valid_y, class_num=5)
        print(f'epoch: {e} tloss: {tloss_mean:.3f} tacc: {acc:.3f} vloss: {vloss_mean:.3f} vacc: {valid_acc:.3f}, W: {vacc0:.3f}  N1: {vacc1:.3f} N2: {vacc2:.3f} N3: {vacc3:.3f} R: {vacc4:.3f}')

        record = pd.DataFrame({'epoch': [e], 'tloss': [tloss_mean.item()], 'tacc': [acc.detach()], 'vloss': [vloss_mean.item()], 'vacc': [valid_acc.detach()], 
                               'tW':[acc0], 'tN1':[acc1], 'tN2':[acc2], 'tN3':[acc3], 'tR':[acc4],
                               'vW':[vacc0], 'vN1':[vacc1], 'vN2':[vacc2], 'vN3':[vacc3], 'vR':[vacc4]})

        
        save_model(Model, file=os.path.join(save_file, f"sleepstageLSTM_{e}.pth"))

        test_con, stop_tag, test_record = test(model=Model, device=device, epoch=e, mode=mode)
        
        record = pd.concat([record, test_record], axis=1)
        
        Record.append(record)
        Record_ = pd.concat(Record)
        Record_.to_csv(record_path, index=False)

        if stop_tag:
            np.save(f"./SubLearning_SleepStage/ReCord/Test_record/{mode}{e}best_Test_confusion_martix_{vision}.npy", test_con)
            break

    Record = pd.concat(Record)
    
    save_model(Model, file=os.path.join(save_file, f"sleepstageLSTM_last.pth"))

    return Record

            
def test(model=None, device="cpu", n_batch=2048,  model_frame=None, mode="vae"):
    if model is None:
        save_file = os.path.join(SAVE_FILE, f"{mode}")
        model = load_model(model_frame, os.path.join(save_file, f"sleepstageLSTM_last.pth")).to(device)
        model.eval()
    else:
        model.to(device)
        model.eval()

    test_p, test_y = [], []
    for id_b in range(chunks_dict['test']): 
        test_Dataset = DataSet_V1(picked='test', pick_chunk = id_b+1, mode=mode)
        testloader = DataLoader(test_Dataset, shuffle=False, batch_size=n_batch)
        for idx, (test, labels) in enumerate(testloader):
            test, labels = test.float().to(device), labels.float().to(device)
            test_l = model(test)
            test_p.append(torch.argmax(test_l, dim=1).to("cpu"))
            test_y.append(labels.to("cpu"))

    test_p = torch.cat(test_p, dim=0)
    test_y = torch.cat(test_y, dim=0)

    test_acc = (test_p == test_y).to(torch.float32).mean()
    con, [acc0, acc1, acc2, acc3, acc4] = get_metrics(test_p, test_y, 5)
    print(f"==== test :  W: {acc0:.3f}  N1: {acc1:.3f} N2: {acc2:.3f} N3: {acc3:.3f} R: {acc4:.3f} total: {test_acc:.3f}")
    res = pd.DataFrame({'W':[acc0], 'N1':[acc1], 'N2':[acc2], 'N3':[acc3], 'R':[acc4], "test total":[test_acc]})
    return con, test_acc >= 0.82, res



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--mode", type=str, default="vae")
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch", type=int, default=5000)
    
    opts = parser.parse_args()

    train_record = train(n_epoch=opts.epoch, n_batch=opts.batch, lr=opts.lr, class_weight=[1,1,1,1,1], mode=opts.mode)
    train_record.to_csv(f"./SubLearning_SleepStage/ReCord/Train_record/{opts.mode}/train_record_{vision}.csv", index=False)
    
    confusion_martix, _, _ = test(model_frame=SleepStage_Model(), device='cuda',  mode=opts.mode)
    np.save(f"./SubLearning_SleepStage/ReCord/Test_record/{opts.mode}_Test_confusion_martix.npy", confusion_martix)

