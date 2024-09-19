import torch 
import argparse
import os
import time
import torch
import torch.utils.data
import pandas as pd
from torch import optim
import numpy as np
# from torch.utils.tensorboard import SummaryWriter

from .baseline.ckpt import save_model, init_model
from .baseline.viz import batch_imgs

from .baseline.transformer import EEGTransformer
from .baseline.dataset import Dataset_Chunks

torch.autograd.set_detect_anomaly(False)


def save_loss_per_line(target_file, line, header):
    if os.path.isfile(target_file):
        # read first and check if empty
        with open(target_file, "r") as fi:
            dat = [line.strip() for line in fi.readlines() if line.strip() != ""]

        # new records
        if len(dat) == 0 or dat[0] != header:
            with open(target_file, "w") as fo:
                print(header, file=fo)
                print(line, file=fo)
        else:
            with open(target_file, "a") as fo:
                print(line, file=fo)
    else:
        with open(target_file, "w") as fo:
            print(header, file=fo)
            print(line, file=fo)


class Estimator(object):
    def __init__(self, in_model, n_gpus, ckpt_file=None):
        self.model, self.aux, self.device = init_model(in_model, n_gpus, ckpt_file)
    
    @staticmethod
    def pearson_index(x, y, dim=-1):
        xy = x * y
        xx = x * x
        yy = y * y

        mx = x.mean(dim)
        my = y.mean(dim)
        mxy = xy.mean(dim)
        mxx = xx.mean(dim)
        myy = yy.mean(dim)

        r = (mxy - mx * my) / torch.sqrt((mxx - mx ** 2) * (myy - my ** 2))
        return r

    def train(self, input_loader, model_dir, im_file, n_epoch=100, lr=1e-3, n_print=1000):

        loss_file = os.path.join(model_dir, "train_loss.csv")

        current_epoch = self.aux.get("current_epoch", 0)
        current_step = self.aux.get("current_step", 0)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
       
        self.model.train()
        start_time = time.time()

        for ie in range(n_epoch):
            current_epoch = current_epoch + 1
            for idx, input_x in enumerate(input_loader, 0):
                current_step = current_step + 1
                input_x = input_x[0].cuda()

                zims, xbar = self.model(input_x)
                
                loss = torch.nn.functional.mse_loss(input_x, xbar, reduction="mean")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if current_step % n_print == 0:
                    pr = self.pearson_index(input_x, xbar)
                    error = input_x - xbar
                    error = error.abs().mean()
                  
                    cycle_time = (time.time() - start_time) / n_print

                    values = (current_epoch, current_step, cycle_time,
                              loss.to(torch.device("cpu")).detach().numpy(),
                              pr.mean().to(torch.device("cpu")).detach().numpy(),
                              error.to(torch.device("cpu")).detach().numpy(),
                              )

                    names = ["current_epoch", "current_step", "cycle_time",
                             "loss", "pr", "error"]

                    print("[Epoch %d, Step %d]: (%.3f s / cycle])\n"
                          "  loss: %.3f;"
                          "  pr: %.3f; mae: %.3f.\n"
                          % values)

                   
                    start_time = time.time()

                    n_float = len(values) - 2
                    fmt_str = "%d,%d" + ",%.3f" * n_float
                    save_loss_per_line(loss_file, fmt_str % values, ",".join(names))

            out_ckpt_file = os.path.join(model_dir, "ckpt_epoch_%d.ckpt" % current_epoch)
            save_model(self.model, out_file=out_ckpt_file,
                       auxiliary=dict(current_step=current_step,
                                      current_epoch=current_epoch))
            batch_imgs(input_x.to(torch.device("cpu")).detach().numpy(),
                                     xbar.to(torch.device("cpu")).detach().numpy(),
                                     256, 4, 2, fig_size=(8, 5), im_file=os.path.join(im_file, "im_epoch_%d" % current_epoch))
        # writer.close()


if __name__ == "__main__":
    m_dir = "./SubAnalyze_BaseLine/train/models"
    im_dir = "./SubAnalyze_BaseLine/train/vizs"
    parser = argparse.ArgumentParser(description='Training Model')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--token", type=int, default=12)
    parser.add_argument("--nhead", type=int, default=3)
    parser.add_argument("--layers", type=int, default=2)
    opts = parser.parse_args()
    
    m_dir = os.path.join(m_dir, f"dmodel-{opts.token}_block-{opts.layers}_head-{opts.nhead}")
    if not os.path.exists(m_dir):
        os.makedirs(m_dir)

    im_dir = os.path.join(im_dir, f"dmodel-{opts.token}_block-{opts.layers}_head-{opts.nhead}")
    if not os.path.exists(im_dir):
        os.makedirs(im_dir)

    # config model
    model = EEGTransformer(token=opts.token, nhead=opts.nhead, num_layers=opts.layers)

    # init estimator
    est = Estimator(model, 1, None)
   
    # load dataset
    train_ds = Dataset_Chunks(1024)

    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True,
                                               batch_size=1,
                                               num_workers=0)
    est.train(input_loader=train_loader,
              model_dir=m_dir,
              im_file=im_dir, n_print=50, lr=opts.lr)

        
