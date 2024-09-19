# -*- coding: utf-8 -*-
import argparse
import itertools
import os
import sys
import time

import torch
import torch.utils.data
import yaml
from torch import optim
from torch.utils.tensorboard import SummaryWriter

_CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(_CURRENT_DIR)

from opts.dataset import ClipDataset
from opts.ckpt import save_model, init_model
from opts.viz import batch_imgs
from net.modelA import VAEEG
from net.losses import recon_loss, kl_loss

torch.autograd.set_detect_anomaly(True)


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
        self.model, self.aux, self.device = init_model(in_model,
                                                       n_gpus=n_gpus,
                                                       ckpt_file=ckpt_file)

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

    def train(self, input_loader, model_dir, n_epoch, lr, beta, n_print):
        summary_dir = os.path.join(model_dir, "save")
        if not os.path.isdir(summary_dir):
            os.makedirs(summary_dir)

        loss_file = os.path.join(model_dir, "train_loss.csv")
        writer = SummaryWriter(summary_dir)

        current_epoch = self.aux.get("current_epoch", 0)
        current_step = self.aux.get("current_step", 0)

       
        optimizer = optim.RMSprop(itertools.chain(self.model.encoder.parameters(),
                                                  self.model.decoder.parameters()),
                                  lr=lr)

        self.model.train()
        start_time = time.time()

        for ie in range(n_epoch):
            current_epoch = current_epoch + 1
            for idx, input_x in enumerate(input_loader, 0):
                current_step = current_step + 1
                input_x = input_x.cuda()

              
                mu, log_var, xbar = self.model(input_x)
              
                kld = kl_loss(mu, log_var)
                rec = recon_loss(input_x, xbar)
                loss = beta * kld + rec

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if current_step % n_print == 0:
                  
                    writer.add_scalar('loss', loss, current_step)
                    writer.add_scalar('kld_loss', kld, current_step)
                    writer.add_scalar('rec_loss', rec, current_step)

                    error = input_x - xbar
                    error = error.abs().mean()
                    writer.add_scalar('mae error', error, current_step)

                    pr = self.pearson_index(input_x, xbar)
                    writer.add_scalar('pearsonr', pr.mean(), current_step)

                    cycle_time = (time.time() - start_time) / n_print

                    values = (current_epoch, current_step, cycle_time,
                              loss.to(torch.device("cpu")).detach().numpy(),
                              # gen_loss.to(torch.device("cpu")).detach().numpy(),
                              kld.to(torch.device("cpu")).detach().numpy(),
                              rec.to(torch.device("cpu")).detach().numpy(),
                              error.to(torch.device("cpu")).detach().numpy(),
                              pr.mean().to(torch.device("cpu")).detach().numpy())

                    names = ["current_epoch", "current_step", "cycle_time",
                             "loss", "kld_loss", "rec_loss",
                             "error", "pr"]

                    print("[Epoch %d, Step %d]: (%.3f s / cycle])\n"
                          "  loss: %.3f; kld_loss: %.3f; rec: %.3f;\n"
                          "  mae error: %.3f; pr: %.3f.\n"
                          % values)

                    img = batch_imgs(input_x.to(torch.device("cpu")).detach().numpy()[:, 0, :],
                                     xbar.to(torch.device("cpu")).detach().numpy()[:, 0, :],
                                     256, 4, 2, fig_size=(8, 5))
                    writer.add_image("signal", img, current_step, dataformats="HWC")
                    start_time = time.time()

                    n_float = len(values) - 2
                    fmt_str = "%d,%d" + ",%.3f" * n_float
                    save_loss_per_line(loss_file, fmt_str % values, ",".join(names))

            out_ckpt_file = os.path.join(model_dir, "ckpt_epoch_%d.ckpt" % current_epoch)
            save_model(self.model, out_file=out_ckpt_file,
                       auxiliary=dict(current_step=current_step,
                                      current_epoch=current_epoch))
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training Model')
    parser.add_argument('--yaml_file', type=str, required=True,
                        help="configures, path of .yaml file")
    parser.add_argument('--z_dim', type=int, required=True,
                        help="z_dim")
    opts = parser.parse_args()

    with open(opts.yaml_file, 'r') as file:
        configs = yaml.safe_load(file)

    train_params = configs["Train"]
    model_params = configs["Model"]
    dataset_params = configs["DataSet"]
    model_params["z_dim"] = opts.z_dim

    if not os.path.isdir(train_params["model_dir"]):
        os.makedirs(train_params["model_dir"])

    # config model
    model = VAEEG(in_channels=model_params["in_channels"],
                     z_dim=model_params["z_dim"],
                     negative_slope=model_params["negative_slope"],
                     decoder_last_lstm=model_params["decoder_last_lstm"])

    # init estimator
    est = Estimator(model, train_params["n_gpus"], train_params["ckpt_file"])
    m_dir = os.path.join(train_params["model_dir"],
                         "%s_z%d" % (os.path.basename(os.path.splitext(opts.yaml_file)[0]),
                                     opts.z_dim)
                         )

    # load dataset
    train_ds = ClipDataset(data_dir=dataset_params["data_dir"],
                           band_name=model_params["name"],
                           clip_len=dataset_params["clip_len"])

    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True,
                                               batch_size=dataset_params["batch_size"],
                                               drop_last=True,
                                               num_workers=0)
    est.train(input_loader=train_loader,
              model_dir=m_dir,
              n_epoch=train_params["n_epoch"],
              lr=train_params["lr"],
              beta=train_params["beta"],
              n_print=train_params["n_print"]
              )
