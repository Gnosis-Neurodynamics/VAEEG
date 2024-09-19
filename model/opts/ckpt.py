# -*- coding: utf-8 -*-
import os
import warnings

import torch
import torch.nn as nn
from ..opts import check_type, type_assert


@type_assert(model=nn.Module, ckpt_file=str)
def load_model(model, ckpt_file):
    """

    Args:
        model: nn.Module
            your net
        ckpt_file: str
            path of checkpoint file

    Returns
        dict for auxiliary information

    """
    state_dict = torch.load(ckpt_file, map_location=torch.device('cpu'))
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state_dict["model"])
    else:
        model.load_state_dict(state_dict["model"])

    return state_dict["auxiliary"]


@type_assert(model=nn.Module)
def save_model(model, out_file, auxiliary=None):
    """

    Args:
        model: nn.Module
            your net
        out_file: str
            path of checkpoint file
        auxiliary: dict or None
            other information to record

    Returns
        None
    """
    check_type("auxiliary", auxiliary, [dict, type(None)])
    data = dict()

    if auxiliary is None:
        data["auxiliary"] = {}
    else:
        data["auxiliary"] = auxiliary

    if isinstance(model, nn.DataParallel):
        data["model"] = model.module.state_dict()
    else:
        data["model"] = model.state_dict()

    torch.save(data, out_file)


@type_assert(m=nn.Module, n_gpus=int)
def init_model(m, n_gpus, ckpt_file=None):
    check_type("ckpt_file", ckpt_file, [str, type(None)])

    n_gpu = min(max(n_gpus, 0), torch.cuda.device_count())
    device = "cuda" if n_gpu > 0 else "cpu"

    if isinstance(ckpt_file, str):
        if os.path.isfile(ckpt_file):
            print("Initial model from %s" % ckpt_file)
            aux = load_model(m, ckpt_file)
        else:
            warnings.warn("The given checkpoint file is not found: %s. "
                          "Initial model randomly instead." % ckpt_file)
            aux = {}
    else:
        print("Initial model randomly")
        aux = {}

    msg = "Assign model to %s device." % device

    if n_gpu > 1:
        msg += "Using %d gpus." % n_gpus
    elif n_gpu == 1:
        msg += "Using 1 gpu."

    print(msg)
    m.to(device)

    if n_gpu > 1:
        om = nn.DataParallel(m, list(range(n_gpu)))
    else:
        om = m
    return om, aux, device
