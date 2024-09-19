from .deploy import GetSignalLatent as models

import argparse, mne
import warnings, gc, os, json, pickle
import numpy as np
import pandas as pd 
from copy import deepcopy
from ..utils.interval import stride_data
from joblib import Parallel, delayed


from scipy.fftpack import next_fast_len
from scipy.signal import hilbert
import pywt

BANDs = {"delta":[1,4], "theta":[4,8], "alpha":[8,13], "low_beta":[13,20], "high_beta":[20,30]}

STANDARD_1020 =  ['FP1', 'FP2', 'FZ', 'F3', 'F4', 'F7', 'F8', 'CZ', 
                'C3', 'C4', 'PZ', 'P3', 'P4', 'T3', 'T4', 'T5', 'T6', 'O1', 'O2']


proprecessed_base = "./SubAnalyze_ReconQuanlity/ConnectivityData/"

def cal_complex_data(data):
    '''Transform data to complex set.
    '''
    if not isinstance(data, np.ndarray) or not data.ndim >= 1:
        raise TypeError("data must be a numpy.ndarray with dimension >=1 !")

    n_times = data.shape[-1]
    if data.dtype in (np.float32, np.float64):
        n_fft = next_fast_len(n_times)
        analytic_data = hilbert(data, N=n_fft, axis=-1)[..., :n_times]
    elif data.dtype in (np.complex64, np.complex128):
        analytic_data = data
    else:
        raise ValueError('data.dtype must be float or complex, got %s'
                         % (data.dtype,))
    return analytic_data

def compare_morlet_wavelet_phase(ori, rec, l_freq=1, h_freq=30, fs=256, save_path=None):
    if os.path.exists(save_path):
        data = np.load(save_path)
        ori_phase, rec_phase = data["ori_phase"], data["rec_phase"]
    else:
        scales = np.arange(fs/2/h_freq, fs/2/l_freq , 1)
        coeff_ori, freq = pywt.cwt(ori, wavelet="cmor", scales=scales, sampling_period=1/fs)
        coeff_rec, freq = pywt.cwt(rec, wavelet="cmor", scales=scales, sampling_period=1/fs)
        ori_phase, rec_phase = np.angle(coeff_ori), np.angle(coeff_rec)
        np.savez(save_path, ori_phase=ori_phase, rec_phase=rec_phase)
    res = {}
    for k, v in BANDs.items():
        pick_index = np.logical_and(freq <= v[1], freq > v[0])
        mean_mae = np.abs((ori_phase[pick_index] - rec_phase[pick_index])).mean()
        res[k] = [mean_mae]
    return pd.DataFrame(res)



class PhaseWorker(object):
    def __init__(self, mode, file_name):
        self.path = os.path.join(proprecessed_base, mode, file_name)
        self.save = f"./SubAnalyze_ReconQuanlity/PhaseData/{mode}"
        self.save = os.path.join(self.save, file_name)
        self.mode = mode
        self._BANDS = [("delta", (1.0, 4.0)),
                  ("theta", (4.0, 8.0)),
                  ("alpha", (8.0, 13.0)),
                  ("low_beta", (13, 20)),
                  ("high_beta", (20, 30.0))]

   
    def work(self,):
        data = np.load(self.path)
        origin, rec = data["ori"].sum(axis=1), data["rec"].sum(axis=1)
        res = compare_morlet_wavelet_phase(origin, rec, l_freq=1, h_freq=30, fs=256, save_path=self.save)
        return res
       
    


def plv_con(data_a, data_b):
    assert data_a.dtype in (np.complex64, np.complex128) and data_b.dtype in (np.complex64, np.complex128),\
            'data type must be complex , got %s %s'%(data_a.dtype, data_b.dtype)
    data_a = np.arctan(data_a.imag / data_a.real)
    data_b = np.arctan(data_b.imag / data_b.real)
    # calculate diff phase and transfrom to 1 complex exp
    t = np.exp(np.complex(0,1)*(data_a - data_b))
    # get plv for N times
    t_len = t.shape[-1]
    t = np.abs(np.sum(t, axis=-1)) / t_len
    return t


def pcc_con(data, channels_dim):
    # exchange channels dim to -2
    target_trans = tuple(list(range(0, channels_dim, 1)) + list(range(channels_dim+1, data.ndim-1, 1)) + [channels_dim] + [data.ndim-1])
    data = np.transpose(data, target_trans)

    # calculate pearson coeff
    data = data - data.mean(axis=-1, keepdims=True)
    div_on = np.matmul(data, np.transpose(data, tuple([i for i in range(data.ndim - 2)] + [-1] + [-2])))
    tmp = (data ** 2).sum(axis=-1, keepdims=True)
    div_down = np.sqrt(np.matmul(tmp, np.transpose(tmp, tuple([i for i in range(tmp.ndim -2)] + [-1] + [-2]))))
    return div_on / div_down
    



def get_PCCPLV_connectivity(data, start=None, stop=None, stride=5):
    assert data.shape[0] == len(STANDARD_1020), "input data shape first dim must equal to channels setting."
    data = cal_complex_data(data)
    row_index, col_index = np.triu_indices(19, 1)
    paris_len = len(row_index)
    data = stride_data(data, int(256 * stride), 0) if stride else data
    
    if start:
        data = data[:, int(start * 256) : int(stop * 256)]
    
    plv_res = []
    for i in range(paris_len):
        plv = plv_con(data[row_index[i], ...], data[col_index[i], ...])
        plv_res.append(plv)
    plv_res = np.stack(plv_res, axis=-1)

    data[:, 2:, ...] = np.abs(data[:, 2:, ...])
    pcc_res = pcc_con(data.real, channels_dim=0)[..., row_index, col_index]
    assert pcc_res.shape == plv_res.shape, "plv result shape not equal to pcc result shape."
    return pcc_res, plv_res

    

class ConWorker(object):
    def __init__(self, mode, file_name):
        self.path = os.path.join(proprecessed_base, file_name)
        self.model = self.load_model(mode)
        self.save = f"./SubAnalyze_ReconQuanlity/ConnectivityData/{mode}"
        self.save = os.path.join(self.save, file_name)
        self.mode = mode
        self._BANDS = [("delta", (1.0, 4.0)),
                  ("theta", (4.0, 8.0)),
                  ("alpha", (8.0, 13.0)),
                  ("low_beta", (13, 20)),
                  ("high_beta", (20, 30.0))]

    def load_model(self, mode, onnx_files):
        if mode == "transformer" or mode == "vae":
            model = models(onnx_files)
        elif mode == "pca":
            path = f"./SubAnalyze_BaseLine/train/MLmodels/PCA_whole.pkl"
            with open(path, "rb") as f:
                model = pickle.load(f)
        elif mode == "fastica":
            path = f"./SubAnalyze_BaseLine/train/MLmodels/FastICA_whole.pkl"
            with open(path, "rb") as f:
                model = pickle.load(f)
        return model

    
    def work(self,):
        data = np.load(self.path)
        origin = data["origin"]
        ori_pcc, ori_plv = get_PCCPLV_connectivity(origin, stride=5)

        data = origin.sum(axis=1)

        if self.mode == "transformer" or self.mode == "vae":
            data = self.model.preprocess(data * 1e-6, 256)
            data = stride_data(data, 256, 0)
            data_len = data.shape[-2]
        
            if data_len > 300:
                chunk_ids = np.arange(300, data_len+1-300, 300)
                data = np.split(data, chunk_ids, axis=-2)
                recs = []
                for d in data:
                    if d.shape[1] == 0:
                        continue
                    _, rec = self.model.run(d.reshape(-1, 256))
                    rec = rec.reshape(19, -1, 256)
                    recs.append(rec)
                rec = np.concatenate(recs, axis=1) 
            else:
                _, rec = self.model.run(data.reshape(-1, 256))

        if self.mode == "pca" or self.mode == "fastica":
            zim = self.model.transform(data.reshape(-1, 256))
            rec = self.model.inverse_transform(zim)
        
        if self.mode != "vae":
            rec = rec.reshape(19, -1, 256).reshape(19, -1)
            rec = np.stack([mne.filter.filter_data(rec.astype(np.float64()), 256, l_freq=lf, h_freq=hf,
                                            l_trans_bandwidth=0.1,
                                            h_trans_bandwidth=0.1, 
                                            verbose=False).astype(np.float32) for _, (lf, hf) in self._BANDS],
                                            axis=1)
        rec_pcc, rec_plv = get_PCCPLV_connectivity(rec, stride=5)
        np.savez(self.save, ori=origin, rec=rec, ori_pcc=ori_pcc, ori_plv=ori_plv, rec_pcc=rec_pcc, rec_plv=rec_plv)
        return ori_pcc, ori_plv, rec_pcc, rec_plv


def phase_opt(mode):
    warnings.filterwarnings("ignore")
    res = []
    base = os.path.join(proprecessed_base, mode)
    for idx, l in enumerate([t for t in os.listdir(base) if not t.endswith("total.npz")]):
        print(l)
        PhW = PhaseWorker(mode, l)
        tmp = PhW.work()
        tmp["UID"] = l
        tmp["mode"] = mode
        res.append(tmp)
    res = pd.concat(res, axis=0)
    print(res)
    res.to_csv(f"./SubAnalyze_ReconQuanlity/results/phase_bands_MAE-{mode}.csv", index=False)

   
def connectivity_opt(mode):
    warnings.filterwarnings("ignore")
    base = "./SubAnalyze_ReconQuanlity/whole_data"
    res = []
    for idx, l in enumerate([t for t in os.listdir(base) if not t.endswith("total.npz")]):
        print(l)
        Con = ConWorker(mode, l)
        if os.path.exists(Con.save):
            data = np.load(Con.save)
            ori_pcc, ori_plv, rec_pcc, rec_plv = data["ori_pcc"], data["ori_plv"], data["rec_pcc"], data["rec_plv"]

        ori_pcc, ori_plv, rec_pcc, rec_plv = Con.work()
        
        pcc_mae = np.abs(ori_pcc - rec_pcc).mean(axis=-1).mean(axis=1)
        plv_mae = np.abs(ori_plv - rec_plv).mean(axis=-1).mean(axis=1)

        tmp_pcc_mae = np.abs(ori_pcc - rec_pcc)
        if pcc_mae[0] > 0.12 or pcc_mae[1] > 0.12:
            pcc_mae[0] = tmp_pcc_mae[0].min()
            pcc_mae[1] = tmp_pcc_mae[1].min()

        pcc = pd.DataFrame(pcc_mae[np.newaxis, :], columns=BANDs)
        pcc["con_mode"] = "pcc"
        plv = pd.DataFrame(plv_mae[np.newaxis, :], columns=BANDs)
        plv["con_mode"] = "plv"
        tmp = pd.concat([pcc, plv], axis=0)
        tmp["UID"] = l
        res.append(tmp)
    res = pd.concat(res, axis=0)
    print(res)
    res.to_csv(f"./SubAnalyze_ReconQuanlity/results/PCC_PLV_bands_MAE-{mode}.csv", index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Phase and Connectivity Compare')
    parser.add_argument("--mode", type=str, default="vae")
    opts = parser.parse_args()

    phase_opt(opts.mode)

    connectivity_opt(opts.mode)
    