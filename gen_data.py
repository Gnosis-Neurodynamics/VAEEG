# -*- coding: utf-8 -*-
import gc
import os
import random
import time

import lighten.utils.interval as lui
import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from .utils import load_raw
from tqdm import tqdm


class DataGen(object):
    _SFREQ = 256.0

    max_clips = 500
    _l_trans_bandwidth = 0.1
    _h_trans_bandwidth = 0.1

    _BANDS = {"whole": (1.0, 30.0),
              "delta": (1.0, 4.0),
              "theta": (4.0, 8.0),
              "alpha": (8.0, 13.0),
              "low_beta": (13, 20),
              "high_beta": (20, 30.0)}

    def __init__(self, in_file, out_dir, out_prefix, ch_mapper=None):
        self.in_file = in_file
        self.ch_mapper = ch_mapper
        self.out_dir = out_dir
        self.out_prefix = out_prefix
        self.data, self.ch_names = self.get_filter_data(verbose=False)

    @staticmethod
    def _check_channel_names(raw_obj, ch_mapper, verbose):
        rc_1 = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1',
                'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ',
                'CZ', 'PZ', 'A1', 'A2']
        rc_2 = ['FP1', 'FP2', 'F3-0', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1',
                'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ',
                'CZ', 'PZ', 'A1', 'A2']

        # rename channel
        if ch_mapper is None:
            mapper = {cname: cname.upper() for cname in raw_obj.ch_names}
        elif isinstance(ch_mapper, dict):
            mapper = ch_mapper
        else:
            raise RuntimeError("ch_mapper must be None or dict.")

        raw_obj.rename_channels(mapper, verbose=verbose)

        ch_names = set(raw_obj.ch_names)

        if set(rc_1).issubset(ch_names):
            raw_obj.pick_channels(rc_1, ordered=True)
        elif set(rc_2).issubset(ch_names):
            raw_obj.pick_channels(rc_2, ordered=True)
            raw_obj.rename_channels(mapping={"F3-0": "F3"}, verbose=verbose)
        else:
            raise RuntimeError("Channel Error")

    def get_filter_data(self, verbose=None):
        raw = load_raw(self.in_file, verbose=verbose)
        self._check_channel_names(raw_obj=raw,
                                  ch_mapper=self.ch_mapper,
                                  verbose=verbose)

        raw.load_data(verbose=verbose)
        raw.resample(self._SFREQ, verbose=verbose)

        raw.set_eeg_reference(ref_channels="average", verbose=verbose)

        data0 = raw.get_data()
        filter_results = {}

        for key, (lf, hf) in self._BANDS.items():
            filter_results[key] = mne.filter.filter_data(data0, self._SFREQ, l_freq=lf, h_freq=hf,
                                                         l_trans_bandwidth=self._l_trans_bandwidth,
                                                         h_trans_bandwidth=self._h_trans_bandwidth,
                                                         verbose=verbose).astype(np.float32)
        ch_names = raw.ch_names

        del raw, data0
        gc.collect()
        return filter_results, ch_names

    @staticmethod
    def _filter_intervalset(input_intervalset, threshold):
        out = []
        for item in input_intervalset:
            if item.upper_bound - item.lower_bound > threshold:
                out.append(item)
        return out

    def get_regions(self, seg_len=5.0, amp_threshold=400, merge_len=1.0, drop=60.0):
        """
        :param seg_len: unit s, keep > seg_len
        :param amp_threshold: unit uV, keep < amp_threshold
        :param merge_len: unit s
        :param drop: unit s, drop regions at the first and in the end.
        :return:
        """
        data = self.data["whole"]
        m_threshold = int(merge_len * self._SFREQ)
        seg_threshold = int(seg_len * self._SFREQ)
        start = int(drop * self._SFREQ)
        end = data.shape[1] - int(drop * self._SFREQ)

        whole_r = lui.IntervalSet([lui.Interval(lower_bound=start, upper_bound=end)])

        flag = np.abs(data) * 1e6 > amp_threshold
        art = lui.find_continuous_area_2d(flag)
        c_art = [lui.merge_continuous_area(s, threshold=m_threshold) for s in art]
        c_clean = [whole_r.difference(s) for s in c_art]

        keep_clean = [self._filter_intervalset(s, seg_threshold) for s in c_clean]

        out = []
        for idx, item_set in enumerate(keep_clean, 0):
            for item in item_set:
                tmp = np.arange(item.lower_bound, item.upper_bound + 1, seg_threshold)
                if len(tmp) > 1:
                    for idj in range(len(tmp) - 1):
                        out.append([idx, self.ch_names[idx], tmp[idj], tmp[idj + 1]])

        df_clean = pd.DataFrame(out, columns=["idx", "ch_names", "clip_start", "clip_stop"])
      

        udata = np.stack([self.data["whole"],
                          self.data["delta"],
                          self.data["theta"],
                          self.data["alpha"],
                          self.data["low_beta"],
                          self.data["high_beta"]], axis=0) * 1.0e6

        ts = list(zip(df_clean.idx, df_clean.clip_start, df_clean.clip_stop))

        # 只保留至少200段信号
        if len(ts) >= 200:
            if len(ts) > self.max_clips:
                ts = random.sample(ts, self.max_clips)

            outputs = []
            for p in ts:
                idx, start, stop = p
                tmp = udata[:, idx, start:stop]
                outputs.append(tmp)

            outputs = np.stack(outputs, axis=0)
            out_file = os.path.join(self.out_dir, "%s.npy" % self.out_prefix)
            np.save(out_file, outputs)


def worker(in_file, out_dir, out_prefix):
    try:
        dg = DataGen(in_file, out_dir, out_prefix)
        dg.get_regions()
    except:
        flag = False
        time.sleep(3.0)
    else:
        flag = True
    return in_file, flag


if __name__ == "__main__":
    in_file = "selection_data.csv"
    out_dir = "./new_data/clips"
    log_file = "./new_data/log.csv"

    n_jobs = 8

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    dt = pd.read_csv(in_file)
    tasks = [(os.path.join(pr, fid) + ".EEG", uid)
             for uid, pr, fid in zip(dt.UID, dt.DataRoot, dt.FileID)]

    res = Parallel(n_jobs=n_jobs)(delayed(worker)(
        in_file, out_dir, out_fid) for in_file, out_fid in tqdm(tasks))

    dt_log = pd.DataFrame(res, columns=["path", "status"])
    dt_log.to_csv(log_file, index=False, encoding="utf-8-sig")
