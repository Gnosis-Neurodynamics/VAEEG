# -*- coding: utf-8 -*-
import os

import mne
import numpy as np
import onnxruntime


class GetSignalLatent(object):
    def __init__(self, onnx_file, cuda_execution=None, provider_options=None):
        """
        Convert the input signal to a latent or reconstruct it from the latent.
        The input signal must meet the following constraints:
            (1) sample frequency 256.0 Hz
            (2) values in uV
            (3) shape [N, 1, 256]
            (4) frequency band [1.0, 30.0] Hz
            (5) np.float32
        Args:
            cuda_execution: none or boolean, to use cuda priority if available. (default None)
        """

        if isinstance(cuda_execution, (type(None), bool)):
            if cuda_execution is None:
                is_cuda = True
            elif cuda_execution is True:
                is_cuda = True
            else:
                is_cuda = False
        else:
            raise ValueError("cuda_execution need to be a boolean value or None.")

        _onnx_name = onnx_file

        if not os.path.isfile(_onnx_name):
            raise RuntimeError("model file `%s` is missing.Please check." % _onnx_name)

        ava_providers = onnxruntime.get_available_providers()
        if 'CPUExecutionProvider' not in ava_providers:
            raise RuntimeError("no cpu providers, unbelievable!")

        if ('CUDAExecutionProvider' in ava_providers) and is_cuda:
            print("Using Cuda.")
            this_providers = ['CUDAExecutionProvider']
            this_provider_options = provider_options

        else:
            print("Using CPU.")
            this_providers = ['CPUExecutionProvider']
            this_provider_options = provider_options

        self._sess = onnxruntime.InferenceSession(_onnx_name, providers=this_providers,
                                                  provider_options=this_provider_options)

    def run(self, x):
        """

        :param x: outputs from preprocess and clip into 1 second. shape (N, 5, 256)
        :return:
            z: latent vector, (N, 50).
                features for delta (1.0-4.0 Hz), theta (4.0-8.0 Hz),
                alpha (8.0-13.0 Hz), low_beta (13.0-20.0 Hz), high_beta (20.0-30.0 Hz)
                 are 1-8, 9-18, 19-30, 31-40, 41-50.
            reconstruction: reconstructed signal of band 1.0-30.0 Hz using z vector.
        """
        output_names = [item.name for item in self._sess.get_outputs()]
        input_names = [item.name for item in self._sess.get_inputs()]
        zim, rec = self._sess.run(output_names=output_names,
                                    input_feed={input_names[0]: x})

        return zim, rec

    @staticmethod
    def preprocess(inputs, current_sfreq):
        """
        Please using this option to preprocess your raw signal data for the model.

        Input a long raw signal (N, L), resample to 256.0 Hz, rescale to uV unit and
        filter into five bands in this option and then sum at axis 1:
            delta: 1.0-4.0 Hz
            theta: 4.0-8.0 Hz
            alpha: 8.0-13.0 Hz
            low_beta: 13.0-20.0 Hz
            high_beta: 20.0-30.0 Hz

        Output (N, L)


        Parameters
        ----------
        inputs: numpy.ndaaray, signal in V unit. Here, the shape of the array is [N, L], L is suggested be large
                to get rid out of edge effect.
        current_sfreq: sample frequency for input signal (Hz)

        Returns
        -------
        Signals for frequency bands 1~30 Hz"after processing.
            numpy.ndaaray, np.float32, in μV
        """
        target_sfreq = 256.0
        _l_trans_bandwidth = 0.1
        _h_trans_bandwidth = 0.1
        _scale = 1.0e6

        _BANDS = [("delta", (1.0, 4.0)),
                  ("theta", (4.0, 8.0)),
                  ("alpha", (8.0, 13.0)),
                  ("low_beta", (13, 20)),
                  ("high_beta", (20, 30.0))]

        inputs = np.array(inputs, np.float64)
        inputs = inputs * _scale

        if current_sfreq != target_sfreq:
            ratio = target_sfreq / current_sfreq
            inputs = mne.filter.resample(inputs, ratio, verbose=False)

        outputs = [mne.filter.filter_data(inputs, target_sfreq, l_freq=lf, h_freq=hf,
                                          l_trans_bandwidth=_l_trans_bandwidth,
                                          h_trans_bandwidth=_h_trans_bandwidth, 
                                          verbose=False).astype(np.float32) for _, (lf, hf) in _BANDS]

        return np.stack(outputs, axis=1).sum(axis=1)
    