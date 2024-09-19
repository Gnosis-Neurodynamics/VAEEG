import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid


def get_signal_plot(input_y, output_y, sfreq=256, fig_size=(8, 5)):
    """
    :param input_y: (N,)
    :param output_y: (N,)
    :param sfreq:
    :param fig_size:
    :return:
    """
    if not (isinstance(input_y, np.ndarray) and input_y.ndim == 1 and input_y.shape == output_y.shape):
        raise RuntimeError("y is not supported.")

    fig = plt.figure(figsize=fig_size)
    ax = plt.subplot(111)

    xt = np.arange(0, input_y.shape[0]) / sfreq

    ax.plot(xt, input_y, label="input")
    ax.plot(xt, output_y, label="output")
    ax.legend(fontsize="large")
    ax.grid(axis="x", linestyle="-.", linewidth=1, which="both")
    ax.set_ylabel("amp (uV)", fontdict={"fontsize": 15})
    ax.set_xlabel("time (s)", fontdict={"fontsize": 15})
    ax.tick_params(labelsize=15)
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)
    return img


def get_signal_plots(input_y, output_y, sfreq, fig_size=(8, 5)):
    if not (isinstance(input_y, np.ndarray) and input_y.ndim == 2 and input_y.shape == output_y.shape):
        raise RuntimeError("y is not supported.")

    out = map(lambda a: get_signal_plot(a[0], a[1], sfreq, fig_size), zip(input_y, output_y))
    return np.stack(list(out), axis=0)


def batch_imgs(input_y, output_y, sfreq, num, n_row, fig_size=(8, 5)):
    z = get_signal_plots(input_y[0:num, :], output_y[0:num, :], sfreq, fig_size)
    img = make_grid(torch.tensor(np.transpose(z, (0, 3, 1, 2))), nrow=n_row, pad_value=0, padding=4)
    a = img.numpy()

    # HWC
    return np.ascontiguousarray(np.transpose(a, (1, 2, 0)))
