import torch


def recon_loss(x, x_bar):
    """
    reconstruct loss
    :param x:
    :param x_bar:
    :return:
    """
    value = torch.nn.functional.mse_loss(x, x_bar, reduction="mean")
    return value


def kl_loss(mu, log_var):
    """
    Compute KL loss

    𝐾𝐿(𝑁(𝜇,𝜎^2),𝑁(0,1)) = -0.5*(log𝜎^2 - 𝜎^2 - 𝜇^2 + 1), 𝜎 > 0

    using log_var
    t = log𝜎^2

    KL= -0.5 * ( t - e^t - u^2 + 1)

    :param mu:
    :param log_var:
    :return:
    """
    value = torch.mean(-0.5 * (1 + log_var - torch.exp(log_var) - mu ** 2))
    return value
