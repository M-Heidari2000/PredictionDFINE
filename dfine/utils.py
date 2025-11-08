import torch
from torch.distributions import kl_divergence
from torch.distributions import MultivariateNormal


def pearson_corr(
    true: torch.Tensor,
    pred: torch.Tensor
):

    # mean and std along time dimension
    true_mean = true.mean(dim=0, keepdim=True)  # (1, B, D)
    pred_mean = pred.mean(dim=0, keepdim=True)
    true_std = true.std(dim=0, unbiased=False, keepdim=True)
    pred_std = pred.std(dim=0, unbiased=False, keepdim=True)

    # covariance across time
    cov = ((true - true_mean) * (pred - pred_mean)).mean(dim=0)  # (B, D)

    corr = cov / (true_std.squeeze(0) * pred_std.squeeze(0) + 1e-8)  # (B, D)
    return corr.mean()

def compute_consistency(
    prior: MultivariateNormal,
    posterior: MultivariateNormal,
):
    prior_mean = prior.loc
    posterior_mean = posterior.loc
    mean_consistency = (
        2 * (prior_mean - posterior_mean).norm(dim=1, p=2) /
        (prior_mean.norm(dim=1, p=2) + posterior_mean.norm(dim=1, p=2)  + 1e-6)
    ).mean()
    kl_consistency = kl_divergence(posterior, prior).mean()

    return mean_consistency, kl_consistency