import torch
import einops
import wandb
from typing import Optional
from tqdm import tqdm
from .utils import pearson_corr
from tqdm import tqdm
from argparse import Namespace
from torch.distributions import MultivariateNormal
from .models import (
    Encoder,
    Dynamics,
    Decoder,
    ZDecoder,
)


def test_k_step_prediction(
    args: Namespace,
    encoder: Encoder,
    decoder: Decoder,
    z_decoder: ZDecoder,
    dynamics_model: Dynamics,
    z: torch.Tensor,
    y: torch.Tensor,
    prediction_k: Optional[int]=None,
):
    if prediction_k is None:
        prediction_k = args.prediction_k

    with torch.no_grad():

        encoder.eval()
        decoder.eval()
        dynamics_model.eval()
        z_decoder.eval()

        device = next(encoder.parameters()).device
        y, z = y.to(device), z.to(device)

        L, B, y_dim = y.shape
        _, _, z_dim = z.shape

        a = encoder(einops.rearrange(y, "l b y -> (l b) y")).loc
        a = einops.rearrange(a, "(l b) a -> l b a", b=B)

        y_pred = torch.zeros(
            (L - prediction_k - 1, B, y_dim),
            device=y.device,
        )

        z_pred = torch.zeros(
            (L - prediction_k - 1, B, z_dim),
            device=y.device,
        )

        # initial belief over x0: N(0, I)
        dist = MultivariateNormal(
            loc=torch.zeros((B, args.x_dim), device=y.device),
            covariance_matrix=torch.eye(args.x_dim, device=y.device).repeat([B, 1, 1])
        )

        for t in tqdm(range(1, L - prediction_k)):
            dist = dynamics_model.dynamics_update(dist=dist)
            dist = dynamics_model.measurement_update(dist=dist, a=a[t])

            pred_dist = dist
            for _ in range(prediction_k):
                pred_dist = dynamics_model.dynamics_update(dist=pred_dist)
            pred_a = dynamics_model.get_a(pred_dist.loc)
            y_pred[t-1] = decoder(pred_a)
            z_pred[t-1] = z_decoder(pred_dist.loc)

        y_true = y[1+prediction_k:]
        z_true = z[1+prediction_k:]

        corr_y = pearson_corr(true=y_true, pred=y_pred)
        corr_z = pearson_corr(true=z_true, pred=z_pred)

        wandb.log(
            {
                "y correlation (averaged over channels)": corr_y.item(),
                "z correlation (averaged over channels)": corr_z.item(),
                "prediction k": prediction_k,
            },
        )

        return y_pred, z_pred