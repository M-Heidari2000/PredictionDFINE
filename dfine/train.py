import torch
import wandb
import einops
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from argparse import Namespace
from torch.distributions import MultivariateNormal
from .memory import ReplayBuffer
from .utils import compute_consistency
from torch.nn.utils import clip_grad_norm_
from .models import (
    Encoder,
    Decoder,
    Dynamics,
    ZDecoder,
)


def train_backbone(
    args: Namespace,
    train_buffer: ReplayBuffer,
    test_buffer: ReplayBuffer,
):

    # define models and optimizer
    device = "cuda" if (torch.cuda.is_available() and not args.disable_gpu) else "cpu"

    encoder = Encoder(
        y_dim=train_buffer.y_dim,
        a_dim=args.a_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    decoder = Decoder(
        y_dim=train_buffer.y_dim,
        a_dim=args.a_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    dynamics_model = Dynamics(
        x_dim=args.x_dim,
        a_dim=args.a_dim,
    ).to(device)

    wandb.watch([encoder, dynamics_model, decoder], log="all", log_freq=10)

    all_params = (
        list(encoder.parameters()) +
        list(decoder.parameters()) + 
        list(dynamics_model.parameters())
    )

    optimizer = torch.optim.Adam(all_params, lr=args.lr, eps=args.eps, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=args.num_updates
    )

    # train and test loop
    print(f"training on {device} ...")
    for update in tqdm(range(args.num_updates)):
        
        # train
        encoder.train()
        decoder.train()
        dynamics_model.train()

        y, _, _ = train_buffer.sample(batch_size=args.batch_size, chunk_length=args.chunk_length)

        # convert to tensor, transform to device, reshape to time-first
        y = torch.as_tensor(y, device=device)
        y = einops.rearrange(y, "b l y -> l b y")
        a = encoder(einops.rearrange(y, "l b y -> (l b) y"))
        a = einops.rearrange(a, "(l b) a -> l b a", b=args.batch_size)

        # initial belief over x0: N(0, I)
        posterior_dist = MultivariateNormal(
            loc=torch.zeros((args.batch_size, args.x_dim), device=device),
            covariance_matrix=torch.eye(args.x_dim, device=device).expand(args.batch_size, -1, -1)
        )
        y_pred_loss = 0.0
        y_filter_loss = 0.0
        mean_consistency = 0.0
        kl_consistency = 0.0

        for t in range(1, args.chunk_length - args.prediction_k):
            prior_dist = dynamics_model.dynamics_update(dist=posterior_dist)
            posterior_dist = dynamics_model.measurement_update(dist=prior_dist, a=a[t])
            consistencies = compute_consistency(prior=prior_dist, posterior=posterior_dist)
            mean_consistency += consistencies[0]
            kl_consistency += consistencies[1]

            filter_a = dynamics_model.get_a(posterior_dist.loc)
            y_filter_loss += nn.MSELoss()(decoder(filter_a), y[t])

            # tensors to hold predictions of future ys
            pred_y = torch.zeros((args.prediction_k, args.batch_size, train_buffer.y_dim), device=device)

            pred_dist = posterior_dist

            for k in range(args.prediction_k):
                pred_dist = dynamics_model.dynamics_update(dist=pred_dist)
                pred_a = dynamics_model.get_a(pred_dist.loc)
                pred_y[k] = decoder(pred_a)

            true_y = y[t+1: t+1+args.prediction_k]
            true_y_flatten = einops.rearrange(true_y, "k b y -> (k b) y")
            pred_y_flatten = einops.rearrange(pred_y, "k b y -> (k b) y")
            y_pred_loss += nn.MSELoss()(pred_y_flatten, true_y_flatten)

        # y prediction loss
        y_pred_loss /= (args.chunk_length - args.prediction_k - 1)

        # y filter loss
        y_filter_loss /= (args.chunk_length - args.prediction_k - 1)

        # autoencoder loss
        a_flatten = einops.rearrange(a, "l b a -> (l b) a")
        y_flatten = einops.rearrange(y, "l b y -> (l b) y")
        y_recon = decoder(a_flatten)
        ae_loss = nn.MSELoss()(y_recon, y_flatten)

        # consistency loss
        mean_consistency /= (args.chunk_length - args.prediction_k - 1)
        kl_consistency /= (args.chunk_length - args.prediction_k - 1)

        total_loss = y_pred_loss + args.filtering_weight * y_filter_loss

        optimizer.zero_grad()
        total_loss.backward()

        # freeze decoder parameters with some probability
        if torch.rand(()) < args.decoder_freeze_prob:
            for p in decoder.parameters():
                p.grad = None

        clip_grad_norm_(all_params, args.clip_grad_norm)
        optimizer.step()
        scheduler.step()

        wandb.log({
            "train/y prediction loss": y_pred_loss.item(),
            "train/y filter loss": y_filter_loss.item(),
            "train/ae loss": ae_loss.item(),
            "train/total loss": total_loss.item(),
            "train/mean consistency": mean_consistency.item(),
            "train/kl consistency": kl_consistency.item(),
            "global_step": update,
        })
            
        if update % args.test_interval == 0:
            # test
            with torch.no_grad():
                encoder.eval()
                decoder.eval()
                dynamics_model.eval()

                y, _, _ = test_buffer.sample(batch_size=args.batch_size, chunk_length=args.chunk_length)

                # convert to tensor, transform to device, reshape to time-first
                y = torch.as_tensor(y, device=device)
                y = einops.rearrange(y, "b l y -> l b y")
                a = encoder(einops.rearrange(y, "l b y -> (l b) y"))
                a = einops.rearrange(a, "(l b) a -> l b a", b=args.batch_size)

                # initial belief over x0: N(0, I)
                posterior_dist = MultivariateNormal(
                    loc=torch.zeros((args.batch_size, args.x_dim), device=device),
                    covariance_matrix=torch.eye(args.x_dim, device=device).expand(args.batch_size, -1, -1)
                )
                y_pred_loss = 0.0
                y_filter_loss = 0.0
                mean_consistency = 0.0
                kl_consistency = 0.0

                for t in range(1, args.chunk_length - args.prediction_k):
                    prior_dist = dynamics_model.dynamics_update(dist=posterior_dist)
                    posterior_dist = dynamics_model.measurement_update(dist=prior_dist, a=a[t])
                    consistencies = compute_consistency(prior=prior_dist, posterior=posterior_dist)
                    mean_consistency += consistencies[0]
                    kl_consistency += consistencies[1]

                    filter_a = dynamics_model.get_a(posterior_dist.loc)
                    y_filter_loss += nn.MSELoss()(decoder(filter_a), y[t])

                    # tensors to hold predictions of future ys
                    pred_y = torch.zeros((args.prediction_k, args.batch_size, train_buffer.y_dim), device=device)

                    pred_dist = posterior_dist

                    for k in range(args.prediction_k):
                        pred_dist = dynamics_model.dynamics_update(dist=pred_dist)
                        pred_a = dynamics_model.get_a(pred_dist.loc)
                        pred_y[k] = decoder(pred_a)

                    true_y = y[t+1: t+1+args.prediction_k]
                    true_y_flatten = einops.rearrange(true_y, "k b y -> (k b) y")
                    pred_y_flatten = einops.rearrange(pred_y, "k b y -> (k b) y")
                    y_pred_loss += nn.MSELoss()(pred_y_flatten, true_y_flatten)

                # y prediction loss
                y_pred_loss /= (args.chunk_length - args.prediction_k - 1)

                # y filter loss
                y_filter_loss /= (args.chunk_length - args.prediction_k - 1)

                # autoencoder loss
                a_flatten = einops.rearrange(a, "l b a -> (l b) a")
                y_flatten = einops.rearrange(y, "l b y -> (l b) y")
                y_recon = decoder(a_flatten)
                ae_loss = nn.MSELoss()(y_recon, y_flatten)

                # consistency loss
                mean_consistency /= (args.chunk_length - args.prediction_k - 1)
                kl_consistency /= (args.chunk_length - args.prediction_k - 1)

                total_loss = y_pred_loss + args.filtering_weight * y_filter_loss

                wandb.log({
                    "test/y prediction loss": y_pred_loss.item(),
                    "test/y filter loss": y_filter_loss.item(),
                    "test/ae loss": ae_loss.item(),
                    "test/total loss": total_loss.item(),
                    "test/mean consistency": mean_consistency.item(),
                    "test/kl consistency": kl_consistency.item(),
                    "global_step": update,
                })
                

    save_dir = Path(args.log_dir) / args.run_id
    torch.save(encoder.state_dict(), save_dir / "encoder.pth")
    torch.save(decoder.state_dict(), save_dir / "decoder.pth")
    torch.save(dynamics_model.state_dict(), save_dir / "dynamics_model.pth")

    return encoder, decoder, dynamics_model


def train_z_decoder(
    args: Namespace,
    encoder: Encoder,
    dynamics_model: Dynamics,
    train_buffer: ReplayBuffer,
    test_buffer: ReplayBuffer,
):
    device = "cuda" if (torch.cuda.is_available() and not args.disable_gpu) else "cpu"

    z_decoder = ZDecoder(
        x_dim=args.x_dim,
        z_dim=train_buffer.z_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    # freeze backbone models
    for p in encoder.parameters():
        p.requires_grad = False

    for p in dynamics_model.parameters():
        p.requires_grad = False

    encoder.eval()
    dynamics_model.eval()

    wandb.watch([z_decoder], log="all", log_freq=10)

    all_params = list(z_decoder.parameters())
    optimizer = torch.optim.Adam(all_params, lr=args.lr, eps=args.eps, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=args.num_updates
    )

    # train and test loop
    print(f"training on {device} ...")
    for update in tqdm(range(args.num_updates)):    
        # train
        z_decoder.train()

        y, z, _ = train_buffer.sample(batch_size=args.batch_size, chunk_length=args.chunk_length)

        # convert to tensor, transform to device, reshape to time-first
        y = torch.as_tensor(y, device=device)
        y = einops.rearrange(y, "b l y -> l b y")
        a = encoder(einops.rearrange(y, "l b y -> (l b) y"))
        a = einops.rearrange(a, "(l b) a -> l b a", b=args.batch_size)
        z = torch.as_tensor(z, device=device)
        z = einops.rearrange(z, "b l z -> l b z")

        # initial belief over x0: N(0, I)
        posterior_dist = MultivariateNormal(
            loc=torch.zeros((args.batch_size, args.x_dim), device=device),
            covariance_matrix=torch.eye(args.x_dim, device=device).expand(args.batch_size, -1, -1)
        )
        z_filter_loss = 0.0

        for t in range(1, args.chunk_length):
            prior_dist = dynamics_model.dynamics_update(dist=posterior_dist)
            posterior_dist = dynamics_model.measurement_update(dist=prior_dist, a=a[t])
            z_filter_loss += nn.MSELoss()(z_decoder(posterior_dist.loc), z[t])

        z_filter_loss /= (args.chunk_length - 1)

        optimizer.zero_grad()
        z_filter_loss.backward()

        clip_grad_norm_(all_params, args.clip_grad_norm)
        optimizer.step()
        scheduler.step()

        wandb.log({
            "train/z filter loss": z_filter_loss.item(),
            "global_step": update,
        })
            
        if update % args.test_interval == 0:
            # test
            with torch.no_grad():
                z_decoder.eval()
                        
                y, z, _ = test_buffer.sample(batch_size=args.batch_size, chunk_length=args.chunk_length)

                # convert to tensor, transform to device, reshape to time-first
                y = torch.as_tensor(y, device=device)
                y = einops.rearrange(y, "b l y -> l b y")
                a = encoder(einops.rearrange(y, "l b y -> (l b) y"))
                a = einops.rearrange(a, "(l b) a -> l b a", b=args.batch_size)
                z = torch.as_tensor(z, device=device)
                z = einops.rearrange(z, "b l z -> l b z")

                # initial belief over x0: N(0, I)
                posterior_dist = MultivariateNormal(
                    loc=torch.zeros((args.batch_size, args.x_dim), device=device),
                    covariance_matrix=torch.eye(args.x_dim, device=device).expand(args.batch_size, -1, -1)
                )
                z_filter_loss = 0.0

                for t in range(1, args.chunk_length):
                    prior_dist = dynamics_model.dynamics_update(dist=posterior_dist)
                    posterior_dist = dynamics_model.measurement_update(dist=prior_dist, a=a[t])
                    z_filter_loss += nn.MSELoss()(z_decoder(posterior_dist.loc), z[t])

                z_filter_loss /= (args.chunk_length - 1)

                wandb.log({
                    "test/z filter loss": z_filter_loss.item(),
                    "global_step": update,
                })

    save_dir = Path(args.log_dir) / args.run_id
    torch.save(z_decoder.state_dict(), save_dir / "z_decoder.pth")

    return z_decoder