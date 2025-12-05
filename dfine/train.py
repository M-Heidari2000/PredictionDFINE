import torch
import wandb
import einops
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from argparse import Namespace
from torch.distributions import MultivariateNormal
from .memory import ReplayBuffer
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
        
        y, _, _ = train_buffer.sample(batch_size=args.batch_size, chunk_length=args.chunk_length)
        # convert to tensor, transform to device, reshape to time-first
        y = torch.as_tensor(y, device=device)
        y = einops.rearrange(y, "b l y -> l b y")
        q_a_samples = encoder(einops.rearrange(y, "l b y -> (l b) y")).rsample()
        q_a_samples = einops.rearrange(
            q_a_samples,
            "(l b) a -> l b a",
            b=args.batch_size
        )

        # Initial distribution N(0, I)
        q_x = [MultivariateNormal(
            loc=torch.zeros((args.batch_size, args.x_dim), dtype=torch.float32, device=device),
            covariance_matrix=torch.diag_embed(torch.ones((args.batch_size, args.x_dim), device=device, dtype=torch.float32)),
        ) for _ in range(args.chunk_length)] 

        # Kalman filtering
        for t in range(1, args.chunk_length):
            q_x[t] = dynamics_model.posterior_step(
                dist=q_x[t-1],
                a=q_a_samples[t],
            )
        
        loss1 = 0.0
        loss2 = 0.0
        loss3 = 0.0

        for t in range(args.overshoot_d+1, args.chunk_length):
            # first loss term
            y_recon = decoder(q_a_samples[t])
            loss1 += nn.MSELoss()(y_recon, y[t])

            # second loss term
            # q(x_{t-d}|a_{1:t-d}, u_{0:t-d-1})
            past_q_x = q_x[t-args.overshoot_d]

            # q(x_t|a_{1:t}, u_{0:t-1})
            current_q_x = q_x[t]

            loss2 += dynamics_model.compute_kl_loss(
                past_q_x=past_q_x,
                current_q_x=current_q_x,
                d=args.overshoot_d,
            ).clamp(min=args.kl_free_nats).mean()

            # third loss term
            # q_a
            current_q_a = encoder(y[t])
            
            loss3 += dynamics_model.compute_logratio_loss(
                current_q_x=current_q_x,
                current_q_a=current_q_a,
                current_q_a_sample=q_a_samples[t],
            ).clamp(min=args.a_free_nats).mean()

        loss1 /= (args.chunk_length - args.overshoot_d - 1)
        loss2 /= (args.chunk_length - args.overshoot_d - 1)
        loss3 /= (args.chunk_length - args.overshoot_d - 1)

        loss = loss1 + args.kl_beta * loss2 + args.a_beta * loss3
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(all_params, args.clip_grad_norm)
        optimizer.step()
        scheduler.step()

        wandb.log({
            "train/loss1": loss1.item(),
            "train/loss2": loss2.item(),
            "train/loss3": loss3.item(),
            "train/total loss": loss.item(),
            "global_step": update+1,
        })

        # test
        if update % args.test_interval == 0:
            encoder.eval()
            decoder.eval()
            dynamics_model.eval()

            with torch.no_grad():
                y, _, _= test_buffer.sample(batch_size=args.batch_size, chunk_length=args.chunk_length)
                # convert to tensor, transform to device, reshape to time-first
                y = torch.as_tensor(y, device=device)
                y = einops.rearrange(y, "b l y -> l b y")
                q_a_samples = encoder(einops.rearrange(y, "l b y -> (l b) y")).rsample()
                q_a_samples = einops.rearrange(
                    q_a_samples,
                    "(l b) a -> l b a",
                    b=args.batch_size
                )

                # Initial distribution N(0, I)
                q_x = [MultivariateNormal(
                    loc=torch.zeros((args.batch_size, args.x_dim), dtype=torch.float32, device=device),
                    covariance_matrix=torch.diag_embed(torch.ones((args.batch_size, args.x_dim), device=device, dtype=torch.float32)),
                ) for _ in range(args.chunk_length)]

                # Kalman filtering
                for t in range(1, args.chunk_length):
                    q_x[t] = dynamics_model.posterior_step(
                        dist=q_x[t-1],
                        a=q_a_samples[t],
                    )
                
                loss1 = 0.0
                loss2 = 0.0
                loss3 = 0.0

                for t in range(args.overshoot_d+1, args.chunk_length):
                    # first loss term
                    y_recon = decoder(q_a_samples[t])
                    loss1 += nn.MSELoss()(y_recon, y[t])

                    # second loss term
                    # q(x_{t-d}|a_{1:t-d}, u_{0:t-d-1})
                    past_q_x = q_x[t-args.overshoot_d]

                    # q(x_t|a_{1:t}, u_{0:t-1})
                    current_q_x = q_x[t]

                    loss2 += dynamics_model.compute_kl_loss(
                        past_q_x=past_q_x,
                        current_q_x=current_q_x,
                        d=args.overshoot_d,
                    ).clamp(min=args.kl_free_nats).mean()

                    # third loss term
                    # q_a
                    current_q_a = encoder(y[t])
                    
                    loss3 += dynamics_model.compute_logratio_loss(
                        current_q_x=current_q_x,
                        current_q_a=current_q_a,
                        current_q_a_sample=q_a_samples[t],
                    ).clamp(min=args.a_free_nats).mean()

                loss1 /= (args.chunk_length - args.overshoot_d - 1)
                loss2 /= (args.chunk_length - args.overshoot_d - 1)
                loss3 /= (args.chunk_length - args.overshoot_d - 1)

                loss = loss1 + args.kl_beta * loss2 + args.a_beta * loss3

                wandb.log({
                    "test/loss1": loss1.item(),
                    "test/loss2": loss2.item(),
                    "test/loss3": loss3.item(),
                    "test/total loss": loss.item(),
                    "global_step": update+1,
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