import torch
import torch.nn as nn
from typing import Optional
from torch.distributions import MultivariateNormal
import torch.nn.init as init


class Encoder(nn.Module):

    """
        q(a_t|y_t)
    """

    def __init__(
        self,
        y_dim: int,
        a_dim: int,
        hidden_dim: int,
        min_var: Optional[float] = 1e-3,
    ):
        super().__init__()

        self.min_var = min_var
        self.a_dim = a_dim
        self.y_dim = y_dim

        self.mlp_layers = nn.Sequential(
            nn.Linear(y_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean_head = nn.Linear(hidden_dim, a_dim)
        self.cov_head = nn.Linear(hidden_dim, a_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, y):
        hidden = self.mlp_layers(y)
        mean = self.mean_head(hidden)
        cov = nn.functional.softplus(self.cov_head(hidden)) + self.min_var
        cov = torch.diag_embed(cov)
        return MultivariateNormal(loc=mean, covariance_matrix=cov)
    

class Decoder(nn.Module):
    """
        p(y_t|a_t)
    """

    def __init__(
        self,
        a_dim: int,
        y_dim: int,
        hidden_dim: Optional[int],
        min_var: Optional[float] = 1e-3,
    ):
        
        super().__init__()
        self.min_var = min_var

        self.mlp_layers = nn.Sequential(
            nn.Linear(a_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, y_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, a):
        return self.mlp_layers(a)


class ZDecoder(nn.Module):
    """
        x_t -> z_t
    """

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        hidden_dim: Optional[int]=None,
    ):
        super().__init__()

        hidden_dim = hidden_dim if hidden_dim is not None else 2*x_dim

        self.mlp_layers = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x):
        return self.mlp_layers(x)


class Dynamics(nn.Module):
    
    """
        dynamics including prior rollouts and posterior inference
    """

    def __init__(
        self,
        x_dim: int,
        a_dim: int,
        min_var: float=1e-4,
    ):
        super().__init__()

        self.x_dim = x_dim
        self.a_dim = a_dim
        self._min_var = min_var

        # Dynamics matrices
        self.A = nn.Parameter(torch.eye(self.x_dim))
        self.C = nn.Parameter(torch.randn(self.a_dim, self.x_dim))

        # Transition noise covariance (diagonal)
        self.nx = nn.Parameter(torch.randn(self.x_dim))
        # Observation noise covariance (diagonal)
        self.na = nn.Parameter(torch.randn(self.a_dim))

    @property
    def Na(self):
        Na = torch.diag(nn.functional.softplus(self.na) + self._min_var)    # shape: a a
        return Na
    
    @property
    def Nx(self):
        Nx = torch.diag(nn.functional.softplus(self.nx) + self._min_var)    # shape: x x
        return Nx

    def make_psd(self, P, eps=1e-6):
        b = P.shape[0]
        P = 0.5 * (P + P.transpose(-1, -2))
        P = P + eps * torch.eye(P.size(-1), device=P.device).expand([b, -1, -1])
        return P
    
    def dynamics_update(
        self,
        dist: MultivariateNormal,
    ):
        """
            infer q(x_t|a_{1:t-1})
            inputs:
                dist:
                    q(x_{t-1}|a_{1:t-1})
                    - mean: b x
                    - cov: b x x
        """

        mean = dist.loc
        cov = dist.covariance_matrix

        next_mean = mean @ self.A.T
        next_cov = self.A @ cov @ self.A.T + self.Nx
        next_cov = self.make_psd(next_cov)

        return MultivariateNormal(loc=next_mean, covariance_matrix=next_cov)
    
    def measurement_update(
        self,
        dist: MultivariateNormal,
        a: torch.Tensor,
    ):
        """
            infer q(x_t|a_{1:t})
            dist:
                - q(x_t|a_{1:t-1})
                - mean: b x
                - cov: b x x
            a: 
                - a_t
                - b a
        """

        mean = dist.loc
        cov = dist.covariance_matrix

        K = cov @ self.C.T @ torch.linalg.pinv(self.C @ cov @ self.C.T + self.Na)
        next_mean = mean + ((a - mean @ self.C.T).unsqueeze(1) @ K.transpose(1, 2)).squeeze(1)
        next_cov = (torch.eye(self.x_dim, device=mean.device) - K @ self.C) @ cov
        next_cov = self.make_psd(next_cov)

        return MultivariateNormal(loc=next_mean, covariance_matrix=next_cov)
    
    def posterior_step(
        self,
        dist: MultivariateNormal,
        a,
    ):
        """
            infer q(x_t|a_{1:t})
            inputs:
                dist: q(x_{t-1}|a_{1:t-1})
                a: a_t
        """

        dist = self.dynamics_update(dist=dist)
        dist = self.measurement_update(dist=dist, a=a)

        return dist
    
    def compute_kl_loss(
        self,
        past_q_x: MultivariateNormal,
        current_q_x: MultivariateNormal,
        d: int,
    ):
        """
            compute the kl loss in ELBO
            inputs:
                past_q_x: q(x_{t-d}|a_{1:t-d})
                current_q_x = q(x_t|a_{1:t})
                d: overshooting distance
        """
        
        mu_p = past_q_x.loc
        sigma_p = past_q_x.covariance_matrix
        mu_x = current_q_x.loc
        sigma_x = current_q_x.covariance_matrix

        mu_d_bar = mu_p
        sigma_d = torch.zeros_like(past_q_x.covariance_matrix)

        for t in range(d):
            mu_d_bar = mu_d_bar @ self.A.T
            sigma_d = self.A @ sigma_d @ self.A.T + self.Nx

        sigma_d = self.make_psd(sigma_d)
        sigma_d_inv = torch.linalg.pinv(sigma_d)

        # logdet term
        logdet = sigma_d.logdet() - sigma_x.logdet()

        # trace 1 term
        trc1 = torch.einsum("bij,bji->b", sigma_d_inv, sigma_x)

        # trace 2 term
        A_d = self.A.matrix_power(d)
        trc2 = torch.einsum(
            "bij,bji->b",
            sigma_d_inv,
            A_d @ sigma_p @ A_d.T
        )

        # quad term
        delta = mu_d_bar - mu_x
        quad = torch.einsum(
            "bi,bij,bj->b",
            delta,
            sigma_d_inv,
            delta
        )

        return 0.5 * (logdet + trc1 + trc2 + quad)


    def compute_logratio_loss(
        self,
        current_q_x: MultivariateNormal,
        current_q_a: MultivariateNormal,
        current_q_a_sample: torch.Tensor,
    ):
        """
            computes the third term in the ELBO
            inputs:
                - current_q_a_sample: a sample from q(a_t | y_t) which current_q_x is computed based on
                - current_q_x: q(x_t | a_{1:t})
                - current_q_a: q(a_t | y_t)
        """

        mu_x = current_q_x.loc
        sigma_x = current_q_x.covariance_matrix

        # entropy term
        entropy = current_q_a.entropy()

        # log det term
        logdet = self.Na.logdet()

        # quadratic term
        delta = current_q_a_sample - mu_x @ self.C.T

        quad = torch.einsum(
            "bi,ij,bj->b",
            delta,
            torch.linalg.pinv(self.Na),
            delta
        )

        # trace term
        trc = torch.einsum(
            "ij,bij->b",
            self.C.T @ torch.linalg.pinv(self.Na) @ self.C,
            sigma_x,
        )

        return -entropy + 0.5*(logdet + quad + trc)