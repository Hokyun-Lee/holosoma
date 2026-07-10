"""Gaussian diffusion (DDPM training, DDPM/DDIM sampling) for motion windows.

Not specified in the paper (arXiv:2604.17335): beta schedule, number of
training diffusion steps, and epsilon-vs-x0 parameterization. Defaults follow
the official MDM implementation (cosine schedule, T=1000, x0 prediction);
epsilon prediction and a linear schedule are available via config.

The paper deploys 2-step denoising; here full-step DDPM is the validated
path and few-step DDIM (including 2 steps) is exposed as experimental.
"""

from __future__ import annotations

import math
from typing import Callable

import torch

ModelFn = Callable[..., torch.Tensor]  # model(x_t, t, **cond) -> prediction


def make_beta_schedule(schedule: str, timesteps: int) -> torch.Tensor:
    if schedule == "cosine":  # Nichol & Dhariwal 2021
        steps = timesteps + 1
        s = 0.008
        x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos((x / timesteps + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(0, 0.999).float()
    if schedule == "linear":
        scale = 1000.0 / timesteps
        return torch.linspace(scale * 1e-4, scale * 0.02, timesteps, dtype=torch.float32)
    raise ValueError(f"Unknown beta schedule: {schedule}")


class GaussianDiffusion:
    def __init__(self, timesteps: int = 1000, schedule: str = "cosine", param: str = "x0"):
        if param not in ("x0", "eps"):
            raise ValueError(f"param must be 'x0' or 'eps', got {param}")
        self.timesteps = timesteps
        self.schedule = schedule
        self.param = param

        betas = make_beta_schedule(schedule, timesteps)
        alphas = 1.0 - betas
        self.betas = betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = self.alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1.0 - self.alphas_cumprod).sqrt()
        # DDPM posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef_x0 = betas * self.alphas_cumprod_prev.sqrt() / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef_xt = (1.0 - self.alphas_cumprod_prev) * alphas.sqrt() / (1.0 - self.alphas_cumprod)

    def _gather(self, coef: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
        out = coef.to(t.device).gather(0, t).float()
        return out.view(t.shape[0], *([1] * (ndim - 1)))

    # -- forward (noising) process -------------------------------------------

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            self._gather(self.sqrt_alphas_cumprod, t, x0.ndim) * x0
            + self._gather(self.sqrt_one_minus_alphas_cumprod, t, x0.ndim) * noise
        )

    def sample_timesteps(self, batch_size: int, device, generator: torch.Generator | None = None) -> torch.Tensor:
        return torch.randint(0, self.timesteps, (batch_size,), device=device, generator=generator)

    # -- parameterization conversions ----------------------------------------

    def pred_to_x0(self, pred: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.param == "x0":
            return pred
        # Note: with param="eps" the x0-space training loss is amplified by
        # 1/sqrt(alpha_bar_t) at large t (an implicit SNR re-weighting);
        # the default and validated parameterization is "x0".
        sqrt_ac = self._gather(self.sqrt_alphas_cumprod, t, x_t.ndim)
        sqrt_om = self._gather(self.sqrt_one_minus_alphas_cumprod, t, x_t.ndim)
        return (x_t - sqrt_om * pred) / sqrt_ac.clamp_min(1e-8)

    def x0_to_eps(self, x0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        sqrt_ac = self._gather(self.sqrt_alphas_cumprod, t, x_t.ndim)
        sqrt_om = self._gather(self.sqrt_one_minus_alphas_cumprod, t, x_t.ndim)
        return (x_t - sqrt_ac * x0) / sqrt_om.clamp_min(1e-8)

    # -- reverse (sampling) process ------------------------------------------

    @torch.no_grad()
    def ddpm_sample(
        self,
        model_fn: ModelFn,
        shape: tuple[int, ...],
        device,
        generator: torch.Generator | None = None,
        **cond,
    ) -> torch.Tensor:
        """Full-step ancestral DDPM sampling."""
        x = torch.randn(shape, device=device, generator=generator)
        for step in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), step, device=device, dtype=torch.long)
            pred = model_fn(x, t, **cond)
            x0 = self.pred_to_x0(pred, x, t)
            mean = (
                self._gather(self.posterior_mean_coef_x0, t, x.ndim) * x0
                + self._gather(self.posterior_mean_coef_xt, t, x.ndim) * x
            )
            if step > 0:
                var = self._gather(self.posterior_variance, t, x.ndim)
                noise = torch.randn(shape, device=device, generator=generator)
                x = mean + var.sqrt() * noise
            else:
                x = mean
        return x

    @torch.no_grad()
    def ddim_sample(
        self,
        model_fn: ModelFn,
        shape: tuple[int, ...],
        device,
        num_steps: int = 50,
        eta: float = 0.0,
        generator: torch.Generator | None = None,
        init_noise: torch.Tensor | None = None,
        **cond,
    ) -> torch.Tensor:
        """DDIM sampling with ``num_steps`` steps (eta=0 -> deterministic given
        the initial noise). num_steps=2 reproduces the paper's deployment
        setting and is experimental in this implementation."""
        num_steps = min(num_steps, self.timesteps)
        step_indices = torch.linspace(self.timesteps - 1, 0, num_steps).round().long()
        step_indices = torch.unique(step_indices).flip(0)  # descending t

        x = init_noise if init_noise is not None else torch.randn(shape, device=device, generator=generator)
        for i, step in enumerate(step_indices):
            t = torch.full((shape[0],), int(step), device=device, dtype=torch.long)
            pred = model_fn(x, t, **cond)
            x0 = self.pred_to_x0(pred, x, t)
            eps = self.x0_to_eps(x0, x, t)

            if i + 1 < len(step_indices):
                t_prev = int(step_indices[i + 1])
                ac_prev = self.alphas_cumprod[t_prev].to(device)
            else:
                ac_prev = torch.tensor(1.0, device=device)
            ac_t = self.alphas_cumprod[int(step)].to(device)

            sigma = eta * ((1 - ac_prev) / (1 - ac_t)).clamp_min(0).sqrt() * (1 - ac_t / ac_prev).clamp_min(0).sqrt()
            dir_xt = (1 - ac_prev - sigma**2).clamp_min(0).sqrt() * eps
            x = ac_prev.sqrt() * x0 + dir_xt
            if eta > 0 and i + 1 < len(step_indices):
                x = x + sigma * torch.randn(shape, device=device, generator=generator)
        return x
