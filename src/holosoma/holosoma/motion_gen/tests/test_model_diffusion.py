import torch

from holosoma.motion_gen.diffusion import GaussianDiffusion
from holosoma.motion_gen.model import MotionDiffusionTransformer


def _tiny_model(feature_dim=78):
    return MotionDiffusionTransformer(
        feature_dim=feature_dim, past_frames=2, future_frames=25,
        terrain_dim=16, d_model=32, n_layers=1, n_heads=2, d_ff=64, dropout=0.0,
    )


def test_model_forward_shape():
    torch.manual_seed(0)
    model = _tiny_model()
    B = 4
    out = model(
        torch.randn(B, 25, 78), torch.randint(0, 10, (B,)),
        torch.randn(B, 2, 78), torch.randn(B, 2), torch.randn(B, 16),
    )
    assert out.shape == (B, 25, 78)
    assert torch.isfinite(out).all()


def test_model_condition_masking_changes_output():
    torch.manual_seed(0)
    model = _tiny_model().eval()
    B = 2
    args = (torch.randn(B, 25, 78), torch.zeros(B, dtype=torch.long),
            torch.randn(B, 2, 78), torch.randn(B, 2), torch.randn(B, 16))
    with torch.no_grad():
        full = model(*args)
        dropped = model(*args, drop_past=torch.ones(B, dtype=torch.bool))
    assert not torch.allclose(full, dropped)


def test_q_sample_shape_and_endpoints():
    diff = GaussianDiffusion(timesteps=100)
    x0 = torch.randn(3, 25, 78)
    noise = torch.randn_like(x0)
    x_t = diff.q_sample(x0, torch.zeros(3, dtype=torch.long), noise)
    assert x_t.shape == x0.shape
    # at t=0 the sample stays close to x0
    assert (x_t - x0).abs().mean() < 0.2


def test_param_conversion_consistency():
    diff = GaussianDiffusion(timesteps=100, param="eps")
    x0 = torch.randn(3, 25, 78)
    noise = torch.randn_like(x0)
    t = torch.tensor([10, 50, 90])
    x_t = diff.q_sample(x0, t, noise)
    # eps -> x0 recovers the original clean sample
    assert torch.allclose(diff.pred_to_x0(noise, x_t, t), x0, atol=1e-4)
    # x0 -> eps recovers the noise
    assert torch.allclose(diff.x0_to_eps(x0, x_t, t), noise, atol=1e-4)


def _run_sampler(param, num_steps=None):
    torch.manual_seed(0)
    model = _tiny_model().eval()
    diff = GaussianDiffusion(timesteps=10, param=param)
    cond = dict(
        past=torch.randn(2, 2, 78), heading=torch.randn(2, 2), terrain=torch.randn(2, 16)
    )

    def model_fn(x_t, t, **_):
        return model(x_t, t, cond["past"], cond["heading"], cond["terrain"])

    shape = (2, 25, 78)
    if num_steps is None:
        return diff.ddpm_sample(model_fn, shape, torch.device("cpu"))
    return diff.ddim_sample(model_fn, shape, torch.device("cpu"), num_steps=num_steps)


def test_ddpm_and_ddim_sampling_shapes():
    for param in ("x0", "eps"):
        assert _run_sampler(param).shape == (2, 25, 78)
        assert _run_sampler(param, num_steps=5).shape == (2, 25, 78)
        assert _run_sampler(param, num_steps=2).shape == (2, 25, 78)  # paper deployment steps


def test_ddim_deterministic_given_seed():
    torch.manual_seed(0)
    model = _tiny_model().eval()
    diff = GaussianDiffusion(timesteps=10)

    def model_fn(x_t, t, **_):
        return model(x_t, t, torch.zeros(1, 2, 78), torch.zeros(1, 2), torch.zeros(1, 16))

    outs = []
    for _ in range(2):
        g = torch.Generator().manual_seed(42)
        outs.append(diff.ddim_sample(model_fn, (1, 25, 78), torch.device("cpu"), num_steps=5, generator=g))
    assert torch.allclose(outs[0], outs[1])
