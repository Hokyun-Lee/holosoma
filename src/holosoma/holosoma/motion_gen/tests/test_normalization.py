import torch

from holosoma.motion_gen.normalization import FeatureNormalizer


def test_roundtrip_and_save_load(tmp_path):
    g = torch.Generator().manual_seed(0)
    mean = torch.randn(78, generator=g)
    std = torch.rand(78, generator=g) + 0.5
    norm = FeatureNormalizer(mean, std)

    x = torch.randn(4, 25, 78, generator=g)
    assert torch.allclose(norm.denormalize(norm.normalize(x)), x, atol=1e-5)

    path = tmp_path / "stats.npz"
    norm.save(path)
    norm2 = FeatureNormalizer.load(path)
    assert torch.allclose(norm.mean, norm2.mean)
    assert torch.allclose(norm.std, norm2.std)


def test_zero_variance_dims_not_scaled():
    mean = torch.zeros(10)
    std = torch.zeros(10)  # constant features (e.g. zero terrain)
    norm = FeatureNormalizer(mean, std)
    x = torch.randn(3, 10)
    assert torch.allclose(norm.normalize(x), x)
