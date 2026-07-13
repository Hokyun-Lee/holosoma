"""CPU tests for deterministic terrain curriculum generation and assignment."""

from __future__ import annotations

import importlib
import math
import sys
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from holosoma.config_values.terrain import (
    terrain_locomotion_curriculum,
    terrain_locomotion_mix,
)
from holosoma.motion_gen.terrain import ScanGrid
from holosoma.simulator.shared.terrain import Terrain


def _small_curriculum_cfg(*, num_rows: int = 3, num_cols: int = 8):
    cfg = terrain_locomotion_curriculum.terrain_term
    return replace(
        cfg,
        border_size=1,
        terrain_length=4.0,
        terrain_width=4.0,
        num_rows=num_rows,
        num_cols=num_cols,
        curriculum_layout=replace(cfg.curriculum_layout, spawn_clearance_radius=0.5),
    )


def test_curriculum_preset_is_equal_four_type_10_by_20_layout() -> None:
    cfg = terrain_locomotion_curriculum.terrain_term
    assert cfg.num_rows == 10
    assert cfg.num_cols == 20
    assert cfg.curriculum_layout.enabled
    assert cfg.curriculum_layout.terrain_types == ["flat", "box", "stair", "hurdle"]
    assert cfg.terrain_config == {"flat": 0.25, "box": 0.25, "stair": 0.25, "hurdle": 0.25}


def test_curriculum_columns_are_balanced_and_rows_are_linear_monotone() -> None:
    terrain = Terrain(_small_curriculum_cfg(), num_robots=1)

    assert terrain.terrain_types_by_column == (
        "flat",
        "box",
        "stair",
        "hurdle",
        "flat",
        "box",
        "stair",
        "hurdle",
    )
    counts = [terrain.terrain_types_by_column.count(name) for name in terrain.terrain_type_names]
    assert max(counts) - min(counts) == 0
    np.testing.assert_allclose(terrain.row_difficulties, [0.0, 0.5, 1.0])
    assert np.all(np.diff(terrain.row_difficulties) >= 0.0)
    assert terrain.origin_grid.shape == (3, 8, 3)


@pytest.mark.parametrize("terrain_type", ["box", "stair", "hurdle"])
def test_curriculum_primitives_are_positive_clear_and_height_sensitive(terrain_type: str) -> None:
    terrain = Terrain(_small_curriculum_cfg(num_rows=2, num_cols=4), num_robots=1)
    easy = terrain.make_terrain(terrain_type, difficulty=0.0).height_field_raw
    hard = terrain.make_terrain(terrain_type, difficulty=1.0).height_field_raw

    assert easy.min() == 0
    assert easy.max() > 0
    assert hard.min() == 0
    assert hard.max() > easy.max()

    center_x, center_y = easy.shape[0] // 2, easy.shape[1] // 2
    # The small fixture overrides the radius from 1.0 m to 0.5 m.
    clearance = math.ceil(_small_curriculum_cfg().curriculum_layout.spawn_clearance_radius / 0.1)
    central = easy[
        center_x - clearance : center_x + clearance + 1,
        center_y - clearance : center_y + clearance + 1,
    ]
    assert np.count_nonzero(central) == 0

    if terrain_type in {"stair", "hurdle"}:
        assert np.array_equal(easy, easy.T)
        assert np.any(easy[center_x, center_y + clearance + 1 :] > 0)
        assert np.any(easy[center_x + clearance + 1 :, center_y] > 0)


def test_default_layout_still_dispatches_to_randomized_terrain(monkeypatch) -> None:
    calls = {"random": 0, "curriculum": 0}

    def fake_randomized(self) -> None:
        calls["random"] += 1

    def fake_curriculum(self) -> None:
        calls["curriculum"] += 1

    monkeypatch.setattr(Terrain, "randomized_terrain", fake_randomized)
    monkeypatch.setattr(Terrain, "curriculum_terrain", fake_curriculum)
    cfg = replace(
        terrain_locomotion_mix.terrain_term,
        border_size=1,
        terrain_length=2.0,
        terrain_width=2.0,
        num_rows=1,
        num_cols=1,
    )
    terrain = Terrain(cfg, num_robots=1)

    assert not terrain.curriculum_enabled
    assert calls == {"random": 1, "curriculum": 0}


def test_locomotion_balances_types_and_updates_subset_origins(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "holosoma.utils.draw", ModuleType("holosoma.utils.draw"))
    TerrainLocomotion = importlib.import_module(
        "holosoma.managers.terrain.terms.locomotion"
    ).TerrainLocomotion

    num_envs = 23
    scene = SimpleNamespace(env_origins=torch.zeros(num_envs, 3))
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        simulator=SimpleNamespace(scene=scene),
    )
    monkeypatch.setattr(
        "holosoma.managers.terrain.terms.locomotion.warp_utils.convert_to_wp_mesh",
        lambda *_args, **_kwargs: object(),
    )
    term = TerrainLocomotion(_small_curriculum_cfg(), env)
    scene.env_origins.copy_(term.env_origins)

    counts = torch.bincount(term.terrain_type_ids, minlength=len(term.terrain_type_names))
    assert int(counts.max() - counts.min()) <= 1
    assert term.num_curriculum_levels == 3
    assert torch.count_nonzero(term.terrain_levels) == 0
    names_by_column = term.terrain.terrain_types_by_column
    for env_id in range(num_envs):
        assert names_by_column[int(term.terrain_columns[env_id])] == term.terrain_type_names[
            int(term.terrain_type_ids[env_id])
        ]

    term.configure_local_height_scan(ScanGrid())
    term._local_height_scan_valid[:] = True
    before = term.env_origins.clone()
    env_ids = torch.tensor([1, 7, 22])
    levels = torch.tensor([1, 2, 1])
    term.set_curriculum_origins(env_ids, levels)

    unchanged = torch.ones(num_envs, dtype=torch.bool)
    unchanged[env_ids] = False
    assert torch.equal(term.terrain_levels[env_ids], levels)
    assert torch.equal(term.env_origins[unchanged], before[unchanged])
    assert torch.equal(scene.env_origins, term.env_origins)
    assert not bool(torch.any(term.local_height_scan_valid[env_ids]))
    assert bool(torch.all(term.local_height_scan_valid[unchanged]))
