"""Closed-loop generated-motion command: the paper's fine-tuning setup.

A frozen diffusion motion generator is called *inside* the RL training loop:
every ``replan_interval_s`` (paper: 0.5 s = 2 Hz) the tracker's own measured
state (past 2 frames at 50 Hz) conditions the generator, which produces the
next 0.5 s reference window that the tracking rewards then follow. This
replaces the fixed motion clip of :class:`MotionCommand`.

Reuse strategy:
    - Episode resets reuse the parent's machinery on a *seed* motion file
      (``motion_file``): robot states are initialized from seed frames +
      noise, giving diverse dynamically-feasible starts; the conditioning
      history is seeded from the same frames.
    - All reference properties consumed by rewards/observations/termination
      are overridden to read from per-env generated windows.

Known approximations (documented; the generator predicts no per-body
orientations, matching the paper's representation):
    - reference body orientations / angular velocities are unavailable ->
      the ``motion_relative_body_orientation_error_exp`` and
      ``motion_global_body_ang_vel`` rewards must be zero-weighted in the
      experiment preset (see ``g1_29dof_wbt_gen``); buffers expose identity
      quats / root angular velocity as placeholders for metrics only.
    - the reference-frame (torso) orientation is reconstructed analytically
      from the generated root orientation and the waist yaw/roll/pitch
      angles (G1 chain: z-, x-, y-axis revolute joints).
"""

from __future__ import annotations

import math
from typing import Any

import torch
from loguru import logger

from holosoma.config_types.command import GeneratedMotionConfig
from holosoma.managers.command.base import CommandTermBase
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.motion_gen.features import quat_conjugate, quat_mul, quat_normalize, quat_yaw
from holosoma.motion_gen.sampling import MotionGenerator, MotionGeneratorInput
from holosoma.utils.rotations import quat_apply, quat_inverse, yaw_quat
from holosoma.utils.rotations import quat_mul as quat_mul_sim


def _wxyz(q_xyzw: torch.Tensor) -> torch.Tensor:
    return q_xyzw[..., [3, 0, 1, 2]]


def _xyzw(q_wxyz: torch.Tensor) -> torch.Tensor:
    return q_wxyz[..., [1, 2, 3, 0]]


def _axis_angle_quat_wxyz(axis: tuple[float, float, float], angle: torch.Tensor) -> torch.Tensor:
    half = 0.5 * angle
    s = torch.sin(half)
    return torch.stack(
        [torch.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dim=-1
    )


class GeneratedMotionCommand(MotionCommand):
    def __init__(self, cfg: Any, env):
        # Bypass MotionCommand.__init__: it coerces dicts to plain MotionConfig,
        # which rejects the generator-specific fields.
        CommandTermBase.__init__(self, cfg, env)
        self._env = env
        mc = cfg.params["motion_config"]
        self.motion_cfg = mc if isinstance(mc, GeneratedMotionConfig) else GeneratedMotionConfig(**mc)
        self.gen_cfg: GeneratedMotionConfig = self.motion_cfg
        self.init_pose_cfg = self.motion_cfg.noise_to_initial_pose
        self._seed_mode = True  # parent reset/setup path reads the seed motion

    # ------------------------------------------------------------------ setup

    def setup(self) -> None:
        super().setup()

        assert self.gen_cfg.generator_checkpoint, "generator_checkpoint must be set"
        self.generator = MotionGenerator.from_checkpoint(
            self.gen_cfg.generator_checkpoint, device=str(self.device), use_ema=self.gen_cfg.use_ema
        )
        trainable_generator_params = sum(
            p.numel() for p in self.generator.model.parameters() if p.requires_grad
        )
        if trainable_generator_params != 0:
            raise RuntimeError(
                "GeneratedMotionCommand requires a frozen generator, but "
                f"{trainable_generator_params} parameters still require gradients."
            )
        gen_data_cfg = self.generator.cfg.data
        self.layout = self.generator.layout
        self._past = gen_data_cfg.past_frames
        self._horizon = gen_data_cfg.future_frames
        self._feat_dim = self.layout.dim
        self._use_sim_terrain_scan = self.gen_cfg.use_sim_terrain_scan
        if self._use_sim_terrain_scan:
            if not gen_data_cfg.use_terrain_scan:
                raise ValueError(
                    "use_sim_terrain_scan requires a generator checkpoint trained with terrain scans"
                )
            if gen_data_cfg.terrain_dim != gen_data_cfg.scan_grid.dim:
                raise ValueError(
                    f"Generator terrain_dim {gen_data_cfg.terrain_dim} does not match "
                    f"scan grid dim {gen_data_cfg.scan_grid.dim}"
                )
            self._terrain_state = self._env.terrain_manager.get_state("locomotion_terrain")
            self._terrain_state.configure_local_height_scan(gen_data_cfg.scan_grid)

        env_fps = 1.0 / self._env.dt
        if abs(env_fps - gen_data_cfg.fps) > 1e-3:
            raise ValueError(
                f"Env control rate {env_fps:.1f} Hz != generator fps {gen_data_cfg.fps}; "
                "closed-loop conditioning assumes one reference frame per policy step."
            )
        self._replan_steps = max(1, round(self.gen_cfg.replan_interval_s / self._env.dt))
        if self._replan_steps > self._horizon:
            raise ValueError(
                f"replan_interval {self.gen_cfg.replan_interval_s}s = {self._replan_steps} steps "
                f"exceeds the generator horizon ({self._horizon} frames)."
            )

        # --- name-based index mappings (sim <-> generator layout) ---
        sim_dofs = list(self._env.simulator.dof_names)
        lay_joints = list(self.layout.joint_names)
        assert set(sim_dofs) == set(lay_joints), "sim dof names != generator joint names"
        self._lay_j_from_sim = torch.tensor(
            [sim_dofs.index(n) for n in lay_joints], device=self.device
        )  # sim-order tensor indexed with this -> layout order
        self._sim_j_from_lay = torch.tensor(
            [lay_joints.index(n) for n in sim_dofs], device=self.device
        )  # layout-order tensor indexed with this -> sim order

        tracked = list(self.motion_cfg.body_names_to_track)
        lay_bodies = list(self.layout.body_names)
        assert set(tracked) == set(lay_bodies), (
            "body_names_to_track must equal the generator's body set for closed-loop training"
        )
        self._lay_b_from_tracked = torch.tensor(
            [tracked.index(n) for n in lay_bodies], device=self.device
        )
        self._tracked_b_from_lay = torch.tensor(
            [lay_bodies.index(n) for n in tracked], device=self.device
        )
        self._torso_in_tracked = tracked.index(self.motion_cfg.body_name_ref[0])
        self._waist_lay_idx = [
            lay_joints.index(n) for n in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
        ]

        # --- per-env generated window buffers (quats stored xyzw, sim convention) ---
        N, H, J, B = self.num_envs, self._horizon, len(sim_dofs), len(tracked)
        dev = self.device
        self._win_root_pos = torch.zeros(N, H, 3, device=dev)
        self._win_root_quat = torch.zeros(N, H, 4, device=dev)
        self._win_root_lin_vel = torch.zeros(N, H, 3, device=dev)
        self._win_root_ang_vel = torch.zeros(N, H, 3, device=dev)
        self._win_joint_pos = torch.zeros(N, H, J, device=dev)
        self._win_joint_vel = torch.zeros(N, H, J, device=dev)
        self._win_body_pos = torch.zeros(N, H, B, 3, device=dev)
        self._win_body_lin_vel = torch.zeros(N, H, B, 3, device=dev)
        self._win_ref_quat = torch.zeros(N, H, 4, device=dev)
        self.window_idx = torch.zeros(N, dtype=torch.long, device=dev)
        self._history = torch.zeros(N, self._past, self._feat_dim, device=dev)
        self._headings = torch.zeros(N, 2, device=dev)
        self._arange = torch.arange(N, device=dev)
        if not self._use_sim_terrain_scan:
            self._terrain_zeros = torch.zeros(N, gen_data_cfg.terrain_dim, device=dev)

        self._seed_mode = False
        logger.info(
            f"[GeneratedMotionCommand] frozen generator @ {self.gen_cfg.generator_checkpoint} "
            f"(step {self.generator.checkpoint_step}), replan every {self._replan_steps} steps, "
            f"{self.gen_cfg.denoise_steps}-step denoising, heading={self.gen_cfg.heading_mode}, "
            f"terrain={'sim_scan' if self._use_sim_terrain_scan else 'stage5_flat_zeros'}, "
            "trainable generator params=0"
        )

    # ------------------------------------------------------------ state -> features

    def _pack(self, root_pos, root_quat_xyzw, joint_pos_sim, body_pos_tracked) -> torch.Tensor:
        """Pack sim-convention states into generator features (world, wxyz)."""
        return torch.cat(
            [
                root_pos,
                _wxyz(root_quat_xyzw),
                joint_pos_sim[..., self._lay_j_from_sim],
                body_pos_tracked[..., self._lay_b_from_tracked, :].flatten(-2),
            ],
            dim=-1,
        )

    def _sim_features(self) -> torch.Tensor:
        """Current measured robot state as generator features (num_envs, D)."""
        sim = self._env.simulator
        return self._pack(
            sim.robot_root_states[:, :3],
            sim.robot_root_states[:, 3:7],
            sim.dof_pos,
            sim._rigid_body_pos[:, self.tracked_body_indexes, :],
        )

    def _seed_features(self, env_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        """Seed-motion frames as generator features (for history right after reset)."""
        origins = self._env.simulator.scene.env_origins[env_ids]
        root_pos = self.motion.body_pos_w[time_steps, 0] + origins
        root_quat = self.motion.body_quat_w[time_steps, 0]  # xyzw
        joint_pos = self.motion.joint_pos[time_steps]  # sim dof order
        body_pos = self.motion.body_pos_w[time_steps][:, self.tracked_body_indexes] + origins[:, None, :]
        return self._pack(root_pos, root_quat, joint_pos, body_pos)

    # ------------------------------------------------------------------ replan

    def _update_terrain_from_sim(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Refresh scans from the measured simulator root pose (xyzw at this boundary)."""
        root_states = self._env.simulator.robot_root_states[env_ids]
        root_xy = root_states[:, :2]
        root_yaw = quat_yaw(_wxyz(root_states[:, 3:7]))
        return self._terrain_state.update_local_height_scan(root_xy, root_yaw, env_ids)

    @torch.no_grad()
    def _replan(self, env_ids: torch.Tensor) -> None:
        past = self._history[env_ids].clone()
        if self.gen_cfg.past_noise_std > 0:
            # Paper: conditions are perturbed during training for robustness.
            past = past + torch.randn_like(past) * self.gen_cfg.past_noise_std
            qs = self.layout.root_quat_slice
            past[..., qs] = quat_normalize(past[..., qs])

        if self._use_sim_terrain_scan:
            # WBT refreshes the shared cache every policy step for tracker
            # observations. Query the due subset again from the measured root
            # so direct command stepping cannot silently use stale data.
            self._update_terrain_from_sim(env_ids)
            valid = self._terrain_state.local_height_scan_valid[env_ids]
            if not bool(valid.all()):
                invalid_count = int((~valid).sum())
                raise RuntimeError(f"Cannot replan with {invalid_count} stale local terrain scans")
            terrain_height = self._terrain_state.local_height_scan[env_ids]
        else:
            terrain_height = self._terrain_zeros[env_ids]

        out = self.generator.generate(
            MotionGeneratorInput(
                past_motion=past,
                target_heading=self._headings[env_ids],
                terrain_height=terrain_height,
            ),
            num_steps=self.gen_cfg.denoise_steps,
            deterministic=False,
        )

        dt = self._env.dt
        root_quat_wxyz = out.root_quat  # (n, H, 4)
        self._win_root_pos[env_ids] = out.root_pos
        self._win_root_quat[env_ids] = _xyzw(root_quat_wxyz)
        self._win_joint_pos[env_ids] = out.joint_pos[..., self._sim_j_from_lay]
        self._win_body_pos[env_ids] = out.body_pos[:, :, self._tracked_b_from_lay]

        # Velocities by finite difference, seeded with the last measured frame.
        prev_feat = self._history[env_ids, -1]
        prev_root_pos = prev_feat[:, self.layout.root_pos_slice].unsqueeze(1)
        prev_quat_wxyz = prev_feat[:, self.layout.root_quat_slice].unsqueeze(1)
        prev_joint = prev_feat[:, self.layout.joint_pos_slice][..., self._sim_j_from_lay].unsqueeze(1)
        prev_body = prev_feat[:, self.layout.body_pos_slice].view(len(env_ids), 1, -1, 3)[
            :, :, self._tracked_b_from_lay
        ]

        pos_seq = torch.cat([prev_root_pos, out.root_pos], dim=1)
        self._win_root_lin_vel[env_ids] = (pos_seq[:, 1:] - pos_seq[:, :-1]) / dt
        joint_seq = torch.cat([prev_joint, self._win_joint_pos[env_ids]], dim=1)
        self._win_joint_vel[env_ids] = (joint_seq[:, 1:] - joint_seq[:, :-1]) / dt
        body_seq = torch.cat([prev_body, self._win_body_pos[env_ids]], dim=1)
        self._win_body_lin_vel[env_ids] = (body_seq[:, 1:] - body_seq[:, :-1]) / dt

        quat_seq = quat_normalize(torch.cat([quat_normalize(prev_quat_wxyz), root_quat_wxyz], dim=1))
        q_rel = quat_mul(quat_seq[:, 1:], quat_conjugate(quat_seq[:, :-1]))
        q_rel = torch.where(q_rel[..., :1] < 0, -q_rel, q_rel)
        self._win_root_ang_vel[env_ids] = 2.0 * q_rel[..., 1:] / dt  # small-angle rotvec rate

        # Reference (torso) orientation from root quat + waist chain (z, x, y axes).
        wy = out.joint_pos[..., self._waist_lay_idx[0]]
        wr = out.joint_pos[..., self._waist_lay_idx[1]]
        wp = out.joint_pos[..., self._waist_lay_idx[2]]
        torso_wxyz = quat_mul(
            quat_mul(quat_mul(root_quat_wxyz, _axis_angle_quat_wxyz((0, 0, 1), wy)),
                     _axis_angle_quat_wxyz((1, 0, 0), wr)),
            _axis_angle_quat_wxyz((0, 1, 0), wp),
        )
        self._win_ref_quat[env_ids] = _xyzw(quat_normalize(torso_wxyz))

    # ------------------------------------------------------------------ reset / step

    def reset(self, env_ids: torch.Tensor | None) -> None:
        env_ids = self._ensure_index_tensor(env_ids)
        if env_ids.numel() == 0:
            return
        self._seed_mode = True
        try:
            super().reset(env_ids)  # robot state <- seed motion frame + noise
        finally:
            self._seed_mode = False

        # Conditioning history from the seed frames (t-past+1 .. t).
        t = self.time_steps[env_ids]
        start = self.motion.motion_start_idx[self.motion_ids[env_ids]]
        for k in range(self._past):
            tk = torch.clamp(t - (self._past - 1 - k), min=start)
            self._history[env_ids, k] = self._seed_features(env_ids, tk)

        if self._use_sim_terrain_scan:
            self._terrain_state.invalidate_local_height_scan(env_ids)

        if self.gen_cfg.heading_mode == "random":
            yaw = torch.rand(env_ids.numel(), device=self.device) * 2 * math.pi
            self._headings[env_ids] = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
        else:  # "current": keep the facing direction implied by the past frames
            q = self._history[env_ids, -1, self.layout.root_quat_slice]
            w, x, y, z = q.unbind(-1)
            yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            self._headings[env_ids] = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)

        self._replan(env_ids)
        self.window_idx[env_ids] = 0

    def step(self) -> None:
        # Measured-state history advances every policy step (50 Hz).
        self._history[:, :-1] = self._history[:, 1:].clone()
        self._history[:, -1] = self._sim_features()

        self.window_idx += 1
        due = torch.where(self.window_idx >= self._replan_steps)[0]
        if due.numel() > 0:
            self._replan(due)
            self.window_idx[due] = 0

        # Relative body poses (same computation as MotionCommand.step, without
        # clip bookkeeping): "if the reference were re-anchored to where the
        # robot currently is (xy + yaw of the ref body), where should each
        # tracked body be?"
        use_root = (self._env.episode_length_buf == 0).unsqueeze(1).float()
        ref_pos_w = self.root_pos_w * use_root + self.ref_pos_w * (1 - use_root)
        ref_quat_w = self.root_quat_w * use_root + self.ref_quat_w * (1 - use_root)
        robot_ref_pos_w = self.robot_root_pos_w * use_root + self.robot_ref_pos_w * (1 - use_root)
        robot_ref_quat_w = self.robot_root_quat_w * use_root + self.robot_ref_quat_w * (1 - use_root)

        n_track = len(self.motion_cfg.body_names_to_track)
        ref_pos_r = ref_pos_w[:, None, :].repeat(1, n_track, 1)
        ref_quat_r = ref_quat_w[:, None, :].repeat(1, n_track, 1)
        rob_pos_r = robot_ref_pos_w[:, None, :].repeat(1, n_track, 1)
        rob_quat_r = robot_ref_quat_w[:, None, :].repeat(1, n_track, 1)

        delta_quat_w = yaw_quat(quat_mul_xyzw(rob_quat_r, quat_inverse(ref_quat_r, w_last=True)), w_last=True)
        self.body_quat_relative_w = quat_mul_xyzw(delta_quat_w, self.body_quat_w)
        delta_pos_w_height = ref_pos_r - rob_pos_r
        delta_pos_w_height[..., :2] = 0.0
        self.body_pos_relative_w = (
            rob_pos_r + delta_pos_w_height + quat_apply(delta_quat_w, self.body_pos_w - ref_pos_r, w_last=True)
        )

    def update_metrics(self) -> None:
        super().update_metrics()
        if not self._use_sim_terrain_scan:
            return
        scan = self._terrain_state.local_height_scan
        root_states = self._env.simulator.robot_root_states
        current_yaw = quat_yaw(_wxyz(root_states[:, 3:7]))
        yaw_delta = current_yaw - self._terrain_state.local_height_scan_root_yaw
        self.metrics["terrain/scan_abs_mean"] = scan.abs().mean(dim=-1)
        self.metrics["terrain/scan_range"] = scan.max(dim=-1).values - scan.min(dim=-1).values
        self.metrics["terrain/scan_anchor_xy_error"] = torch.linalg.vector_norm(
            root_states[:, :2] - self._terrain_state.local_height_scan_root_xy,
            dim=-1,
        )
        self.metrics["terrain/scan_anchor_yaw_error"] = torch.atan2(
            torch.sin(yaw_delta),
            torch.cos(yaw_delta),
        ).abs()

    # ------------------------------------------------------------------ reference properties

    def _at_idx(self, buf: torch.Tensor) -> torch.Tensor:
        return buf[self._arange, self.window_idx.clamp(max=self._horizon - 1)]

    @property
    def joint_pos(self) -> torch.Tensor:
        if self._seed_mode:
            return MotionCommand.joint_pos.fget(self)  # type: ignore[attr-defined]
        return self._at_idx(self._win_joint_pos)

    @property
    def joint_vel(self) -> torch.Tensor:
        if self._seed_mode:
            return MotionCommand.joint_vel.fget(self)  # type: ignore[attr-defined]
        return self._at_idx(self._win_joint_vel)

    @property
    def body_pos_w(self) -> torch.Tensor:
        if self._seed_mode:
            return MotionCommand.body_pos_w.fget(self)  # type: ignore[attr-defined]
        return self._at_idx(self._win_body_pos)

    @property
    def body_quat_w(self) -> torch.Tensor:
        if self._seed_mode:
            return MotionCommand.body_quat_w.fget(self)  # type: ignore[attr-defined]
        # No per-body orientations from the generator: expose the reference
        # torso orientation for every tracked body. Only zero-weighted rewards
        # and logging metrics consume this.
        return self._at_idx(self._win_ref_quat)[:, None, :].expand(-1, self._win_body_pos.shape[2], -1)

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        if self._seed_mode:
            return MotionCommand.body_lin_vel_w.fget(self)  # type: ignore[attr-defined]
        return self._at_idx(self._win_body_lin_vel)

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        if self._seed_mode:
            return MotionCommand.body_ang_vel_w.fget(self)  # type: ignore[attr-defined]
        return self._at_idx(self._win_root_ang_vel)[:, None, :].expand(-1, self._win_body_pos.shape[2], -1)

    @property
    def ref_pos_w(self) -> torch.Tensor:
        if self._seed_mode:
            return MotionCommand.ref_pos_w.fget(self)  # type: ignore[attr-defined]
        return self._at_idx(self._win_body_pos)[:, self._torso_in_tracked]

    @property
    def ref_quat_w(self) -> torch.Tensor:
        if self._seed_mode:
            return MotionCommand.ref_quat_w.fget(self)  # type: ignore[attr-defined]
        return self._at_idx(self._win_ref_quat)

    @property
    def ref_lin_vel_w(self) -> torch.Tensor:
        if self._seed_mode:
            return MotionCommand.ref_lin_vel_w.fget(self)  # type: ignore[attr-defined]
        return self._at_idx(self._win_body_lin_vel)[:, self._torso_in_tracked]

    @property
    def ref_ang_vel_w(self) -> torch.Tensor:
        if self._seed_mode:
            return MotionCommand.ref_ang_vel_w.fget(self)  # type: ignore[attr-defined]
        return self._at_idx(self._win_root_ang_vel)

    @property
    def root_pos_w(self) -> torch.Tensor:
        if self._seed_mode:
            return MotionCommand.root_pos_w.fget(self)  # type: ignore[attr-defined]
        return self._at_idx(self._win_root_pos)

    @property
    def root_quat_w(self) -> torch.Tensor:
        if self._seed_mode:
            return MotionCommand.root_quat_w.fget(self)  # type: ignore[attr-defined]
        return self._at_idx(self._win_root_quat)

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        if self._seed_mode:
            return MotionCommand.root_lin_vel_w.fget(self)  # type: ignore[attr-defined]
        return self._at_idx(self._win_root_lin_vel)

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        if self._seed_mode:
            return MotionCommand.root_ang_vel_w.fget(self)  # type: ignore[attr-defined]
        return self._at_idx(self._win_root_ang_vel)


def quat_mul_xyzw(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return quat_mul_sim(a, b, w_last=True)
