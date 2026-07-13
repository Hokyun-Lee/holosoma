# Motion Generator Implementation Notes

Staged reproduction of the diffusion motion generator and tracker from
**"Learning Whole-Body Humanoid Locomotion via Motion Generation and Motion
Tracking"** (arXiv:2604.17335), built inside HoloSoma for the Unitree G1.
Korean user manual: `docs/motion_generator_ko.html`.

## Papers and sources investigated

| Source | What was used | License |
|---|---|---|
| arXiv:2604.17335 (main paper) | Generator design facts (below). The official project page exposes Paper/Video/Poster only and **no public code** (rechecked 2026-07-13). | — |
| MDM, Tevet et al. ICLR 2023 (arXiv:2209.14916) | Architecture family + defaults (8 layers, 4 heads, d=512, ff=1024, dropout 0.1, GELU; x0-prediction; cosine schedule, T=1000). Ideas only, no code copied. | MIT |
| PARC, Xu et al. SIGGRAPH 2025 | Cited by the paper for geometric losses; used as conceptual reference for velocity/consistency losses. | — |
| OmniRetarget (arXiv:2509.26633; this repo is its codebase) | Data conversion pipeline (`convert_data_format_mj.py` logic), climb/chair motions from the HF dataset. | Code Apache-2.0 (repo LICENSE), dataset MIT |
| BeyondMimic (HybridRobotics/whole_body_tracking) | LAFAN1 CSV column layout confirmation (root pos xyz, quat **xyzw**, 29 joints @30 fps), 30→50 fps resampling approach. Ideas only. | MIT |
| LAFAN1 G1 retarget (HF lvhaidong/LAFAN1_Retargeting_Dataset) | 6 locomotion clips. | CC BY-NC-ND 4.0 (non-commercial, no redistribution) |
| OMOMO large-box demo (shipped with HoloSoma) | 1 whole-body clip. | OMOMO upstream CC BY-NC 4.0 |

## Facts stated in the paper vs. implementation choices

**Stated in the paper (Sec. III):** per-frame features = root position R³ +
root orientation R⁴ + joint positions R²³ + body-link positions R^{23×3};
horizon 0.5 s = 25 frames; conditioning = 2 past frames + target heading
vector (from base pose difference) + terrain height scan; MDM-family
architecture; reconstruction loss + velocity / joint-consistency / terrain-
penetration geometric losses; ~1 h augmented training data; 2-step denoising
at deployment; receding-horizon re-planning every 0.25 s; TensorRT ~0.02 s on
Jetson Thor.

**Not published (all values below are implementation choices):** hidden
dims/layers/heads, beta schedule, T, epsilon-vs-x0, optimizer, batch size,
LR, EMA, loss weights, normalization frame, terrain scan resolution, CFG,
condition-noise magnitudes/distributions, and FK loss reduction/weight.

| Item | Paper | This implementation | Basis |
|---|---|---|---|
| Joints | 23 (G1 23-DoF) | **29** (HoloSoma G1 29-DoF model) | current robot model; wrist DOFs included |
| Body positions | 23 links | **14 links** (HoloSoma WBT `body_names_to_track`) | aligns generated bodies with the existing tracker rewards |
| Feature dim / frame | 99 | **78** = 3+4+29+42 | consequence of the above |
| Canonical frame | unspecified | anchor = last past frame; xy→0, yaw→0, z absolute; quats wxyz sign-continuous | common practice (MDM/PARC-style) |
| Heading | "from base pose difference" | unit xy vector anchor→last future frame, fallback (1,0) below 5 cm displacement | paper gives no encoding details |
| Terrain scan | used, resolution unknown | **289-dim** (17×17 @0.1 m), x∈[-0.3,1.3], y∈[-0.8,0.8], heading-aligned absolute height; trained on 150 OmniRetarget terrain clips (Phase B); real simulator scans connected in Stage 6 | resolution/range are implementation choices; Stage 5 remains the zero-scan flat baseline, while Stage 7 adds the balanced obstacle curriculum |
| Architecture | "MDM [26,30]" | MDM defaults 512/8/4/1024 | official MDM implementation |
| Diffusion | unspecified | DDPM T=1000 cosine, x0-prediction (eps available) | MDM convention |
| Few-step inference | 2 steps (deployment) | DDIM, any step count; **2-step validated** at 0.0475 m val body MPJPE and used by the closed-loop tracker | 50-step remains the offline evaluation protocol |
| Losses | recon + velocity + joint consistency + terrain penetration | recon split (root pos/quat/joint/body) + quat-norm + velocity + bone-length consistency + **true differentiable G1 FK consistency (Stage 8, opt-in)** + foot-slide (contact proxy) + scan/flat terrain penetration | weights and FK model/subset are choices; contact labels do not exist → contact-consistency loss disabled |
| Data | ~1 h augmented | **165.3 min, 195 clips** (paperscale); the original 6.8 min/11-clip profile remains available | implementation scale-up for a single RTX 4090 |
| Fine-tuning w/ tracker | closed loop | **implemented**: frozen generator, measured two-frame state, real simulator terrain scan, 0.5 s replanning, 2-step DDIM | Stage 5 flat baseline; Stage 6 terrain dataflow; Stage 7 history/reward/curriculum; robust Stage-8 generator still awaits converged terrain PPO |

## Data pipeline

- 11 clips, 50 fps WBT format, total ≈ 6.8 min:
  train (9): lafan1 walk1_subject1 / walk3_subject2 / dance1_subject2 /
  run2_subject1 / sprint1_subject2, omni_climb_02, omni_climb_14,
  omni_chair_scene_06, omomo_largebox_003;
  val (2): lafan1_walk4_subject1, omni_climb_09. Split is motion-level.
- LAFAN1 clips trimmed to ≤60 s each (`data_manifest.py` frame ranges) to stay
  in the 3–10 min scope.
- Conventions verified against real files: WBT npz `joint_pos` (T,36) =
  [root pos, root quat **wxyz**, 29 joints in MuJoCo model order]; body arrays
  have 51 bodies with `world` at index 0; OmniRetarget raw qpos is
  [quat wxyz, pos, joints]; LAFAN1 CSV quat is **xyzw** (converted).
- `convert_data_format_mj.py`'s `MotionLoader` misreads npz `fps` stored as
  frames-per-second (`round(1/30)=0` → ZeroDivisionError). Not fixed there
  (kept user code untouched); the new headless converter reads fps correctly.

## New / changed files

Core (`src/holosoma/holosoma/motion_gen/`): `features.py` (representation,
quat utils, canonicalization), `normalization.py`, `dataset.py`,
`model.py`, `diffusion.py`, `losses.py`, `configs.py` (smoke/debug/
baseline_4090 presets), `training.py`, `evaluation.py`, `sampling.py`
(MotionGenerator inference API + receding horizon), `export.py`,
`visualization.py`, `data_manifest.py`, `scripts/` (inspect_motion_npz,
download_data, prepare_motions, make_splits, train, evaluate, sample),
`tests/` (36 CPU tests incl. end-to-end training smoke).

Other: `src/holosoma_retargeting/.../data_conversion/convert_data_format_mj_headless.py`
(batch FK conversion, joint-limit dump), `.../data_conversion/view_motion_mj.py`
(MuJoCo kinematic replay / GIF render of any qpos or WBT npz),
`demo_scripts/prepare_motion_gen_data.sh`,
`.gitignore` (+`/data/`), `README.md` (short section),
`docs/motion_generator_ko.html`, this file.

No existing files were modified except `.gitignore` and `README.md` (additive).

## Commands actually executed and results

| Command | Result |
|---|---|
| `bash demo_scripts/prepare_motion_gen_data.sh` | passed — 11 processed clips @50 fps, joint_limits.json, splits.json |
| sanity check (pelvis/feet heights, contact ratios on 4 clips) | passed — pelvis ≈0.78 m walking, feet min z ≈0.035 m |
| `pytest src/holosoma/holosoma/motion_gen/tests/ -q` (hssim env, CPU) | **36 passed** (incl. regression tests for the review fixes; one DDIM step-ordering bug found by the tests and fixed) |
| `python -m holosoma.motion_gen.scripts.train smoke` (GPU) | passed — 30 steps, ckpt + samples + plots written |
| `python -m holosoma.motion_gen.scripts.train debug` (GPU, 3000 steps, ~34 s) | passed — overfit metrics improve monotonically (body MPJPE 0.076→0.040 m, root 0.057→0.031 m); no NaN, joint-limit violations 0 |
| `python -m holosoma.motion_gen.scripts.train baseline_4090` (200k-step run, 64 min) | passed — ~52 steps/s, 1.3 GB torch-allocated VRAM (2.9 GB process). Val (unseen walk4 + climb_09, DDIM 50): best around 4k–10k steps; **overfits afterwards** |
| `evaluate.py` ckpt_00010000 vs final (val split) | passed — 10k: body MPJPE **0.166 m** / root 0.142 m; 200k: 0.237 m / 0.233 m (train at 200k: 0.005 m → memorization). Based on this the baseline preset default was reduced to max_steps=50k with val_interval=1k |
| `sample.py --mode window` and `--mode rollout` (baseline & debug ckpts) | passed — plots + `*_gen_raw.npz` + `*_gen_qpos.npz` written (a heading/device bug found here was fixed; see review section) |
| `convert_data_format_mj_headless.py --input-file <rollout qpos npz>` | passed — generated rollout round-trips into the full 51-body WBT schema npz |
| `view_motion_mj.py --motion <rollout gen_qpos.npz> --video gen.gif` | passed — generated rollout replays kinematically on the G1 MuJoCo model (offscreen render verified; interactive viewer same code path) |
| `view_motion_mj.py --motion <gen> --gt <processed clip> --gt-start 0` | passed — generated + translucent GT robot replay together in one scene (MjSpec attach, verified offscreen: past frames overlap exactly, divergence visible afterwards) |
| `mypy --config-file mypy.ini src/holosoma/holosoma/motion_gen` | passed — no issues in 30 files |

## Independent implementation review (subagent) and fixes

A separate reviewer verified the DDPM/DDIM math numerically (cosine schedule,
posterior coefficients, x0/eps conversions, EMA direction, normalization
consistency, canonicalization roundtrip — all confirmed correct) and found
issues that were then fixed and covered by regression tests:

1. Heading rotation shape bug — `generate()` crashed whenever
   `target_heading` was passed (fixed in `sampling.py`; test added).
2. Device mixing in `receding_horizon` and in `sample.py` (CPU input ×
   CUDA model) — fixed; rollout output now returns on the input device.
3. Yaw ±π branch cut could flip the canonical quaternion sign between
   near-identical windows (sign-multimodal diffusion targets) — canonical
   windows are now re-anchored to w≥0 at the anchor frame (test added).
4. Condition dropout changed from independent-per-condition to joint
   (MDM-style) so the unconditional branch used by CFG is actually trained.
5. Generated features now repack a re-normalized root quaternion before
   receding-horizon feedback/export (yaw extraction assumes unit norm).
6. Non-finite-loss skip no longer desynchronizes the LR scheduler from the
   step counter; `param="eps"`'s x0-space loss amplification at large t is
   clamped and documented (x0 remains the validated default).
7. Earlier, the test suite itself caught a DDIM step-ordering bug
   (ascending-t iteration → NaN), fixed before any training run.

The published baseline_4090 run was restarted after these fixes.

## Stage 2: paper-scale data expansion (2026-07-10, docs/motion_generator_scaleup_ko.html)

The 11-clip run generalized poorly (val body MPJPE ≈ 0.166 m), so the dataset
was expanded to a "paperscale" profile matching/exceeding the paper's ~1 h:
34 full-length LAFAN1 G1 clips (fallAndGetUp excluded — locomotion scope,
implementation choice) + all 145 OmniRetarget climb clips (29 takes × 5
z-scales — the dataset's own terrain-height augmentation) + 15
robot-object-terrain clips + the OMOMO demo = **195 clips, 165.3 min
measured**, train 186 / val 9. Val holds out whole sequence groups to prevent
leakage: walk4 (all), run1 (both subjects), climb_09 (all z-scales),
scene_04. Same architecture/config as baseline (only data + step schedule
changed): preset `paperscale_4090`, artifacts under
`data/motion_gen/{raw_qpos,processed,metadata}_paperscale` and
`splits/splits_paperscale.json`; scripts gained `--profile`, the prep shell
script takes the profile as first argument. Known residual duplication:
different subjects of the same LAFAN1 sequence share choreography inside the
train split.

Results (measured 2026-07-11, 200k steps in 63.9 min, no overfitting — val
improved monotonically to the end): on the *same* val clips and protocol as
stage 1 (walk4 + climb_09, DDIM 50, fixed seeds), val body MPJPE
**0.166 → 0.0411 m (4.0x)**, root position error 0.142 → 0.0264 m, joint
error 0.258 → 0.0937 rad, foot slide 0.276 → 0.142 m/s. Full 9-clip val:
MPJPE 0.0546 m vs train 0.0288 m (healthy ~1.9x gap). Only the data changed;
model/losses/training code identical. `view_motion_mj.py` gained
contact/constraint disabling for kinematic replay (two overlapping robots
crashed the constraint solver with a FactorizeHessian fatal error).

## Stage 3: terrain-conditioned generator, Phase B (2026-07-11, docs/motion_generator_terrain_ko.html)

Real terrain height scans extracted from OmniRetarget's multi-box terrain
models (per-climb/scene URDFs referencing 8-vertex box .obj meshes baked in
world coordinates; z-scale variants via the URDF scale attribute). New
`motion_gen/terrain.py` computes heights analytically (max over yaw-rotated
box footprints) and samples heading-aligned scans; grid is a forward-biased
17x17 @0.1 m (289 dims, x in [-0.3,1.3], y in [-0.8,0.8]) — implementation
choice, the paper gives no scan resolution. `add_terrain_scans.py` attaches
`terrain_height`/`terrain_grid` keys to the processed npz (150 clips) and
validates motion-terrain alignment: **150/150 clips pass** (no feet >5 cm
below terrain; min clearances match the ~3.5 cm ankle-origin height).
The penetration loss now penalizes bodies below the bilinearly interpolated
scan surface (grid-outside bodies excluded); flat clips keep the z<0 form.
`terrain_4090` preset = paperscale + use_terrain_scan (terrain_dim 289).
Rollout sampling can re-sample scans along the generated root via
`sample.py --terrain-urdf` (receding_horizon terrain_fn callback). Takes
(10 clips) have no terrain models -> zero scans; the chair object itself is
NOT part of scene scans (documented).

Results (measured 2026-07-11, 200k steps / 66 min, no overfitting): on the
terrain val subset (climb_09 x5 z-scales + scene_04), MPJPE is unchanged
within noise vs the zero-scan paperscale model (0.0457 vs 0.0432 m) but the
**measured terrain penetration of generated bodies drops 3.5x (0.21 -> 0.06
mm)**; flat val unchanged (0.0546 vs 0.0566 m). Qualitative: a climb_09
rollout with `--terrain-urdf` scan re-sampling climbs onto the terrain box
(verified in the MuJoCo render; `view_motion_mj.py` gained a
`--terrain-urdf` option that draws the terrain boxes as static geoms).
Honest read: at a 0.5 s horizon the 2 past frames already imply most of the
terrain state, so scan conditioning mainly improves physical plausibility
here; its steering value should appear in receding-horizon deployment over
unseen terrain layouts (post-tracker integration).

## Stage 4: RL motion tracker on generated motion (2026-07-11, docs/motion_generator_tracker_ko.html)

The generated-motion export path was validated against the existing HoloSoma
WBT RL task with **zero task-code changes**: a 20 s receding-horizon rollout
(walk4 val clip start, terrain_4090 generator, deterministic DDIM 50) was
converted to the full WBT schema and fed to `exp:g1-29dof-wbt` via the
motion_file override. Smoke run (64 envs, 10 PPO iterations, Isaac Sim
headless, logger:disabled) passed — motion loads, PPO trains, checkpoints
and per-checkpoint ONNX exports are produced
(`logs/WholeBodyTracking/20260711_021527-*/model_00009.pt/.onnx`).
A full-scale run (4096 envs, save_interval 500) was intentionally stopped at
30% (9,000/30,000 iterations, ~21 iter/min measured) per user request.
Results at iter 9,012: average episode length 88 -> **500 steps (full 10 s
episodes, no falls)**; relative body position/orientation rewards
0.985/0.908 (max 1.0); body lin/ang velocity 0.870/0.805; global ref
position/orientation 0.334/0.486 still rising (absolute-drift tightening is
the remaining-70%/tuning territory). Policy checkpoint:
`logs/WholeBodyTracking/20260711_021742-*/model_09000.pt` (replayable via
`eval_agent.py --checkpoint ...`). Closed-loop generator-tracker
fine-tuning remains future work (stage 5).

## Stage 5: closed-loop generator-in-the-loop fine-tuning (2026-07-11, docs/motion_generator_closedloop_ko.html)

Implements the paper's actual fine-tuning scheme: the frozen generator is
called *inside* the RL loop — every 0.5 s (2 Hz) the measured robot state
(past 2 frames at 50 Hz) of all 4096 envs is batch-fed to the generator
(2-step DDIM; pre-measured quality 0.0475 m val MPJPE ≈ 50-step), producing
the next 25-frame reference window that rewards/observations/termination
consume. New `GeneratedMotionCommand` (managers/command/terms/wbt_gen.py)
subclasses `MotionCommand`: seed-motion resets reused for initialization and
conditioning history; name-based sim↔generator mappings (Isaac joint order ↔
MuJoCo order, xyzw ↔ wxyz); per-episode random target heading; Gaussian
conditioning noise (σ=0.01, paper's robustness idea, scale is a choice);
reference velocities by finite difference; torso reference orientation
reconstructed from root quat ⊗ waist z/x/y joint chain. Documented
approximations: no per-body reference orientations/angular velocities (the
generator representation has none, as in the paper) → the two rewards
needing them are zero-weighted in the new `exp:g1-29dof-wbt-gen` preset.
Config: `GeneratedMotionConfig` (config_types/command.py), presets in
config_values/wbt/g1/{command_gen,experiment_gen}.py (+2-line registry).
Smoke (64 envs, 10 iters) passed first try. Full run fine-tuned from the
stage-4 tracker checkpoint (model_09000.pt) for 3,000 iterations (stopped at
iter 12,048, converged/plateaued). Results: episode length dropped to 47
immediately after the offline->closed-loop switch (distribution shift from
fixed-clip to self-conditioned reference + random heading), recovered to
full 500-step episodes within ~350 iterations. Final vs the stage-4 offline
policy: relative body position reward 0.973 (offline 0.985, comparable);
**global ref position reward 0.466 vs 0.334 (+40%)** — closed-loop
re-conditions the reference on measured state every 0.5 s, so absolute
drift cannot accumulate the way it does tracking a fixed clip; global
orientation 0.477 (offline 0.486, comparable); body lin vel 0.848 (offline
0.870, comparable). Generator-call overhead negligible (~21 iter/min,
same as offline; 9.3 GB GPU). The resulting policy tracks generator output
under a random per-episode heading command — effectively a heading-steerable
locomotion controller. Checkpoint:
`logs/WholeBodyTracking/20260711_122501-*/model_12000.pt`.

A second run trained the same `exp:g1-29dof-wbt-gen` preset **from scratch**
(`training.checkpoint: null`, PPO `resume: null`) for all 30,000 iterations.
It completed without logged errors in about 18 h 40 min with 4096 environments,
producing 61 PT checkpoints and 61 ONNX exports; final checkpoint:
`logs/WholeBodyTracking/20260711_144420-*/model_29999.pt`. TensorBoard final
scalars at iteration 29,999 versus the offline-9k + closed-loop-3k run endpoint
at iteration 12,048 (latest persisted checkpoint: 12,000) were: average episode length **489.2 vs 492.7** (max 500),
relative body position **0.918 vs 0.972**, global reference position **0.465
vs 0.466**, global reference orientation **0.464 vs 0.477**, and body linear
velocity **0.844 vs 0.848**. Thus the offline-pretrained route used 12k total
PPO iterations (only 3k closed-loop adaptation) yet matched or exceeded the
30k closed-loop-only run, suggesting better sample efficiency in this
implementation. These are rolling training scalars rather than a common
independent evaluation; both runs use seed 42, but their reset seed motions
differ (walk4 vs largebox). It is therefore not a statistical superiority
claim; matched multi-seed ablation remains necessary.

During `eval_agent.py`, generated motion does not advance through a clip list:
the command stochastically replans a 25-frame window every 0.5 s. A uniform
random world heading (`heading_mode="random"`) is sampled only on reset.
Although training episodes last 10 s, `get_eval_config()` overrides the eval
episode length to 100,000 s, so the heading does not automatically change
every 10 s; a fall/termination resets it. Pass
`--simulator.config.sim.max-episode-length-s 10` to deliberately cycle random
headings every 10 s, or select `heading_mode="current"` to preserve the
reset-facing direction. Stage 5 has no fixed-vector selector or manual
immediate-reset key, and Isaac Sim's W/S/Q/E command keys do not feed
`GeneratedMotionCommand`.

## Stage 6: simulator terrain connection (2026-07-13, docs/motion_generator_stage6_ko.html)

Stage 5 remains an unchanged flat baseline, while the new
`exp:g1-29dof-wbt-gen-terrain` preset uses the existing procedural trimesh and
the terrain-trained generator. The simulator scan contract is taken directly
from the generator checkpoint: root-XY/root-yaw local frame, x
`[-0.3, 1.3]`, y `[-0.8, 0.8]`, 0.1 m spacing, 17x17=289 values,
x-major/y-fastest flattening, raw absolute world-Z metres. A GPU Warp-raycast
diagnostic containing a box, stairs, and a hurdle measured flat/obstacle/yaw
maximum errors of 1.53e-5 / 1.53e-5 / 2.29e-5 m and saved a debug NPZ.
These grid values are implementation choices inherited from Stage 3, not
paper-specified constants.

`TerrainLocomotion` now owns one per-environment scan cache shared by the
frozen generator and tracker. The WBT observation callback refreshes it every
50 Hz policy step from measured simulator root states; the 2 Hz generator
replan refreshes its due subset again and rejects stale scans. Isaac root
quaternions are converted from xyzw at the simulator boundary; motion-gen
internals remain wxyz. Tracker observations append only the current 289-value
scan: actor 154->443 and critic 286->575. Raw values enter the observation
term at scale 1 and PPO applies its empirical normalization.

Warm-starting the Stage-5 `model_12000.pt` required a deliberately narrow,
opt-in `checkpoint_load_mode="expand_input"`: only the appended input suffix
of each first Linear may grow. Old columns are copied exactly; new columns and
AdamW moments are zero-filled; new normalizer entries start at mean 0 and
variance/std 1. The old scalar normalizer count is retained to preserve old
statistics, so the new channels initially remain near raw metre scale and
adapt slowly (an implementation choice and future ablation). Strict loading
remains the global and flat-preset default.

Real Isaac validation with 64 envs resumed the old checkpoint for two PPO
iterations (`20260713_075346-*`), saved the fully expanded model/normalizers/
optimizer, and changed the initially-zero terrain weights (actor max 0.002571,
critic max 0.003237). A second diagnostic run logged actual scan abs mean
0.0290 m, range 0.0368 m, and zero displayed root-XY/yaw anchor errors. With
the same history, heading, seed, and 2-step deterministic sampler, replacing a
flat scan by a synthetic 30 cm obstacle changed generated features by mean
0.00879 / max 0.06617. The post-change flat regression strict-loaded the old
154/286 checkpoint and completed one PPO iteration (`20260713_075823-*`).
Related tests: 51 passed, Ruff clean. The current terrain mix is still the old
flat/rough/depression set; positive box/stair/hurdle curriculum is Stage 7.3.

Reset-alignment correction (2026-07-13): the original statement that reset
roots and seed bodies used the "same terrain tile" was too broad. Scene/env
origins were synchronized with terrain tile origins, but the command still
added the motion file's accumulated world-space root XY. Curriculum tiles are
8 m by 8 m (4 m half-width per axis), so a random phase could initialize the
robot/reference on an adjacent tile even though the scene and scan used one
consistent world frame. In the 8,095-frame generated seed, 6,933 frames
(85.645%) had pelvis radius greater than 1 m and 4,793 (59.209%) lay outside at
least one 4 m half-width; x ranged `[-6.530, 0.616]` m and y
`[-0.639, 4.688]` m. Actual 1,024-env episode-start audits found radius>1 m /
outside-tile counts of 883/575 in the legacy Stage-9 `model_12000.pt` state and
865/610 in `model_14999.pt`. The maximum residual to the nearest raw phase was
0.0663 m, consistent with configured reset noise.

Commit `884a26b` adds an opt-in reset-time XY translation that places the
selected root phase at the assigned env origin while preserving the subsequent
relative trajectory, plus a future-frame phase horizon for fixed references.
Commit `d748e46` keeps generated-terrain training at horizon 0 because
`GeneratedMotionCommand` does not advance the seed clip after reset and must
continue to accept the short legacy seed. Final semantics are: Stage-5 flat
`reanchor=false, horizon=0`; generated-terrain training
`reanchor=true, horizon=0`; A fixed-reference training
`reanchor=true, horizon=500`; and Stage-10 evaluation forces all variants to
`reanchor=true` with an effective 500-step horizon at the default 10 s/50 Hz.
The flat smoke `20260713_124232-*` completed one PPO iteration with the legacy
false/0 contract. The terrain smoke `20260713_124259-*` completed two
iterations with true/0, simulator scan, frozen generator, expanded tracker and
optimizer, and saved `model_12001.pt`. The related suite reported 88 passed and
the modified Python files passed Ruff. Pre-fix terrain runs remain valid
integrity/dataflow artifacts but are superseded as terrain-performance
checkpoints.

## Stage 7.2: world-heading reward (2026-07-13, docs/motion_generator_stage7_ko.html)

The terrain generated-motion preset now adds a separately configurable
`motion_heading_alignment` term using measured world-frame root velocity and
the generator's unit world-heading condition:
`dot(v_xy, d_xy) / (norm(v_xy) + epsilon)`. The Stage-5 flat preset has no new
reward term. Default terrain weight 1.0 and epsilon 1e-6 are implementation
choices; CLI weight 0 disables the term. Logged direction error uses pi/2 for
speed below 0.05 m/s (also a configurable implementation choice) because
travel direction is undefined at rest; this convention does not alter the
reward equation.

Both 64-env Isaac runs completed one PPO iteration from the legacy tracker:
ON `20260713_080555-*` and OFF `20260713_080625-*`. The ON TensorBoard file has
instantaneous heading error/reward/speed plus independent Episode and
RawEpisode heading tags; the OFF file retains diagnostic metrics but omits
only the episodic reward tags. The measured ON values at iteration 12,000 were
error 1.4921 rad, raw alignment 0.0440, and speed 0.4722 m/s. The terrain scan,
443/575 checkpoint expansion, and frozen generator remained active. Unit and
CLI tests: 8 passed. Terrain curriculum (7.3) remains after the selective
history work below.

## Stage 7.1: selective five-frame proprioception (2026-07-13, docs/motion_generator_stage7_ko.html)

`ObsTermCfg` now supports an optional per-term `history_length`. Existing
groups with no override retain the legacy group-level flattening exactly. In
the terrain generated-motion preset, base angular velocity, projected gravity,
29 joint positions, and 29 joint velocities use five frames; generated
references and the terrain scan remain current-frame only. Buffers run at the
50 Hz policy rate in oldest-to-newest order, zero-pad unavailable initial
frames, zero only the reset environment rows, and do not advance during the
bootstrap `modify_history=False` observation.

To preserve checkpoint columns, the Stage-6 current observation remains an
unchanged prefix and only the four older frames are appended in a suffix.
Projected gravity current is added after that old prefix. Actor input therefore
grows 443->702 and critic input 575->834. First-layer weights and Adam moments
use the existing zero-suffix migration. New terrain/gravity current normalizer
columns start with identity statistics, while the 244 appended lag columns for
base angular velocity and joint position/velocity copy the matching current
feature's mean/variance/std. These ordering, zero-padding, and projected-gravity
noise=0 choices are implementation choices because the paper does not publish
them. Per-term suffixes are explicitly rejected when symmetry augmentation is
enabled; this WBT preset has symmetry disabled. Deployment must reproduce the
same stateful history outside the training environment. The inherited scalar
normalizer count is about 1.18e9, so genuinely new terrain/gravity statistics
adapt slowly; this keeps exact-resume behavior for bounded inputs but remains a
normalizer warm-up ablation.

A real 64-env Isaac terrain run resumed Stage-5 `model_12000.pt`, expanded to
702/834, completed iterations 12,000-12,001, and saved finite model,
normalizer, and optimizer tensors (`20260713_081949-*`). A separate Stage-5
flat regression strict-loaded the original 154/286 policy and completed one
iteration unchanged (`20260713_082014-*`). CPU regression: 62 passed; Ruff and
diff checks clean.

## Stage 7.3: balanced terrain curriculum (2026-07-13, docs/motion_generator_stage7_ko.html)

The terrain generated-motion preset now uses an opt-in deterministic 10-row by
20-column layout. Columns cycle flat/box/stair/hurdle, giving each type exactly
25% of environments, and rows linearly increase difficulty from 0 to 1. Each
environment keeps its type and column for the whole run; only its row changes.
Obstacle geometry is positive-only and a central 1 m half-width square is
forced to exactly zero height for safe reset. Box height spans 0.05-0.30 m;
stair maximum elevation and hurdle height span 0.05-0.35 m. Their dimensions,
10 levels, and the concentric Chebyshev-square geometry are implementation
choices exposed in config because the paper does not publish them. The old
random-mix and Stage-5 flat terrain paths remain unchanged.

In the original Stage-7 implementation, per-environment progression evaluated
only comparable completed episodes: a timeout after at least 90% of the
configured episode length was a curriculum success proxy, a fall termination
was a failure, and the first randomized rollout fragment was ignored. The
Stage-7 diagnostic counts below are therefore per-terrain survival-timeout
proxies, **not obstacle-crossing results**, and must remain interpreted with
that historical meaning. Five consecutive proxy successes promoted one row
and two consecutive failures demoted one row, clamped to levels 0-9. These
thresholds are implementation choices/config values. A new pre-reset
curriculum lifecycle applied the terrain origin and invalidated its scan before
the command reset, so generator history and terrain scan agreed with the new
tile.

The Stage-7 logs included mean level, every level fraction, per-type cumulative
success and episode counts/rates/fractions, and type-fraction min/max/range.
Its checkpoint state version 1 contained levels, fixed type IDs, streaks,
actual-step guards, initial-fragment eligibility and per-type counts; tensor
schema validation preceded mutation and restore reapplied every terrain
origin. PPO resume started a fresh physical episode via `reset_all()`, so
durable levels, streaks and cumulative counts continued while the in-progress
episode guard reset on the first reset. Legacy tracker checkpoints had no such
state and correctly initialized at level 0. The state still does not contain a
terrain geometry fingerprint; resuming after changing geometry under the same
type names is unsupported.

Follow-up crossing semantic correction (`321afb7`, after the Stage-7
diagnostics): after command reset, each episode captures measured root XY
`p_0` and the normalized episode-start `motion_command.target_heading_w`
direction `d_0`. At every step, progress is the signed projection
`dot(p_t - p_0, d_0)`; the gate retains `max(0, max_t progress_t)`, so lateral
or backward displacement cannot satisfy it. The configured threshold is
`crossing_distance_m=1.5`, an unpublished implementation choice. A curriculum
success now requires a comparable non-initial fragment, no fall/non-timeout
termination, a timeout after at least 90% of the episode, and signed forward
progress of at least 1.5 m. A termination is a failure; a surviving timeout
below 1.5 m is also a progression failure. Survival, crossing, success, and
failure counts/rates are logged independently per terrain type and per
type×level. This defines the metric; it does not demonstrate convergence or an
improvement in obstacle success.

The current checkpoint schema is version 3 with
`progress_semantics=target_heading_signed_v1`,
`episode_target_heading_w`, and `max_episode_forward_progress_m`. Restore
validates unit headings and tensor schemas. Version-2 state used radial
distance, which cannot be converted to signed forward progress: migration
preserves durable levels, streaks, and other outcome state, but resets crossing
counts and the in-progress forward gate to zero and starts from the current
command heading.

The complete 200-tile mesh generated successfully (3,821,135 vertices,
7,672,002 faces), with column modulo four assigning type, difficulty spanning
0 to 1, and all origin Z values at zero. A 64-env shortened 0.2 s diagnostic
completed four PPO iterations (`20260713_084509-*`) and reached 63 environments
at level 1; exact type fractions stayed [0.25, 0.25, 0.25, 0.25]. Resuming its
`model_12003.pt` (`20260713_084603-*`) continued levels and cumulative counts
rather than resetting them. A separate standard-10-second run completed two
iterations and crossed the 25-policy-step / 0.5 s generator replan boundary
(`20260713_084654-*`). Generator parameters remained frozen and the actual
state, five-frame proprioception and simulator scan paths stayed active. A
post-lifecycle Stage-5 flat regression again strict-loaded 154/286 inputs and
completed one PPO iteration with `stage5_flat_zeros` (`20260713_084859-*`).
CPU regression: 80 passed; Ruff and diff checks clean. The shortened curriculum
run validated the historical Stage-7 progression mechanics, not obstacle
performance or convergence. Flat environments also carry a row index although
their geometry is unchanged, so global mean level is a sampling-state
diagnostic rather than average physical obstacle difficulty. The later signed
crossing/type×level channels provide Stage 9/10 evaluation semantics, but no
converged comparison or success improvement is claimed here.

## Stage 8: generator robustness (2026-07-13, docs/motion_generator_stage8_ko.html)

### Task 8.1: structured condition noise — completed

- [x] noise magnitudes are configurable
- [x] root quaternions remain normalized and sign-continuous
- [x] clean/noisy validation uses matched diffusion randomness
- [x] noisy-condition samples remain finite and do not collapse

`ConditionNoiseCfg` applies perturbations in physical units to the two past
frames and terrain condition only, before normalization. It never corrupts the
future target or the clean scan used by geometric losses, and inference does
not add it implicitly. `terrain_robust_4090` uses root-position 0.01 m,
root-orientation 0.02 rad, joint-position 0.01 rad, body-position 0.01 m,
terrain point-height 0.01 m, scan bias 0.01 m, local-xy 0.02 m and yaw 0.02 rad
Gaussian standard deviations, plus 5% point dropout. Quaternion noise is
axis-angle left-composed in internal wxyz order, normalized, and sign-aligned
across time. Terrain xy/yaw error is a bilinear warp of the existing
x-major/y-fastest 17x17 scan. All these values and distributions are
**implementation choices**: the paper does not publish them. Every default is
zero, preserving Stage 1-7 presets.

Validation computes clean and noisy loss/sample metrics from the same batch,
diffusion timestep/noise, and exact same initial DDIM noise (condition RNG has
a separate fixed seed). The old checkpoint parser maps the missing nested
noise config to all-zero defaults. Because the Stage-3 checkpoint ended its
200k-step cosine schedule at zero LR, the robustness preset has an explicit
`resume_weights_only=True`: it loads model/EMA/normalizer but restarts optimizer,
scheduler, and step. Full-resume behavior remains the default elsewhere.

At batch size 256, eight fixed validation batches, seed 123, and 2-step DDIM,
the old `terrain_4090/final.pt` scored clean/noisy total loss
0.009626818/0.012202425 and clean/noisy body MPJPE
0.0345846/0.0601754 m. After 10,000 robustness+FK steps the corresponding
values were 0.009751713/0.010175588 and 0.0385989/0.0499975 m. The noisy-clean
MPJPE gap therefore narrowed from 0.0255908 to 0.0113986 m without non-finite
or collapsed motion, while clean MPJPE worsened by about 4 mm. This is a
measured robustness/clean-accuracy trade-off from one seed, not a general
superiority claim.

Changed files: `motion_gen/condition_noise.py`, `configs.py`, `training.py`,
`sampling.py`, `tests/test_condition_noise.py`, and
`tests/test_train_smoke.py` (implementation commit `cbae008`).

### Task 8.2: differentiable FK consistency — completed

- [x] differentiable 29-DoF G1 FK is implemented
- [x] FK coordinate loss and body-error metric are logged
- [x] clean and noisy FK errors decreased after fine-tuning
- [x] the measured joint/root-head versus body-head conflict decreased

`G1ForwardKinematics` is a parameter-free torch implementation of the exact
29-hinge tree in
`src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof.xml`.
Constants are pinned to source SHA-256
`8c586e4747da85804180fe44d8692e0fd8231356728b6327e256dca498087a78`;
joint/body names and source hash fail fast. Inputs accept arbitrary batch
dimensions, root wxyz quaternion, root position, and 29 joint angles; output is
the 14 tracked body origins in generator order. Across 256 random poses,
MuJoCo parity maximum L2 error was 1.095e-15 m in float64 and 6.35e-7 m in
float32. A pre-training calibration on eight real dataset windows measured
mean 6.324e-8 m and max 3.806e-7 m, below the configurable 1 mm tolerance.

`LossWeights.fk_consistency` defaults to zero for Stage 1-7 compatibility.
The separate `terrain_robust_fk_4090` preset sets it to 0.1 and minimizes
coordinate MSE between the independently predicted body positions and FK of
the predicted root/joints; `fk_body_error_m` logs mean body-origin L2 distance.
The MJCF, 14-body subset, coordinate-MSE reduction, weight 0.1, and 1 mm
tolerance are **implementation choices** because the paper does not publish
them. The fixed comparison above reduced clean FK error
0.00698697 -> 0.00690262 m (1.2%) and noisy FK error
0.00717166 -> 0.00694394 m (3.2%). The change is small but in the intended
direction under both conditions; it does not by itself establish better
contacts or obstacle traversal.

Changed files: `motion_gen/kinematics.py`, `tests/test_kinematics.py`
(foundation commit `dc5c952`), plus `losses.py`, `configs.py`, `training.py`,
and `tests/test_losses.py` (integration commit `cbae008`).

### Task 8.3: fixed-condition terrain sensitivity — evaluation completed

- [x] foot height, root height, knee flexion, and penetration proxy compared
- [x] obstacle height causes a threshold-level physical response
- [ ] desired obstacle-clearing response is **not** demonstrated

`evaluate_terrain_sensitivity.py` fixes the same walk4 history at frame 5750,
anchor heading `[0.976827, -0.214030]`, seed 123, and 2-step DDIM, then changes
only the height scan among flat, 0.30 m, and 0.60 m rectangular obstacles. The
obstacle occupies anchor-local x `[0.30, 0.90]`, y `[-0.40, 0.40]` (63 grid
points). Each condition is generated in a separate size-one call so batch RNG
semantics cannot produce a false difference. JSON and NPZ artifacts are under
`logs/motion_gen/terrain_sensitivity/{stage8_3_corrected_baseline,stage8_3_robust_fk}`.
The evaluator and its tests are commit `c69e557`.

For the robust+FK checkpoint, flat/0.30/0.60 m values were: forward travel
0.39762/0.29656/0.25834 m; final root z
0.76797/0.77176/0.77946 m; left-foot max z
0.15133/0.13856/0.12956 m; right-foot max z
0.12808/0.12185/0.10174 m; left-knee max flexion
1.40717/1.36853/1.35608 rad; right-knee
1.24641/1.15923/0.96134 rad. The link-origin penetration proxy max was
0/0.25264/0.36192 m and its body rate 0/3.714/9.714%. The configured physical
response classifier is true and root final height is monotonic, but feet and
knees change in the wrong direction and the proxy rises. Robust fine-tuning
does reduce the old checkpoint's 0.60 m max proxy from 0.43775 to 0.36192 m,
which is still poor. This proxy compares the 14 generated body origins with
scan height; it is **not** collision-shape, contact, or mesh penetration.
Thus Task 8.3 confirms terrain sensitivity, not successful obstacle clearing.
The rectangle, 0.30/0.60 m heights, thresholds (root/foot 0.01 m, knee 0.05
rad, travel 0.02 m), start frame, and seed are all **implementation choices**.

Commands actually executed:

```bash
PY=~/.holosoma_deps/miniconda3/envs/hssim/bin/python

$PY - <<'PY'
from holosoma.motion_gen.configs import terrain_robust_fk_4090
from holosoma.motion_gen.training import Trainer
cfg = terrain_robust_fk_4090()
cfg.run_name = "terrain_robust_fk_baseline_eval"
cfg.resume = "logs/motion_gen/terrain_4090/checkpoints/final.pt"
cfg.max_steps, cfg.batch_size, cfg.num_workers = 10000, 256, 4
cfg.val_batches, cfg.val_sample_steps = 8, 2
Trainer(cfg).validate()
PY

$PY -m holosoma.motion_gen.scripts.train terrain_robust_fk_4090 \
  --resume logs/motion_gen/terrain_4090/checkpoints/final.pt \
  --max-steps 10000 --val-sample-steps 2 --val-batches 8

$PY -m holosoma.motion_gen.scripts.evaluate_terrain_sensitivity \
  --checkpoint logs/motion_gen/terrain_4090/checkpoints/final.pt \
  --output-dir logs/motion_gen/terrain_sensitivity/stage8_3_corrected_baseline
$PY -m holosoma.motion_gen.scripts.evaluate_terrain_sensitivity \
  --checkpoint logs/motion_gen/terrain_robust_fk_4090/checkpoints/final.pt \
  --output-dir logs/motion_gen/terrain_sensitivity/stage8_3_robust_fk

$PY -m pytest src/holosoma/holosoma/motion_gen/tests -q
# 77 passed, 1 skipped
~/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python -m pytest \
  src/holosoma/holosoma/motion_gen/tests/test_kinematics.py -q
# 7 passed
R=~/.holosoma_deps/miniconda3/envs/hsmujoco/bin/ruff
$R check src/holosoma/holosoma/motion_gen/{configs.py,condition_noise.py,kinematics.py,losses.py,sampling.py,training.py} \
  src/holosoma/holosoma/motion_gen/scripts/evaluate_terrain_sensitivity.py \
  src/holosoma/holosoma/motion_gen/tests/{test_condition_noise.py,test_kinematics.py,test_losses.py,test_terrain_sensitivity.py,test_train_smoke.py}
# All checks passed
```

The final checkpoint is
`logs/motion_gen/terrain_robust_fk_4090/checkpoints/final.pt` (290,483,143
bytes), SHA-256
`7c63764b771ec43fb5d463d77b6860eee8e46f1db5c0b196554f98cc527ed5fa`.
This is the frozen generator used by the completed Stage-9/10 tracker runs.
The clean/robustness trade-off still needs tuning, and real collision/contact
or obstacle crossing must be measured in simulation rather than inferred from
the Stage-8 body-origin proxy.

## Stage 9: terrain closed-loop PPO and corrected production (2026-07-14, docs/motion_generator_stage9_ko.html)

`exp:g1-29dof-wbt-gen-terrain` keeps Stage 5 unchanged and closes the intended
terrain loop. At 50 Hz it packs simulator root/joint/14-body measurements and a
shared `(N, 289)` local scan; after the one seed-clip bootstrap, every replan
requires two measured history frames. Every 25 policy steps (0.5 s, 2 Hz) the
frozen robust+FK generator produces 25 reference frames with two DDIM steps.
The tracker sees five-frame proprioception plus the current reference and scan,
and only actor/critic parameters enter PPO. Runtime aborts if any generator
parameter is trainable. The generator uses internal `wxyz`; Isaac `xyzw` is
converted only at the boundary.

Curriculum schema v3 uses
`progress_semantics="target_heading_signed_v1"`: maximum non-negative
projection from episode-start XY along the episode-start heading. A crossing
requires no fall/non-timeout termination, at least 90% of the 500-step timeout,
and at least 1.5 m signed progress. Five successes raise a level and two
failures lower it within L0..L9. These gates, obstacle geometry, scan grid,
1,024 envs, and 3,000-update budget are implementation choices, not values
published in arXiv:2604.17335.

The old 1,024-env run `20260713_100222-*` and old A/E/F runs remain valid only
for process/checkpoint integrity: their random phase XY could place the robot
on a neighboring 8 m tile. They are superseded for every terrain performance
claim. The corrected D run is:

- directory:
  `logs/WholeBodyTracking/20260713_124451-g1_29dof_wbt_ablation_d_full_terrain_ft-locomotion/`
- fair checkpoint `model_12500.pt`, SHA-256
  `2dd0b09c5b6141e1eb0dd627da32102bcbd32865f6a09659d09dd4847a9ba9e0`
- final checkpoint `model_14999.pt`, SHA-256
  `d5850aad610b9bebf1613cd9473853225dbc06b23f007230d05e92bd328efad8`
- natural final save/end: 2026-07-13 23:21:27 KST; both iteration fields
  equal 14,999; actor/critic inputs are 702/834.

The corrected final checkpoint has zero non-finite elements in model
(17 tensors / 1,119,803 elements), optimizer (51 / 2,239,623), normalizers
(8 / 4,610), and environment state (22 / 13,533). Runtime and saved config
agree on robust+FK generator step 10,000/EMA, zero trainable generator
parameters, simulator scan, measured-history guard, random heading, 25-step
replan, two denoise steps, `replan_interval_s=0.5`,
`use_sim_terrain_scan=true`, `require_fully_measured_history=true`,
`past_noise_std=0.0`, and training `reanchor=true / phase_horizon_steps=0`.
Sampling during training was not the deterministic evaluation mode.

Reset audit found only configured noise: startup 1,024 starts had maximum
absolute x/y/radial tile-center offsets 0.049998/0.049995/0.069307 m, with
0/1,024 outside radius 1 m or the tile. A further 150 logged resets had maxima
0.049652/0.049930/0.067165 m and 0/150 outside the tile. TensorBoard step
14,999 contains all 488 expected finite scalar tags. Selected final values are:
environment/train episode length 478.753876/482.649994, reward/noise/KL
25.910803/0.353401/0.013253, relative-body/global-position/global-orientation/
body-linear-velocity rewards 0.930758/0.462019/0.423966/0.769190,
body/reference-position/reference-rotation errors
0.047407 m/0.051963 m/0.125945 rad, heading error/raw/episode reward
0.597559 rad/0.734315/0.695071, signed progress 1.179505 m, mean curriculum
level 6.484131, contact-any/body-count 0.051717/0.053792 per rollout step,
measured history 1.998006, denoise steps 2, observed replan interval 23.743694,
and scan anchor XY/yaw errors 0/1.46368e-7 rad.

The corrected fixed-L1, seed-42 evaluation gives this fair-to-final recovery:

| metric | D fair `12500` | D final `14999` |
|---|---:|---:|
| mean episode length | 415.06 | 473.86 |
| success / fall | 54% / 17% | 82% / 4% |
| flat / box / stair / hurdle success | 96 / 24 / 92 / 4% | 100 / 84 / 92 / 52% |
| episode any-contact OR | 77/100 | 99/100 |
| contact-step fraction | 1,078/41,506 = 2.597% | 2,440/47,386 = 5.149% |
| undesired-contact bodies/step | 1,182/41,506 = 0.02848 | 2,516/47,386 = 0.05310 |

Thus Stage-9 checklist state is: actual state `[x]`, actual terrain `[x]`,
frozen generator `[x]`, 2-step replan `[x]`, episode-length recovery `[x]`,
obstacle success increase `[x]`, combined fall/collision decrease `[ ]`, and a
causal tracker-correction case `[ ]`. Falls improved, but all contact measures
rose. The last checkbox stays open because the six validated exemplars are
body-origin height proxies, not collision geometry or evidence of policy
intent.

The six exemplar visual sets (NPZ plus `_visual.json`/`_visual.png`) are A,
D-fair, and E under `.../corrected_seed42_v1/common_l1/`, D-final under
`.../d_final_l1/`, and D-final 30/60 cm under the two `stress_*` directories.
The D-final box exemplar has reference left-ankle proxy 0.0441646 m and robot
proxy zero (reference z 0.0358449 m, queried terrain z 0.0800095 m). Reference
and robot query XY differ, so it is not a clean-crossing or causal-correction
case. B/C/F/rough had no threshold-qualified frame.

## Stage 10: completed ablation and evaluation (2026-07-14, docs/motion_generator_stage10_ko.html)

All eight Stage-10 result checklist items are complete in the narrow sense that
the planned runs, exact quotas, artifacts, and latency output exist. Completion
does not mean every hypothesis was supported. The common comparison uses the
same Stage-5 BASE, SHA-256
`fafb20e89962935d4c5980a6531c9cc538cf71fea541496a8723995e48c4819b`.
A/D/E/F use 501 logged iterations; B/C use zero updates. Corrected checkpoints
are:

- fixed-reference baseline `[x]`; generator-only baseline `[x]`;
  fine-tuning ablation `[x]`; tracker-terrain-observation ablation `[x]`;
  heading-reward ablation `[x]`; 30/60 cm evaluation `[x]`; unseen rough
  evaluation `[x]`; RTX 4090 two-step inference latency `[x]`.

- A:
  `logs/WholeBodyTracking/20260713_142246-g1_29dof_wbt_ablation_a_fixed_reference_terrain_ft-locomotion/model_12500.pt`,
  SHA-256 `643c00d26bca38e72781bd9ec631ac3345835e171f46b09bb2d17257360521d9`
- D fair/final: paths and hashes listed in Stage 9 above
- E:
  `logs/WholeBodyTracking/20260713_143609-g1_29dof_wbt_ablation_e_generator_terrain_only_ft-locomotion/model_12500.pt`,
  SHA-256 `96928c758b51d2c577b3d80ff548e213c1214c9ddf68b028cfd0f83bb39a2f4b`
- F successful retry:
  `logs/WholeBodyTracking/20260713_145931-g1_29dof_wbt_ablation_f_no_heading_reward_ft-stage10_corrected_seed42/model_12500.pt`,
  SHA-256 `7b398406e2799bbfb895e17c899419b4871e8613aa7a9050520238fe1b086595`

The first F attempt `20260713_145342-*` stopped after about four minutes with
only `model_12000.pt`: W&B offline video tried to launch a missing `ffmpeg` and
raised `FileNotFoundError: 'ffmpeg'` / `Video encoding failed`. The retry set
`logger.video=false` and completed. The failed run and all old pre-fix
A/D/E/F artifacts are integrity-only and excluded from performance tables.

All outputs are rooted at
`logs/terrain_evaluations/stage10/corrected_seed42_v1/`. Ten evaluator calls
exited zero with `complete=true` and 100/100 episodes. Common A-F, D-final,
30 cm, and 60 cm each use exactly 25 flat/box/stair/hurdle episodes; rough uses
100 rough episodes. The preregistered primary protocol uses fixed L1,
evaluation seed 42, training seed 42, uniform valid phase, XY reanchor, horizon
500, deterministic generator and per-env sampling, and generator seed 0.
Evaluation seeds 43 and 44 were added later with the same policy checkpoints
and protocol. `torch_deterministic=false`, so no bitwise claim is made; there
are three evaluation seeds but only one training seed. Timeout records 501
simulator steps while the command advances 500 times before reset.

The fresh strict 10-source/5-group report has no protocol mismatch and is:

- `report/stage10_corrected_seed42_v1_contact_clarified.json`, SHA-256
  `61984e070a206c728a1c9c407342ceda9413a79e86b27529d27aed94ed977fe1`
- `report/stage10_corrected_seed42_v1_contact_clarified.csv`, SHA-256
  `89002781ae0961c3707d626c7c6cee33b0f7bbd000799827a1777339a5005013`
- `report/stage10_corrected_seed42_v1_contact_clarified.md`, SHA-256
  `f92e31354c10ec8a43182a87896090ae31805b9894923e74bf66a15ebcc4d21f`

The earlier `report/stage10_corrected_seed42_v1.{json,csv,md}` prefix is
superseded because its contact-column wording is ambiguous; final citations
must use the `_contact_clarified` prefix above.

Contact columns are deliberately distinct: episode any-contact is an episode
OR; contact-step fraction is contact-positive recorded steps / all recorded
steps; bodies/step is the sum of active undesired-contact bodies / all recorded
steps. None is a mesh-penetration or clean-crossing metric.

| variant | success | fall | survival | episode contact | contact steps | heading rad | body err m | progress m | length |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 30% | 10% | 52% | 93% | 919/37,154 = 2.474% | 1.229 | 0.0556 | 1.336 | 371.54 |
| B | 0% | 0% | 98% | 50% | 178/49,170 = 0.362% | 1.464 | 0.0260 | 0.224 | 491.70 |
| C | 0% | 0% | 98% | 55% | 199/49,170 = 0.405% | 1.472 | 0.0262 | 0.207 | 491.70 |
| D fair | 54% | 17% | 58% | 77% | 1,078/41,506 = 2.597% | 0.467 | 0.0412 | 2.345 | 415.06 |
| E | 66% | 7% | 70% | 69% | 409/43,874 = 0.932% | 0.474 | 0.0418 | 2.247 | 438.74 |
| F | 0% | 0% | 99% | 62% | 179/49,632 = 0.361% | 1.555 | 0.0206 | 0.082 | 496.32 |

C-to-D shows that fine-tuning changes directed progress/success, but also
raises falls/contact. In the seed-42 primary row E exceeds fair-budget D (66%
vs 54% success, 7% vs 17% fall, and lower step contact); evaluation seeds 43
and 44 retain the same success direction. These results do **not** show a
tracker-terrain-scan benefit, but one training seed cannot prove the scan is
harmful. F survives but is stationary (seed-42 progress 0.082 m, heading error
1.555 rad, 0% success), making heading reward essential for directed
locomotion under this protocol. A uses one fixed clip rather than B-F's
random-heading command distribution, limiting its absolute comparison; 10.26%
of its valid windows naturally cross a tile boundary even after reset
reanchoring.

D-final fixed-L1 is 82% success, 4% fall, 89% survival, 99/100 episode contact,
2,440/47,386 = 5.149% contact steps, 2,516/47,386 = 0.05310 bodies/step,
0.465 rad heading error, 0.0411 m body error, 2.052 m progress, and 473.86 mean
steps. Terrain success is flat/box/stair/hurdle 100/84/92/52%.

Stress results are:

- 30 cm all-four L0: 72% success, 6% fall, 86% survival, 100/100 episode
  contact, 4,180/48,304 = 8.654% contact steps, 2.060 m progress. Hurdle alone
  is 0/25 success, 6/25 fall, and 22.462% contact steps.
- 60 cm all-four L0: 69% success, 1% fall, 96% survival, 100/100 episode
  contact, 6,163/48,837 = 12.620% contact steps, 1.984 m progress. Hurdle alone
  is 0/25 success, 0/25 fall, and 39.571% contact steps.
- unseen rough L9: 8% success, 25% fall, 59% survival, 100/100 episode contact,
  5,225/38,378 = 13.615% contact steps, 1.801 rad heading error, 0.067 m body
  error, and 0.820 m progress. This is a collapse.

Terrain-specific success/fall/contact-step percentages for
D-final L1 / 30 cm L0 / 60 cm L0 are: flat
`100/0/3.154 / 100/0/2.371 / 96/0/2.435`; box
`84/8/2.855 / 96/0/5.908 / 84/0/6.149`; stair
`92/0/7.160 / 92/0/5.852 / 96/4/2.952`; and hurdle
`52/8/7.809 / 0/24/22.462 / 0/0/39.571`.

Thus the overall 30/60 cm success comes from flat/box/stair and is not evidence
of a clean hurdle crossing. Fixed 0.30/0.60 m geometry, L0/L9, 100 episodes,
four envs, and 1.5 m threshold are preregistered implementation/evaluation
choices, not paper values.

The RTX 4090 module-only result is
`latency/D_final_pytorch_cuda.json` (PyTorch 2.7.0+cu128, float32, batch one,
two DDIM steps, warm-up 50, measured 500). CUDA-event mean/median/p95 are
3.25656/3.18248/3.64224 ms for the generator,
0.07823/0.07434/0.08369 ms for the tracker, and
3.35356/3.28349/3.67616 ms sequentially. The generator-every-25-steps estimate
is 0.2084916 ms/policy step. This excludes Isaac Sim, raycasts, observation and
reference assembly, transfers/bus, a functional connected closed loop,
ONNX/TensorRT, and deployment end-to-end latency.

W&B project/group is `hkleetony-dyros/WholeBodyTracking` /
`stage10_corrected_seed42`: [D `8kh28k5w`](https://wandb.ai/hkleetony-dyros/WholeBodyTracking/runs/8kh28k5w),
[A `9i00geyd`](https://wandb.ai/hkleetony-dyros/WholeBodyTracking/runs/9i00geyd),
[E `hkqexdqe`](https://wandb.ai/hkleetony-dyros/WholeBodyTracking/runs/hkqexdqe),
[successful F `qrk3845o`](https://wandb.ai/hkleetony-dyros/WholeBodyTracking/runs/qrk3845o),
[evaluation `jcrms4zx`](https://wandb.ai/hkleetony-dyros/WholeBodyTracking/runs/jcrms4zx),
[artifact `stage10-corrected-evaluation-seed42:v0`](https://wandb.ai/hkleetony-dyros/WholeBodyTracking/artifacts/evaluation/stage10-corrected-evaluation-seed42/v0),
and [24-file analysis artifact v0](https://wandb.ai/hkleetony-dyros/WholeBodyTracking/artifacts/evaluation-analysis/stage10-corrected-evaluation-seed42-analysis/v0).
The analysis resume preserved all 387 existing summary keys and added 21
contact/report/proxy keys; it includes all six NPZ/JSON/PNG visual sets. The
failed F offline id `x3xmyiya` was not synchronized. Commit `6c7aacc` makes the
three contact denominators explicit in code and fresh reports.

### Evaluation-seed 43/44 addendum and raw pooling

The seed-42 tables above remain the preregistered primary result. The same
training-seed-42 checkpoints were evaluated again with evaluation seeds 43 and
44. Each seed has ten complete 100-episode sources: four-terrain identities
have exact 25/25/25/25 quotas and rough has 100 episodes. Across seeds 42–44,
the pooled report therefore covers 30 sources and 3,000 episodes, with 75
episodes per terrain/identity or 300 rough episodes. Pooling sums raw
numerators and denominators; it never averages per-seed rates.

| variant | pooled success | per-seed success range | episode any-contact | contact-step raw |
|---|---:|---:|---:|---:|
| A | 76/300 (25.3%) | 22–30% | 269/300 | 2,802/113,174 (2.48%) |
| B | 0/300 | 0–0% | 155/300 | 558/146,513 (0.38%) |
| C | 0/300 | 0–0% | 158/300 | 584/146,514 (0.40%) |
| D fair | 167/300 (55.7%) | 54–57% | 242/300 | 2,802/123,953 (2.26%) |
| E | 198/300 (66.0%) | 64–68% | 219/300 | 1,276/129,297 (0.99%) |
| F | 1/300 (0.3%) | 0–1% | 185/300 | 750/146,953 (0.51%) |
| D final L1 | 229/300 (76.3%) | 73–82% | 297/300 | 8,845/140,944 (6.28%) |
| D final 30 cm | 214/300 (71.3%) | 68–74% | 299/300 | 13,763/141,651 (9.72%) |
| D final 60 cm | 204/300 (68.0%) | 65–70% | 295/300 | 22,190/145,524 (15.25%) |
| D final rough L9 | 26/300 (8.7%) | 8–10% | 298/300 | 15,548/115,790 (13.43%) |

Episode any-contact is still an episode OR; contact-step is a
duration-normalized raw step fraction. The pooled JSON/CSV separately retains
undesired-contact bodies/step numerator and step denominator. These columns do
not establish clean crossings.

All five success contrasts keep the same sign for all three evaluation seeds:

| contrast | seed-delta range | raw-pooled delta |
|---|---:|---:|
| D fair minus C | +54–57 percentage points | +55.7 pp |
| D fair minus F | +54–57 pp | +55.3 pp |
| E minus D fair | +7–12 pp | +10.3 pp |
| D final minus D fair | +16–28 pp | +20.7 pp |
| D final L1 minus rough | +63–74 pp | +67.7 pp |

Thus fine-tuning, heading reward, longer fine-tuning, E-over-D, and rough
collapse are directionally stable to these evaluation seeds. This does not
measure policy training variance: all evaluated checkpoints still come from
one training seed. F remains effectively stationary at 1/300 pooled success,
and E-over-D still fails to demonstrate tracker terrain-scan benefit.

The added output sets and SHA-256 values are:

- seed 43 prefix
  `logs/terrain_evaluations/stage10/corrected_eval_seed43_v1/report/stage10_corrected_eval_seed43_v1_contact_clarified`:
  JSON `fd78092e46b1a9fdcfecfa715a6022c73c17da9901ae30e17519714c545aa34c`,
  CSV `01ded5e633fd4f17486b5f1b60ee3789832198e96a971bac55e742e1e8254b65`,
  Markdown `2812e84c8f80a545ad75a538c2fecdb138aa7f8669f1c7980e0f0e6b153505f7`.
- seed 44 prefix
  `logs/terrain_evaluations/stage10/corrected_eval_seed44_v1/report/stage10_corrected_eval_seed44_v1_contact_clarified`:
  JSON `d877342278fe59719fc0d427a0fb8e2dd4d07e19fa2bb7372d96af0ad0fa1357`,
  CSV `205094147c582c2d8c2c7aa406ed32327fcc570992719570bec9762e8792681e`,
  Markdown `57be995d918fc49aba736221819bbd27bff7f86a1f593a13cebebf2cefe24159`.
- raw-pooled prefix
  `logs/terrain_evaluations/stage10/corrected_eval_seeds42_44_summary/stage10_corrected_eval_seeds42_44_pooled_raw`:
  JSON `fa1202894d95dabe4ebc5eb521bbc7aaa63ab8076e5094721164fabb073f8997`,
  CSV `74254648effe0683cd0941facf3d94a28ec37451fad4470d703479e499d6bbac`,
  Markdown `8a6105a45ce88935868238f3688bf686ea70f0de855f8ca66a056fb726f2df7c`.

W&B provenance is [seed-43 run `qbqazh4x`](https://wandb.ai/hkleetony-dyros/WholeBodyTracking/runs/qbqazh4x)
with [39-file results artifact v0](https://wandb.ai/hkleetony-dyros/WholeBodyTracking/artifacts/evaluation/stage10-corrected-evaluation-seed43-results/v0),
and [seed-44 run `i644g8ug`](https://wandb.ai/hkleetony-dyros/WholeBodyTracking/runs/i644g8ug)
with [42-file results artifact v0](https://wandb.ai/hkleetony-dyros/WholeBodyTracking/artifacts/evaluation/stage10-corrected-evaluation-seed44-results/v0).
Both are finished in the same project/group. The seed-44 artifact metadata
records seeds `[42,43,44]`, 30 sources, and 3,000 episodes, and its `combined/`
directory contains the pooled JSON/CSV/Markdown. Latency is neither pooled nor
rerun; the seed-42 RTX 4090 module benchmark remains the latency result.

## Known limitations / not yet verified

- Corrected terrain-aware closed-loop training and all Stage-10 rows completed.
  Evaluation seeds 42/43/44 provide 3,000 pooled episodes, but every evaluated
  checkpoint still comes from training seed 42. Evaluation replication does
  not capture training-seed variance, so A-F differences and D fair-to-final
  recovery have no multi-training-seed confidence interval.
- E outperforms fair-budget D, so tracker terrain observation benefit is not
  demonstrated. F's near-stationary survival shows heading reward is necessary
  for directed locomotion in this protocol, not necessarily every protocol.
- D-final raises obstacle success and lowers falls, but episode/step/body
  contact all increase from D fair. The combined fall/collision checklist stays
  open. Episode contact is an OR, step contact is duration-normalized, and
  neither proves or disproves a clean crossing.
- In the seed-42 primary row, 30/60 cm hurdle success is 0/25 with
  22.462%/39.571% contact-step fractions. Three-seed pooling gives 30 cm hurdle
  2/75 success with 23.92% contact steps, 60 cm hurdle 0/75 with 41.95%, and
  unseen rough only 26/300 success. Terrain closed-loop behavior is therefore
  not robust to these stresses.
- Six exemplar visuals pass schema/hash/finite/scan validation but remain
  body-origin height proxies. Collision-shape/mesh intersection, contact-force
  replay, and policy-intent causality are not verified.
- 2-step DDIM quality is validated offline, used in closed-loop training, and
  benchmarked module-only on RTX 4090. Simulator end-to-end, ONNX/TensorRT, and
  deployment latency/export remain unverified.
- Paperscale generation reaches 0.0546 m full-val MPJPE (terrain checkpoint
  0.0393 on its earlier fixed validation batch). Stage-8 fixed-history stress
  tests show a real terrain-conditioned response, but foot/knee direction and
  tall-obstacle body-origin penetration proxy remain poor.
- The 50k-step baseline default was not itself re-run end-to-end after being
  reduced from the measured 200k run (same code path, shorter schedule).
- Generated `*_gen_qpos.npz` → full WBT npz conversion re-runs MuJoCo FK; body
  orientations/velocities of the generated motion come from FK + finite
  differences, not from the model.
- Generator ONNX export not attempted; the tracker is exported to ONNX at every
  checkpoint. The generator uses only standard ops
  (Linear/LayerNorm/TransformerEncoder/sinusoidal embeddings), no known
  blockers, but unverified.
- Pre-fix Stage-9/A/E/F and the first failed F artifacts are retained only for
  integrity/provenance. MuJoCo sim-to-sim and real G1 verification remain.

Before running any A-F simulator result, the common primary terrain row was
fixed at curriculum level 1. In the fair-budget D snapshot (`model_12500.pt`),
the obstacle current-level counts were box L0/L1=175/81, stair=200/56, and
hurdle=191/65; no obstacle env was currently at L2, and cumulative L2 episode
counts were only 20/10/2. L2+ is therefore treated only as extrapolation
stress. Flat geometry is level-invariant, so its high level is not evidence of
obstacle competence. This is an implementation/evaluation choice, not a
paper-specified value, and was recorded before inspecting A-F evaluation
results.

## Technical risks for the next stage

1. 29-DoF vs paper's 23-DoF: a future tracker trained on paper-style 23-DoF
   references would need a remap; keeping 29-DoF end-to-end avoids this.
2. Active undesired contact is a contact-force threshold from the reward term,
   while body-origin penetration is a separate height proxy. Neither is exact
   collision-mesh penetration; geometry/contact replay is still needed.
3. Receding-horizon distribution shift: structured condition noise is now
   implemented and narrows the matched noisy-validation gap. Corrected
   closed-loop training and three-evaluation-seed fixed-level evaluation
   completed, but multi-training-seed replication and tall-hurdle/rough
   robustness are still required before calling it converged or robust.
4. LAFAN1 license (CC BY-NC-ND) forbids redistribution of retargeted
   derivatives — generated models trained on it inherit non-commercial terms.
