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
Remaining work is to tune the clean/robustness trade-off and train the Stage-9
tracker closed-loop against this frozen generator. Real collision/contact and
obstacle crossing must be measured in simulation rather than inferred from
the Stage-8 body-origin proxy.

## Stage 9: terrain closed-loop PPO wiring and gate (2026-07-13, docs/motion_generator_stage9_ko.html)

`exp:g1-29dof-wbt-gen-terrain` closes the intended loop without changing the
Stage-5 flat preset: each 50 Hz policy step packs measured simulator state and
the shared `(N, 289)` local terrain scan, non-bootstrap replans require two
measured history frames, and every 25 control steps the frozen robust+FK
generator produces a 25-frame reference with 2-step DDIM. The tracker consumes
five-frame proprioception plus the current reference/scan; only tracker
actor/critic parameters enter PPO. A runtime check fails if any generator
parameter is trainable. Bootstrap uses the seed clip's two frames once; all
later condition-history frames are simulator measurements.

Curriculum outcome state is schema v3 with
`progress_semantics="target_heading_signed_v1"`. Each episode captures its
start XY and normalized target heading, then gates crossing on the maximum
non-negative signed projection along that fixed heading. Success requires a
comparable complete episode, no fall/non-timeout termination, at least 90% of
the 500-step timeout, and at least 1.5 m signed progress. Lateral/backward
travel therefore cannot count as a crossing. Version-2 radial-progress state
cannot be converted: migration preserves durable levels/streaks and resets
crossing/progress state. These gates, thresholds, ten levels, and obstacle
geometry are implementation choices, not published paper values.

The 64-env Isaac gate at
`logs/WholeBodyTracking/20260713_100144-g1_29dof_wbt_gen_terrain_manager-locomotion/model_12001.pt`
(SHA-256
`d4fe42db0349d630fb4ec9d58979cc37867708fb3d68d2913e216c177264ad41`)
completed two PPO iterations, crossed a non-bootstrap replan boundary, loaded
the optimizer, expanded the Stage-5 actor/critic inputs from 154/286 to
702/834, preserved equal 16-env quotas for flat/box/stair/hurdle, saved v3
curriculum state, and logged zero trainable generator parameters. This is a
wiring/checkpoint-migration gate, not evidence of recovery or obstacle
performance.

The 1,024-env, 3,000-update production run at
`logs/WholeBodyTracking/20260713_100222-g1_29dof_wbt_gen_terrain_manager-locomotion/`
completed naturally after its final save at 2026-07-13 20:38:34 KST. The final
tracker is `model_14999.pt` (SHA-256
`7e927e83681e5a5beef73c15dda311faa91fd543cec869a1c2559346b80af10a`),
with both saved iteration fields equal to 14,999. The frozen generator was
`logs/motion_gen/terrain_robust_fk_4090/checkpoints/final.pt` (SHA-256
`7c63764b771ec43fb5d463d77b6860eee8e46f1db5c0b196554f98cc527ed5fa`):
runtime recorded step 10,000, zero trainable generator parameters, simulator
scan input, a 25-control-step replan interval, two denoise steps, random
episode heading, and the fully-measured-history guard. The serialized config
agrees (`replan_interval_s=0.5`, `use_sim_terrain_scan=true`,
`require_fully_measured_history=true`, `past_noise_std=0.0`). No
Traceback/ERROR/OOM/Killed/NaN/Inf marker was present in `train.log`.

A CPU finite audit of the final checkpoint found zero non-finite values in
the model (17 tensors / 1,119,803 elements), optimizer (51 / 2,239,623),
normalizers (8 / 4,610), and curriculum+environment state (22 / 13,533). The
actor/critic input widths are 702/834. At TensorBoard step 14,999, all 488
expected scalar tags were present exactly once and finite. Exact selected
values were: environment/train mean episode length 441.062744/443.559998,
train reward 21.556921, relative-body/global-ref-position/global-ref-orientation/
body-linear-velocity rewards 0.853199/0.419497/0.390161/0.678832,
body-position/reference-position/reference-rotation errors
0.047850 m/0.055799 m/0.133334 rad, heading error 0.629092 rad, heading raw
alignment 0.707746, episode heading reward 0.611144, current signed progress
1.221730 m, contact-any 0.056315, contact-body count 0.067627, mean policy
noise 0.382510, KL 0.010113, measured-history valid count 1.997396, and two
denoise steps. The observed replan interval averaged 23.353069 steps because
episode-reset boundaries shorten it. The correction-case/proxy-improvement
values 0.113688/0.005820 m remain body-origin proxies, not collision geometry.

The final curriculum state is schema v3 with
`progress_semantics="target_heading_signed_v1"`, 1,024 envs, and exactly 256
envs per terrain type. Current L0..L9 distributions (mean level) were flat
`[0,0,0,0,0,0,0,0,7,249]` (8.972656), box
`[17,71,87,54,16,9,2,0,0,0]` (2.062500), stair
`[7,55,81,73,36,2,2,0,0,0]` (2.351562), and hurdle
`[100,134,22,0,0,0,0,0,0,0]` (0.695312). Cumulative
episode/success/survival/crossing/failure counts were respectively
flat 38,654/34,252/35,521/35,187/4,402; box
42,978/30,597/33,259/33,195/12,381; stair
46,166/32,610/34,740/33,621/13,556; and hurdle
42,676/29,601/32,832/32,700/13,075. These adaptive-training aggregates mix
levels and time, and flat geometry is level-invariant; they are not fixed-level
performance estimates. From step 12,500 to 14,999 the rollout contact-any
value changed from 0.055827 to 0.056315, so contact reduction in particular
cannot be claimed. Episode recovery, fall/contact reduction, obstacle-success
improvement, and a causal tracker-correction case remain unchecked pending the
common Stage-10 protocol.

The production artifact above predates the terrain-reset alignment correction.
It remains valid evidence that the 3,000-update process completed naturally,
the generator stayed frozen, the closed-loop dataflow ran, and all audited
tensors/scalars were finite. It is **superseded for terrain performance**:
episode starts inherited raw phase XY instead of being centered on their
assigned 8 m tile. The corrected D run at
`logs/WholeBodyTracking/20260713_124451-g1_29dof_wbt_ablation_d_full_terrain_ft-locomotion/`
is currently in progress, so no final checkpoint result or performance box is
recorded. Its startup `model_12000.pt` env state already provides an actual
reanchor check: across 1,024 starts, maximum absolute x/y offsets from tile
center were 0.049998/0.049995 m, maximum radius was 0.069307 m, and both
radius>1 m and outside-4 m counts were 0/1,024. Only configured reset noise
remains.

The first actual D-final common-evaluator smoke exposed that Isaac Sim returns
`robot_root_states` as a `RootStatesProxy`, which has no direct tensor dtype.
Commit `4f6b71d` slices the proxy before reading the canonical tensor metadata
and fixes eval-config override propagation. Retry 3 then exited successfully
and wrote `complete=true` JSON/CSV with the exact four-episode quota (one each
for flat/box/stair/hurdle at fixed level 1) under
`logs/terrain_evaluations/stage10/smoke/D_final_l1_retry3*`. All four tiny
smoke episodes terminated for bad tracking after 8/8/14/8 steps and none
succeeded. This verifies the real simulator evaluation path, quota, terminal
capture, and serialization only; it is not a performance result. Follow-up
audit also found that it used the legacy forced phase zero (a nearly stationary
seed segment), batch-dependent deterministic DDIM noise, and no motion XY
reanchoring, so it is superseded even as an evaluation protocol example.

Commits `32546da` and `7edf0e1` make Stage-10 evaluation sample a uniform valid
phase and derive DDIM/condition noise independently from env, episode, and
replan identity, with batch-order/subset/reset regression coverage. Commits
`884a26b` and `d748e46` add terrain XY reanchoring and command-specific phase
horizons. The corrected-protocol real Isaac smoke under
`logs/terrain_evaluations/stage10/smoke/D_legacy_tracker_l1_corrected_protocol*`
completed exactly four level-1 episodes (one flat/box/stair/hurdle) and recorded
3/4 successes. This is only a four-episode protocol smoke using the legacy
misaligned-trained `model_14999.pt`, not final terrain performance or an
improvement claim. Its saved correction exemplar is likewise only a body-origin
proxy from that legacy tracker, not the causal Stage-9 completion case.

HoloSoma's timeout predicate is `episode_length_buf > max_episode_length`.
Consequently the evaluator accumulator records a default timeout episode as
501 steps, while the command advances exactly 500 times before reset; the
correct fixed-reference phase horizon is therefore 500. Commit `523b197`
changes only the one-env `max_eval_steps` upper bound to 501 steps per episode.
Its focused suite reported 15 passed and Ruff clean; the broader reset/evaluator
suite reported 88 passed and Ruff clean.

## Stage 10: ablation and evaluation infrastructure (implementation-only, docs/motion_generator_stage10_ko.html)

Six current-config experiment presets define the reduced comparison from the
same Stage-5 BASE `model_12000.pt`: A fixed generated reference with tracker
terrain/history and heading reward; B terrain-blind online generator with the
legacy tracker and no heading reward/update; C full terrain architecture with
no fine-tuning; D full terrain fine-tuning; E generator terrain only (tracker
scan removed); and F full terrain setup with heading-reward weight zero. A, D,
E, and F use the same 501 logged-update reduced budget; B/C use zero new
updates. Pre-fix A (`20260713_114424-*`), D (`20260713_100222-*`), E
(`20260713_115808-*`), and F (`20260713_121418-*`) completed and produced
integrity-valid checkpoints, but all are superseded for terrain performance by
the reset-alignment finding. Corrected D (`20260713_124451-*`) is in progress;
corrected A/E/F training and all A-F common performance evaluation remain
pending.

A is intentionally based on the Stage-4 fixed generated-reference NPZ. Its
heading is inferred from root XY velocity, then a 0.5 s displacement, then
root yaw; the trajectory is not randomly yaw-rotated. Rotating only the
heading would make positions/orientations/velocities inconsistent, while a
correct augmentation must rotate the entire trajectory. A therefore does not
share B-F's random-heading command distribution, limiting especially the
interpretation of heading comparisons. Its 997-frame raw reference has 847
frames (84.955%) outside radius 1 m and 119 (11.936%) outside a 4 m tile
half-width, with x `[-5.267, 0.175]` m and y `[-0.144, 2.501]` m. Without a
future horizon, 499 of 996 possible starts (50.10%) could hit clip end before a
500-step episode. Reanchoring plus horizon 500 removes those two protocol bugs,
but 51 of the 497 valid start windows (10.26%) naturally travel across a tile
boundary. That residual is a structural limitation of A's one fixed clip, not
a reanchor failure.

The actual 1,024-env pre-fix episode-start audit found radius>1 m / outside-tile
counts of 883/575 for legacy Stage-9 `model_12000.pt`, 865/610 for
`model_14999.pt`, 873/108 for old A `model_12500.pt`, 859/613 for old E, and
870/620 for old F. Maximum nearest-raw-phase residual was 0.0663 m, consistent
with reset noise. These exact checkpoint counts are why the old artifacts are
retained for integrity/architecture provenance but excluded from terrain
success/fall/contact comparisons.

`evaluate_wbt_terrain.py` supplies a shared fixed-difficulty protocol and
records schema-versioned JSON, summary CSV, and per-episode CSV with checkpoint
paths/hashes and raw numerators/denominators. The requested episode total must
divide evenly by the number of active terrain types; the evaluator assigns an
exact equal quota (default 100 episodes -> 25 each for flat/box/stair/hurdle)
and excludes vectorized-step overflow episodes from every denominator. It
disables adaptive curriculum/env-state restore, uses the episode-start target
heading for signed progress, and records success/fall/timeout/bad-tracking,
heading/tracking error, contact, episode length, and body-origin penetration
proxies. It now samples a uniform valid motion phase, forces reset XY
reanchoring for every A-F variant, derives a 500-command-advance horizon from
the effective default 10 s simulator config, and derives deterministic
generator noise from env/episode/replan identity so batch order, subsets, and
asynchronous resets do not change another sample. HoloSoma records timeout at
step 501, but the command advances only 500 times before reset; these are
intentionally distinct. Training/preset reanchor/horizon semantics are Stage-5
flat false/0, A true/500, B false/0, and generated terrain C/D/E/F true/0;
Stage-10 evaluation overrides every A-F variant to true/effective-500. An
opt-in `rough` type supports unseen-terrain evaluation without changing the
production four-type training distribution; fixed 0.30/0.60 m obstacle
overrides are evaluation choices and have not been run.

The evaluator can save the first threshold-qualified tracker-correction
exemplar as compressed NPZ plus JSON metadata. Qualification only means that
the reference body-origin penetration proxy is at least 0.02 m and the
reference-minus-robot proxy improves by at least 0.01 m. It is neither
collision-shape/mesh penetration nor causal evidence that the policy
intentionally corrected a bad reference. If no frame qualifies, it records
`found=false` and creates no NPZ. No common evaluation has run, so no exemplar
result exists yet. The corrected-protocol four-episode smoke did save one
proxy exemplar, but because it uses the legacy misaligned-trained tracker and
only four episodes, it is explicitly not accepted as the final causal
correction case.

`benchmark_inference_latency.py` implements warm-up and CUDA-event/host-clock
timing for batch-one 2-step PyTorch generator inference, normalized tracker
actor inference, and their sequential module calls, including checkpoint
hashes and mean/median/p95/min/max/std output. Its scope explicitly excludes
Isaac stepping, terrain raycasts, observation/reference assembly, transfers,
ONNX/TensorRT, and a functional generator-to-tracker closed loop; sequential
timing is only two modules called on one CUDA stream. No RTX 4090 benchmark has
run, so module-only GPU latency and deployment/end-to-end latency remain
pending. The A-F presets, quota/evaluator/serializer, rough opt-in, exemplar
writer, and latency helpers passed their documented CPU tests; those tests do
not constitute simulator evaluation or GPU timing.

## Known limitations / not yet verified

- Terrain-aware closed-loop dataflow, PPO startup, and a full 1,024-env,
  3,000-update production run are validated with a finite final checkpoint,
  but that pre-fix checkpoint is superseded for terrain performance because
  reset phases could start on adjacent 8 m tiles. The corrected D rerun is in
  progress. Adaptive curriculum rollouts are not a fixed-level baseline
  comparison; common-protocol episode recovery, fall/contact reduction,
  positive obstacle-success improvement, and causal tracker correction remain
  unmeasured.
- 2-step DDIM quality is validated offline and used in closed-loop training;
  a module-only PyTorch CUDA benchmark exists but has not been run. Simulator
  end-to-end, ONNX/TensorRT, and deployment latency/export remain unverified.
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
- Pre-fix Stage-10 A/E/F training completed with integrity-valid checkpoints,
  but those artifacts are superseded for terrain performance together with old
  D. Corrected A/E/F training, all preregistered A-F common evaluation, 30/60 cm
  and unseen-rough evaluation, terrain closed-loop convergence, MuJoCo
  sim-to-sim, and sim-to-real remain.

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
2. Contact proxy (height+speed thresholds) is crude; real contact labels from
   physics replay would improve foot-slide supervision.
3. Receding-horizon distribution shift: structured condition noise is now
   implemented and narrows the matched noisy-validation gap, and one terrain
   closed-loop tracker fine-tuning run completed. Fixed-level common evaluation
   and multi-seed training/evaluation are still required before calling it
   converged or robust.
4. LAFAN1 license (CC BY-NC-ND) forbids redistribution of retargeted
   derivatives — generated models trained on it inherit non-commercial terms.
