# Motion Generator Implementation Notes

Staged reproduction of the diffusion motion generator and tracker from
**"Learning Whole-Body Humanoid Locomotion via Motion Generation and Motion
Tracking"** (arXiv:2604.17335), built inside HoloSoma for the Unitree G1.
Korean user manual: `docs/motion_generator_ko.html`.

## Papers and sources investigated

| Source | What was used | License |
|---|---|---|
| arXiv:2604.17335 (main paper) | Generator design facts (below). Project page https://wholebodylocomotion.github.io/ has **no public code** (checked 2026-07-10). | — |
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
LR, EMA, loss weights, normalization frame, terrain scan resolution, CFG.

| Item | Paper | This implementation | Basis |
|---|---|---|---|
| Joints | 23 (G1 23-DoF) | **29** (HoloSoma G1 29-DoF model) | current robot model; wrist DOFs included |
| Body positions | 23 links | **14 links** (HoloSoma WBT `body_names_to_track`) | aligns generated bodies with the existing tracker rewards |
| Feature dim / frame | 99 | **78** = 3+4+29+42 | consequence of the above |
| Canonical frame | unspecified | anchor = last past frame; xy→0, yaw→0, z absolute; quats wxyz sign-continuous | common practice (MDM/PARC-style) |
| Heading | "from base pose difference" | unit xy vector anchor→last future frame, fallback (1,0) below 5 cm displacement | paper gives no encoding details |
| Terrain scan | used, resolution unknown | **289-dim** (17×17 @0.1 m), x∈[-0.3,1.3], y∈[-0.8,0.8], heading-aligned absolute height; trained on 150 OmniRetarget terrain clips (Phase B) | resolution/range are implementation choices; closed-loop simulator still supplies zeros through stage 5 |
| Architecture | "MDM [26,30]" | MDM defaults 512/8/4/1024 | official MDM implementation |
| Diffusion | unspecified | DDPM T=1000 cosine, x0-prediction (eps available) | MDM convention |
| Few-step inference | 2 steps (deployment) | DDIM, any step count; **2-step validated** at 0.0475 m val body MPJPE and used by the closed-loop tracker | 50-step remains the offline evaluation protocol |
| Losses | recon + velocity + joint consistency + terrain penetration | recon split (root pos/quat/joint/body) + quat-norm + velocity + **bone-length consistency** (surrogate for FK consistency; no differentiable FK) + foot-slide (contact proxy) + flat-ground penetration | weights are choices; contact labels do not exist → contact-consistency loss disabled |
| Data | ~1 h augmented | **165.3 min, 195 clips** (paperscale); the original 6.8 min/11-clip profile remains available | implementation scale-up for a single RTX 4090 |
| Fine-tuning w/ tracker | closed loop | **implemented**: frozen generator, measured two-frame state, 0.5 s replanning, 2-step DDIM | stage 5; terrain input remains zero until stage 6 |

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

## Known limitations / not yet verified

- Terrain-aware closed-loop dataflow and PPO startup are validated, but no
  terrain policy has been trained to convergence and positive obstacle success
  is not yet measured.
- 2-step DDIM quality is validated offline and used in closed-loop training;
  deployment latency/export remains unverified.
- Paperscale generation reaches 0.0546 m full-val MPJPE (terrain checkpoint
  0.0393 on its fixed validation batch); unseen simulator terrain robustness
  remains the next test.
- The 50k-step baseline default was not itself re-run end-to-end after being
  reduced from the measured 200k run (same code path, shorter schedule).
- Generated `*_gen_qpos.npz` → full WBT npz conversion re-runs MuJoCo FK; body
  orientations/velocities of the generated motion come from FK + finite
  differences, not from the model.
- Generator ONNX export not attempted; the tracker is exported to ONNX at every
  checkpoint. The generator uses only standard ops
  (Linear/LayerNorm/TransformerEncoder/sinusoidal embeddings), no known
  blockers, but unverified.
- Terrain closed-loop convergence/evaluation, MuJoCo sim-to-sim, and
  sim-to-real remain.

## Technical risks for the next stage

1. 29-DoF vs paper's 23-DoF: a future tracker trained on paper-style 23-DoF
   references would need a remap; keeping 29-DoF end-to-end avoids this.
2. Contact proxy (height+speed thresholds) is crude; real contact labels from
   physics replay would improve foot-slide supervision.
3. Receding-horizon distribution shift: generator conditioned on its own
   outputs drifts; the paper addresses this with noise injection during
   training and closed-loop fine-tuning (not yet implemented).
4. LAFAN1 license (CC BY-NC-ND) forbids redistribution of retargeted
   derivatives — generated models trained on it inherit non-commercial terms.
