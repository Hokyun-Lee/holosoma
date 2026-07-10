# Motion Generator Implementation Notes

First-stage reproduction of the diffusion motion generator from
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
| Terrain scan | used, resolution unknown | 121-dim (11×11 @0.1 m) interface, **zeros only (Phase A)**; encoder + penetration loss wired but never trained with real scans | no terrain scans in the data yet; marked experimental |
| Architecture | "MDM [26,30]" | MDM defaults 512/8/4/1024 | official MDM implementation |
| Diffusion | unspecified | DDPM T=1000 cosine, x0-prediction (eps available) | MDM convention |
| Few-step inference | 2 steps (deployment) | DDIM, any step count; 2-step exposed as experimental | validated path is 50-step DDIM / full DDPM |
| Losses | recon + velocity + joint consistency + terrain penetration | recon split (root pos/quat/joint/body) + quat-norm + velocity + **bone-length consistency** (surrogate for FK consistency; no differentiable FK) + foot-slide (contact proxy) + flat-ground penetration | weights are choices; contact labels do not exist → contact-consistency loss disabled |
| Data | ~1 h augmented | ~6.8 min, 11 clips | user-set first-stage scope (~10 clips, 3–10 min) |
| Fine-tuning w/ tracker | closed loop | **not implemented** (stage 2) | out of scope |

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
(batch FK conversion, joint-limit dump), `demo_scripts/prepare_motion_gen_data.sh`,
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

## Known limitations / not yet verified

- Terrain conditioning is interface-only (zero scans): **not validated**; no
  terrain-aware claims are made.
- 2-step DDIM inference runs but quality is unvalidated (experimental).
- Generation quality at this data scale is limited: best val body MPJPE
  ≈ 0.166 m over a 0.5 s horizon (11 clips ≈ 6.8 min vs the paper's ~1 h);
  the pipeline is validated, quality needs more data.
- The 50k-step baseline default was not itself re-run end-to-end after being
  reduced from the measured 200k run (same code path, shorter schedule).
- Generated `*_gen_qpos.npz` → full WBT npz conversion re-runs MuJoCo FK; body
  orientations/velocities of the generated motion come from FK + finite
  differences, not from the model.
- ONNX export not attempted; model uses only standard ops
  (Linear/LayerNorm/TransformerEncoder/sinusoidal embeddings), no known
  blockers, but unverified.
- Closed-loop tracker fine-tuning, MuJoCo sim-to-sim, sim-to-real: stage 2+.

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
