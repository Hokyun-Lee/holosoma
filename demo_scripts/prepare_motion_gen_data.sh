#!/usr/bin/env bash
# Prepare the ~11 training motions for the diffusion motion generator:
#   download -> normalize to qpos npz -> MuJoCo FK conversion (50 fps WBT format) -> splits
#
# Uses two conda envs created by scripts/setup_all.sh:
#   hssim         (python deps for holosoma core; download/prepare/splits)
#   hsretargeting (mujoco; headless FK conversion)
# Override with HSSIM_PYTHON / HSRETARGETING_PYTHON env vars if needed.

set -e

SOURCE="${BASH_SOURCE[0]:-${(%):-%x}}"
SCRIPT_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

HOLOSOMA_DEPS_DIR="${HOLOSOMA_DEPS_DIR:-$HOME/.holosoma_deps}"
HSSIM_PYTHON="${HSSIM_PYTHON:-$HOLOSOMA_DEPS_DIR/miniconda3/envs/hssim/bin/python}"
HSRETARGETING_PYTHON="${HSRETARGETING_PYTHON:-$HOLOSOMA_DEPS_DIR/miniconda3/envs/hsretargeting/bin/python}"

for P in "$HSSIM_PYTHON" "$HSRETARGETING_PYTHON"; do
    if [ ! -x "$P" ]; then
        echo "Error: python not found at $P (run scripts/setup_all.sh or set the *_PYTHON env vars)"
        exit 1
    fi
done

DATA_ROOT="$PROJECT_ROOT/data/motion_gen"

echo "== 1/4 download selected motions =="
cd "$PROJECT_ROOT"
"$HSSIM_PYTHON" -m holosoma.motion_gen.scripts.download_data --data-root "$DATA_ROOT"

echo "== 2/4 normalize to qpos npz =="
"$HSSIM_PYTHON" -m holosoma.motion_gen.scripts.prepare_motions --data-root "$DATA_ROOT" --repo-root "$PROJECT_ROOT"

echo "== 3/4 MuJoCo FK conversion to 50 fps WBT format =="
cd "$PROJECT_ROOT/src/holosoma_retargeting/holosoma_retargeting"
"$HSRETARGETING_PYTHON" data_conversion/convert_data_format_mj_headless.py \
    --input-dir "$DATA_ROOT/raw_qpos" \
    --output-dir "$DATA_ROOT/processed" \
    --joint-limits-out "$DATA_ROOT/metadata/joint_limits.json"

echo "== 4/4 train/val splits =="
cd "$PROJECT_ROOT"
"$HSSIM_PYTHON" -m holosoma.motion_gen.scripts.make_splits --data-root "$DATA_ROOT"

echo "Done. Processed motions in $DATA_ROOT/processed"
