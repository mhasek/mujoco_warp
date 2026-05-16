#!/usr/bin/env bash
# Capture renderer baselines at two historical commits via git worktrees.
#
# Usage:
#   notebooks/_capture_baselines.sh
#
# Writes to:
#   mujoco_warp/test_data/baselines/before_phase1/
#   mujoco_warp/test_data/baselines/before_spec_fix/
#
# The capture script (_capture_baselines.py) is run from the *current*
# branch (so the fixture XML stays identical across baselines), but
# PYTHONPATH is pointed at a worktree of the historical commit so the
# imported `mujoco_warp` package reflects that commit's renderer.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-$REPO_ROOT/.venv/bin/python}"
CAPTURE_SCRIPT="$REPO_ROOT/notebooks/_capture_baselines.py"
WORKTREE_PARENT="$(mktemp -d -t mjwarp_baseline_XXXXXX)"
trap 'rm -rf "$WORKTREE_PARENT"; git worktree prune' EXIT

OUT_ROOT="$REPO_ROOT/mujoco_warp/test_data/baselines"
mkdir -p "$OUT_ROOT"

BASELINES=(
  "before_phase1:origin/main"
  "before_spec_fix:707d89c~1"
)

for ENTRY in "${BASELINES[@]}"; do
  LABEL="${ENTRY%%:*}"
  REF="${ENTRY##*:}"
  WT="$WORKTREE_PARENT/$LABEL"
  OUT="$OUT_ROOT/$LABEL"

  echo "==> baseline '$LABEL' @ $REF"
  git worktree add --detach "$WT" "$REF" >/dev/null

  # Force the capture script to load mujoco_warp from the worktree, but keep
  # using the current venv's interpreter + deps.
  PYTHONPATH="$WT" "$PY" "$CAPTURE_SCRIPT" --label "$LABEL" --out "$OUT"

  git worktree remove --force "$WT" >/dev/null
done

echo
echo "done. baselines written under:"
ls -d "$OUT_ROOT"/*/
