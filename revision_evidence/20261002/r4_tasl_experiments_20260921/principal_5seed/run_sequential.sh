#!/usr/bin/env bash
set -uo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
PY=/home/njit5/anaconda3/envs/lt/bin/python
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
printf 'controller_started=%s\n' "$(date --iso-8601=seconds)" | tee "$ROOT/status.txt"
for SEED in 42 123 2024 3407 7711; do
  printf 'stage=seed%s status=running started=%s\n' "$SEED" "$(date --iso-8601=seconds)" | tee -a "$ROOT/status.txt"
  "$PY" -u "$ROOT/scripts/exp2/exp2_run.py" --config "$ROOT/configs/seed${SEED}.yaml" --no-baseline > "$ROOT/logs/seed${SEED}.log" 2>&1
  RC=$?
  printf 'stage=seed%s train_rc=%s ended=%s\n' "$SEED" "$RC" "$(date --iso-8601=seconds)" | tee -a "$ROOT/status.txt"
  if [ "$RC" -ne 0 ]; then
    printf 'pipeline_status=failed_at_seed%s_train\n' "$SEED" | tee -a "$ROOT/status.txt"
    exit "$RC"
  fi
  "$PY" "$ROOT/tests/validate_seed.py" "$ROOT/results/fashion_principal_seed${SEED}" >> "$ROOT/logs/seed${SEED}.log" 2>&1
  VRC=$?
  printf 'stage=seed%s validation_rc=%s ended=%s\n' "$SEED" "$VRC" "$(date --iso-8601=seconds)" | tee -a "$ROOT/status.txt"
  if [ "$VRC" -ne 0 ]; then
    printf 'pipeline_status=failed_at_seed%s_validation\n' "$SEED" | tee -a "$ROOT/status.txt"
    exit "$VRC"
  fi
done
printf 'pipeline_status=complete ended=%s\n' "$(date --iso-8601=seconds)" | tee -a "$ROOT/status.txt"
touch "$ROOT/COMPLETE"
