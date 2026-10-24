#!/usr/bin/env bash
# S4 plan §5.5 thread hygiene: run a command with all common numeric-library
# thread pools pinned to 1.  Used as the A/B wrapper for the B3 bench step
# (expected 5–15% on the prefetch pipeline; MUST be A/B measured, never
# assumed — see plan §5.5 and docs/infra/wsl_memory_governance.md).
#
# Usage:
#   thread_caps.sh <cmd ...>
#
# Note: the training loop's own torch intra-op threads are set inside the
# trainer (torch.set_num_threads(1)) when this experiment is on; this wrapper
# only covers the OpenMP/MKL/BLAS/NumExpr side that leaks from numpy/scipy.
set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "usage: thread_caps.sh <cmd ...>" >&2
  exit 2
fi

exec env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 "$@"
