"""``python -m scripts.agent_ops.feature_gate`` -> harness CLI."""

from .equivariance_harness import _main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
