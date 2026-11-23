"""PES2TS V5.6 — deep model architecture benchmark package (new, independent).

This package is created for V5.6-A/B per
``docs/plans/PES2TS_V5_深度模型架构Benchmark方案_v2.1_20261122.md``.
It lives next to ``pes2ts_core`` and does **not** modify or depend on unstable
internals of ``pes2ts_core``.  Only NumPy/Pandas/PyArrow/Torch/PyYAML and the
Python standard library are used (no torch_geometric / mamba_ssm / einops).

Stable internal data interface lives in :mod:`pes2ts_v5.data.manifest`
(``V55DatasetSpec``).  When the V5.5 data/label contract document lands, a
thin adapter maps it onto that interface without changing downstream code.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
