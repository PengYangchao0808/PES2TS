"""Offline toolchain of the predictor product (data, splits, labels, training,
evaluation, experiments, publishing).

Everything under ``pesn.offline`` is development-side only: it may depend on
the runtime set (``api``/``contracts``/``features``/``models``/``inference``/
``bundle``) and on existing G0/G1 read-only contracts through adapters, but
no runtime module may import anything from this subpackage.
"""

from __future__ import annotations
