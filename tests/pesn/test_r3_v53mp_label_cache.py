"""Round-trip tests for the V5.3-MP label cache (save == load bit-exact)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (  # noqa: E402
    CurveV52Labels,
    V52LabelTable,
    V52SampleLabel,
)
from pes2ts_core.pesn.offline.data.key_atom_endpoint_labels_v1 import (  # noqa: E402
    EndpointLabelTableV1,
    EndpointSampleLabelV1,
)
from pes2ts_core.pesn.offline.data.key_atom_label_cache_v53mp import (  # noqa: E402
    LabelCacheError,
    load_label_cache,
    save_label_cache,
)


def _tiny_tables():
    v52_samples = {}
    ep_samples = {}
    for i in range(6):
        sid = f"RXN_0000000001:{i}:RXN_0000000001_fwd:0.08"
        v52_samples[sid] = V52SampleLabel(
            sample_id=sid, curve_id="RXN_0000000001", node_index=i,
            intent_id="RXN_0000000001_fwd", direction="forward",
            fold=1, label_status="ok" if i % 4 else "terminal",
            q=None if i == 5 else -1.0 + 0.4 * i,
            q_mask=i != 5, stop=(1 if i == 3 else (0 if i < 2 else -1)),
            d_key_ts=0.01 * i)
        ep_samples[sid] = EndpointSampleLabelV1(
            sample_id=sid, curve_id="RXN_0000000001", node_index=i,
            intent_id="RXN_0000000001_fwd", direction="forward",
            fold=1, label_status="ok" if i % 4 else "terminal",
            endpoint_label=(1 if i >= 5 else (0 if i <= 1 else -1)),
            d_end_angstrom=0.05 * i, d_key_ts_angstrom=0.01 * i,
            bond_edit_complete=i >= 3, geometry_legal=True,
            ts_row=i == 3, start_row=i == 0, terminal_row=(i % 4 == 0),
            degenerate_curve=False,
            energy_ev_within_window=(None if i % 2 else True))
    curves = {
        "RXN_0000000001": CurveV52Labels(
            curve_id="RXN_0000000001", ts_index=3,
            arc_s=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
            q_canonical=(-1.0, -0.6, -0.2, 0.0, 0.5, 1.0),
            q_zero_arc_reason="",
            node_order=(0, 1, 2, 3, 4, 5)),
        "RXN_0000000002": CurveV52Labels(
            curve_id="RXN_0000000002", ts_index=1,
            arc_s=(0.0, 0.2, 0.4),
            q_canonical=None,
            q_zero_arc_reason="zero_arc_a_ts",
            node_order=(0, 1, 2)),
    }
    v52 = V52LabelTable(by_sample=v52_samples, curves=curves)
    endpoint = EndpointLabelTableV1(
        by_sample=ep_samples, positive_window=0.10, negative_window=0.30)
    return v52, endpoint


def test_round_trip_bit_exact(tmp_path):
    v52, endpoint = _tiny_tables()
    doc = save_label_cache(tmp_path, v52, endpoint,
                           dataset_manifest_sha256="abc123")
    assert doc["schema"] == "v53mp_label_cache_v1"
    v52b, epb = load_label_cache(tmp_path, dataset_manifest_sha256="abc123")
    for sid, lab in v52.by_sample.items():
        other = v52b.by_sample[sid]
        assert other.q == lab.q or (
            other.q is not None and lab.q is not None
            and abs(other.q - lab.q) < 1e-15)
        assert other.q_mask == lab.q_mask
        assert other.stop == lab.stop
        assert other.d_key_ts == lab.d_key_ts
        assert other.fold == lab.fold
        assert other.label_status == lab.label_status
        assert other.direction == lab.direction
    for cid, curve in v52.curves.items():
        other = v52b.curves[cid]
        assert other.ts_index == curve.ts_index
        assert other.node_order == curve.node_order
        assert other.arc_s == curve.arc_s
        assert other.q_canonical == curve.q_canonical
        assert other.q_zero_arc_reason == curve.q_zero_arc_reason
    for sid, lab in endpoint.by_sample.items():
        other = epb.by_sample[sid]
        assert other.endpoint_label == lab.endpoint_label
        assert other.d_end_angstrom == lab.d_end_angstrom
        assert other.bond_edit_complete == lab.bond_edit_complete
        assert other.geometry_legal == lab.geometry_legal
        assert other.ts_row == lab.ts_row
        assert other.start_row == lab.start_row
        assert other.terminal_row == lab.terminal_row
        assert other.degenerate_curve == lab.degenerate_curve
        assert other.energy_ev_within_window == lab.energy_ev_within_window
        assert other.d_key_ts_angstrom == lab.d_key_ts_angstrom
    assert epb.positive_window == 0.10 and epb.negative_window == 0.30


def test_stale_cache_refused(tmp_path):
    v52, endpoint = _tiny_tables()
    save_label_cache(tmp_path, v52, endpoint,
                     dataset_manifest_sha256="abc123")
    with pytest.raises(LabelCacheError) as exc:
        load_label_cache(tmp_path, dataset_manifest_sha256="different")
    assert exc.value.code == "CACHE_STALE"


def test_missing_cache_refused(tmp_path):
    with pytest.raises(LabelCacheError) as exc:
        load_label_cache(tmp_path, dataset_manifest_sha256="abc123")
    assert exc.value.code == "CACHE_MISSING"


def test_mismatched_coverage_refused(tmp_path):
    v52, endpoint = _tiny_tables()
    sid = sorted(v52.by_sample)[0]
    reduced = dict(endpoint.by_sample)
    reduced.pop(sid)
    endpoint = EndpointLabelTableV1(
        by_sample=reduced, positive_window=0.10, negative_window=0.30)
    with pytest.raises(LabelCacheError) as exc:
        save_label_cache(tmp_path, v52, endpoint,
                         dataset_manifest_sha256="abc123")
    assert exc.value.code == "CACHE_COVERAGE"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
