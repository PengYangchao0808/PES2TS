"""V5.4.2 model package (WP1 lane A: unified Backbone interface + A0
adapter + parity harness; ADR-0033 §2.1–§2.2).

Public surface (frozen for WP2/WP3)::

    from pes2ts_core.pesn.models.v542 import (
        BackboneOutput, Backbone, DirectMoveDecoder,
        A0LegacyBackbone, V542Model, load_a0_strict, a0_mapping_table,
    )

    model = V542Model()                       # A0: legacy-exact forward
    out   = model(batch)                      # {u, z, q, endpoint_logit, order_score}
    move  = model.forward_move(batch)         # {u, z}

Scope discipline: this package only ADDS the V542 layer.  The legacy
``StarEGNNV3`` / heads / V541 files, campaign configs, validators and
registries stay untouched and keep loading old checkpoints bit-for-bit.
A1–A5 candidates are NOT implemented here — they arrive in WP2 building
against :class:`Backbone`/:class:`BackboneOutput`.  ``torch.compile`` is
forbidden across the whole package (ADR-0033 §2.1).
"""

from pes2ts_core.pesn.models.v542.backbone_a0 import (
    A0_BACKBONE_VERSION,
    A0LegacyBackbone,
    assert_a0_joint_invariants,
)
from pes2ts_core.pesn.models.v542.decoder import (
    W_V_PARAM,
    DirectMoveDecoder,
    assert_decoder_a_ref_frozen,
)
from pes2ts_core.pesn.models.v542.interfaces import (
    A0_ACTIVE_PARAMS_V542,
    A0_JOINT_PARAMS_V542,
    ACTIVE_PARAM_BUDGET_V542,
    A_REF_V542,
    EDGE_MSG_DIM_V542,
    FORWARD_KEYS_V542,
    FORWARD_MOVE_KEYS_V542,
    HIDDEN_V542,
    N_DECODER_PARAMS_V542,
    N_PROCESS_PARAMS_V542,
    NODE_INPUT_DIM_V542,
    V542_SCHEMA_VERSION,
    Z_DIM_V542,
    Backbone,
    BackboneOutput,
    V542BackboneOutputError,
    V542ForwardKeyError,
    V542InitError,
    V542InterfaceError,
    V542LoadError,
    V542ParamBudgetError,
    assert_active_param_budget,
    assert_a0_param_invariants,
    assert_no_torch_compile,
    validate_backbone_output,
)
from pes2ts_core.pesn.models.v542.loader import (
    A0_LOAD_AUDIT_SCHEMA,
    A0_WV_SOURCE_KEYS,
    a0_mapping_table,
    load_a0_strict,
)
from pes2ts_core.pesn.models.v542.model import (
    V542_MODEL_VERSION,
    V542Model,
    measured_params_v542_joint,
)

__all__ = [
    "A0_ACTIVE_PARAMS_V542",
    "A0_BACKBONE_VERSION",
    "A0_JOINT_PARAMS_V542",
    "A0_LOAD_AUDIT_SCHEMA",
    "A0_WV_SOURCE_KEYS",
    "ACTIVE_PARAM_BUDGET_V542",
    "A_REF_V542",
    "EDGE_MSG_DIM_V542",
    "FORWARD_KEYS_V542",
    "FORWARD_MOVE_KEYS_V542",
    "HIDDEN_V542",
    "N_DECODER_PARAMS_V542",
    "N_PROCESS_PARAMS_V542",
    "NODE_INPUT_DIM_V542",
    "V542_SCHEMA_VERSION",
    "W_V_PARAM",
    "Z_DIM_V542",
    "A0LegacyBackbone",
    "Backbone",
    "BackboneOutput",
    "DirectMoveDecoder",
    "V542BackboneOutputError",
    "V542ForwardKeyError",
    "V542InitError",
    "V542InterfaceError",
    "V542LoadError",
    "V542Model",
    "V542ParamBudgetError",
    "V542_MODEL_VERSION",
    "a0_mapping_table",
    "assert_active_param_budget",
    "assert_a0_joint_invariants",
    "assert_a0_param_invariants",
    "assert_decoder_a_ref_frozen",
    "assert_no_torch_compile",
    "load_a0_strict",
    "measured_params_v542_joint",
    "validate_backbone_output",
]
