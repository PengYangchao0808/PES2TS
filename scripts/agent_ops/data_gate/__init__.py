"""Data-gate tooling for the PES2TS 30h R3 campaign (dry-run lane A).

Submodules
----------
``r3_table_schema``
    Declarative column/dtype/key schemas for the six R3 tables plus the
    split/data/teacher manifests (30h plan v1.1 §3.1, §7.1).
``validate_r3_tables``
    CLI that machine-checks Data Gate tests T0.1a/b/c and T0.4a/b.

This package never reads ``data/``, ``outputs/`` or ``artifacts/``; it only
describes and validates table layouts given an explicit ``--tables-dir``.
"""
