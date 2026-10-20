"""Supervisor-only enforcement tooling (WP-L2): write allowlist + deadline control.

* ``allowlist`` -- per-file merge allowlist from ``parallel_config.json``
  ``write_scopes`` + E0 additions, plus the frozen-file sha256 denylist.
* ``deadline`` -- absolute campaign window + phase ``deadline_utc`` enforcement
  (overdue phase -> ``phase:blocked``; remaining budget < eta -> refuse dispatch).
"""
