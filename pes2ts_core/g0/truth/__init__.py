"""Truth-access subpackage.

Holds the only sanctioned readers for the quarantined transition-state and IRC
artifacts.  Nothing is re-exported here: every access goes through
``truth_reader`` with an explicit ``allow_truth=True`` and is written to the
audit log.
"""
