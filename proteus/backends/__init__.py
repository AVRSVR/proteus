"""Optional heavy backends.

Nothing here is imported by the core package. Each module imports its
dependency lazily so that Proteus remains installable and testable without
PyRosetta (licence-gated) or a structure predictor (multi-gigabyte weights).
"""
