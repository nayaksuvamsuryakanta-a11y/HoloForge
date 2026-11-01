"""Check whether fast-simplification is importable in the active venv."""
try:
    import fast_simplification as f

    print("fast_simplification version:", getattr(f, "__version__", "unknown"))
except ImportError as exc:
    print("IMPORT FAILED:", exc)