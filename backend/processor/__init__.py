"""
DeepWiki Code Processor — standalone wiki generation pipeline.

Usage:
    python -m backend.processor.code_processor --repo=URL --branch=main --mode=local
    python -m backend.processor.code_processor --config=run.json
"""

# Lazy imports to avoid RuntimeWarning when running with python -m
# (importing code_processor here would cause it to appear in sys.modules
# before its __main__ block executes)
__all__ = ["main", "run_code_processor"]


def __getattr__(name):
    if name in __all__:
        from backend.processor.code_processor import main, run_code_processor
        return {"main": main, "run_code_processor": run_code_processor}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
