"""
DeepWiki Code Processor — standalone wiki generation pipeline.

Usage:
    python -m backend.processor.code_processor --repo=URL --branch=main --mode=local
    python -m backend.processor.code_processor --config=run.json
"""

from backend.processor.code_processor import main, run_code_processor

__all__ = ["main", "run_code_processor"]
