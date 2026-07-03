"""Compatibility wrapper for the maintained SOC analyzer.

Older code imported ``EmlSOCAnalyzer`` from ``src.analyze``. The real
implementation now lives in ``src.analyzer``; this module keeps that public
import working without duplicating analysis logic.
"""

from src.analyzer import EmlSOCAnalyzer

__all__ = ["EmlSOCAnalyzer"]
