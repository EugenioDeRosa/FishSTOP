"""Compatibility wrapper for SOC flag generation."""

from src.analyzer.soc_analyzer import EmlSOCAnalyzer


def build_flags(report: dict) -> list[dict]:
    """Build SOC flags using the maintained analyzer implementation."""
    return EmlSOCAnalyzer._build_flags(report)


__all__ = ["build_flags"]
