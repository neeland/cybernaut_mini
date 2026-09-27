"""sentiment_distil pipeline: label matrix → greedy ensemble → OLS students."""

from cybernaut_mini.pipelines.sentiment_distil.pipeline import create_pipeline

__all__ = ["create_pipeline"]
