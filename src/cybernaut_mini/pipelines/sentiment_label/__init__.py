"""sentiment_label pipeline: stories → labeler-pool matrix → reports."""

from cybernaut_mini.pipelines.sentiment_label.pipeline import create_pipeline

__all__ = ["create_pipeline"]
