"""Incremental embedding pipeline: scan, hash, embed the delta, upsert, prune."""

__version__ = "0.8.2"


class EmbedPipeError(Exception):
    """Base class for every error this package raises on purpose."""


class ConfigError(EmbedPipeError):
    """Bad flags, a missing optional dependency, or a mismatched collection."""


class ManifestError(EmbedPipeError):
    """The manifest on disk cannot be trusted and must not be guessed at."""
