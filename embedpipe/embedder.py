"""Embedders behind one interface, plus an offline one.

The real embedder loads sentence-transformers and, with it, torch. The change
detection and batching code is what regresses, and it does not need a model, so
the model sits behind a Protocol and the tests use HashEmbedder.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

import numpy as np

from . import ConfigError

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_DIM = 384


def fingerprint(
    kind: str, *, model: str = DEFAULT_MODEL, revision: str | None = None, dim: int = DEFAULT_DIM
) -> str:
    """The identity of a vector space, derived from flags and nothing else.

    plan has to know which model a run would use, and loading a model to read
    one string back off it would make planning cost gigabytes of download. So
    the fingerprint is a pure function of the flags, and the real embedder
    checks the loaded model against the declared dim rather than reporting its
    own. A declared dim that is wrong is then an error at startup instead of a
    manifest that disagrees with the collection.
    """
    if kind == "hash":
        return f"hash-not-a-model@v1/d{dim}"
    if kind == "sentence-transformers":
        return f"{model}@{revision or 'unpinned'}/d{dim}"
    raise ConfigError(f"unknown embedder: {kind}")


@runtime_checkable
class Embedder(Protocol):
    @property
    def fingerprint(self) -> str:
        """Identity of the vector space, recorded in the manifest per document."""

    @property
    def dim(self) -> int: ...

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of unit vectors."""


class HashEmbedder:
    """Deterministic vectors from a seeded generator. No semantics at all.

    This exists so the pipeline can be run and tested end to end with no model
    download and no network. It is useful for plumbing, batching and resume
    behaviour. It is useless for retrieval, and a collection built with it will
    return nonsense for every query, which is why its fingerprint says so.
    """

    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        if dim <= 0:
            raise ConfigError("hash embedder dim must be positive")
        self._dim = dim
        self.calls = 0
        self.texts_seen = 0

    @property
    def fingerprint(self) -> str:
        return fingerprint("hash", dim=self._dim)

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        self.texts_seen += len(texts)
        out = np.empty((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            digest = hashlib.sha256(text.encode("utf-8")).digest()[:8]
            rng = np.random.default_rng(int.from_bytes(digest, "big"))
            vector = rng.standard_normal(self._dim).astype(np.float32)
            out[row] = vector / max(float(np.linalg.norm(vector)), 1e-12)
        return out


class SentenceTransformerEmbedder:
    """sentence-transformers, with the revision recorded where one is given."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        revision: str | None = None,
        dim: int = DEFAULT_DIM,
        device: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - needs the hf extra
            raise ConfigError(
                "sentence-transformers is not installed. pip install -e '.[hf]'"
            ) from exc
        self._model = SentenceTransformer(model_name, revision=revision, device=device)
        loaded_dim = int(self._model.get_sentence_embedding_dimension())
        if loaded_dim != dim:
            raise ConfigError(
                f"{model_name} produces {loaded_dim}-dimensional vectors, the run declared "
                f"{dim}. Pass --dim {loaded_dim}."
            )
        self._name = model_name
        self._revision = revision or "unpinned"
        self._dim = loaded_dim

    @property
    def fingerprint(self) -> str:
        return fingerprint(
            "sentence-transformers", model=self._name, revision=self._revision, dim=self._dim
        )

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - needs a model
        vectors = self._model.encode(
            texts,
            batch_size=len(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def build_embedder(
    kind: str, *, model: str = DEFAULT_MODEL, revision: str | None = None, dim: int = DEFAULT_DIM
) -> Embedder:
    if kind == "hash":
        return HashEmbedder(dim=dim)
    if kind == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name=model, revision=revision, dim=dim)
    raise ConfigError(f"unknown embedder: {kind}")
