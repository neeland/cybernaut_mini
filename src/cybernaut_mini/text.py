"""Text normalization, tokenization, and entity extraction.

The regex fallback is the tested default: NFKC -> lowercase -> ``[a-z0-9]+`` tokens ->
stopword removal, no lemmatization, no entities. The spaCy path activates only when
``en_core_web_sm`` is importable and ``use_spacy`` allows it.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_WS_RE = re.compile(r"\s+")

_STOPWORDS_RAW = """
    a about above after again against all am an and any are aren as at be because been
    before being below between both but by can cannot could couldn did didn do does
    doesn doing don down during each few for from further had hadn has hasn have haven
    having he her here hers herself him himself his how i if in into is isn it its
    itself just ll m ma me might mightn more most must mustn my myself needn no nor not
    now o of off on once only or other our ours ourselves out over own re s same shan
    she should shouldn so some such t than that the their theirs them themselves then
    there these they this those through to too under until up ve very was wasn we were
    weren what when where which while who whom why will with won would wouldn y you
    your yours yourself yourselves
    """

STOPWORDS: frozenset[str] = frozenset(_STOPWORDS_RAW.split())


def normalize(text: str) -> str:
    """NFKC-normalize and collapse whitespace, preserving case."""
    return _WS_RE.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def lexical_form(text: str) -> str:
    return normalize(text).lower()


@lru_cache(maxsize=1)
def _load_spacy() -> Any | None:
    try:
        import spacy

        return spacy.load("en_core_web_sm")
    except Exception:  # any failure (missing package or model) selects the regex path
        return None


class TextProcessor:
    """Tokenizer/lemmatizer/entity-extractor with a deterministic regex fallback.

    ``use_spacy=None`` auto-detects; tests pass ``False`` to pin the fallback path.
    """

    def __init__(self, use_spacy: bool | None = None) -> None:
        self._nlp = _load_spacy() if use_spacy in (None, True) else None
        if use_spacy is True and self._nlp is None:
            msg = "spaCy model en_core_web_sm requested but not available"
            raise RuntimeError(msg)

    @property
    def backend(self) -> str:
        return "spacy" if self._nlp is not None else "regex"

    def tokenize(self, text: str) -> list[str]:
        """All lowercase alphanumeric tokens, stopwords included."""
        return _TOKEN_RE.findall(lexical_form(text))

    def content_tokens(self, text: str) -> list[str]:
        """Stopword-free tokens; lemmatized when spaCy is active."""
        if self._nlp is not None:
            doc = self._nlp(normalize(text))
            result: list[str] = []
            for token in doc:
                if token.is_punct:
                    continue
                for piece in _TOKEN_RE.findall(token.lemma_.lower()):
                    if piece not in STOPWORDS:
                        result.append(piece)
            return result
        return [token for token in self.tokenize(text) if token not in STOPWORDS]

    def entities(self, text: str) -> list[str]:
        """Normalized named entities; the regex fallback returns none."""
        if self._nlp is None:
            return []
        doc = self._nlp(normalize(text))
        return [lexical_form(ent.text) for ent in doc.ents if ent.text.strip()]
