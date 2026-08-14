"""The one standardized tokenization interface the post describes.

Stage 2 of the pipeline. One class, one method, every language: NFKC normalisation ->
sentence boundary detection -> casefolding -> word segmentation -> stop-word removal ->
stemming, with the language-specific machinery (pySBD, jieba, MeCab, PyStemmer) selected
underneath and never exposed to the caller.

The acceptance test is the post's own worked example. "What lessons from bacteria and
yeast actually translate into safer gene-editing medicines?" must yield exactly
``{yeast, bacteria, safer, lesson, translat, gene, medicin, edit, actual}`` — which
pins down the whole ordering: hyphens split (``gene``/``edit``), function words are
dropped before stemming, and the survivors are Snowball-stemmed.

Determinism and safety
----------------------
The output is a pure function of ``(text, language, constructor options)`` plus the
pinned versions of pySBD/jieba/PyStemmer. No clock, no randomness, no I/O, no network.

The cached pySBD segmenters, PyStemmer stemmers and MeCab tagger are stateful C or
Python objects that are **not thread-safe** — give each thread its own tokenizer. They
are also unpicklable, so ``__getstate__`` drops the caches and ``__setstate__`` restores
empty ones; a pickled tokenizer round-trips and rebuilds its caches lazily, which also
makes the class safe to hand to ``multiprocessing`` or a Kedro parallel runner.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 2, "Multilingual
    Tokenization": "the question and expansions are tokenized. This involves sentence
    boundary detection, text segmentation, text inflection, Unicode normalization, word
    stemming, stop-word removal... We ended up creating a standardized interface that
    wraps probably a dozen or more different NLP packages and abstracts away all
    language-specific complexities." Local copy:
    ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions:
    - ``tokens`` and ``stems`` are parallel and post-stopword: element ``i`` of one is
      the surface form of element ``i`` of the other. Keeping the removed function words
      would make the two lists disagree with each other or with the post's example, and
      a caller that wants them can set ``remove_stopwords=False``.
    - ``sentences`` keeps original case and original spacing (NFKC-normalised only),
      because sentences are shown to humans and fed to snippet builders, whereas tokens
      are index keys. Only the token path is casefolded.
    - The language tag is an input, not something this stage infers: stage 1 of the
      pipeline (fastText) owns detection. ``language=None`` falls back to a script guess
      so a caller with no tag still gets sane CJK segmentation rather than mojibake.
    - Casefolding is applied to all scripts. It is a no-op for Han, kana and Hangul, and
      ``str.casefold`` (not ``lower``) is used so German ``ß``/``SS`` and Greek final
      sigma collide in the index.

Alternatives rejected:
    - A function instead of a class: simpler to call, but the expensive objects (jieba's
      prefix dictionary, a pySBD profile, a Snowball stemmer) would be rebuilt per call
      or hidden in module-level globals shared across threads. The class makes the unsafe
      state explicitly owned.
    - Returning ``list``s in the dataclass: mutable output invites a caller to mutate a
      cached value. Tuples in a frozen dataclass make the result hashable and safe to
      reuse.
    - Deduplicating tokens: the post's worked example lists each token once, but term
      frequency is exactly what BM25 needs downstream, so duplicates are kept and the
      test compares sets.
    - A ``functools.lru_cache`` over ``tokenize``: attractive for repeated queries, but
      it would pin unbounded query text in memory in a long-lived process, and the caller
      knows better than we do whether its workload repeats.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .languages import UNDETERMINED, guess_script_language, normalize_language
from .segmentation import build_sentence_segmenter, segment_words, split_sentences
from .stemming import SNOWBALL_LANGUAGES, build_stemmer, stem_words
from .stopwords import SUPPORTED_LANGUAGES as STOPWORD_LANGUAGES
from .stopwords import stopwords_for

__all__ = ["MultilingualTokenizer", "Tokenized"]


@dataclass(frozen=True, slots=True)
class Tokenized:
    """The result of tokenizing one text.

    Attributes:
        language: canonical ISO 639-1 tag actually used, or ``"und"``.
        sentences: NFKC-normalised sentences, original case.
        tokens: casefolded surface forms that survived stop-word removal.
        stems: stems of ``tokens``; identical to ``tokens`` for unstemmed languages.
            Always the same length as ``tokens``, index for index.
    """

    language: str
    sentences: tuple[str, ...] = ()
    tokens: tuple[str, ...] = ()
    stems: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.tokens) != len(self.stems):  # pragma: no cover - internal invariant
            raise ValueError("tokens and stems must be parallel")

    def __len__(self) -> int:
        return len(self.tokens)

    def unique_stems(self) -> tuple[str, ...]:
        """Stems in first-appearance order, deduplicated."""
        return tuple(dict.fromkeys(self.stems))


@dataclass(eq=False)
class MultilingualTokenizer:
    """Language-agnostic tokenizer: one interface, a dozen libraries underneath.

    Args:
        stem: apply Snowball stemming where Snowball supports the language.
        remove_stopwords: drop shipped function words before stemming.
        min_token_length: discard surface tokens shorter than this. The default of 1
            keeps single CJK characters and single letters, which carry meaning in ways
            a Latin-centric default of 2 or 3 would silently destroy.
        use_mecab: ``None`` auto-detects the optional ``cjk`` extra, ``True`` requires it
            (falling back if the import fails), ``False`` pins the bigram path so tests
            are identical on machines with and without MeCab.
        extra_stopwords: per-language additions, e.g. ``{"en": ["hereby"]}``.

    Instances are cheap to construct and expensive to share: the caches they build are
    not thread-safe. One per thread.
    """

    stem: bool = True
    remove_stopwords: bool = True
    min_token_length: int = 1
    use_mecab: bool | None = None
    extra_stopwords: Mapping[str, Sequence[str]] | None = None
    _segmenters: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _stemmers: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _stopwords: dict[str, frozenset[str]] = field(default_factory=dict, init=False, repr=False)
    _tagger: list[Any] = field(default_factory=list, init=False, repr=False)

    # -------------------------------------------------------------- public API --

    def tokenize(self, text: str, language: str | None = "en") -> Tokenized:
        """Normalise, split, segment, filter and stem ``text``.

        ``language`` accepts any spelling stage 1 might produce (``"en"``, ``"eng"``,
        ``"en-US"``, ``"English"``). ``None`` guesses from the script and only to choose
        a segmentation strategy — Latin script then yields ``"und"``, meaning no stemming
        and no stop-word list.
        """
        normalized = self.normalize(text)
        lang = self._resolve_language(normalized, language)
        if not normalized:
            return Tokenized(language=lang)

        sentences = tuple(split_sentences(normalized, self._segmenter(lang)))
        tagger = self._mecab_tagger() if lang == "ja" else None

        # Segment on ORIGINAL case: the stop-word filter needs to tell the acronym "US"
        # from the pronoun "us", and casefolding here would destroy that evidence before
        # it is ever consulted. Casefolding happens inside _filter, after the decision.
        surface: list[str] = []
        for sentence in sentences or (normalized,):
            surface.extend(segment_words(sentence, lang, tagger))

        tokens = self._filter(surface, lang)
        stems = stem_words(tokens, self._stemmer(lang)) if self.stem else list(tokens)
        return Tokenized(
            language=lang,
            sentences=sentences,
            tokens=tuple(tokens),
            stems=tuple(stems),
        )

    def sentences(self, text: str, language: str | None = "en") -> list[str]:
        """Sentence-split ``text`` only (NFKC-normalised, original case)."""
        normalized = self.normalize(text)
        lang = self._resolve_language(normalized, language)
        if not normalized:
            return []
        return split_sentences(normalized, self._segmenter(lang))

    def words(self, text: str, language: str | None = "en") -> list[str]:
        """Surface tokens only: normalise, casefold, segment, filter. No stemming."""
        normalized = self.normalize(text)
        lang = self._resolve_language(normalized, language)
        if not normalized:
            return []
        tagger = self._mecab_tagger() if lang == "ja" else None
        return self._filter(segment_words(normalized, lang, tagger), lang)

    def stems(self, text: str, language: str | None = "en") -> list[str]:
        """Convenience wrapper returning only the stems."""
        return list(self.tokenize(text, language).stems)

    @staticmethod
    def normalize(text: str) -> str:
        """NFKC-normalise and strip. Full-width ``Ｇ`` becomes ``G``, ``　`` becomes a space.

        NFKC is the compatibility form deliberately: it folds full-width Latin, circled
        and superscript digits, and the ligatures that pasted PDF text is full of onto
        their plain equivalents, so the index does not hold three spellings of "gene".
        """  # noqa: RUF002 — the fullwidth characters are the example being given.
        if not text:
            return ""
        return unicodedata.normalize("NFKC", text).strip()

    @property
    def stemmed_languages(self) -> frozenset[str]:
        """Languages this build can stem."""
        return SNOWBALL_LANGUAGES

    @property
    def stopword_languages(self) -> frozenset[str]:
        """Languages with a shipped stop-word list."""
        return STOPWORD_LANGUAGES

    @property
    def mecab_available(self) -> bool:
        """Whether the optional ``cjk`` extra supplied a usable Japanese tagger."""
        return self._mecab_tagger() is not None

    # ------------------------------------------------------------- internals --

    def _resolve_language(self, text: str, language: str | None) -> str:
        lang = normalize_language(language)
        if lang == UNDETERMINED:
            return guess_script_language(text)
        return lang

    def _filter(self, tokens: Sequence[str], language: str) -> list[str]:
        """Casefold, drop short tokens, drop stop-words — but spare acronyms.

        ``tokens`` arrive in their ORIGINAL case. An all-caps token of two or more
        characters is treated as an acronym and exempted from stop-word removal, because
        ``US``/``IT``/``WHO``/``EU``/``UN``/``AI`` are among the most frequent entities in
        a news corpus and casefolding collides every one of them with a stop-word. The
        post's stage 4 instruction template is explicitly about retrieving documents that
        "focus on the same named entities as the question", so silently deleting the
        entity is a direct hit on the pipeline's stated objective.

        The exemption is switched off when capitalisation carries no information — an
        ALL-CAPS HEADLINE makes every token look like an acronym — so the caps signal is
        only trusted when the text is not predominantly upper-case.

        Output stays casefolded. The acronym is emitted as ``us``, not ``US``, and that is
        deliberate: documents pass through this same tokenizer, so a document mentioning
        "US" also indexes ``us`` while the pronoun "us" is still removed from both sides.
        The index term therefore means "the acronym", and query and document agree.
        """
        stops = self._stopword_set(language) if self.remove_stopwords else frozenset()
        minimum = max(1, self.min_token_length)
        trust_caps = stops and not self._caps_uninformative(tokens)

        kept: list[str] = []
        for token in tokens:
            folded = token.casefold()
            if len(folded) < minimum:
                continue
            if folded in stops and not (trust_caps and self._is_acronym(token)):
                continue
            kept.append(folded)
        return kept

    @staticmethod
    def _is_acronym(token: str) -> bool:
        """``US`` and ``WHO`` yes; ``Who``, ``us``, ``A``, ``2024`` and ``細菌`` no.

        Requires at least two cased characters that are all upper-case. The cased-count
        check keeps digit- and CJK-only tokens out: ``"細菌".isupper()`` is False, but
        ``"2024".upper() == "2024"`` would fool a naive comparison.
        """
        cased = [ch for ch in token if ch.isalpha()]
        return len(cased) >= 2 and all(ch.isupper() for ch in cased)

    @staticmethod
    def _caps_uninformative(tokens: Sequence[str], *, threshold: float = 0.8) -> bool:
        """True when so much of the text is upper-case that caps stop being a signal.

        A headline like "US SANCTIONS HIT IT SECTOR" would otherwise exempt every word
        from stop-word removal. Below four cased tokens the ratio is too noisy to act on,
        so short all-caps queries keep the exemption — "US IT" should still work.
        """
        cased = [t for t in tokens if any(ch.isalpha() for ch in t)]
        if len(cased) < 4:
            return False
        upper = sum(1 for t in cased if t.isupper())
        return upper / len(cased) >= threshold

    def _stopword_set(self, language: str) -> frozenset[str]:
        cached = self._stopwords.get(language)
        if cached is None:
            extra: Iterable[str] | None = None
            if self.extra_stopwords is not None:
                extra = self.extra_stopwords.get(language)
            cached = stopwords_for(language, extra)
            self._stopwords[language] = cached
        return cached

    def _segmenter(self, language: str) -> Any | None:
        if language not in self._segmenters:
            self._segmenters[language] = build_sentence_segmenter(language)
        return self._segmenters[language]

    def _stemmer(self, language: str) -> Any | None:
        if language not in self._stemmers:
            self._stemmers[language] = build_stemmer(language)
        return self._stemmers[language]

    def _mecab_tagger(self) -> Any | None:
        if not self._tagger:
            self._tagger.append(self._build_tagger())
        return self._tagger[0]

    def _build_tagger(self) -> Any | None:
        if self.use_mecab is False:
            return None
        try:
            import MeCab  # type: ignore[import-untyped]
        except Exception:
            return None
        args = "-Owakati"
        try:
            import ipadic  # type: ignore[import-untyped]

            args = f"-Owakati {ipadic.MECAB_ARGS}"
        except Exception:  # pragma: no cover - a system dictionary may be configured
            pass
        try:
            tagger = MeCab.Tagger(args)
            tagger.parse("")  # forces the dictionary to load, so failure surfaces here
            return tagger
        except Exception:  # pragma: no cover - broken install falls back to bigrams
            return None

    # ------------------------------------------------------ pickling support --

    def __getstate__(self) -> dict[str, Any]:
        """Drop the unpicklable, thread-unsafe caches; options alone define behaviour."""
        return {
            "stem": self.stem,
            "remove_stopwords": self.remove_stopwords,
            "min_token_length": self.min_token_length,
            "use_mecab": self.use_mecab,
            "extra_stopwords": (
                None if self.extra_stopwords is None else dict(self.extra_stopwords)
            ),
        }

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.stem = bool(state.get("stem", True))
        self.remove_stopwords = bool(state.get("remove_stopwords", True))
        self.min_token_length = int(state.get("min_token_length", 1))
        self.use_mecab = state.get("use_mecab")
        self.extra_stopwords = state.get("extra_stopwords")
        self._segmenters = {}
        self._stemmers = {}
        self._stopwords = {}
        self._tagger = []
