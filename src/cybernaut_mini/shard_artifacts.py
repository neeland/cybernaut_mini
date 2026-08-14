"""The three per-shard routing artifacts the blog lists but this repo did not have.

A shard in Cybernaut-1 is not just a list of document ids. The post enumerates what
lives inside one, and three of those entries are routing *filters* rather than
rankers: a bloom filter of important phrases, a trained Zstandard compression
dictionary, and the shard's vocabulary. This module builds all three as pure
functions over already-extracted text, and defines the JSON serialisation contract
that lets a shard manifest carry them.

Nothing here loads or writes an index; the builders take text in and return small
value objects out, so they can be exercised at a scale (1,000 shards) where nothing
that materialises a corpus can be.

Determinism
-----------
Every artifact is byte-identical across processes for the same input. The trap is
the bloom filter: rBloom hashes with :func:`hash` by default, and CPython salts
:func:`hash` for ``str`` and ``bytes`` with ``PYTHONHASHSEED``, so a filter built in
one process is meaningless in the next. rBloom guards its own ``save_bytes`` against
the *built-in* hash, but a caller-supplied ``lambda o: hash(o)`` slips straight
through the guard and silently produces a different bit pattern per process. This
module therefore never lets a hash function be chosen: it pins BLAKE2b-128
(:func:`stable_hash`), records that choice in the payload, and refuses to reload a
payload written under a different one.

Zstandard is deterministic on its own terms once training is single-threaded: with
``threads=0`` the COVER parameter search is a fixed computation over the samples, and
a ``dict_id`` of 0 makes zstd derive the id from the dictionary content rather than
generating a random one.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — "Bloom Filters: used to
    quickly check if phrases exist"; "Each shard has a bloom filter that keeps track
    of important phrases it contains. If a selected shard hasn't seen any of the
    user's search intents, then that shard should get downranked"; "Each shard has a
    trained Zstandard text compression dictionary. The shard that can compress the
    input question the best is more likely to contain the most similar content";
    "Vocabulary: the set of unique words in this shard" (shard 11,343 is quoted at a
    vocabulary size of 64,992). Local copy:
    ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions:
    - "Important phrases" is left undefined by the post. The builder takes whatever
      phrase strings the caller considers important — shard keywords, entities,
      titles — and only fixes how they are *normalised* (NFKC, whitespace-collapsed,
      lowercased, via :func:`cybernaut_mini.text.lexical_form`), so that a query
      intent and a stored phrase agree on spelling. Both sides must go through this
      module or the filter answers nonsense.
    - "Compress the input question the best" is read as a gain against the same text
      compressed with no dictionary at all, not as a raw compressed length. A raw
      length is dominated by the length of the question, which is identical for every
      shard in a given query but makes the number useless for comparing across
      queries; the gain is a ratio in a fixed range.
    - The vocabulary is capped (65,536 terms, just above the 64,992 the post quotes
      for a real shard) and the cap drops the *rarest* terms. A shard vocabulary is a
      recall signal for routing, and the tail of a news shard is hapax legomena — OCR
      noise, ids, mangled tokens — which carry the least routing value per byte.
    - Sample order is part of the training input: zstd's dictionary depends on it.
      Callers pass shard texts in the order the shard stores them (this repo orders
      documents by id), so the artifact is stable without this module sorting a few
      megabytes of text on every build.

Alternatives rejected:
    - Letting the caller pass ``hash_func`` through to rBloom, as rBloom's own API
      does. Rejected: it is exactly the footgun described above, and there is no
      legitimate reason for two shards in one index to hash phrases differently.
    - Pickling the ``Bloom`` object. Rejected: the payload must be JSON so a
      manifest can carry it, and a pickle is neither inspectable nor version-stable.
    - Storing the vocabulary as a hash set of digests, which is smaller. Rejected:
      the post describes the vocabulary as a readable property of a shard, the term
      strings are needed to explain a routing decision in a trace, and sorted plain
      terms compress well anyway.
    - Reusing the ``zlib``-against-a-3-title-summary approximation already in
      ``routing.py``. Rejected: it trains on nothing and measures how much a title
      string helps compress a question, which is a much weaker signal than a
      dictionary trained on the whole shard. This module is the real thing; wiring it
      into routing is a separate change.
"""

from __future__ import annotations

import base64
import hashlib
from bisect import bisect_left
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import zstandard
from rbloom import Bloom

from cybernaut_mini.models import canonical_dumps
from cybernaut_mini.text import lexical_form

#: Payload schema version. Bump when a field's meaning changes, not when one is added.
ARTIFACTS_PAYLOAD_VERSION = "1"

#: Identifier for the pinned hash function, stored in every bloom payload.
STABLE_HASH_NAME = "blake2b-128"

#: rBloom requires a hash in the signed 128-bit range; BLAKE2b at 16 bytes fills it.
_STABLE_HASH_BYTES = 16

#: False-positive rate for a shard's phrase filter. At 1% a shard that has seen none
#: of a five-intent query still answers "no" 95% of the time, which is enough to
#: downrank it, while the filter stays ~9.6 bits per phrase.
DEFAULT_FALSE_POSITIVE_RATE = 0.01

#: Upper bound on a trained dictionary. zstd trims it when the samples do not justify
#: the size, so this is a ceiling and not a fixed cost: at 1,000 shards the worst case
#: is 16 MB of dictionaries for the whole index.
DEFAULT_DICT_SIZE = 16 * 1024

#: Compression level used for both sides of the compression score. Fixed, because a
#: score compared across shards must not vary with the level, and because a
#: precomputed dictionary is bound to the level it was precomputed for.
DEFAULT_COMPRESSION_LEVEL = 3

#: Vocabulary cap. The post quotes 64,992 unique words for shard 11,343, so a real
#: shard sits just under this; the cap exists to bound a pathological shard, not to
#: truncate a normal one.
VOCABULARY_MAX_TERMS = 65_536


def stable_hash(obj: Any) -> int:
    """Process-stable 128-bit hash of ``obj``'s string form, for rBloom.

    CPython's :func:`hash` is salted per process for ``str``/``bytes``, so a bloom
    filter built with it cannot be persisted and reloaded — the same phrase lands in
    different buckets in the next process. BLAKE2b has no salt, so it does.
    """
    digest = hashlib.blake2b(str(obj).encode("utf-8"), digest_size=_STABLE_HASH_BYTES).digest()
    return int.from_bytes(digest, "big", signed=True)


def normalize_phrase(phrase: str) -> str:
    """Canonical spelling of a phrase, shared by the builder and the query side.

    Reuses the repo's :func:`cybernaut_mini.text.lexical_form` (NFKC, collapsed
    whitespace, lowercased) so that a phrase stored at build time and an intent
    produced at query time agree without a second normalisation policy existing.
    """
    return lexical_form(phrase)


# ------------------------------------------------------------------ #
# (a) Bloom filter over important phrases                             #
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class PhraseBloom:
    """A shard's phrase filter plus the parameters needed to rebuild it exactly."""

    #: The live filter. Always constructed with :func:`stable_hash`.
    bloom: Bloom
    #: Number of distinct normalised phrases actually inserted.
    n_phrases: int
    #: Capacity the filter was sized for (``>= n_phrases``, at least 1).
    expected_items: int
    #: Target false-positive rate the filter was sized for.
    false_positive_rate: float

    def __contains__(self, phrase: str) -> bool:
        """True if ``phrase`` may be in the shard (subject to the false-positive rate)."""
        return normalize_phrase(phrase) in self.bloom

    @property
    def size_in_bits(self) -> int:
        """Bucket count of the underlying filter."""
        size: int = self.bloom.size_in_bits
        return size

    def to_payload(self) -> dict[str, Any]:
        """JSON-safe payload; the bit array travels base64-encoded."""
        return {
            "bits": base64.b64encode(self.bloom.save_bytes()).decode("ascii"),
            "expected_items": self.expected_items,
            "false_positive_rate": self.false_positive_rate,
            "hash": STABLE_HASH_NAME,
            "n_phrases": self.n_phrases,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> PhraseBloom:
        """Rebuild from :meth:`to_payload`, refusing a payload hashed differently.

        The hash check is the point of the field: a filter reloaded under a different
        hash function does not error, it just answers wrongly forever.
        """
        hash_name = payload.get("hash")
        if hash_name != STABLE_HASH_NAME:
            msg = (
                f"bloom payload was written with hash {hash_name!r}, but this build "
                f"can only read {STABLE_HASH_NAME!r}; the filter would answer wrongly"
            )
            raise ValueError(msg)
        bloom = Bloom.load_bytes(base64.b64decode(payload["bits"]), stable_hash)
        return cls(
            bloom=bloom,
            n_phrases=int(payload["n_phrases"]),
            expected_items=int(payload["expected_items"]),
            false_positive_rate=float(payload["false_positive_rate"]),
        )


def build_phrase_bloom(
    phrases: Iterable[str],
    *,
    false_positive_rate: float = DEFAULT_FALSE_POSITIVE_RATE,
    expected_items: int | None = None,
) -> PhraseBloom:
    """Build a shard's phrase filter.

    Phrases are normalised and deduplicated first, so the filter is sized for the
    number of distinct phrases and is independent of the order they arrive in.
    ``expected_items`` overrides that sizing when the caller expects to add more
    later; it is never allowed below the number of phrases inserted, because
    undersizing silently inflates the false-positive rate.

    Raises :class:`ValueError` for a false-positive rate outside ``(0, 1)``.
    """
    if not 0.0 < false_positive_rate < 1.0:
        msg = f"false_positive_rate must be in (0, 1), got {false_positive_rate!r}"
        raise ValueError(msg)

    unique = {normalize_phrase(p) for p in phrases}
    unique.discard("")
    # rBloom rejects a capacity of 0, and an empty shard filter still has to exist so
    # that routing can score it (as zero coverage) instead of special-casing None.
    capacity = max(1, len(unique), expected_items or 0)

    bloom = Bloom(capacity, false_positive_rate, stable_hash)
    for phrase in sorted(unique):
        bloom.add(phrase)

    return PhraseBloom(
        bloom=bloom,
        n_phrases=len(unique),
        expected_items=capacity,
        false_positive_rate=false_positive_rate,
    )


def bloom_contains_any(phrase_bloom: PhraseBloom, intents: Iterable[str]) -> float:
    """Fraction of ``intents`` the shard may contain, in ``[0, 1]``.

    The post's rule is "if a selected shard hasn't seen any of the user's search
    intents, then that shard should get downranked", so 0.0 is the downrank signal
    and the intermediate values grade how much of the query the shard covers. A
    filter can only be wrong in one direction, so this is an upper bound on true
    coverage. No intents means nothing to test, which scores 0.0.
    """
    normalised = [normalize_phrase(intent) for intent in intents]
    candidates = [intent for intent in normalised if intent]
    if not candidates:
        return 0.0
    hits = sum(1 for intent in candidates if intent in phrase_bloom.bloom)
    return hits / len(candidates)


# ------------------------------------------------------------------ #
# (b) Trained Zstandard dictionary                                    #
# ------------------------------------------------------------------ #


def _baseline_compressor(level: int) -> zstandard.ZstdCompressor:
    """A fresh dictionary-free compressor for one scoring call.

    Deliberately not cached across calls. python-zstandard does not guarantee that two
    methods of one ``ZstdCompressor`` may be called from different threads at the same
    time, and the natural use of this module is scoring 1,000 shards for one question
    across the box's 8 cores — exactly the shape that would share one instance between
    threads. Measured cost of constructing one instead of reusing it: 0.79 us per call
    (3.81 us vs 3.02 us for a ~230-byte question at level 3), i.e. under a millisecond
    across a whole 1,000-shard fan-out, which is not worth a data race.
    """
    return zstandard.ZstdCompressor(
        level=level,
        write_checksum=False,
        write_content_size=False,
    )


def train_zstd_dictionary(
    texts: Iterable[str],
    *,
    dict_size: int = DEFAULT_DICT_SIZE,
) -> bytes | None:
    """Train a shard's compression dictionary, or return None if zstd refuses.

    zstd will not train on too few or too small samples — it raises ``ZstdError:
    cannot train dict: Src size is incorrect``, and measured here it needs roughly
    8 samples of a few hundred bytes before it will produce anything. A shard that
    small is legitimate (the tail of a k-means split), so this degrades to ``None``
    rather than failing a build; :func:`compression_score` then scores that shard 0.0
    and routing falls back on its other signals.

    ``threads=0`` keeps training single-threaded, which is what makes the output
    reproducible; ``dict_id=0`` makes zstd derive the dictionary id from the
    dictionary's own content instead of generating a random one.

    Returns the raw dictionary bytes, suitable for base64 in a manifest.
    """
    if dict_size <= 0:
        msg = f"dict_size must be positive, got {dict_size!r}"
        raise ValueError(msg)

    # Annotated wide because zstandard types the parameter as an invariant list of the
    # buffer union; a bare list[bytes] is not assignable to it.
    samples: list[bytes | bytearray | memoryview[int]] = [
        text.encode("utf-8") for text in texts if text
    ]
    if not samples:
        return None

    try:
        trained = zstandard.train_dictionary(dict_size, samples, threads=0, dict_id=0)
    except zstandard.ZstdError:
        # The documented failure: not enough material to find repeated content.
        return None

    raw: bytes = trained.as_bytes()
    return raw or None


def load_zstd_dictionary(
    raw: bytes,
    *,
    level: int = DEFAULT_COMPRESSION_LEVEL,
) -> zstandard.ZstdCompressionDict:
    """Wrap raw dictionary bytes, precomputed for ``level``.

    Precomputing costs a one-off cost per shard and removes it from every query.
    A precomputed dictionary is bound to its level, so pass the same ``level`` to
    :func:`compression_score`.
    """
    dictionary = zstandard.ZstdCompressionDict(raw)
    dictionary.precompute_compress(level=level)
    return dictionary


def compression_score(
    dictionary: bytes | zstandard.ZstdCompressionDict | None,
    text: str,
    *,
    level: int = DEFAULT_COMPRESSION_LEVEL,
) -> float:
    """How much better this shard's dictionary compresses ``text``. Higher is better.

    The score is the fractional saving against the same text compressed with no
    dictionary::

        (len(zstd(text)) - len(zstd(text, dict))) / len(zstd(text))

    so 0.0 means the dictionary bought nothing. Measured on this repo's two synthetic
    60-document shards, a question scores about +0.27 against its own shard's
    dictionary and about -0.07 against the other topic's, so the sign alone separates
    them. Negative values are real — an unrelated dictionary costs a few bytes of
    header for entries it never uses — and are clamped at -1.0 so the signal stays in
    a bounded range for fusion. A missing dictionary or an empty question scores 0.0,
    which is the same "no evidence" value as a useless one.

    The frame is written without a dictionary id, so shards are compared on content
    alone and not on a 4-byte header difference.
    """
    payload = text.encode("utf-8")
    if not payload.strip() or dictionary is None:
        return 0.0

    baseline = len(_baseline_compressor(level).compress(payload))
    if baseline == 0:
        return 0.0

    dict_data = (
        zstandard.ZstdCompressionDict(dictionary) if isinstance(dictionary, bytes) else dictionary
    )
    compressor = zstandard.ZstdCompressor(
        level=level,
        dict_data=dict_data,
        write_checksum=False,
        write_content_size=False,
        write_dict_id=False,
    )
    with_dict = len(compressor.compress(payload))
    return max(-1.0, (baseline - with_dict) / baseline)


# ------------------------------------------------------------------ #
# (c) Vocabulary                                                      #
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class Vocabulary:
    """The set of unique words in a shard, sorted and capped."""

    #: Kept terms, lexicographically sorted.
    terms: tuple[str, ...]
    #: Distinct terms seen before the cap was applied.
    unique_terms: int
    #: True when the cap dropped terms, so a reader knows absence is not proof.
    truncated: bool

    def __contains__(self, term: str) -> bool:
        # Binary search over the sorted tuple rather than a set. A set would be O(1)
        # but costs a second copy of every term reference; at 1,000 shards x 65k terms
        # that is gigabytes of index-resident overhead to save a few microseconds on a
        # lookup that runs once per query token per shard.
        position = bisect_left(self.terms, term)
        return position < len(self.terms) and self.terms[position] == term

    def __len__(self) -> int:
        return len(self.terms)

    def coverage(self, tokens: Iterable[str]) -> float:
        """Fraction of ``tokens`` present in this vocabulary, in ``[0, 1]``.

        Unlike the bloom filter this answer is exact (modulo truncation), which makes
        it the cheap check to run before trusting a probabilistic one.
        """
        wanted = list(tokens)
        if not wanted:
            return 0.0
        return sum(1 for token in wanted if token in self) / len(wanted)

    def to_payload(self) -> dict[str, Any]:
        return {
            "terms": list(self.terms),
            "truncated": self.truncated,
            "unique_terms": self.unique_terms,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Vocabulary:
        return cls(
            terms=tuple(payload["terms"]),
            unique_terms=int(payload["unique_terms"]),
            truncated=bool(payload["truncated"]),
        )


def build_vocabulary(
    token_streams: Iterable[Iterable[str]],
    *,
    max_terms: int = VOCABULARY_MAX_TERMS,
) -> Vocabulary:
    """Collect a shard's unique tokens, capped to the most frequent ``max_terms``.

    Takes streams of already-tokenized text (one stream per document) rather than raw
    text, because tokenization policy lives in :mod:`cybernaut_mini.text` and a shard
    must use the same one as the index it belongs to.

    When the cap bites, terms are ranked by frequency and ties broken
    lexicographically, so the result never depends on document order; the kept terms
    are then stored sorted.
    """
    if max_terms <= 0:
        msg = f"max_terms must be positive, got {max_terms!r}"
        raise ValueError(msg)

    counts: Counter[str] = Counter()
    for stream in token_streams:
        counts.update(token for token in stream if token)

    unique_terms = len(counts)
    if unique_terms <= max_terms:
        return Vocabulary(terms=tuple(sorted(counts)), unique_terms=unique_terms, truncated=False)

    kept = sorted(counts, key=lambda term: (-counts[term], term))[:max_terms]
    return Vocabulary(terms=tuple(sorted(kept)), unique_terms=unique_terms, truncated=True)


# ------------------------------------------------------------------ #
# (d) Serialisation contract                                          #
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class ShardArtifacts:
    """The three routing artifacts of one shard, as a single serialisable unit.

    This is the shape a ``ShardManifest`` field would carry. It round-trips through
    JSON — base64 for the two binary blobs — and :meth:`canonical_json` is
    byte-identical for two builds over the same input.
    """

    phrase_bloom: PhraseBloom
    zstd_dictionary: bytes | None
    vocabulary: Vocabulary

    def to_payload(self) -> dict[str, Any]:
        dictionary = self.zstd_dictionary
        return {
            "phrase_bloom": self.phrase_bloom.to_payload(),
            "version": ARTIFACTS_PAYLOAD_VERSION,
            "vocabulary": self.vocabulary.to_payload(),
            "zstd_dictionary": (
                None if dictionary is None else base64.b64encode(dictionary).decode("ascii")
            ),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ShardArtifacts:
        version = payload.get("version")
        if version != ARTIFACTS_PAYLOAD_VERSION:
            msg = (
                f"shard artifacts payload version {version!r} is not readable by this "
                f"build (expected {ARTIFACTS_PAYLOAD_VERSION!r})"
            )
            raise ValueError(msg)
        encoded = payload["zstd_dictionary"]
        return cls(
            phrase_bloom=PhraseBloom.from_payload(payload["phrase_bloom"]),
            zstd_dictionary=None if encoded is None else base64.b64decode(encoded),
            vocabulary=Vocabulary.from_payload(payload["vocabulary"]),
        )

    def canonical_json(self) -> str:
        """Deterministic JSON, through the same writer as every other artifact."""
        return canonical_dumps(self.to_payload())


def build_shard_artifacts(
    *,
    phrases: Iterable[str],
    texts: Iterable[str],
    token_streams: Iterable[Iterable[str]],
    false_positive_rate: float = DEFAULT_FALSE_POSITIVE_RATE,
    dict_size: int = DEFAULT_DICT_SIZE,
    max_terms: int = VOCABULARY_MAX_TERMS,
) -> ShardArtifacts:
    """Build all three artifacts for one shard.

    Keyword-only, because ``phrases``, ``texts`` and ``token_streams`` are three
    different views of the same shard and a positional call site would be unreadable.
    Each argument is consumed once, so generators are fine.
    """
    return ShardArtifacts(
        phrase_bloom=build_phrase_bloom(phrases, false_positive_rate=false_positive_rate),
        zstd_dictionary=train_zstd_dictionary(texts, dict_size=dict_size),
        vocabulary=build_vocabulary(token_streams, max_terms=max_terms),
    )
