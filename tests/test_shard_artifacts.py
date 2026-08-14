"""Tests for cybernaut_mini.shard_artifacts."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor

import pytest
import zstandard

from cybernaut_mini.shard_artifacts import (
    ARTIFACTS_PAYLOAD_VERSION,
    STABLE_HASH_NAME,
    VOCABULARY_MAX_TERMS,
    PhraseBloom,
    ShardArtifacts,
    Vocabulary,
    bloom_contains_any,
    build_phrase_bloom,
    build_shard_artifacts,
    build_vocabulary,
    compression_score,
    load_zstd_dictionary,
    normalize_phrase,
    stable_hash,
    train_zstd_dictionary,
)

# ------------------------------------------------------------------ #
# Corpus helpers: two lexically disjoint topics, so the compression   #
# dictionary has something real to separate.                          #
# ------------------------------------------------------------------ #

_GENOMICS = textwrap.dedent(
    """
    crispr cas9 nuclease sgrna guide rna plasmid vector transfection homologous
    recombination genome editing spacer sequence codon optimization mutagenesis
    transgenesis knockout knockin protospacer adjacent motif double strand break
    non homologous end joining repair template electroporation colony screening
    """
).split()

_MARKETS = textwrap.dedent(
    """
    inflation central bank interest rate policy bond yield curve equity portfolio
    hedge treasury dividend earnings quarterly guidance revenue margin buyback
    volatility index futures contract settlement clearing counterparty liquidity
    spread basis point tightening easing recession forecast consumer sentiment
    """
).split()


def _passage(vocabulary: list[str], rng: random.Random, words: int = 300) -> str:
    return " ".join(rng.choice(vocabulary) for _ in range(words))


def _shard_texts(vocabulary: list[str], seed: int, n: int = 60) -> list[str]:
    rng = random.Random(seed)
    return [_passage(vocabulary, rng) for _ in range(n)]


_GENOMICS_QUESTION = (
    "how does a cas9 nuclease use an sgrna guide rna to make a double strand break "
    "that homologous recombination then repairs with a repair template"
)
_MARKETS_QUESTION = (
    "what happens to the bond yield curve and equity portfolio volatility when the "
    "central bank raises the policy interest rate to fight inflation"
)


# ------------------------------------------------------------------ #
# stable_hash: the salted-hash trap                                   #
# ------------------------------------------------------------------ #


def test_stable_hash_fits_signed_128_bits() -> None:
    """rBloom requires the hash to land in the signed 128-bit range."""
    limit = 2**127
    for value in ("crispr", "", "a" * 10_000, "éè accents", 12345):
        assert -limit <= stable_hash(value) < limit


def test_stable_hash_is_not_python_hash() -> None:
    """A regression guard: the moment this equals hash(), persistence is broken."""
    assert stable_hash("crispr genome editing") != hash("crispr genome editing")


_SUBPROCESS_DIGEST_SCRIPT = """
import hashlib, json, sys
from cybernaut_mini.shard_artifacts import build_phrase_bloom, bloom_contains_any

phrases = json.loads(sys.argv[1])
bloom = build_phrase_bloom(phrases)
payload = json.dumps(bloom.to_payload(), sort_keys=True).encode()
print(hashlib.sha256(payload).hexdigest())
print(bloom_contains_any(bloom, ["crispr genome editing"]))
"""

_SUBPROCESS_SALTED_SCRIPT = """
import hashlib
from rbloom import Bloom

# The trap this module exists to avoid: a caller-supplied hash that wraps CPython's
# salted hash() passes rbloom's own guard against the *built-in* hash object.
bloom = Bloom(64, 0.01, lambda obj: hash(obj))
for phrase in ("crispr genome editing", "sgrna design", "plasmid vector"):
    bloom.add(phrase)
print(hashlib.sha256(bloom.save_bytes()).hexdigest())
"""


def _run_script(script: str, *args: str, hash_seed: str) -> list[str]:
    result = subprocess.run(
        [sys.executable, "-c", script, *args],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONHASHSEED": hash_seed, "PATH": "/usr/bin:/bin"},
    )
    return result.stdout.split()


def test_bloom_payload_is_identical_across_processes_and_hash_seeds() -> None:
    """The salted-hash trap, proven absent: fresh processes, different PYTHONHASHSEED.

    If the filter hashed with CPython's ``hash()`` these digests would differ, because
    ``hash()`` for ``str`` is salted per process.
    """
    phrases = ["CRISPR Genome Editing", "sgRNA design", "plasmid vector", "codon optimization"]
    encoded = json.dumps(phrases)

    in_process = build_phrase_bloom(phrases)
    expected = hashlib.sha256(
        json.dumps(in_process.to_payload(), sort_keys=True).encode()
    ).hexdigest()

    digests = []
    for seed in ("0", "1", "424242"):
        digest, coverage = _run_script(_SUBPROCESS_DIGEST_SCRIPT, encoded, hash_seed=seed)
        digests.append(digest)
        assert coverage == "1.0"

    assert digests == [expected, expected, expected]


def test_salted_hash_really_would_have_broken_this() -> None:
    """Shows the trap is real, so the test above is not testing nothing."""
    digests = {_run_script(_SUBPROCESS_SALTED_SCRIPT, hash_seed=seed)[0] for seed in "123"}
    assert len(digests) == 3, "expected a salted hash to produce a different filter per process"


# ------------------------------------------------------------------ #
# (a) Bloom filter                                                    #
# ------------------------------------------------------------------ #


def test_present_phrase_is_found_regardless_of_spelling() -> None:
    bloom = build_phrase_bloom(["CRISPR Genome Editing", "sgRNA design"])
    assert "crispr genome editing" in bloom
    assert "  CRISPR   Genome\tEditing " in bloom
    assert "SGRNA DESIGN" in bloom


def test_absent_phrases_are_almost_always_rejected() -> None:
    """1,000 absent phrases against a 1% filter: allow a generous slack, not zero."""
    phrases = [f"present phrase {i}" for i in range(500)]
    bloom = build_phrase_bloom(phrases, false_positive_rate=0.01)

    absent = [f"absent phrase {i}" for i in range(1000)]
    false_positives = sum(1 for phrase in absent if phrase in bloom)

    assert false_positives < 40, f"{false_positives}/1000 false positives at a 1% target"
    assert all(phrase in bloom for phrase in phrases)


def test_bloom_sizing_ignores_duplicates_and_order() -> None:
    first = build_phrase_bloom(["alpha beta", "gamma", "ALPHA  BETA"])
    second = build_phrase_bloom(["gamma", "alpha beta"])
    assert first.n_phrases == 2
    assert first.to_payload() == second.to_payload()


def test_bloom_expected_items_can_oversize_but_never_undersize() -> None:
    phrases = [f"phrase {i}" for i in range(20)]
    oversized = build_phrase_bloom(phrases, expected_items=1000)
    exact = build_phrase_bloom(phrases)
    assert oversized.expected_items == 1000
    assert oversized.size_in_bits > exact.size_in_bits

    undersized = build_phrase_bloom(phrases, expected_items=1)
    assert undersized.expected_items == 20, "must not size below the phrases inserted"


def test_empty_phrase_set_builds_a_usable_empty_filter() -> None:
    bloom = build_phrase_bloom([])
    assert bloom.n_phrases == 0
    assert "anything" not in bloom
    assert bloom_contains_any(bloom, ["anything", "at all"]) == 0.0


def test_blank_phrases_are_dropped() -> None:
    bloom = build_phrase_bloom(["   ", "", "real phrase"])
    assert bloom.n_phrases == 1


@pytest.mark.parametrize("rate", [0.0, 1.0, -0.1, 1.5])
def test_invalid_false_positive_rate_raises(rate: float) -> None:
    with pytest.raises(ValueError, match="false_positive_rate"):
        build_phrase_bloom(["x"], false_positive_rate=rate)


def test_bloom_contains_any_is_a_coverage_fraction() -> None:
    bloom = build_phrase_bloom(["crispr", "plasmid", "nuclease"])
    assert bloom_contains_any(bloom, ["crispr", "plasmid"]) == 1.0
    assert bloom_contains_any(bloom, ["crispr", "zylophristine assay"]) == 0.5
    assert bloom_contains_any(bloom, ["zylophristine assay"]) == 0.0
    assert bloom_contains_any(bloom, []) == 0.0
    assert bloom_contains_any(bloom, ["  ", ""]) == 0.0


def test_bloom_downranks_a_shard_that_has_seen_no_intent() -> None:
    """The post's actual rule, expressed as a comparison between two shards."""
    genomics = build_phrase_bloom(["crispr genome editing", "sgrna design", "plasmid vector"])
    markets = build_phrase_bloom(["bond yield curve", "central bank policy", "equity portfolio"])
    intents = ["crispr genome editing", "sgrna design"]
    assert bloom_contains_any(genomics, intents) == 1.0
    assert bloom_contains_any(markets, intents) == 0.0


def test_bloom_survives_an_explicit_save_and_reload() -> None:
    bloom = build_phrase_bloom(["crispr genome editing", "plasmid vector"])
    restored = PhraseBloom.from_payload(json.loads(json.dumps(bloom.to_payload())))
    assert "crispr genome editing" in restored
    assert restored.n_phrases == bloom.n_phrases
    assert restored.expected_items == bloom.expected_items
    assert restored.false_positive_rate == bloom.false_positive_rate
    assert restored.to_payload() == bloom.to_payload()


def test_bloom_payload_written_with_another_hash_is_refused() -> None:
    payload = build_phrase_bloom(["crispr"]).to_payload()
    payload["hash"] = "siphash-2-4"
    with pytest.raises(ValueError, match="answer wrongly"):
        PhraseBloom.from_payload(payload)
    assert STABLE_HASH_NAME == "blake2b-128"


# ------------------------------------------------------------------ #
# (b) Zstandard dictionary                                            #
# ------------------------------------------------------------------ #


def test_dictionary_trains_on_a_realistic_shard() -> None:
    dictionary = train_zstd_dictionary(_shard_texts(_GENOMICS, seed=1))
    assert dictionary is not None
    assert len(dictionary) > 0
    # Round-trips as a usable zstd dictionary, not just as opaque bytes.
    assert zstandard.ZstdCompressionDict(dictionary).dict_id() != 0


def test_matching_text_compresses_better_than_unrelated_text() -> None:
    """The post's claim: the shard that compresses the question best is the right one."""
    genomics = train_zstd_dictionary(_shard_texts(_GENOMICS, seed=1))
    markets = train_zstd_dictionary(_shard_texts(_MARKETS, seed=2))
    assert genomics is not None
    assert markets is not None

    on_topic = compression_score(genomics, _GENOMICS_QUESTION)
    off_topic = compression_score(genomics, _MARKETS_QUESTION)
    assert on_topic > off_topic

    # And symmetrically, so the result is a property of the pairing, not of one text.
    assert compression_score(markets, _MARKETS_QUESTION) > compression_score(
        markets, _GENOMICS_QUESTION
    )

    # Routing picks the shard with the higher score for each question.
    assert compression_score(genomics, _GENOMICS_QUESTION) > compression_score(
        markets, _GENOMICS_QUESTION
    )
    assert compression_score(markets, _MARKETS_QUESTION) > compression_score(
        genomics, _MARKETS_QUESTION
    )


def test_compression_score_stays_in_a_bounded_range() -> None:
    dictionary = train_zstd_dictionary(_shard_texts(_GENOMICS, seed=1))
    assert dictionary is not None
    score = compression_score(dictionary, _GENOMICS_QUESTION)
    assert -1.0 <= score <= 1.0
    assert score > 0.1, f"a matching question should compress measurably better, got {score}"


def test_compression_score_is_zero_without_a_dictionary_or_text() -> None:
    dictionary = train_zstd_dictionary(_shard_texts(_GENOMICS, seed=1))
    assert compression_score(None, _GENOMICS_QUESTION) == 0.0
    assert compression_score(dictionary, "") == 0.0
    assert compression_score(dictionary, "   \n ") == 0.0


def test_precomputed_dictionary_scores_the_same_as_raw_bytes() -> None:
    raw = train_zstd_dictionary(_shard_texts(_GENOMICS, seed=1))
    assert raw is not None
    assert compression_score(load_zstd_dictionary(raw), _GENOMICS_QUESTION) == compression_score(
        raw, _GENOMICS_QUESTION
    )


@pytest.mark.parametrize(
    "texts",
    [
        [],
        [""],
        ["hi"],
        ["hi", "there"],
        ["a short line"] * 4,
    ],
    ids=["empty", "blank", "one-tiny", "two-tiny", "four-short"],
)
def test_tiny_input_degrades_to_none_instead_of_raising(texts: list[str]) -> None:
    """zstd refuses to train on too little material; a small shard is not a build failure."""
    assert train_zstd_dictionary(texts) is None


def test_a_none_dictionary_still_lets_a_shard_be_scored() -> None:
    """The degradation has to be usable by the caller, not just non-fatal."""
    artifacts = build_shard_artifacts(phrases=["tiny shard"], texts=["hi"], token_streams=[["hi"]])
    assert artifacts.zstd_dictionary is None
    assert compression_score(artifacts.zstd_dictionary, _GENOMICS_QUESTION) == 0.0
    assert artifacts.canonical_json()  # and it still serialises


def test_invalid_dict_size_raises() -> None:
    with pytest.raises(ValueError, match="dict_size"):
        train_zstd_dictionary(_shard_texts(_GENOMICS, seed=1), dict_size=0)


def test_dictionary_training_is_deterministic() -> None:
    texts = _shard_texts(_GENOMICS, seed=1)
    assert train_zstd_dictionary(texts) == train_zstd_dictionary(list(texts))


def test_scoring_is_safe_and_identical_under_concurrency() -> None:
    """Routing fans one question out over many shards at once; that must not race.

    python-zstandard does not promise that one ``ZstdCompressor`` may be used from two
    threads simultaneously, so :func:`compression_score` must not share one. A shared
    compressor here surfaces as a wrong length or a raised error, not as a hang.
    """
    dictionary = train_zstd_dictionary(_shard_texts(_GENOMICS, seed=1))
    assert dictionary is not None
    expected = compression_score(dictionary, _GENOMICS_QUESTION)

    with ThreadPoolExecutor(max_workers=8) as pool:
        scores = list(
            pool.map(lambda _: compression_score(dictionary, _GENOMICS_QUESTION), range(400))
        )

    assert scores == [expected] * 400


# ------------------------------------------------------------------ #
# (c) Vocabulary                                                      #
# ------------------------------------------------------------------ #


def test_vocabulary_is_sorted_unique_and_order_independent() -> None:
    forward = build_vocabulary([["gamma", "alpha"], ["beta", "alpha"]])
    reversed_streams = build_vocabulary([["alpha", "beta"], ["alpha", "gamma"]])
    assert forward.terms == ("alpha", "beta", "gamma")
    assert list(forward.terms) == sorted(forward.terms)
    assert forward.unique_terms == 3
    assert forward.truncated is False
    assert forward == reversed_streams


def test_vocabulary_drops_empty_tokens() -> None:
    vocabulary = build_vocabulary([["", "alpha", ""]])
    assert vocabulary.terms == ("alpha",)


def test_vocabulary_membership_and_coverage() -> None:
    vocabulary = build_vocabulary([["alpha", "beta", "gamma"]])
    assert "alpha" in vocabulary
    assert "delta" not in vocabulary
    assert len(vocabulary) == 3
    assert vocabulary.coverage(["alpha", "delta"]) == 0.5
    assert vocabulary.coverage(["alpha", "beta"]) == 1.0
    assert vocabulary.coverage([]) == 0.0


def test_vocabulary_cap_keeps_the_most_frequent_terms() -> None:
    streams = [["common"] * 5, ["frequent"] * 3, ["rare"], ["rarer"]]
    vocabulary = build_vocabulary(streams, max_terms=2)
    assert vocabulary.terms == ("common", "frequent")
    assert vocabulary.unique_terms == 4
    assert vocabulary.truncated is True


def test_vocabulary_cap_breaks_frequency_ties_lexicographically() -> None:
    """Ties must not resolve by document order, or the artifact stops being stable."""
    streams = [["zeta"], ["alpha"], ["mu"]]
    assert build_vocabulary(streams, max_terms=2).terms == ("alpha", "mu")
    assert build_vocabulary(list(reversed(streams)), max_terms=2).terms == ("alpha", "mu")


def test_vocabulary_cap_default_is_above_the_blog_figure() -> None:
    """The post quotes 64,992 unique words for shard 11,343; a real shard must fit."""
    assert VOCABULARY_MAX_TERMS > 64_992


def test_vocabulary_cap_holds_at_a_realistic_shard_size() -> None:
    """90,000 distinct terms — above the 64,992 the post quotes — must cap, not blow up."""
    streams = [
        [f"term{(doc * 1_000 + i) % 90_000}" for i in range(1_000)] for doc in range(100)
    ]
    first = build_vocabulary(streams)
    second = build_vocabulary([list(reversed(stream)) for stream in reversed(streams)])

    assert first.unique_terms == 90_000
    assert first.truncated is True
    assert len(first) == VOCABULARY_MAX_TERMS
    assert list(first.terms) == sorted(first.terms)
    assert first == second, "cap selection must not depend on document order"


@pytest.mark.parametrize("max_terms", [0, -1])
def test_invalid_max_terms_raises(max_terms: int) -> None:
    with pytest.raises(ValueError, match="max_terms"):
        build_vocabulary([["alpha"]], max_terms=max_terms)


# ------------------------------------------------------------------ #
# (d) Serialisation contract                                          #
# ------------------------------------------------------------------ #


def _genomics_artifacts() -> ShardArtifacts:
    texts = _shard_texts(_GENOMICS, seed=1)
    return build_shard_artifacts(
        phrases=["CRISPR Genome Editing", "sgRNA design", "plasmid vector"],
        texts=texts,
        token_streams=[text.split() for text in texts],
    )


def test_artifacts_round_trip_through_json_byte_identically() -> None:
    artifacts = _genomics_artifacts()
    encoded = artifacts.canonical_json()

    restored = ShardArtifacts.from_payload(json.loads(encoded))

    assert restored.canonical_json() == encoded
    assert restored.zstd_dictionary == artifacts.zstd_dictionary
    assert restored.vocabulary == artifacts.vocabulary
    assert "crispr genome editing" in restored.phrase_bloom
    assert bloom_contains_any(restored.phrase_bloom, ["sgrna design"]) == 1.0
    assert compression_score(restored.zstd_dictionary, _GENOMICS_QUESTION) == compression_score(
        artifacts.zstd_dictionary, _GENOMICS_QUESTION
    )


def test_two_builds_over_the_same_input_are_byte_identical() -> None:
    assert _genomics_artifacts().canonical_json() == _genomics_artifacts().canonical_json()


_SUBPROCESS_ARTIFACTS_SCRIPT = """
import hashlib, sys
sys.path.insert(0, {tests_dir!r})
from test_shard_artifacts import _genomics_artifacts
print(hashlib.sha256(_genomics_artifacts().canonical_json().encode()).hexdigest())
"""


def test_artifacts_are_byte_identical_in_a_fresh_process() -> None:
    """The determinism claim that matters: a rebuild tomorrow must match today's bytes."""
    script = _SUBPROCESS_ARTIFACTS_SCRIPT.format(tests_dir=str(__file__.rsplit("/", 1)[0]))
    expected = hashlib.sha256(_genomics_artifacts().canonical_json().encode()).hexdigest()
    digests = {_run_script(script, hash_seed=seed)[0] for seed in ("0", "7", "999")}
    assert digests == {expected}


def test_payload_is_json_serialisable_with_no_binary_leakage() -> None:
    payload = _genomics_artifacts().to_payload()
    assert json.loads(json.dumps(payload)) == payload
    assert isinstance(payload["zstd_dictionary"], str)
    assert isinstance(payload["phrase_bloom"]["bits"], str)
    assert payload["version"] == ARTIFACTS_PAYLOAD_VERSION


def test_unknown_payload_version_is_refused() -> None:
    payload = _genomics_artifacts().to_payload()
    payload["version"] = "99"
    with pytest.raises(ValueError, match="not readable by this build"):
        ShardArtifacts.from_payload(payload)


def test_vocabulary_and_bloom_round_trip_independently() -> None:
    vocabulary = build_vocabulary([["alpha", "beta"]], max_terms=1)
    assert Vocabulary.from_payload(json.loads(json.dumps(vocabulary.to_payload()))) == vocabulary

    bloom = build_phrase_bloom(["alpha beta"])
    restored = PhraseBloom.from_payload(json.loads(json.dumps(bloom.to_payload())))
    assert restored.to_payload() == bloom.to_payload()


def test_normalize_phrase_matches_the_repo_lexical_form() -> None:
    """Build side and query side must not drift into two normalisation policies."""
    from cybernaut_mini.text import lexical_form

    for raw in ("CRISPR  Genome\tEditing", "  sgRNA  ", "Bond Yield Curve"):
        assert normalize_phrase(raw) == lexical_form(raw)
