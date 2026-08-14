"""Stage 1 (language detection and translation) — offline against a warm model cache.

No test here opens a socket *of its own*: translation is exercised through an injected
fake client, and the no-credential path is asserted without a client existing at all.
Detection is offline only once ``lid.176.bin`` is in the ``fasttext-langdetect`` cache;
``ftlangdetect`` downloads that 126 MB file over the network the first time it is asked
for one. This file never lets that download happen implicitly — the cache is checked as
a file on disk before any detection runs, and a cold cache is a hard failure naming the
command that warms it.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 1. The post's
    worked example is the assertion at the heart of this file: the English question
    passes through untranslated, and the same question in Japanese is
    "細菌と酵母から得られる教訓は、より安全な遺伝子編集医薬品にどのように応用されるのでしょうか？".
    Local copy: ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions:
    - A missing model is a broken test environment, not a tolerable condition. The
      worked examples are the acceptance criteria for this stage, and a green run that
      silently skipped them proves nothing, so ``requires_detector`` FAILS rather than
      skips. The escape hatch is explicit and has to be chosen by a human:
      ``CYBERNAUT_MINI_ALLOW_MISSING_FASTTEXT=1`` turns the failure back into a skip
      for a machine that genuinely cannot fetch the model.
    - The detection tests pin the exact codes ``lid.176`` returns, including the wrong
      ones it returns for two-character input. The model file is frozen and the call
      takes no sampling parameters, so these are stable; pinning them is what makes
      the tests capable of failing if a damping curve or a mapping table is ever
      introduced between fastText and ``LanguageResult``.

Alternatives rejected:
    - Monkeypatching fastText out entirely: fast and hermetic, but it would test the
      wrapper against a mock of the exact component whose real behaviour — the newline
      crash — this stage exists to absorb.
    - Letting ``requires_detector`` call ``detect_language`` on a cold cache to find
      out whether the model works: that turns a missing model into a silent 126 MB
      download in the middle of a "fully offline" test suite. Checking for the file
      first keeps the failure honest and instant.
"""  # noqa: RUF002 — the quoted Japanese is the post's own text, fullwidth mark and all.

from __future__ import annotations

import inspect
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest
from ftlangdetect.detect import _default_cache_dir as _ftlangdetect_cache_dir

from cybernaut_mini.query.s1_language import (
    API_KEY_ENV_VARS,
    DEFAULT_BASE_URL,
    DEFAULT_TRANSLATION_MODEL,
    UNKNOWN_LANGUAGE,
    MissingAPIKeyError,
    NullTranslator,
    OpenAICompatibleTranslator,
    PreparedQuestion,
    TranslationError,
    Translator,
    detect_language,
    language_name,
    normalise_for_detection,
    normalise_lang_code,
    prepare_question,
)
from cybernaut_mini.query.s1_language import detect as detect_module

ENGLISH_QUESTION = (
    "What lessons from bacteria and yeast actually translate into safer gene-editing medicines?"
)
#: The post's own Japanese rendering, character for character. Its fullwidth question
#: mark is part of the quotation, so RUF001 is silenced rather than the text edited.
JAPANESE_QUESTION = (
    "細菌と酵母から得られる教訓は、より安全な遺伝子編集医薬品にどのように応用されるのでしょうか？"  # noqa: RUF001
)

ALL_KEY_ENV_VARS = (*API_KEY_ENV_VARS, "ANTHROPIC_API_KEY")

#: The model ``detect_language`` uses: ``low_memory=False`` selects the full 126 MB
#: ``lid.176.bin``, not the compressed ``.ftz``.
MODEL_FILENAME = "lid.176.bin"

#: Setting this to a truthy value downgrades "the model is missing" from a failure to
#: a skip. It exists for a machine that genuinely cannot reach
#: ``dl.fbaipublicfiles.com``; it must never be set in CI, because the two blog
#: worked-example tests are the acceptance criteria for this stage.
ALLOW_MISSING_MODEL_ENV = "CYBERNAUT_MINI_ALLOW_MISSING_FASTTEXT"

_WARM_CACHE_HINT = (
    "warm the cache once with: "
    "uv run python -c \"from ftlangdetect import detect; print(detect('hello world'))\""
)


def cached_model_path() -> Path:
    """Where ``ftlangdetect`` looks for ``lid.176.bin``.

    Mirrors ``ftlangdetect.detect._default_cache_dir``: ``$FTLANG_CACHE`` when set,
    otherwise ``<tempdir>/fasttext-langdetect``. Read at call time, because a test may
    legitimately point ``FTLANG_CACHE`` somewhere else.
    """
    override = os.environ.get("FTLANG_CACHE")
    if override:
        return Path(override).expanduser() / MODEL_FILENAME
    return Path(tempfile.gettempdir()) / "fasttext-langdetect" / MODEL_FILENAME


@pytest.fixture(scope="module")
def requires_detector() -> None:
    """Assert the detector really works, or fail loudly — never skip by default.

    ``detect_language`` degrades a missing model to "und" on purpose, which means a
    detection test written against it would quietly assert nothing on a cold cache.
    Two of the tests behind this fixture ARE the post's worked examples, so a run that
    skipped them is not evidence of anything. The model file is therefore checked as a
    file (no implicit download), and then actually exercised.
    """
    model = cached_model_path()
    if not model.exists():
        message = (
            f"fastText {MODEL_FILENAME} is not cached at {model}. The stage 1 "
            f"worked-example tests cannot run without it and will not be skipped "
            f"silently: {_WARM_CACHE_HINT}. On a machine that genuinely cannot "
            f"download it, set {ALLOW_MISSING_MODEL_ENV}=1 to skip these tests "
            f"deliberately."
        )
        if os.environ.get(ALLOW_MISSING_MODEL_ENV, "").strip():
            pytest.skip(message)
        pytest.fail(message, pytrace=False)

    result = detect_language("this is an ordinary english sentence")
    if result.lang != "en":
        pytest.fail(
            f"{model} exists but detection returned {result.lang!r} instead of 'en'; "
            f"the cached model is unusable (truncated download?). Delete it and "
            f"{_WARM_CACHE_HINT}.",
            pytrace=False,
        )


def test_cached_model_path_matches_ftlangdetect() -> None:
    """The fixture's cache-path mirror must not drift from ftlangdetect's own.

    If it does, ``requires_detector`` would check the wrong file and could pass on a
    cold cache — reintroducing exactly the silent-skip hole this guards.
    """
    assert cached_model_path().parent == _ftlangdetect_cache_dir()
    assert cached_model_path().name == MODEL_FILENAME


@pytest.fixture
def no_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guarantee a credential-free environment, whatever the developer has exported."""
    for name in ALL_KEY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class _FakeMessage:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str | None) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str | None) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, owner: _FakeOpenAI) -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> _FakeResponse:
        self._owner.calls.append(kwargs)
        return _FakeResponse(self._owner.reply)


class _FakeChat:
    def __init__(self, owner: _FakeOpenAI) -> None:
        self.completions = _FakeCompletions(owner)


class _FakeOpenAI:
    """Stands in for ``openai.OpenAI`` — records every request, never opens a socket."""

    def __init__(self, api_key: str, base_url: str, reply: str | None) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.reply = reply
        self.calls: list[dict[str, Any]] = []
        self.chat = _FakeChat(self)


def _fake_translator(
    reply: str | None, **kwargs: Any
) -> tuple[OpenAICompatibleTranslator, list[_FakeOpenAI]]:
    """A translator wired to a fake client, plus the list its calls land in."""
    clients: list[_FakeOpenAI] = []

    def factory(api_key: str, base_url: str) -> _FakeOpenAI:
        client = _FakeOpenAI(api_key, base_url, reply)
        clients.append(client)
        return client

    translator = OpenAICompatibleTranslator(client_factory=factory, **kwargs)
    return translator, clients


class _ExplodingTranslator:
    """Fails the test if stage 1 calls it — proves a short-circuit really short-circuits."""

    def translate(self, text: str, target_lang: str, source_lang: str | None = None) -> str:
        raise AssertionError(f"translator must not be called (target={target_lang!r})")


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


@pytest.mark.usefixtures("requires_detector")
def test_english_question_detects_en() -> None:
    result = detect_language(ENGLISH_QUESTION)
    assert result.lang == "en"
    assert 0.0 < result.confidence <= 1.0
    assert result.is_known


@pytest.mark.usefixtures("requires_detector")
def test_japanese_question_detects_ja() -> None:
    result = detect_language(JAPANESE_QUESTION)
    assert result.lang == "ja"
    assert 0.0 < result.confidence <= 1.0


@pytest.mark.usefixtures("requires_detector")
def test_newlines_do_not_crash_detection() -> None:
    """fastText rejects newlines outright; the stage must absorb that, not forward it."""
    multiline = "What lessons from bacteria and yeast\nactually translate into\n\tsafer medicines?"
    assert detect_language(multiline).lang == "en"
    assert detect_language(multiline) == detect_language(normalise_for_detection(multiline))


def test_detection_normalises_before_calling_fasttext(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prove the newline fix is ours, not the installed ftlangdetect build's."""
    seen: list[str] = []

    def strict_detect(text: str, low_memory: bool = False, k: int = 1) -> dict[str, Any]:
        if "\n" in text or "\r" in text:
            msg = "predict processes one line at a time (remove '\\n')"
            raise ValueError(msg)
        seen.append(text)
        return {"lang": "en", "score": 0.99}

    monkeypatch.setattr(detect_module, "_ft_detect", strict_detect)
    result = detect_language("line one\r\nline two\n\n\tline three")
    assert result == detect_module.LanguageResult("en", 0.99)
    assert seen == ["line one line two line three"]


@pytest.mark.parametrize("text", ["", "   ", "\n", "\t\n  \r\n"])
def test_empty_input_is_undetermined(text: str) -> None:
    result = detect_language(text)
    assert result.lang == UNKNOWN_LANGUAGE
    assert result.confidence == 0.0
    assert not result.is_known


@pytest.mark.usefixtures("requires_detector")
@pytest.mark.parametrize(
    ("text", "expected_lang"),
    [
        # "hi" and "3.14" are genuinely WRONG answers, and that is the documented
        # behaviour: short input is detected anyway rather than rejected, and the
        # model's own low confidence is what a caller thresholds on. Pinning the wrong
        # answers is what makes this test able to fail if the stage ever starts
        # second-guessing the model.
        ("hi", "ca"),
        ("3.14", "no"),
        ("ok", "en"),
        ("細菌", "zh"),
        ("https://example.com", "en"),
    ],
)
def test_very_short_text_is_detected_not_rejected(text: str, expected_lang: str) -> None:
    result = detect_language(text)
    raw = detect_module._ft_detect(text, low_memory=False, k=1)

    assert result.lang == expected_lang
    # Undamped: the confidence is fastText's own score, not a rescaling of it.
    assert result.confidence == pytest.approx(float(raw["score"]))
    assert result.is_known


@pytest.mark.usefixtures("requires_detector")
def test_mixed_script_text_returns_one_language() -> None:
    result = detect_language(f"{ENGLISH_QUESTION} {JAPANESE_QUESTION}")
    assert result.lang in {"en", "ja"}
    assert 0.0 < result.confidence <= 1.0


@pytest.mark.usefixtures("requires_detector")
def test_detection_is_deterministic() -> None:
    assert detect_language(ENGLISH_QUESTION) == detect_language(ENGLISH_QUESTION)
    assert detect_language(JAPANESE_QUESTION) == detect_language(JAPANESE_QUESTION)


def test_detector_failure_degrades_to_und(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(text: str, low_memory: bool = False, k: int = 1) -> dict[str, Any]:
        raise OSError("model file missing")

    monkeypatch.setattr(detect_module, "_ft_detect", boom)
    assert detect_language(ENGLISH_QUESTION).lang == UNKNOWN_LANGUAGE


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({"lang": "EN", "score": 1.5}, ("en", 1.0)),
        ({"lang": "__label__ja", "score": -0.2}, ("ja", 0.0)),
        ({"lang": "", "score": 0.9}, (UNKNOWN_LANGUAGE, 0.0)),
        ({"lang": "fr", "score": "not-a-number"}, ("fr", 0.0)),
        ([{"lang": "de", "score": 0.5}], ("de", 0.5)),
        ([], (UNKNOWN_LANGUAGE, 0.0)),
        ("garbage", (UNKNOWN_LANGUAGE, 0.0)),
    ],
)
def test_odd_detector_payloads_are_coerced(
    monkeypatch: pytest.MonkeyPatch, raw: Any, expected: tuple[str, float]
) -> None:
    monkeypatch.setattr(detect_module, "_ft_detect", lambda *a, **k: raw)
    result = detect_language("something")
    assert (result.lang, result.confidence) == expected


def test_normalise_for_detection_collapses_whitespace() -> None:
    assert normalise_for_detection("  a\n\nb\t c \r\n") == "a b c"
    assert normalise_for_detection("   ") == ""


@pytest.mark.parametrize("value", [None, 3.14, b"bytes are not str", ["a", "b"], object()])
def test_non_string_input_degrades_to_und_instead_of_raising(value: Any) -> None:
    """The module promises it is never the thing that crashes the query path.

    A ``None`` from a caller that forgot to default an optional field used to raise
    ``TypeError`` out of the regex, which contradicted that promise outright.
    """
    result = detect_language(value)
    assert result == detect_module.LanguageResult(UNKNOWN_LANGUAGE, 0.0)
    assert not result.is_known
    assert normalise_for_detection(value) == ""


@pytest.mark.parametrize("value", [None, 3.14, b"bytes are not str"])
def test_prepare_question_survives_non_string_input(value: Any) -> None:
    """Stage 1 degrades a bad call to a blank question, and stays all-``str``."""
    prepared = prepare_question(value, "ja", _ExplodingTranslator())
    assert prepared.original == ""
    assert prepared.text_for_retrieval == ""
    assert prepared.language == UNKNOWN_LANGUAGE
    assert prepared.confidence == 0.0
    assert prepared.translated is False
    assert prepared.retrieval_language == UNKNOWN_LANGUAGE
    assert isinstance(prepared.original, str)
    assert isinstance(prepared.text_for_retrieval, str)


# --------------------------------------------------------------------------
# Translators
# --------------------------------------------------------------------------


def test_null_translator_is_identity() -> None:
    translator = NullTranslator()
    assert translator.translate(ENGLISH_QUESTION, "ja") == ENGLISH_QUESTION
    assert translator.translate(ENGLISH_QUESTION, "ja", "en") == ENGLISH_QUESTION
    assert translator.translate("", "fr", "en") == ""


def test_implementations_match_the_protocol_signature_and_behaviour() -> None:
    """Check the shape ``isinstance`` cannot see.

    ``Translator`` is ``runtime_checkable``, and a runtime-checkable Protocol's
    ``isinstance`` compares attribute NAMES only — it never looks at whether the
    attribute is callable, let alone at its parameters. The first assertion below
    proves that hole is real; the rest is the check that actually constrains the
    implementations.
    """
    imposter = type("NotATranslator", (), {"translate": 3})()
    assert isinstance(imposter, Translator)  # the weak check is fooled...
    assert not callable(imposter.translate)  # ...by something that cannot translate.

    expected = inspect.signature(Translator.translate)
    for impl in (NullTranslator(), OpenAICompatibleTranslator()):
        assert callable(impl.translate)
        assert inspect.signature(type(impl).translate) == expected
        # And it really returns the text unchanged on the no-work path, with no key
        # and no client — a str out, not just an attribute that exists.
        assert impl.translate("Bonjour", "fr", "fr") == "Bonjour"


@pytest.mark.usefixtures("no_api_keys")
def test_translator_is_constructible_without_any_key() -> None:
    translator = OpenAICompatibleTranslator()
    assert translator.model == DEFAULT_TRANSLATION_MODEL
    assert translator.base_url == DEFAULT_BASE_URL


@pytest.mark.usefixtures("no_api_keys")
def test_missing_key_raises_and_names_the_env_var() -> None:
    translator = OpenAICompatibleTranslator()
    with pytest.raises(MissingAPIKeyError) as excinfo:
        translator.translate(ENGLISH_QUESTION, "ja", "en")
    message = str(excinfo.value)
    assert "OPENROUTER_API_KEY" in message
    assert "OPENAI_API_KEY" in message
    assert isinstance(excinfo.value, TranslationError)


def test_anthropic_key_is_not_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ALL_KEY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-openai-shaped")
    with pytest.raises(MissingAPIKeyError, match="OPENROUTER_API_KEY"):
        OpenAICompatibleTranslator().translate(ENGLISH_QUESTION, "ja", "en")


@pytest.mark.parametrize("env_var", API_KEY_ENV_VARS)
def test_key_is_read_from_env_at_call_time(monkeypatch: pytest.MonkeyPatch, env_var: str) -> None:
    for name in ALL_KEY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    translator, clients = _fake_translator(JAPANESE_QUESTION)

    # Constructed with no key present; the key appears only afterwards.
    monkeypatch.setenv(env_var, f"key-from-{env_var}")
    assert translator.translate(ENGLISH_QUESTION, "ja", "en") == JAPANESE_QUESTION
    assert clients[0].api_key == f"key-from-{env_var}"


def test_openrouter_key_wins_over_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    translator, clients = _fake_translator(JAPANESE_QUESTION)
    translator.translate(ENGLISH_QUESTION, "ja", "en")
    assert clients[0].api_key == "router-key"


def test_explicit_api_key_beats_the_environment() -> None:
    translator, clients = _fake_translator(
        JAPANESE_QUESTION, api_key="explicit-key", environ={"OPENROUTER_API_KEY": "env-key"}
    )
    translator.translate(ENGLISH_QUESTION, "ja", "en")
    assert clients[0].api_key == "explicit-key"


def test_translate_wiring_reproduces_the_blog_example() -> None:
    """The post's worked example, end to end through the real client code path."""
    translator, clients = _fake_translator(
        JAPANESE_QUESTION, environ={"OPENROUTER_API_KEY": "test-key"}
    )
    output = translator.translate(ENGLISH_QUESTION, "ja", "en")

    assert output == JAPANESE_QUESTION
    client = clients[0]
    assert client.base_url == DEFAULT_BASE_URL
    request = client.calls[0]
    assert request["model"] == DEFAULT_TRANSLATION_MODEL
    assert request["temperature"] == 0.0
    messages = request["messages"]
    assert messages[0]["role"] == "system"
    assert ENGLISH_QUESTION in messages[1]["content"]
    assert "Japanese" in messages[1]["content"]
    assert "English" in messages[1]["content"]
    # An unchanged key reuses the same client rather than rebuilding a connection pool.
    translator.translate("another question", "ja", "en")
    assert len(clients) == 1
    assert len(client.calls) == 2


def test_unchanged_key_reuses_the_client_changed_key_rebuilds_it() -> None:
    """The real contract behind "the credential is read at call time".

    Reading it once and memoising the client would make the promise true only for the
    first call. The user case this exists for: an API key written into ``.env`` while
    the process is already running, or a key rotated by a credential helper.
    """
    environ = {"OPENROUTER_API_KEY": "key-v1"}
    translator, clients = _fake_translator(JAPANESE_QUESTION, environ=environ)

    translator.translate(ENGLISH_QUESTION, "ja", "en")
    translator.translate(ENGLISH_QUESTION, "ja", "en")
    assert [c.api_key for c in clients] == ["key-v1"]  # unchanged key -> one client

    environ["OPENROUTER_API_KEY"] = "key-v2"
    translator.translate(ENGLISH_QUESTION, "ja", "en")
    assert [c.api_key for c in clients] == ["key-v1", "key-v2"]  # rotated -> rebuilt

    translator.translate(ENGLISH_QUESTION, "ja", "en")
    assert [c.api_key for c in clients] == ["key-v1", "key-v2"]  # and cached again

    # The new client is the one doing the work; the stale one is not called again.
    assert len(clients[0].calls) == 2
    assert len(clients[1].calls) == 2


def test_key_appearing_mid_session_is_picked_up_without_a_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No key at first call, key exported afterwards, second call succeeds."""
    for name in ALL_KEY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    translator, clients = _fake_translator(JAPANESE_QUESTION)

    with pytest.raises(MissingAPIKeyError):
        translator.translate(ENGLISH_QUESTION, "ja", "en")
    assert clients == []

    monkeypatch.setenv("OPENROUTER_API_KEY", "key-added-to-dotenv-mid-session")
    assert translator.translate(ENGLISH_QUESTION, "ja", "en") == JAPANESE_QUESTION
    assert [c.api_key for c in clients] == ["key-added-to-dotenv-mid-session"]


def test_translate_accepts_a_custom_model_and_base_url() -> None:
    translator, clients = _fake_translator(
        "Hola",
        model="gpt-4o-mini",
        base_url="http://localhost:11434/v1",
        environ={"OPENAI_API_KEY": "local"},
    )
    assert translator.translate("Hello", "es", "en") == "Hola"
    assert clients[0].base_url == "http://localhost:11434/v1"
    assert clients[0].calls[0]["model"] == "gpt-4o-mini"


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (f"```\n{JAPANESE_QUESTION}\n```", JAPANESE_QUESTION),
        (f"```japanese\n{JAPANESE_QUESTION}\n```", JAPANESE_QUESTION),
        (f"  {JAPANESE_QUESTION}  ", JAPANESE_QUESTION),
    ],
)
def test_model_output_is_unwrapped(reply: str, expected: str) -> None:
    translator, _ = _fake_translator(reply, environ={"OPENROUTER_API_KEY": "k"})
    assert translator.translate(ENGLISH_QUESTION, "ja", "en") == expected


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        # Regression: a space-free translation on the fence's own line is NOT an info
        # string. Both of these used to be emptied out and then raise "empty content".
        (f"```{JAPANESE_QUESTION}\n```", JAPANESE_QUESTION),
        ("```Bonjour\n```", "Bonjour"),
        # Regression: an unfenced first line was treated as an info string and dropped.
        ("```細菌と酵母\nより安全な医薬品```", "細菌と酵母\nより安全な医薬品"),
        # A real info string is still removed.
        ('```json\n{"a": 1}\n```', '{"a": 1}'),
    ],
)
def test_fence_stripping_never_eats_the_translation(reply: str, expected: str) -> None:
    translator, _ = _fake_translator(reply, environ={"OPENROUTER_API_KEY": "k"})
    assert translator.translate(ENGLISH_QUESTION, "ja", "en") == expected


@pytest.mark.parametrize("reply", ["", "   ", None])
def test_empty_model_output_raises_translation_error(reply: str | None) -> None:
    translator, _ = _fake_translator(reply, environ={"OPENROUTER_API_KEY": "k"})
    with pytest.raises(TranslationError, match="empty content"):
        translator.translate(ENGLISH_QUESTION, "ja", "en")


def test_unreadable_response_raises_translation_error() -> None:
    class _BadClient:
        def __init__(self, *args: Any) -> None:
            self.chat = self

        @property
        def completions(self) -> Any:
            return self

        def create(self, **kwargs: Any) -> object:
            return object()

    translator = OpenAICompatibleTranslator(
        client_factory=lambda key, url: _BadClient(),
        environ={"OPENROUTER_API_KEY": "k"},
    )
    with pytest.raises(TranslationError, match="unreadable response"):
        translator.translate(ENGLISH_QUESTION, "ja", "en")


@pytest.mark.usefixtures("no_api_keys")
@pytest.mark.parametrize(
    ("text", "target", "source"),
    [
        (ENGLISH_QUESTION, "en", "en"),
        ("", "ja", "en"),
        ("   ", "ja", "en"),
        (ENGLISH_QUESTION, "", None),
    ],
)
def test_nothing_to_translate_short_circuits_without_a_key(
    text: str, target: str, source: str | None
) -> None:
    """No key, no client, no network — and no error, because no work was needed."""
    assert OpenAICompatibleTranslator().translate(text, target, source) == text


def test_language_name_maps_codes_and_passes_unknowns_through() -> None:
    assert language_name("ja") == "Japanese"
    assert language_name("EN") == "English"
    assert language_name("xx") == "xx"


# --------------------------------------------------------------------------
# prepare_question — the stage as the pipeline sees it
# --------------------------------------------------------------------------


@pytest.mark.usefixtures("requires_detector")
def test_english_question_passes_through_untranslated() -> None:
    """The post's first worked example: English in, English out, no translation."""
    prepared = prepare_question(ENGLISH_QUESTION)
    assert prepared == PreparedQuestion(
        original=ENGLISH_QUESTION,
        language="en",
        confidence=prepared.confidence,
        text_for_retrieval=ENGLISH_QUESTION,
        translated=False,
        target_lang=None,
    )
    assert prepared.retrieval_language == "en"


@pytest.mark.usefixtures("requires_detector")
def test_english_question_translated_to_japanese() -> None:
    """The post's second worked example, with the translator faked out."""
    translator, clients = _fake_translator(JAPANESE_QUESTION, environ={"OPENROUTER_API_KEY": "k"})
    prepared = prepare_question(ENGLISH_QUESTION, "ja", translator)

    assert prepared.language == "en"
    assert prepared.target_lang == "ja"
    assert prepared.translated is True
    assert prepared.text_for_retrieval == JAPANESE_QUESTION
    assert prepared.original == ENGLISH_QUESTION
    assert prepared.retrieval_language == "ja"
    # The detected source language was handed to the translator as a hint.
    assert "English" in clients[0].calls[0]["messages"][1]["content"]


@pytest.mark.usefixtures("requires_detector")
def test_same_language_target_never_calls_the_translator() -> None:
    prepared = prepare_question(ENGLISH_QUESTION, "en", _ExplodingTranslator())
    assert prepared.translated is False
    assert prepared.text_for_retrieval == ENGLISH_QUESTION
    assert prepared.retrieval_language == "en"


@pytest.mark.usefixtures("requires_detector")
@pytest.mark.parametrize("target", ["EN", "en-US", "en_GB", " en "])
def test_target_language_tags_are_normalised(target: str) -> None:
    prepared = prepare_question(ENGLISH_QUESTION, target, _ExplodingTranslator())
    assert prepared.target_lang == "en"
    assert prepared.translated is False


@pytest.mark.usefixtures("requires_detector")
def test_no_translator_degrades_to_the_original_text() -> None:
    prepared = prepare_question(ENGLISH_QUESTION, "ja")
    assert prepared.target_lang == "ja"
    assert prepared.translated is False
    assert prepared.text_for_retrieval == ENGLISH_QUESTION
    assert prepared.retrieval_language == "en"


@pytest.mark.usefixtures("requires_detector")
def test_null_translator_leaves_translated_false() -> None:
    prepared = prepare_question(ENGLISH_QUESTION, "ja", NullTranslator())
    assert prepared.translated is False
    assert prepared.text_for_retrieval == ENGLISH_QUESTION


@pytest.mark.usefixtures("requires_detector", "no_api_keys")
def test_missing_key_surfaces_through_prepare_question() -> None:
    """A translator that was explicitly passed in must not fail silently."""
    with pytest.raises(MissingAPIKeyError, match="OPENROUTER_API_KEY"):
        prepare_question(ENGLISH_QUESTION, "ja", OpenAICompatibleTranslator())


@pytest.mark.usefixtures("requires_detector")
def test_japanese_question_prepares_as_ja() -> None:
    prepared = prepare_question(JAPANESE_QUESTION)
    assert prepared.language == "ja"
    assert prepared.text_for_retrieval == JAPANESE_QUESTION
    assert prepared.translated is False


@pytest.mark.parametrize("text", ["", "   \n "])
def test_blank_question_prepares_without_translating(text: str) -> None:
    prepared = prepare_question(text, "ja", _ExplodingTranslator())
    assert prepared.language == UNKNOWN_LANGUAGE
    assert prepared.confidence == 0.0
    assert prepared.translated is False
    assert prepared.text_for_retrieval == text
    assert prepared.retrieval_language == UNKNOWN_LANGUAGE


@pytest.mark.usefixtures("requires_detector")
def test_prepare_question_is_deterministic() -> None:
    first = prepare_question(ENGLISH_QUESTION, "ja", NullTranslator())
    second = prepare_question(ENGLISH_QUESTION, "ja", NullTranslator())
    assert first == second


def test_normalise_lang_code() -> None:
    assert normalise_lang_code(None) is None
    assert normalise_lang_code("") is None
    assert normalise_lang_code("   ") is None
    assert normalise_lang_code("ja-JP") == "ja"
    assert normalise_lang_code("PT_BR") == "pt"
    assert normalise_lang_code(" Zh ") == "zh"
