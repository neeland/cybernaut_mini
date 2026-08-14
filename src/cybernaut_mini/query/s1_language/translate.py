"""Stage 1b: translate a question into the language the results are requested in.

Three implementations behind one protocol: :class:`NullTranslator` (the offline
default, an identity function), :class:`OpenAICompatibleTranslator` (a real call to
any OpenAI-shaped ``/v1`` endpoint — OpenRouter, OpenAI, or a local server), and
whatever a caller supplies of its own. The protocol is what the pipeline depends on,
so the query path never imports a network client it may not be able to use.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 1: "if results are
    requested in a different language, translate that question into the requested
    language", with `gemma-3n-e4b-it` (140 languages, via OpenRouter) named as the
    translator. The post's worked example translates "What lessons from bacteria and
    yeast actually translate into safer gene-editing medicines?" into Japanese.
    Local copy: ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions:
    - The credential is read from the environment on EVERY call, never at import time
      and never once-and-cached. Importing this module, and constructing the
      translator, must work on a machine with no keys at all — the whole offline test
      suite depends on that, and so does a user who only ever runs the deterministic
      path. "Every call" is the part that is easy to get wrong: a key added to
      ``.env`` mid-session, or rotated by a credential helper, has to take effect on
      the next translation, so the resolved key is compared against the one the
      cached client was built with and the client is rebuilt only when they differ.
    - ``OPENROUTER_API_KEY`` is preferred over ``OPENAI_API_KEY`` because the default
      ``base_url`` is OpenRouter's, which is the provider the post itself uses.
      ``ANTHROPIC_API_KEY`` is deliberately NOT consulted: Anthropic's API is not
      OpenAI-shaped at this base URL, so accepting it would produce a 401 at the far
      end instead of the actionable error raised here.
    - Translation is a single non-streaming chat completion at ``temperature=0``.
      That is as deterministic as a hosted model gets; true reproducibility is not
      available from any remote provider, so callers that need byte-stability cache
      the output rather than trusting the endpoint.
    - The model is instructed to return the translation and nothing else, and the
      response is stripped of surrounding whitespace and of a wrapping code fence.
      Small instruction-tuned models do occasionally add one; no other repair is
      attempted, because silently editing a translation is worse than returning it.
    - Language codes are passed to the model as English language names where known
      (``ja`` → ``Japanese``) and as the bare code otherwise. A 140-language name
      table is not worth carrying; the model resolves unfamiliar codes itself.

Alternatives rejected:
    - The ``anthropic`` SDK, or a provider-specific client per vendor: rejected
      because one OpenAI-compatible client already covers OpenRouter (the post's
      provider), OpenAI, vLLM, Ollama and llama.cpp behind a single ``base_url``.
    - Reading the key in ``__init__``: the obvious place, and it gives a faster
      failure. Rejected because it makes the class unconstructible in tests and in
      any process that builds the pipeline before deciding whether to translate.
    - A local translation model (NLLB / M2M-100 via transformers): offline and
      reproducible, but it pulls torch into the core install for a stage the post
      explicitly implements with a hosted LLM.
    - Auto-instantiating a real client when a key happens to be present: rejected
      because a stage that silently starts making paid network calls depending on the
      environment is not something a test suite can defend.
    - Caching the client after the first successful key lookup and never revisiting
      it: one less environment read per call, but it quietly downgrades "read at call
      time" to "read at first call", which is a different and much weaker promise —
      a key exported after the first translation would never be seen and there would
      be no way to reset the translator short of rebuilding it.
    - Rebuilding the client on every call instead: correct but wasteful, since the
      OpenAI client owns a connection pool that a fresh instance throws away. Keying
      the cache on the resolved credential gets both properties.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "API_KEY_ENV_VARS",
    "DEFAULT_BASE_URL",
    "DEFAULT_TRANSLATION_MODEL",
    "MissingAPIKeyError",
    "NullTranslator",
    "OpenAICompatibleTranslator",
    "TranslationError",
    "Translator",
    "language_name",
]

#: The post's translator: gemma-3n-e4b-it, served by OpenRouter.
DEFAULT_TRANSLATION_MODEL = "google/gemma-3n-e4b-it"

#: OpenRouter's OpenAI-compatible endpoint. Point this at ``https://api.openai.com/v1``
#: or ``http://localhost:11434/v1`` to use another provider with the same client.
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

#: Consulted in order at call time. ANTHROPIC_API_KEY is intentionally absent.
API_KEY_ENV_VARS: tuple[str, ...] = ("OPENROUTER_API_KEY", "OPENAI_API_KEY")

#: English names for the codes lid.176 emits most often. Anything missing is passed
#: to the model as its bare code, which the 140-language translator resolves fine.
_LANGUAGE_NAMES: Mapping[str, str] = {
    "ar": "Arabic",
    "bn": "Bengali",
    "cs": "Czech",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sv": "Swedish",
    "sw": "Swahili",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh": "Chinese",
}

_SYSTEM_PROMPT = (
    "You are a translation engine inside a search pipeline. You translate the user's "
    "search question into the requested language and output nothing else: no "
    "explanation, no transliteration, no quotation marks, no preamble. Preserve named "
    "entities, numbers, and the question's interrogative form."
)


class TranslationError(RuntimeError):
    """Raised when a translation could not be produced."""


class MissingAPIKeyError(TranslationError):
    """Raised when no usable API key is present in the environment."""


def language_name(code: str) -> str:
    """English name for an ISO language *code*, or the code itself if unknown."""
    return _LANGUAGE_NAMES.get(code.strip().lower(), code.strip())


@runtime_checkable
class Translator(Protocol):
    """Anything that can turn *text* into *target_lang*."""

    def translate(self, text: str, target_lang: str, source_lang: str | None = None) -> str:
        """Return *text* rendered in *target_lang*.

        *source_lang* is a hint, not a requirement: implementations that can detect
        the source themselves may ignore it.
        """
        ...


class NullTranslator:
    """Identity translator: returns the input unchanged.

    The default everywhere. It is what stage 1 uses when the requested language
    already matches the detected one, when no language was requested, and when the
    process is offline — in all three cases the correct output is the original
    question, so this is a real implementation rather than a stub.
    """

    def translate(self, text: str, target_lang: str, source_lang: str | None = None) -> str:
        return text


#: What a Markdown fence info string looks like: a short, lowercase, space-free ASCII
#: token ("json", "text", "ja"). Deliberately narrow — a translation can also be a
#: single space-free token ("Bonjour") or a whole space-free CJK sentence, and
#: mistaking either for an info string deletes the translation.
_FENCE_INFO_RE = re.compile(r"^[a-z0-9_+.#-]{1,20}$")


def _strip_code_fence(text: str) -> str:
    """Unwrap a ```-fenced block if the model wrapped its answer in one."""
    stripped = text.strip()
    if not stripped.startswith("```") or not stripped.endswith("```"):
        return stripped
    inner = stripped[3:-3]
    first_newline = inner.find("\n")
    if first_newline == -1:
        return inner.strip()
    first_line = inner[:first_newline].strip()
    body = inner[first_newline + 1 :]
    # ```text\n...\n``` — drop the info string, but only when the first line really
    # looks like one AND dropping it leaves something behind. ```\n...\n```, and
    # ```Bonjour\n```, and a fence followed by a space-free CJK sentence, all keep
    # everything.
    if _FENCE_INFO_RE.match(first_line) and body.strip():
        return body.strip()
    return inner.strip()


class OpenAICompatibleTranslator:
    """Translate via any OpenAI-compatible chat-completions endpoint.

    Constructing this is free and offline: no key is read, no client is built, no
    network is touched until :meth:`translate` is called. The credential is then
    re-resolved on every call, and the underlying client is rebuilt whenever the
    resolved credential differs from the one it was built with — so a key added to
    the environment mid-session is picked up without a restart, and an unchanged key
    reuses the same client rather than paying for a new one per translation.

    Parameters
    ----------
    model:
        Model id at the endpoint. Defaults to the post's ``google/gemma-3n-e4b-it``.
    base_url:
        OpenAI-compatible ``/v1`` root. Defaults to OpenRouter's.
    api_key:
        Explicit key. When ``None`` (the default) the environment is consulted on
        every call, in the order of :data:`API_KEY_ENV_VARS`.
    temperature:
        Sampling temperature; ``0.0`` for the most reproducible output available.
    timeout:
        Per-request timeout in seconds, passed to the client.
    client_factory:
        Builds the underlying client from ``(api_key, base_url)``. Defaults to
        ``openai.OpenAI``. This is the seam tests use to prove the wiring without a
        network, and the seam a caller uses to inject a pre-configured client.
    environ:
        Mapping consulted for the API key. Defaults to :data:`os.environ`, re-read on
        every call so a key exported — or changed — after construction is picked up
        without rebuilding the translator.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_TRANSLATION_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout: float = 60.0,
        client_factory: Callable[[str, str], Any] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.timeout = timeout
        self._api_key = api_key
        self._client_factory = client_factory
        self._environ = environ
        self._client: Any | None = None
        #: The credential ``self._client`` was built with. Compared against a freshly
        #: resolved key on every call so a rotated key rebuilds the client.
        self._client_api_key: str | None = None

    # -- credential handling ------------------------------------------------

    def _resolve_api_key(self) -> str:
        if self._api_key:
            return self._api_key
        environ: Mapping[str, str] = os.environ if self._environ is None else self._environ
        for name in API_KEY_ENV_VARS:
            value = environ.get(name, "").strip()
            if value:
                return value
        names = " or ".join(API_KEY_ENV_VARS)
        msg = (
            f"No API key for {self.base_url}: set {names} in the environment "
            f"(export {API_KEY_ENV_VARS[0]}=...) or pass api_key= to "
            f"{type(self).__name__}. ANTHROPIC_API_KEY is not accepted here — this "
            f"client speaks the OpenAI chat-completions protocol."
        )
        raise MissingAPIKeyError(msg)

    def _default_client_factory(self, api_key: str, base_url: str) -> Any:
        """Build a real ``openai.OpenAI`` client.

        Imported here rather than at module scope so that importing stage 1 costs
        nothing on the offline path.
        """
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover — openai is a core dependency
            msg = "the 'openai' package is required to translate; install it with `uv sync`"
            raise TranslationError(msg) from exc
        return OpenAI(api_key=api_key, base_url=base_url, timeout=self.timeout)

    def _get_client(self) -> Any:
        """The client for the credential that is in the environment *right now*.

        Resolving the key is a dict lookup, so it happens on every call; building the
        client is not, so it happens only when the key that would be used differs from
        the key the cached client already holds. A key that appears mid-session (a
        freshly written ``.env``), or one that is rotated, therefore takes effect on
        the next translation, while an unchanged key keeps reusing one client and its
        connection pool.
        """
        api_key = self._resolve_api_key()
        if self._client is None or self._client_api_key != api_key:
            factory = self._client_factory or self._default_client_factory
            self._client = factory(api_key, self.base_url)
            self._client_api_key = api_key
        return self._client

    # -- translation --------------------------------------------------------

    def build_messages(
        self, text: str, target_lang: str, source_lang: str | None = None
    ) -> list[dict[str, str]]:
        """The exact prompt sent to the model, exposed so tests can assert on it."""
        target = language_name(target_lang)
        origin = (
            f"The text is written in {language_name(source_lang)}. "
            if source_lang and source_lang != "und"
            else ""
        )
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"{origin}Translate the following search question into {target}. "
                    f"Output only the translation.\n\n{text}"
                ),
            },
        ]

    def translate(self, text: str, target_lang: str, source_lang: str | None = None) -> str:
        """Translate *text* into *target_lang*, returning the model's output.

        Short-circuits — with no key lookup and no network call — when there is
        nothing to translate: blank text, a blank target, or a target that already
        equals the source.
        """
        if not text.strip() or not target_lang.strip():
            return text
        if source_lang and source_lang.strip().lower() == target_lang.strip().lower():
            return text

        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=self.build_messages(text, target_lang, source_lang),
            temperature=self.temperature,
        )
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            msg = f"translation endpoint returned an unreadable response: {response!r}"
            raise TranslationError(msg) from exc

        translated = _strip_code_fence(content or "")
        if not translated:
            msg = f"translation endpoint returned empty content for model {self.model!r}"
            raise TranslationError(msg)
        return translated
