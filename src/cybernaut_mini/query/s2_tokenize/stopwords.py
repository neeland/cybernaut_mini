"""Per-language stop-word lists, with a no-op for languages we do not ship.

Stop-word removal is the step that turns the post's worked example from twelve words
into nine tokens: ``what``, ``from``, ``and`` and ``into`` are dropped before stemming.
Removal is done on the *surface* form rather than the stem, because a stemmer maps
several stopwords onto strings that are not themselves stopwords, and matching stems
against a surface list produces surprising misses.

Lists are frozensets of casefolded surface forms. A language with no list returns an
empty set, so the pipeline degrades to "keep everything" instead of failing.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — stage 2, "Multilingual
    Tokenization", which lists "stop-word removal" among the classic NLP steps and shows
    the function words missing from the worked example's token list. Local copy:
    ``docs/blog-archive/the-road-to-cybernaut-1.md``.

Assumptions:
    - The post never publishes its lists, so these are conventional Snowball/NLTK-style
      sets, trimmed of anything that would break the worked example. ``actually`` is
      deliberately *kept* — several published English lists drop it, but the post's
      expected tokens contain ``actual``, so a list containing "actually" would fail.
    - Contraction fragments (``don``, ``ll``, ``ve``, ``t``) are included because the
      word regex splits on the apostrophe, so "don't" arrives as two tokens.
    - Inflected forms are spelled out for the fusional languages. Removal happens before
      stemming, so a list carrying only Snowball's ``какой``/``какая`` would let the
      plural ``какие`` and the oblique ``каким`` straight through; the Russian and German
      lists therefore enumerate the declensions of the determiners rather than relying on
      a stem to collapse them.
    - Japanese and Chinese lists cover particles and function words only. Without a POS
      tagger there is no way to catch inflectional tails, so the Japanese list also holds
      common two-kana fragments, because the MeCab-free segmenter emits hiragana as
      bigrams. ``どの`` is deliberately *not* a stopword: the post's Japanese worked
      example lists it as a token.
    - English digits and single Latin letters are not stopwords. They are cheap in the
      index and occasionally load-bearing ("vitamin d", "type 2").

Alternatives rejected:
    - Pulling lists from NLTK/spaCy at runtime: both need a download, which breaks the
      offline requirement and makes the token set depend on what a machine cached.
    - Corpus-derived stopwords (top-N by document frequency): better calibrated for a
      real index and worth doing at build time, but it makes the tokenizer's output
      depend on a corpus, and a query tokenizer must be a pure function of its input.
    - Removing stopwords after stemming: fewer entries to maintain, but stems collide
      (``us`` -> ``us`` is a stopword, ``US`` the country is not), so it silently deletes
      content words.

      Ordering alone did not save us from that collision. Casefolding used to happen
      before segmentation, so ``US``/``IT``/``WHO`` reached this filter indistinguishable
      from ``us``/``it``/``who`` and were deleted from every English query — on a news
      corpus, three of the most frequent entities in the collection. The tokenizer now
      segments in original case and exempts all-caps tokens here; see
      ``MultilingualTokenizer._filter`` for the rule and its failure modes. The lesson is
      that a stop-word list cannot be evaluated apart from the case pipeline feeding it.
"""

# This module is nothing but word lists in seven scripts, so RUF001 ("ambiguous unicode
# character") fires on every Cyrillic and CJK entry. The characters are the data.
# ruff: noqa: RUF001

from __future__ import annotations

from collections.abc import Iterable

_EN = """
a about above after again against all almost also although always am among an and another
any anyone anything are aren around as at be because been before being below between both
but by came can cannot could couldn did didn do does doesn doing don done down during each
either else enough etc even ever every everyone everything few for from further get got had
hadn has hasn have haven having he hence her here hers herself him himself his how however
i if in indeed instead into is isn it its itself just ll m ma may maybe me might mightn mine
more moreover most mostly much must mustn my myself needn neither never nevertheless no nor
not nothing now o of off often on once one only onto or other others otherwise ought our
ours ourselves out over own per perhaps rather re s same shall shan she should shouldn since
so some someone something still such t than that the their theirs them themselves then
thence there therefore these they thing things this those though through thus to too toward
towards under unless until unto up upon us use used using ve very via was wasn we well were
weren what whatever when whenever where whereas wherever whether which while who whoever
whom whose why will with within without won would wouldn y yet you your yours yourself
yourselves
"""

_ES = """
a al algo algunas algunos ante antes como con contra cual cuando de del desde donde dos e el
ella ellas ellos en entre era erais eran eras eres es esa esas ese eso esos esta estaba
estado estan estas este esto estos ha habia han hasta hay la las le les lo los mas me mi mis
mucho muy nada ni no nos nosotros o os otra otras otro otros para pero poco por porque que
quien se sea segun ser si sin sobre son su sus también tanto te tiene tienen todo todos tu
tus un una uno unos vosotros y ya yo
"""

_FR = """
a au aux avec ce ces dans de des du elle en est et eux il ils je la le les leur lui ma mais
me meme mes moi mon ne nos notre nous on ou par pas pour qu que qui sa se ses son sur ta te
tes toi ton tu un une vos votre vous y d l n s c j qu être avoir plus cette ceux dont
"""

_DE = """
aber alle als also am an auch auf aus bei bin bis bist da damit dann das dass dein deine dem
den der des dessen dich die dies diese dieser dieses doch dort du durch ein eine einem einen
einer eines er es euch euer für hat hatte hier hin ich ihr ihre im in ist ja jede jedem jeden
jeder kann kein keine mein meine mit nach nicht noch nun nur ob oder ohne sein seine sich sie
sind so soll über um und uns unser vom von vor war waren was weil weiter wenn wer werden wie
wieder will wir wird wo zu zum zur
"""

_RU = """
а без бы был была были было быть в вам вас весь во вот все всего всех вы где да для до его
ее если есть еще же за здесь и из из-за или им их к как кто ко когда которое которой которые
который которых какая какие каких каким какое какой какую ли либо мне может мои мой мы на над
надо наш не него нее нет ни них но ну о об однако он она они оно от очень по под при с со так
также такая такие такое такой там те тем то того тоже той только том тот ту ты у уже что
чтобы чье чья эта эти это я
"""

_ZH = """
的 了 和 是 就 都 而 及 与 这 那 你 我 他 她 它 们 在 有 个 上 下 不 也 很 到 说 要 会 着 没有 什么
怎么 为什么 呢 吗 吧 啊 把 被 给 从 对 于 之 其 或 但 如果 因为 所以 一个 可以 已经 还是 以及 以
"""

# Japanese without MeCab arrives as kanji/hiragana bigrams, so this list holds both real
# particles and the two-kana fragments that segmentation actually emits.
_JA = """
の に は を た が で て と し れ さ ある いる も する から な こと として い や れる など なっ ない
この ため その あっ よう また もの という あり まで られ なる へ か だ これ によって により おり
より による ず なり られる において ば なかっ なく しかし について だっ たり ます です
され でし しょ ょう うか のか ので には にお ける てい いる であ です ます そし して また では でも
という いう うこ こと とが がで きる れて てき きた たら らな なけ ければ ばな らない
"""


def _make(raw: str) -> frozenset[str]:
    return frozenset(word.casefold() for word in raw.split())


STOPWORDS: dict[str, frozenset[str]] = {
    "en": _make(_EN),
    "es": _make(_ES),
    "fr": _make(_FR),
    "de": _make(_DE),
    "ru": _make(_RU),
    "zh": _make(_ZH),
    "ja": _make(_JA),
}
"""Casefolded stop-word sets keyed by ISO 639-1 code."""

SUPPORTED_LANGUAGES: frozenset[str] = frozenset(STOPWORDS)

_EMPTY: frozenset[str] = frozenset()


def stopwords_for(language: str, extra: Iterable[str] | None = None) -> frozenset[str]:
    """Return the stop-word set for ``language``, empty when none is shipped.

    ``extra`` is casefolded and unioned in, so a caller can extend a shipped list or
    supply one for a language this module does not cover.
    """
    base = STOPWORDS.get(language, _EMPTY)
    if extra is None:
        return base
    added = frozenset(word.casefold() for word in extra if word)
    return base | added
