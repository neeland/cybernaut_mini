"""The Resolver: turn a promoted surface into the post's entity record, with receipts.

Two implementations of one flow. :class:`DeterministicResolver` consults three free
tools — Wikipedia, the Wikidata entity API, and the SEC's cached
``company_tickers.json`` — merges what they return, and accepts the result only when
at least two independent sources agree on who the entity is.
:class:`ResolverAgent` runs the same three tools under an LLM's control through the
repo's OpenAI-compatible client, for the cases where deterministic matching is not
enough; its output passes through the *same* acceptance rule, so the model can
propose but never self-certify.

Every network access is doubly opt-in: each tool caches to JSON files and a cache
miss raises unless ``CYBERNAUT_MINI_ENTITIES_NETWORK=1`` is exported. Accepted
entities are stored keyed by Wikidata QID, which is what makes cross-collection
resolution "extremely cache friendly": the second collection to suggest JPMorgan is
answered by one SELECT.

Blog ref: https://nosible.com/blog/can-faceted-search-at-web-scale-self-organize —
    "The Resolver Agent uses a bunch of tools (Search, Wikipedia, LinkedIn, Market
    APIs, etc.) to work out who this entity is. This step is obviously slow and
    expensive, but it's also extremely cache friendly." The record schema is copied
    field-for-field from the post's JPMorgan example
    (``configs/entities/jpmorgan.json``). Local copy:
    ``docs/blog-archive/can-faceted-search-at-web-scale-self-organize.md``.

Assumptions:
    - The tool set is Wikipedia + Wikidata + SEC EDGAR [gap-analysis choice]: the
      post's LinkedIn and paid market APIs (FIGI, EODHD, OpenCorporates, SERP) have
      no free equivalent, so their fields stay ``None`` and are documented as stubs.
      Wikidata properties carry the identifiers the post shows: ISIN ``P946``, LEI
      ``P1278``, CIK ``P5531``, subsidiaries ``P355``, website ``P856``, and the
      ticker as a ``P249`` qualifier on the stock-exchange claim ``P414``.
    - Acceptance needs >=2 *independent* sources agreeing on the entity's identity
      [inferred — the post never states its acceptance rule]. Two sources agree when
      they share a ticker, a CIK, or a suffix-stripped normalised name; requiring
      two makes a single hallucinated or mismatched lookup insufficient.
    - ``ticker_eodhd`` is rendered EODHD-style as ``<TICKER>.US`` when the ticker
      came from the SEC file — an SEC listing is a US listing, which is exactly the
      information that suffix encodes.
    - ``cik_code`` is zero-padded to ten digits, matching both the post's record and
      Wikidata's own formatting of P5531.
    - Caches are canonical-JSON files keyed by a slug of the query (or the QID), so
      a warm cache makes every tool fully offline and deterministic; the repo ships
      real cached lookups for the post's own worked example under
      ``data/01_raw/entities/``.
    - The LLM agent is judged by its tools: evidence is captured as the tool calls
      execute, and acceptance is recomputed from that evidence, never from the
      model's confidence. A record whose QID no tool returned is rejected outright.

Alternatives rejected:
    - Filling the geographic/sector fields from EDGAR presence ("SEC ⇒ US"): true
      for the country in almost every case, but it is one source asserting a fact no
      second source confirms, and the acceptance rule exists precisely to forbid
      that; the fields stay ``None`` until a real second source is wired in.
    - A search-engine tool as the post's "Search": every free option is either
      rate-limited into uselessness or an API key away from being a paid API;
      Wikipedia's own search endpoint covers the discovery role for the entities a
      news corpus surfaces.
    - Letting the agent loop until it declares itself done: an unbounded tool loop
      with a paid endpoint is a cost bug waiting to happen; the loop is capped at
      ``max_rounds`` and the final message must already be the JSON record.
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from cybernaut_mini.entities.store import EntityStore
from cybernaut_mini.entities.tagger import normalise_surface
from cybernaut_mini.models import canonical_dumps
from cybernaut_mini.query.s1_language.translate import API_KEY_ENV_VARS, DEFAULT_BASE_URL

__all__ = [
    "ACCEPTANCE_MIN_SOURCES",
    "DEFAULT_CACHE_ROOT",
    "DEFAULT_EDGAR_PATH",
    "DEFAULT_RESOLVER_MODEL",
    "NETWORK_ENV",
    "DeterministicResolver",
    "EdgarCompanyTickers",
    "NetworkDisabledError",
    "Resolution",
    "ResolverAgent",
    "ResolverError",
    "ResolverRecord",
    "SourceEvidence",
    "WikidataTool",
    "WikipediaTool",
    "entity_id_for",
    "resolve_and_store",
    "sources_in_agreement",
]

#: Export this to allow the tools to touch the network on a cache miss.
NETWORK_ENV = "CYBERNAUT_MINI_ENTITIES_NETWORK"

#: Independent sources that must agree before an entity is accepted.
ACCEPTANCE_MIN_SOURCES = 2

#: Where the repo keeps real cached tool responses (committed for the worked example).
DEFAULT_CACHE_ROOT = Path("data/01_raw/entities")

#: The SEC's public ticker↔CIK file, cached once and read offline thereafter.
DEFAULT_EDGAR_PATH = Path("data/01_raw/edgar/company_tickers.json")

#: A small instruction-tuned model is enough to drive three tools.
DEFAULT_RESOLVER_MODEL = "qwen/qwen-2.5-7b-instruct"

_USER_AGENT = "cybernaut-mini/0.1 (research replica; entities resolver)"

_WIKIDATA_API = "https://www.wikidata.org/w/api.php"
_WIKIDATA_ENTITY = "https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
_WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
_WIKIPEDIA_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"


class ResolverError(RuntimeError):
    """Raised when resolution cannot proceed."""


class NetworkDisabledError(ResolverError):
    """A tool needed the network but the opt-in gate is not set."""


class ResolverRecord(BaseModel):
    """The post's Resolver output schema, field for field.

    Every key of the published JPMorgan record appears here with the same name.
    Fields the free tool set cannot fill (LinkedIn, FIGI, EODHD, OpenCorporates,
    SERP, geography, GICS sectors) default to ``None`` and are stubs, exactly as
    documented in the module docstring.
    """

    model_config = ConfigDict(extra="forbid")

    name_orig: str | None = None
    name_kgid: str | None = None
    name_wiki: str | None = None
    name_figi: str | None = None
    name_eodhd: str | None = None
    name_lei: str | None = None
    ticker_wiki: str | None = None
    ticker_figi: str | None = None
    ticker_eodhd: str | None = None
    ticker_serp: str | None = None
    kgid: str | None = None
    wiki_qid: str | None = None
    cik_code: str | None = None
    isin_code: str | None = None
    cusip_code: str | None = None
    lei_code: str | None = None
    figi_code: str | None = None
    open_corp_code: str | None = None
    ein_code: str | None = None
    exchange: str | None = None
    mic_code: str | None = None
    exch_code: str | None = None
    ccy_name: str | None = None
    ccy_code: str | None = None
    ccy_symbol: str | None = None
    wiki_page: str | None = None
    website: str | None = None
    linkedin: str | None = None
    is_company: bool | None = None
    is_public: bool | None = None
    is_private: bool | None = None
    is_delisted: bool | None = None
    continent: str | None = None
    region: str | None = None
    country: str | None = None
    country_iso: str | None = None
    city: str | None = None
    address: str | None = None
    phone_num: str | None = None
    sector: str | None = None
    industry_group: str | None = None
    industry: str | None = None
    sub_industry: str | None = None
    historical_names: list[str] = []
    brand_names: list[str] = []
    subsidiaries: list[str] = []
    short_wiki: str | None = None
    summary_raw: str | None = None
    summary_wiki: str | None = None
    summary_eodhd: str | None = None
    markdown_wiki: str | None = None
    debug_verbose: bool = False


# ------------------------------------------------------------------ #
# Shared plumbing                                                      #
# ------------------------------------------------------------------ #


def _network_allowed(environ: Mapping[str, str] | None) -> bool:
    env: Mapping[str, str] = os.environ if environ is None else environ
    return env.get(NETWORK_ENV, "").strip() not in {"", "0", "false"}


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "empty"


def _http_get_json(url: str, timeout: float = 30.0) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        msg = f"expected a JSON object from {url}"
        raise ResolverError(msg)
    return payload


class _CachedFetcher:
    """Cache-or-fetch for one tool's JSON lookups, network behind the env gate."""

    def __init__(self, cache_dir: Path, environ: Mapping[str, str] | None = None) -> None:
        self.cache_dir = cache_dir
        self._environ = environ

    def get(self, key: str, fetch: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            return dict(cached)
        if not _network_allowed(self._environ):
            msg = (
                f"no cached response at {path} and network is disabled; "
                f"export {NETWORK_ENV}=1 to allow this lookup"
            )
            raise NetworkDisabledError(msg)
        value = fetch()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(canonical_dumps(value) + "\n", encoding="utf-8")
        return value


# ------------------------------------------------------------------ #
# Tools                                                                #
# ------------------------------------------------------------------ #


class WikipediaTool:
    """Wikipedia search + page summary, cached per query/title."""

    name = "wikipedia"

    def __init__(
        self,
        cache_dir: Path = DEFAULT_CACHE_ROOT / "wikipedia",
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self._fetcher = _CachedFetcher(cache_dir, environ)

    def search(self, query: str) -> list[str]:
        """Page titles matching *query*, best first."""

        def fetch() -> dict[str, Any]:
            url = (
                f"{_WIKIPEDIA_API}?action=query&list=search&format=json&srlimit=5"
                f"&srsearch={urllib.parse.quote(query)}"
            )
            raw = _http_get_json(url)
            titles = [str(r["title"]) for r in raw.get("query", {}).get("search", [])]
            return {"query": query, "results": titles}

        return [str(t) for t in self._fetcher.get(f"search_{_slug(query)}", fetch)["results"]]

    def summary(self, title: str) -> dict[str, Any]:
        """``{"title", "extract", "url", "page"}`` for one page."""

        def fetch() -> dict[str, Any]:
            url = _WIKIPEDIA_SUMMARY.format(title=urllib.parse.quote(title.replace(" ", "_")))
            raw = _http_get_json(url)
            return {
                "title": str(raw.get("title", title)),
                "extract": str(raw.get("extract", "")),
                "url": str(raw.get("content_urls", {}).get("desktop", {}).get("page", "")),
                "page": str(raw.get("titles", {}).get("canonical", title.replace(" ", "_"))),
            }

        return self._fetcher.get(f"summary_{_slug(title)}", fetch)


class WikidataTool:
    """Wikidata entity search + normalised entity data, cached per query/QID."""

    name = "wikidata"

    def __init__(
        self,
        cache_dir: Path = DEFAULT_CACHE_ROOT / "wikidata",
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self._fetcher = _CachedFetcher(cache_dir, environ)

    def search(self, query: str) -> list[dict[str, str]]:
        """``[{"id", "label", "description"}]`` matches for *query*, best first."""

        def fetch() -> dict[str, Any]:
            url = (
                f"{_WIKIDATA_API}?action=wbsearchentities&language=en&format=json&limit=5"
                f"&search={urllib.parse.quote(query)}"
            )
            raw = _http_get_json(url)
            results = [
                {
                    "id": str(r.get("id", "")),
                    "label": str(r.get("label", "")),
                    "description": str(r.get("description", "")),
                }
                for r in raw.get("search", [])
            ]
            return {"query": query, "results": results}

        results = self._fetcher.get(f"search_{_slug(query)}", fetch)["results"]
        return [dict(r) for r in results]

    def entity(self, qid: str) -> dict[str, Any]:
        """Normalised entity data for *qid*, subsidiary labels included.

        Keys: ``qid, label, description, official_name, ticker, exchange, isin,
        lei, cik, website, subsidiaries, chief_executive, wiki_page``.
        """

        def fetch() -> dict[str, Any]:
            raw = _http_get_json(_WIKIDATA_ENTITY.format(qid=qid))
            entity = raw["entities"][qid]
            return _normalise_wikidata_entity(qid, entity, self._labels)

        return self._fetcher.get(f"entity_{qid}", fetch)

    def _labels(self, qids: Sequence[str]) -> dict[str, str]:
        if not qids:
            return {}
        url = (
            f"{_WIKIDATA_API}?action=wbgetentities&props=labels&languages=en&format=json"
            f"&ids={urllib.parse.quote('|'.join(qids))}"
        )
        raw = _http_get_json(url)
        labels: dict[str, str] = {}
        for qid, payload in raw.get("entities", {}).items():
            label = payload.get("labels", {}).get("en", {}).get("value")
            if label:
                labels[str(qid)] = str(label)
        return labels


def _claim_values(entity: Mapping[str, Any], prop: str) -> list[Any]:
    values: list[Any] = []
    for claim in entity.get("claims", {}).get(prop, []):
        datavalue = claim.get("mainsnak", {}).get("datavalue")
        if datavalue is not None:
            values.append(datavalue.get("value"))
    return values


def _normalise_wikidata_entity(
    qid: str,
    entity: Mapping[str, Any],
    fetch_labels: Callable[[Sequence[str]], dict[str, str]],
) -> dict[str, Any]:
    """Reduce raw ``Special:EntityData`` JSON to the fields the record needs."""
    ticker: str | None = None
    exchange_qid: str | None = None
    for claim in entity.get("claims", {}).get("P414", []):
        qualifiers = claim.get("qualifiers", {})
        if "P249" in qualifiers and "P582" not in qualifiers:  # active listing only
            ticker = str(qualifiers["P249"][0]["datavalue"]["value"])
            value = claim.get("mainsnak", {}).get("datavalue", {}).get("value", {})
            exchange_qid = value.get("id")
            break
    subsidiary_qids = [str(v["id"]) for v in _claim_values(entity, "P355") if isinstance(v, dict)]
    ceo_values = _claim_values(entity, "P169")
    ceo_qid = str(ceo_values[0]["id"]) if ceo_values and isinstance(ceo_values[0], dict) else None
    label_qids = list(subsidiary_qids)
    if ceo_qid:
        label_qids.append(ceo_qid)
    if exchange_qid:
        label_qids.append(exchange_qid)
    labels = fetch_labels(label_qids)

    def first_str(prop: str) -> str | None:
        values = _claim_values(entity, prop)
        return str(values[0]) if values else None

    official = _claim_values(entity, "P1448")
    websites = [str(v) for v in _claim_values(entity, "P856")]
    return {
        "qid": qid,
        "label": str(entity.get("labels", {}).get("en", {}).get("value", "")),
        "description": str(entity.get("descriptions", {}).get("en", {}).get("value", "")),
        "official_name": (
            str(official[0]["text"]) if official and isinstance(official[0], dict) else None
        ),
        "ticker": ticker,
        "exchange": labels.get(exchange_qid or "", None),
        "isin": first_str("P946"),
        "lei": first_str("P1278"),
        "cik": first_str("P5531"),
        "website": websites[0] if websites else None,
        "subsidiaries": [labels[q] for q in subsidiary_qids if q in labels],
        "chief_executive": labels.get(ceo_qid or "", None),
        "wiki_page": str(
            entity.get("sitelinks", {}).get("enwiki", {}).get("title", "")
        )
        or None,
    }


class EdgarCompanyTickers:
    """The SEC's ``company_tickers.json``, read from a committed local cache.

    The repo ships a real excerpt of the file; :meth:`refresh` re-downloads the full
    file and is the only networked path, behind the same env gate as the other
    tools.
    """

    name = "edgar"

    def __init__(
        self,
        path: Path = DEFAULT_EDGAR_PATH,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.path = path
        self._environ = environ
        self._rows: list[dict[str, Any]] | None = None

    def refresh(self) -> None:
        if not _network_allowed(self._environ):
            msg = f"refreshing {self.path} needs the network; export {NETWORK_ENV}=1"
            raise NetworkDisabledError(msg)
        raw = _http_get_json("https://www.sec.gov/files/company_tickers.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(canonical_dumps(raw) + "\n", encoding="utf-8")
        self._rows = None

    def _load(self) -> list[dict[str, Any]]:
        if self._rows is None:
            if not self.path.exists():
                msg = (
                    f"no cached SEC company_tickers.json at {self.path}; "
                    f"call refresh() with {NETWORK_ENV}=1 to fetch it"
                )
                raise ResolverError(msg)
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._rows = [dict(row) for row in raw.values()]
        return self._rows

    def lookup(self, query: str) -> list[dict[str, Any]]:
        """Rows matching *query* as a ticker (exact) or a company name (prefix)."""
        normalized = _norm_name(query)
        if not normalized:
            return []
        matches: list[dict[str, Any]] = []
        for row in self._load():
            ticker = str(row.get("ticker", ""))
            title_norm = _norm_name(str(row.get("title", "")))
            if normalized == ticker.lower() or (
                len(normalized) > 3
                and (title_norm.startswith(normalized) or normalized.startswith(title_norm))
            ):
                matches.append(row)
        return matches


# ------------------------------------------------------------------ #
# Agreement and acceptance                                             #
# ------------------------------------------------------------------ #

_NAME_NOISE = re.compile(r"[^a-z0-9 ]+")
_NAME_SUFFIXES = (
    " and company",
    " and co",
    " corporation",
    " incorporated",
    " company",
    " limited",
    " corp",
    " inc",
    " ltd",
    " llc",
    " plc",
    " co",
)


def _norm_name(name: str) -> str:
    """Casefold, strip punctuation and trailing corporate suffixes, for agreement."""
    text = _NAME_NOISE.sub(" ", name.casefold().replace("&", " and "))
    text = " ".join(text.split())
    changed = True
    while changed:
        changed = False
        for suffix in _NAME_SUFFIXES:
            if text.endswith(suffix) and len(text) > len(suffix):
                text = text[: -len(suffix)].strip()
                changed = True
    return text


@dataclass(frozen=True)
class SourceEvidence:
    """What one independent source asserted about the entity's identity."""

    source: str
    name: str | None = None
    ticker: str | None = None
    qid: str | None = None
    cik: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


def _agrees(a: SourceEvidence, b: SourceEvidence) -> bool:
    if a.ticker and b.ticker and a.ticker.upper() == b.ticker.upper():
        return True
    if a.qid and b.qid and a.qid == b.qid:
        return True
    if a.cik and b.cik and int(a.cik) == int(b.cik):
        return True
    return bool(a.name and b.name and _norm_name(a.name) == _norm_name(b.name))


def sources_in_agreement(evidence: Sequence[SourceEvidence]) -> tuple[str, ...]:
    """Distinct source names forming the largest identity-agreeing group."""
    best: tuple[str, ...] = ()
    for anchor in evidence:
        agreeing = {e.source for e in evidence if _agrees(anchor, e)}
        agreeing.add(anchor.source)
        group = tuple(sorted(agreeing))
        if len(group) > len(best):
            best = group
    return best


@dataclass(frozen=True)
class Resolution:
    """A Resolver verdict: the record, whether it was accepted, and why."""

    record: ResolverRecord
    accepted: bool
    sources: tuple[str, ...]
    evidence: tuple[SourceEvidence, ...]
    resolved_by: str


def entity_id_for(record: ResolverRecord, surface: str = "") -> str:
    """QID-first entity key; falls back to the normalised name."""
    if record.wiki_qid:
        return record.wiki_qid
    name = record.name_orig or record.name_wiki or surface
    return f"name:{normalise_surface(name)}"


# ------------------------------------------------------------------ #
# Deterministic resolver                                               #
# ------------------------------------------------------------------ #


class DeterministicResolver:
    """Tool lookups, merged mechanically — the fallback that needs no LLM at all."""

    def __init__(
        self,
        wikipedia: WikipediaTool | None = None,
        wikidata: WikidataTool | None = None,
        edgar: EdgarCompanyTickers | None = None,
    ) -> None:
        self.wikipedia = wikipedia if wikipedia is not None else WikipediaTool()
        self.wikidata = wikidata if wikidata is not None else WikidataTool()
        self.edgar = edgar if edgar is not None else EdgarCompanyTickers()

    def gather_evidence(
        self, surface: str
    ) -> tuple[tuple[SourceEvidence, ...], dict[str, Any]]:
        """Run all three tools; a tool that is offline-blocked simply abstains."""
        evidence: list[SourceEvidence] = []
        detail: dict[str, Any] = {}
        try:
            rows = self.edgar.lookup(surface)
        except ResolverError:
            rows = []
        if rows:
            row = rows[0]
            detail["edgar"] = row
            evidence.append(
                SourceEvidence(
                    source="edgar",
                    name=str(row.get("title", "")) or None,
                    ticker=str(row.get("ticker", "")) or None,
                    cik=str(row.get("cik_str", "")) or None,
                    extra=row,
                )
            )
        try:
            matches = self.wikidata.search(surface)
            if matches:
                entity = self.wikidata.entity(matches[0]["id"])
                detail["wikidata"] = entity
                evidence.append(
                    SourceEvidence(
                        source="wikidata",
                        name=str(entity.get("label", "")) or None,
                        ticker=entity.get("ticker"),
                        qid=str(entity.get("qid", "")) or None,
                        cik=entity.get("cik"),
                        extra=entity,
                    )
                )
        except NetworkDisabledError:
            pass
        try:
            titles = self.wikipedia.search(surface)
            if titles:
                summary = self.wikipedia.summary(titles[0])
                detail["wikipedia"] = summary
                evidence.append(
                    SourceEvidence(
                        source="wikipedia",
                        name=str(summary.get("title", "")) or None,
                        extra=summary,
                    )
                )
        except NetworkDisabledError:
            pass
        return tuple(evidence), detail

    def resolve(self, surface: str) -> Resolution:
        evidence, detail = self.gather_evidence(surface)
        sources = sources_in_agreement(evidence)
        accepted = len(sources) >= ACCEPTANCE_MIN_SOURCES
        record = self._build_record(surface, detail, sources)
        return Resolution(
            record=record,
            accepted=accepted,
            sources=sources,
            evidence=evidence,
            resolved_by="resolver-deterministic",
        )

    @staticmethod
    def _build_record(
        surface: str, detail: Mapping[str, Any], sources: tuple[str, ...]
    ) -> ResolverRecord:
        wikidata = dict(detail.get("wikidata", {})) if "wikidata" in sources else {}
        edgar = dict(detail.get("edgar", {})) if "edgar" in sources else {}
        wikipedia = dict(detail.get("wikipedia", {})) if "wikipedia" in sources else {}
        ticker = wikidata.get("ticker") or (str(edgar["ticker"]) if edgar.get("ticker") else None)
        cik_raw = wikidata.get("cik") or (
            str(edgar["cik_str"]) if edgar.get("cik_str") is not None else None
        )
        return ResolverRecord(
            name_orig=surface,
            name_wiki=wikidata.get("label") or (wikipedia.get("title") if wikipedia else None),
            name_lei=wikidata.get("official_name"),
            name_eodhd=str(edgar["title"]) if edgar.get("title") else None,
            ticker_wiki=wikidata.get("ticker"),
            ticker_eodhd=f"{edgar['ticker']}.US" if edgar.get("ticker") else None,
            wiki_qid=wikidata.get("qid"),
            cik_code=str(cik_raw).zfill(10) if cik_raw else None,
            isin_code=wikidata.get("isin"),
            lei_code=wikidata.get("lei"),
            exchange=wikidata.get("exchange"),
            wiki_page=(
                str(wikidata.get("wiki_page") or "").replace(" ", "_") or None
                if wikidata
                else (wikipedia.get("page") if wikipedia else None)
            ),
            website=wikidata.get("website"),
            linkedin=None,  # stub: no free LinkedIn API
            is_company=True if (edgar or wikidata.get("ticker")) else None,
            is_public=True if ticker else None,
            is_private=False if ticker else None,
            subsidiaries=[str(s) for s in wikidata.get("subsidiaries", [])],
            short_wiki=wikipedia.get("extract") if wikipedia else None,
        )


# ------------------------------------------------------------------ #
# LLM agent                                                            #
# ------------------------------------------------------------------ #

_AGENT_SYSTEM_PROMPT = (
    "You are the Resolver Agent of a self-organizing search index. You are given a "
    "suggested entity surface discovered in one collection of documents. Use the "
    "available tools to work out who this entity is, then output ONLY a JSON object "
    "with the resolver record fields you could verify. Use null for fields no tool "
    "confirmed. Do not invent identifiers."
)

_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "wikipedia_lookup",
            "description": "Search Wikipedia and return the best page's title and summary.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wikidata_entity",
            "description": (
                "Search Wikidata and return the top entity's QID, label, ticker, "
                "ISIN, LEI, CIK, website and subsidiaries."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edgar_tickers",
            "description": "Look a name or ticker up in the cached SEC company_tickers.json.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]

_FENCE_RE = re.compile(r"^```[a-z]*\n(.*)\n```$", re.S)


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    match = _FENCE_RE.match(stripped)
    return match.group(1).strip() if match else stripped


class ResolverAgent:
    """The post's Resolver Agent: an LLM driving the three tools, then judged by them.

    Uses the repo's OpenAI-compatible client discipline (same env vars and default
    base URL as stage 1's translator; ``client_factory`` is the offline test seam).
    Evidence is recorded as the model's tool calls execute, and the final record is
    accepted only if :func:`sources_in_agreement` finds
    :data:`ACCEPTANCE_MIN_SOURCES` agreeing sources in that evidence — the model
    cannot vouch for itself.
    """

    def __init__(
        self,
        *,
        deterministic: DeterministicResolver | None = None,
        model: str = DEFAULT_RESOLVER_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout: float = 120.0,
        max_rounds: int = 6,
        client_factory: Callable[[str, str], Any] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.tools = deterministic if deterministic is not None else DeterministicResolver()
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.timeout = timeout
        self.max_rounds = max_rounds
        self._api_key = api_key
        self._client_factory = client_factory
        self._environ = environ

    def _resolve_api_key(self) -> str:
        if self._api_key:
            return self._api_key
        environ: Mapping[str, str] = os.environ if self._environ is None else self._environ
        for name in API_KEY_ENV_VARS:
            value = environ.get(name, "").strip()
            if value:
                return value
        msg = f"ResolverAgent needs {' or '.join(API_KEY_ENV_VARS)}, or an explicit api_key"
        raise ResolverError(msg)

    def _client(self) -> Any:
        api_key = self._resolve_api_key()
        if self._client_factory is not None:
            return self._client_factory(api_key, self.base_url)
        from openai import OpenAI  # local import: constructing the agent stays offline

        return OpenAI(api_key=api_key, base_url=self.base_url, timeout=self.timeout)

    def _execute_tool(
        self, name: str, arguments: Mapping[str, Any], evidence: list[SourceEvidence]
    ) -> dict[str, Any]:
        query = str(arguments.get("query", ""))
        if name == "edgar_tickers":
            rows = self.tools.edgar.lookup(query)
            if rows:
                row = rows[0]
                evidence.append(
                    SourceEvidence(
                        source="edgar",
                        name=str(row.get("title", "")) or None,
                        ticker=str(row.get("ticker", "")) or None,
                        cik=str(row.get("cik_str", "")) or None,
                        extra=row,
                    )
                )
            return {"matches": rows[:3]}
        if name == "wikidata_entity":
            matches = self.tools.wikidata.search(query)
            if not matches:
                return {"matches": []}
            entity = self.tools.wikidata.entity(matches[0]["id"])
            evidence.append(
                SourceEvidence(
                    source="wikidata",
                    name=str(entity.get("label", "")) or None,
                    ticker=entity.get("ticker"),
                    qid=str(entity.get("qid", "")) or None,
                    cik=entity.get("cik"),
                    extra=entity,
                )
            )
            return entity
        if name == "wikipedia_lookup":
            titles = self.tools.wikipedia.search(query)
            if not titles:
                return {"titles": []}
            summary = self.tools.wikipedia.summary(titles[0])
            evidence.append(
                SourceEvidence(
                    source="wikipedia",
                    name=str(summary.get("title", "")) or None,
                    extra=summary,
                )
            )
            return summary
        msg = f"unknown tool {name!r}"
        raise ResolverError(msg)

    def resolve(self, surface: str) -> Resolution:
        client = self._client()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _AGENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Resolve this suggested entity surface: {surface!r}. "
                    "Reply with the JSON record when you are done."
                ),
            },
        ]
        evidence: list[SourceEvidence] = []
        content = ""
        for _ in range(self.max_rounds):
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=_TOOL_SCHEMAS,
                temperature=self.temperature,
            )
            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None) or []
            if not tool_calls:
                content = message.content or ""
                break
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments,
                            },
                        }
                        for call in tool_calls
                    ],
                }
            )
            for call in tool_calls:
                arguments = json.loads(call.function.arguments or "{}")
                result = self._execute_tool(call.function.name, arguments, evidence)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": canonical_dumps(result),
                    }
                )
        else:
            msg = f"ResolverAgent exceeded {self.max_rounds} rounds without a final record"
            raise ResolverError(msg)
        try:
            record = ResolverRecord.model_validate(json.loads(_strip_json_fence(content)))
        except (json.JSONDecodeError, ValueError) as exc:
            msg = f"ResolverAgent's final message is not a valid record: {content[:200]!r}"
            raise ResolverError(msg) from exc
        sources = sources_in_agreement(tuple(evidence))
        accepted = len(sources) >= ACCEPTANCE_MIN_SOURCES
        if record.wiki_qid and record.wiki_qid not in {e.qid for e in evidence}:
            accepted = False  # the model asserted a QID no tool returned
        return Resolution(
            record=record,
            accepted=accepted,
            sources=sources,
            evidence=tuple(evidence),
            resolved_by="resolver-llm",
        )


# ------------------------------------------------------------------ #
# Store integration                                                    #
# ------------------------------------------------------------------ #


def resolve_and_store(
    store: EntityStore,
    surface: str,
    resolver: DeterministicResolver | ResolverAgent,
) -> tuple[str | None, bool]:
    """Resolve *surface* unless its entity is already cached; return (entity_id, cache_hit).

    The QID-keyed cache check happens *before* any resolution work when the surface
    itself was resolved before under a name key, and after resolution otherwise —
    a fresh QID that is already in the store means another collection paid for this
    entity already, and its record is reused as-is.
    """
    name_key = f"name:{normalise_surface(surface)}"
    cached = store.get_entity(name_key)
    if cached is not None:
        return name_key, True
    resolution = resolver.resolve(surface)
    if not resolution.accepted:
        return None, False
    entity_id = entity_id_for(resolution.record, surface)
    if store.get_entity(entity_id) is not None:
        return entity_id, True
    store.put_entity(
        entity_id,
        resolution.record.model_dump(mode="json"),
        resolution.resolved_by,
        resolution.sources,
    )
    return entity_id, False
