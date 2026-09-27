"""The Resolver held to the post's JPMorgan record: schema, acceptance, cache, agent.

Everything runs offline. The tool responses are the *committed real lookups* under
``data/01_raw/entities/`` (Wikipedia + Wikidata, fetched once by the tools' own
code) and the real SEC EDGAR excerpt under ``data/01_raw/edgar/``; the network gate
is passed an empty environ so any cache miss raises instead of touching the wire.
The oracle for record fields is ``configs/entities/jpmorgan.json`` — the post's own
published record, verbatim.

Blog ref: https://nosible.com/blog/can-faceted-search-at-web-scale-self-organize —
    "The Resolver Agent uses a bunch of tools … to work out who this entity is. This
    step is obviously slow and expensive, but it's also extremely cache friendly."
    Local copy: ``docs/blog-archive/can-faceted-search-at-web-scale-self-organize.md``.

Assumptions:
    - Field-for-field fidelity is tested two ways: the pydantic schema's field set
      must equal the published record's key set exactly, and every identifier the
      free tool set can actually verify (QID, CIK, ISIN, LEI, ticker, exchange,
      wiki page) must match the published value byte-for-byte.
    - The LLM agent is exercised through a scripted OpenAI-compatible client (the
      ``client_factory`` seam): the tool calls execute for real against the cached
      tools, and the acceptance verdict is recomputed from that evidence — including
      the case where the model asserts a QID no tool returned.

Alternatives rejected:
    - Mocking the tools for the agent test: the whole point of the acceptance rule
      is that the *tools* certify the model; scripting only the model keeps the
      certification path real.
    - A live-network test variant here: `refresh`/cache-miss paths are covered by
      the gate tests; recording fresh lookups belongs to the opt-in tooling, not CI.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cybernaut_mini.entities.resolver import (
    ACCEPTANCE_MIN_SOURCES,
    DeterministicResolver,
    EdgarCompanyTickers,
    NetworkDisabledError,
    ResolverAgent,
    ResolverError,
    ResolverRecord,
    SourceEvidence,
    WikidataTool,
    WikipediaTool,
    entity_id_for,
    resolve_and_store,
    sources_in_agreement,
)
from cybernaut_mini.entities.store import EntityStore

RECORD_PATH = Path("configs/entities/jpmorgan.json")

#: An empty environ closes the network gate: cache misses raise, never fetch.
OFFLINE: dict[str, str] = {}


def _offline_resolver() -> DeterministicResolver:
    return DeterministicResolver(
        wikipedia=WikipediaTool(environ=OFFLINE),
        wikidata=WikidataTool(environ=OFFLINE),
        edgar=EdgarCompanyTickers(environ=OFFLINE),
    )


def test_record_schema_matches_the_post_field_for_field() -> None:
    published = json.loads(RECORD_PATH.read_text(encoding="utf-8"))
    assert set(ResolverRecord.model_fields) == set(published)


def test_deterministic_resolver_reproduces_the_published_identifiers() -> None:
    published = json.loads(RECORD_PATH.read_text(encoding="utf-8"))
    resolution = _offline_resolver().resolve("JPMorgan Chase")
    assert resolution.accepted
    assert resolution.resolved_by == "resolver-deterministic"
    # All three free sources agree on the identity — comfortably >= the minimum 2.
    assert resolution.sources == ("edgar", "wikidata", "wikipedia")
    assert len(resolution.sources) >= ACCEPTANCE_MIN_SOURCES
    record = resolution.record
    for field in (
        "wiki_qid",  # Q192314
        "cik_code",  # 0000019617 — zero-padded like Wikidata P5531 and the post
        "isin_code",  # US46625H1005 (P946)
        "lei_code",  # 8I5DZWZKVSZI1NUHU748 (P1278)
        "ticker_wiki",  # JPM
        "ticker_eodhd",  # JPM.US — the SEC listing rendered EODHD-style
        "exchange",
        "wiki_page",
        "name_orig",
        "name_wiki",
        "is_company",
        "is_public",
        "is_private",
    ):
        assert getattr(record, field) == published[field], field
    # Subsidiaries come from P355 with real labels; the post's list is longer
    # because it merges historical evidence, so containment is the honest check.
    assert "Chase Bank" in record.subsidiaries
    # Stub fields stay None rather than being guessed from one source.
    assert record.linkedin is None
    assert record.country is None


def test_network_gate_blocks_every_cache_miss(tmp_path: Path) -> None:
    with pytest.raises(NetworkDisabledError, match="CYBERNAUT_MINI_ENTITIES_NETWORK"):
        WikidataTool(cache_dir=tmp_path, environ=OFFLINE).search("acme robotics")
    with pytest.raises(NetworkDisabledError):
        WikipediaTool(cache_dir=tmp_path, environ=OFFLINE).search("acme robotics")
    edgar = EdgarCompanyTickers(path=tmp_path / "missing.json", environ=OFFLINE)
    with pytest.raises(NetworkDisabledError):
        edgar.refresh()
    with pytest.raises(ResolverError, match="company_tickers"):
        edgar.lookup("acme")


def test_sources_in_agreement_identity_keys() -> None:
    edgar = SourceEvidence(source="edgar", name="JPMORGAN CHASE & CO", ticker="JPM", cik="19617")
    wikidata = SourceEvidence(
        source="wikidata", name="JPMorgan Chase", ticker="JPM", qid="Q192314", cik="0000019617"
    )
    wikipedia = SourceEvidence(source="wikipedia", name="JPMorgan Chase & Co.")
    stray = SourceEvidence(source="serp", name="Chase Field", ticker=None)
    # Ticker (case-blind), CIK (numeric), and suffix-stripped name all count.
    assert sources_in_agreement((edgar, wikidata)) == ("edgar", "wikidata")
    assert sources_in_agreement((wikipedia, wikidata)) == ("wikidata", "wikipedia")
    assert sources_in_agreement((edgar, wikidata, wikipedia, stray)) == (
        "edgar",
        "wikidata",
        "wikipedia",
    )
    # One source alone can never reach the acceptance minimum.
    assert len(sources_in_agreement((edgar,))) < ACCEPTANCE_MIN_SOURCES


def test_unresolvable_surface_is_rejected_not_guessed() -> None:
    # No cached lookups exist for this surface, so every networked tool abstains
    # and only EDGAR (offline file, no match) answers: zero agreeing sources.
    resolution = _offline_resolver().resolve("Piestewa Circumference")
    assert not resolution.accepted
    assert resolution.sources == ()


def test_resolve_and_store_is_qid_cache_friendly() -> None:
    resolver = _offline_resolver()
    with EntityStore() as store:
        entity_id, cache_hit = resolve_and_store(store, "JPMorgan Chase", resolver)
        assert entity_id == "Q192314"
        assert not cache_hit
        # A second collection suggesting the same company resolves to the same QID
        # and is answered by the store — the post's "extremely cache friendly".
        entity_id_again, cache_hit_again = resolve_and_store(store, "JPMorgan Chase", resolver)
        assert entity_id_again == "Q192314"
        assert cache_hit_again
        stored = store.get_entity("Q192314")
        assert stored is not None
        assert stored.sources == ("edgar", "wikidata", "wikipedia")
        # A rejected surface stores nothing.
        assert resolve_and_store(store, "Piestewa Circumference", resolver) == (None, False)
        assert store.entity_ids() == ["Q192314"]


def test_entity_id_falls_back_to_normalised_name() -> None:
    assert entity_id_for(ResolverRecord(wiki_qid="Q192314")) == "Q192314"
    assert entity_id_for(ResolverRecord(name_orig="Acme  Robotics")) == "name:acme robotics"
    assert entity_id_for(ResolverRecord(), surface="Acme") == "name:acme"


# ------------------------------------------------------------------ #
# The LLM agent, through the scripted-client seam                      #
# ------------------------------------------------------------------ #


def _tool_call(call_id: str, name: str, query: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps({"query": query})),
    )


def _response(content: str | None, tool_calls: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))]
    )


class _ScriptedClient:
    """An OpenAI-compatible chat client that replays a fixed list of responses."""

    def __init__(self, turns: list[SimpleNamespace]) -> None:
        self._turns = list(turns)
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.requests.append(kwargs)
        return self._turns.pop(0)


def _agent(turns: list[SimpleNamespace]) -> tuple[ResolverAgent, _ScriptedClient]:
    client = _ScriptedClient(turns)
    agent = ResolverAgent(
        deterministic=_offline_resolver(),
        api_key="test-key",
        client_factory=lambda api_key, base_url: client,
        environ=OFFLINE,
    )
    return agent, client


def test_agent_record_is_accepted_when_its_tools_agree() -> None:
    final = json.dumps({"name_orig": "JPMorgan Chase", "wiki_qid": "Q192314"})
    agent, client = _agent(
        [
            _response(
                None,
                [
                    _tool_call("call_1", "wikidata_entity", "jpmorgan chase"),
                    _tool_call("call_2", "edgar_tickers", "jpmorgan chase"),
                ],
            ),
            _response(f"```json\n{final}\n```", []),
        ]
    )
    resolution = agent.resolve("jpmorgan chase")
    assert resolution.resolved_by == "resolver-llm"
    assert resolution.accepted
    assert resolution.sources == ("edgar", "wikidata")
    assert resolution.record.wiki_qid == "Q192314"
    # The scripted transcript really carried the tool results back to the model.
    tool_messages = [
        m for m in client.requests[1]["messages"] if m.get("role") == "tool"
    ]
    assert [m["tool_call_id"] for m in tool_messages] == ["call_1", "call_2"]


def test_agent_cannot_self_certify_a_qid_no_tool_returned() -> None:
    final = json.dumps({"name_orig": "JPMorgan Chase", "wiki_qid": "Q999999999"})
    agent, _ = _agent(
        [
            _response(
                None,
                [
                    _tool_call("call_1", "wikidata_entity", "jpmorgan chase"),
                    _tool_call("call_2", "edgar_tickers", "jpmorgan chase"),
                ],
            ),
            _response(final, []),
        ]
    )
    resolution = agent.resolve("jpmorgan chase")
    assert not resolution.accepted  # tools agreed, but on Q192314 — not this QID


def test_agent_with_no_tool_evidence_is_rejected() -> None:
    final = json.dumps({"name_orig": "JPMorgan Chase"})
    agent, _ = _agent([_response(final, [])])
    resolution = agent.resolve("jpmorgan chase")
    assert not resolution.accepted
    assert resolution.sources == ()


def test_agent_garbage_final_message_raises() -> None:
    agent, _ = _agent([_response("I could not find anything.", [])])
    with pytest.raises(ResolverError, match="not a valid record"):
        agent.resolve("jpmorgan chase")
