"""Generate the deterministic sample corpus for cybernaut-mini evaluation.

Produces two files (idempotent, byte-stable):
  data/sample/documents.jsonl  — 66 documents across 6 topics
  data/sample/judgments.jsonl  — 12 graded relevance judgments (grades 1 or 2)

Design guarantees required by the acceptance tests
---------------------------------------------------
* **Lexical fixture** — ``doc-lex-01`` contains the rare token ``"zylophristine"``
  that appears nowhere else in the corpus.  The judgment ``q01`` ("zylophristine
  compound analysis") is answerable only via BM25 (lexical signal).

* **Paraphrase / dense fixture** — ``doc-para-01`` is titled "Genome editing with
  molecular scissors" and its text uses synonyms ("genes", "editing", "genomes")
  that share character-trigram overlap with the query "gene edit techniques" but
  share no exact stop-word-stripped BM25 tokens with it.  Under the hash embedder
  the dense vector is built from character-trigrams so these docs are recoverable
  by dense-only or hybrid search but not by BM25 alone.

Determinism
-----------
All randomness is seeded from Python's stdlib ``random`` module (seed 42).
No network access, no third-party libraries beyond cybernaut_mini itself.
``canonical_dumps`` (sorted keys) is used for every output line so the files
are byte-for-byte identical across runs in the same Python version.

Run
---
    uv run python scripts/generate_sample_corpus.py
"""

from __future__ import annotations

import random
import sys
from datetime import UTC, datetime
from pathlib import Path

# Make the package importable when run from repo root.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cybernaut_mini.models import Document, Judgment, canonical_dumps

# ------------------------------------------------------------------ #
# Constants / templates                                               #
# ------------------------------------------------------------------ #

_SEED = 42

# Topic definitions: (category, title_templates, text_templates)
# Each template set provides varied vocabulary so BM25 and dense both have signal.
_TOPICS: dict[str, dict[str, list[str]]] = {
    "biotech": {
        "titles": [
            "CRISPR screening maps immune pathways in T-cell activation",
            "Gut microbiome composition predicts inflammatory bowel disease",
            "Single-cell RNA sequencing reveals tumour heterogeneity",
            "Synthetic biology platforms accelerate enzyme design cycles",
            "Protein folding prediction enables rational drug discovery",
            "mRNA delivery platforms extend therapeutic gene expression",
            "Epigenetic reprogramming reverses cellular senescence markers",
            "High-throughput antibody screening identifies neutralising candidates",
            "Organoid models recapitulate liver metabolism in vitro",
            "CRISPR base editing corrects point mutations in haematopoietic cells",
        ],
        "sentences": [
            "Genome-wide association studies reveal novel loci linked to autoimmunity.",
            "Flow cytometry panels quantify T-cell subset activation dynamics.",
            "Lentiviral transduction achieves stable transgene integration rates.",
            "Proximity ligation assays detect protein-protein interactions in situ.",
            "Phage display libraries yield sub-nanomolar affinity antibody fragments.",
            "Lipid nanoparticle formulations improve intracellular nucleic acid delivery.",
            "Metabolic flux analysis maps carbon utilisation in recombinant cell lines.",
            "CRISPR-Cas12 enables multiplexed transcription factor knockout screens.",
            "Chromatin accessibility profiling delineates regulatory element landscapes.",
            "Base editing avoids double-strand DNA breaks during precision correction.",
        ],
    },
    "energy": {
        "titles": [
            "Perovskite solar cells achieve record photovoltaic conversion efficiency",
            "Grid-scale lithium-iron-phosphate batteries stabilise renewable output",
            "Offshore floating wind turbines expand deep-water generation capacity",
            "Green hydrogen electrolysers reach cost parity with fossil alternatives",
            "Demand-response programmes reduce peak load on transmission networks",
            "Molten-salt thermal storage extends concentrated solar plant dispatch",
            "Solid-state electrolytes improve lithium-metal battery safety margins",
            "Wave energy converters harvest tidal kinetic power at scale",
            "Smart inverters enable distributed solar integration into microgrids",
            "Carbon-neutral aviation fuels produced via biomass gasification pathways",
        ],
        "sentences": [
            "Tandem junction architectures stack photovoltaic layers to capture wider spectra.",
            "Battery management systems balance state-of-charge across cell packs.",
            "Offshore substations aggregate power from multiple wind turbine arrays.",
            "Alkaline electrolysis stacks operate at high current densities for efficiency.",
            "Virtual power plants aggregate distributed assets for grid frequency services.",
            "Thermal energy storage decouples solar generation from overnight demand.",
            "Solid electrolyte interfaces suppress lithium dendrite formation during cycling.",
            "Oscillating water columns convert wave pressure into pneumatic generation.",
            "Reactive power compensation stabilises voltage profiles in distribution networks.",
            "Fischer-Tropsch synthesis converts syngas into drop-in hydrocarbon fuels.",
        ],
    },
    "space": {
        "titles": [
            "Perseverance rover collects rock samples from ancient Martian river delta",
            "James Webb Telescope resolves galaxy cluster lensing at cosmic dawn",
            "Reusable heavy-lift rockets reduce orbital launch costs dramatically",
            "Artemis programme returns crewed missions to lunar south pole region",
            "Gravitational wave detectors capture neutron-star merger ringdown signals",
            "Europa Clipper investigates subsurface ocean habitability potential",
            "Space debris mitigation frameworks address low-Earth-orbit congestion",
            "Ion propulsion systems extend deep-space mission delta-v budgets",
            "CubeSat constellations enable real-time global maritime traffic monitoring",
            "Atmospheric entry aeroshell technologies protect payloads during descent",
        ],
        "sentences": [
            "Spectroscopic analysis of Martian regolith detects organic biosignature candidates.",
            "Gravitational lensing magnifies background galaxies beyond the diffraction limit.",
            "Propellant crossfeed systems improve staging mass fractions in reusable launchers.",
            "Polar craters preserve water ice accessible to in-situ resource utilisation.",
            "Matched filtering algorithms extract chirp signals from detector strain data.",
            "Mass spectrometers characterise plume material ejected from icy moons.",
            "Active debris removal spacecraft rendezvous with defunct satellite targets.",
            "Xenon propellant tanks sustain ion thruster operation across multi-year missions.",
            "L-band synthetic aperture radar penetrates cloud cover for ship detection.",
            "Heat shield ablatives dissipate hypersonic reentry thermal loads.",
        ],
    },
    "finance": {
        "titles": [
            "Central bank raises policy rates to combat persistent core inflation",
            "Sovereign bond yields invert signalling recession probability increase",
            "Retail investors amplify momentum via social media coordination signals",
            "Venture capital deployment shifts toward climate-tech sector allocations",
            "Currency hedging strategies reduce emerging-market foreign exchange risk",
            "Algorithmic trading strategies exploit limit-order-book microstructure",
            "Corporate credit spreads widen on refinancing risk and liquidity concerns",
            "Private equity buyouts leverage asset-backed securitisation structures",
            "Stablecoin regulatory frameworks address systemic financial stability risks",
            "Quantitative easing unwind shrinks central bank balance sheets gradually",
        ],
        "sentences": [
            "Forward guidance rhetoric anchors market expectations for rate trajectories.",
            "Duration risk amplifies bond portfolio losses when yield curves steepen.",
            "Options implied volatility surfaces reflect skew in tail-risk pricing.",
            "Momentum factor strategies harvest behavioural bias across equity cross-sections.",
            "Cross-currency basis swaps facilitate synthetic dollar funding arbitrage.",
            "Co-location services reduce latency to exchange matching engine infrastructure.",
            "High-yield spread compression signals improving corporate earnings outlooks.",
            "Leveraged buyout models project exit multiples under varying EBITDA scenarios.",
            "Reserve requirements and capital adequacy ratios constrain bank credit expansion.",
            "Decentralised liquidity pools introduce smart-contract counterparty exposure.",
        ],
    },
    "medicine": {
        "titles": [
            "mRNA vaccine platforms demonstrate broad cross-variant immune protection",
            "CAR-T cell therapies achieve durable remission in haematological malignancies",
            "GLP-1 receptor agonists reduce cardiovascular events in diabetic cohorts",
            "Antibiotic stewardship programmes slow resistance emergence in hospital settings",
            "Liquid biopsy tests detect circulating tumour DNA with high sensitivity",
            "Wearable continuous glucose monitors improve glycaemic control outcomes",
            "Faecal microbiota transplantation resolves recurrent Clostridioides infections",
            "Telemedicine platforms expand specialist access in rural underserved regions",
            "Polygenic risk scores stratify cardiovascular disease prevention populations",
            "Drug-eluting stents reduce restenosis rates following coronary interventions",
        ],
        "sentences": [
            "Spike protein antigen sequences guide bivalent mRNA booster formulation design.",
            "Chimeric antigen receptors direct cytotoxic T-cells to surface tumour antigens.",
            "Glucagon-like peptide agonists modulate pancreatic insulin secretion pathways.",
            "Antimicrobial susceptibility testing guides empirical antibiotic selection choices.",
            "Methylation patterns in circulating cell-free DNA enable tissue-of-origin inference.",
            "Flash glucose monitoring reduces nocturnal hypoglycaemia event frequency.",
            "Donor microbiome engraftment restores gut barrier integrity post-transplant.",
            "Asynchronous store-and-forward teleconsultation reduces patient travel burden.",
            "Genome-wide polygenic score calculations aggregate thousands of variant effects.",
            "Drug polymer matrix coatings release antiproliferative agents over weeks.",
        ],
    },
    "ai": {
        "titles": [
            "Transformer architectures scale language model performance with parameter count",
            "Reinforcement learning from human feedback aligns model outputs to preferences",
            "Sparse mixture-of-experts routing reduces inference cost at large model sizes",
            "Constitutional AI training minimises harmful outputs via self-critique loops",
            "Retrieval-augmented generation grounds language model responses in external facts",
            "Neural scaling laws predict capability emergence at critical parameter thresholds",
            "Multimodal vision-language models unify image and text representation spaces",
            "Chain-of-thought prompting improves multi-step arithmetic reasoning accuracy",
            "Differentially private training protects memorised training data in deployments",
            "Graph neural networks propagate structural inductive biases across node embeddings",
        ],
        "sentences": [
            "Attention mechanisms compute query-key dot-products across full sequence contexts.",
            "Proximal policy optimisation updates actor and critic networks with clipped grads.",
            "Expert routing gating functions assign tokens to feed-forward sub-networks.",
            "Red-teaming adversarially probes model outputs for alignment boundary failures.",
            "Dense passage retrieval encodes queries and passages into shared embedding spaces.",
            "Power-law fits to compute-vs-loss curves extrapolate frontier model performance.",
            "Contrastive image-text pretraining aligns visual and linguistic feature spaces.",
            "Scratchpad token sequences externalise intermediate reasoning computation steps.",
            "DP-SGD adds calibrated Gaussian noise to gradient batches during model training.",
            "Message-passing iterations aggregate neighbour representations through graph layers.",
        ],
    },
}

# ------------------------------------------------------------------ #
# Generation helpers                                                  #
# ------------------------------------------------------------------ #


def _make_date(rng: random.Random, year_range: tuple[int, int] = (2023, 2025)) -> datetime:
    year = rng.randint(*year_range)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    return datetime(year, month, day, tzinfo=UTC)


def _make_body(rng: random.Random, sentences: list[str], n: int = 3) -> str:
    """Pick n distinct sentences (with per-doc seed variation) and join."""
    chosen = rng.sample(sentences, min(n, len(sentences)))
    return " ".join(chosen)


def _generate_documents() -> list[Document]:
    documents: list[Document] = []

    # --- Lexical fixture (MUST be first for deterministic doc-lex-01 id) ---
    documents.append(
        Document(
            id="doc-lex-01",
            title="Zylophristine compound bioassay procedures",
            text=(
                "Zylophristine compound bioassay procedures quantify trace amounts of "
                "zylophristine using spectrophotometric detection in biological matrices. "
                "Sample preparation involves solid-phase extraction prior to zylophristine "
                "quantification via calibrated absorbance measurements."
            ),
            language="en",
            published_at=datetime(2024, 3, 1, tzinfo=UTC),
            metadata={"category": "biotech", "source": "synthetic"},
        )
    )

    # --- Paraphrase / dense fixture ---
    documents.append(
        Document(
            id="doc-para-01",
            title="Genome editing with molecular scissors technologies",
            text=(
                "Genes editing genomes precisely with molecular scissors technologies "
                "enables targeted sequence modifications without off-target cleavage. "
                "Editing accuracy improves when guide sequences are optimised for "
                "minimal homologous mismatches across genomic loci."
            ),
            language="en",
            published_at=datetime(2024, 6, 15, tzinfo=UTC),
            metadata={"category": "biotech", "source": "synthetic"},
        )
    )

    # --- One non-English doc for filter realism ---
    documents.append(
        Document(
            id="doc-de-01",
            title="Solarzellen Effizienz Rekord durch Halbleitermaterialien",
            text=(
                "Solarzellen erreichen neuen Effizienzrekord durch verbesserte "
                "Halbleitermaterialien in Tandem-Schichtarchitekturen. "
                "Photovoltaische Umwandlungsraten steigen auf über dreissig Prozent."
            ),
            language="de",
            published_at=datetime(2024, 1, 5, tzinfo=UTC),
            metadata={"category": "energy", "source": "synthetic"},
        )
    )

    # --- Topical documents (10 per topic = 60 topical docs) ---
    counter = 1
    for topic, data in _TOPICS.items():
        titles = data["titles"]
        sentences = data["sentences"]
        for i, title in enumerate(titles):
            doc_rng = random.Random(_SEED + counter * 31 + i * 7)
            n_sentences = doc_rng.randint(2, 4)
            body = _make_body(doc_rng, sentences, n=n_sentences)
            # Add the title as first sentence fragment for BM25 signal.
            text = f"{title}. {body}"
            pub = _make_date(doc_rng)
            documents.append(
                Document(
                    id=f"doc-{topic[:3]}-{i + 1:02d}",
                    title=title,
                    text=text,
                    language="en",
                    published_at=pub,
                    metadata={"category": topic, "source": "synthetic"},
                )
            )
            counter += 1

    return documents


# ------------------------------------------------------------------ #
# Judgments                                                           #
# ------------------------------------------------------------------ #

def _generate_judgments() -> list[Judgment]:
    """Return exactly 12 graded judgments spread across topics.

    Each judgment has >= 2 relevant docs that genuinely exist in the corpus.
    IDs match the zero-padded format produced by _generate_documents():
      doc-lex-01, doc-para-01, doc-bio-01..10, doc-ene-01..10,
      doc-spa-01..10, doc-fin-01..10, doc-med-01..10, doc-ai-01..10.
    """
    return [
        # q01: lexical query — only doc-lex-01 has the rare token zylophristine
        Judgment(
            query_id="q01",
            question="zylophristine compound analysis spectrophotometric assay",
            relevant_document_ids={
                "doc-lex-01": 2,
                "doc-bio-01": 1,  # bioassay adjacency
            },
        ),
        # q02: dense / paraphrase — gene editing, no exact BM25 token overlap
        Judgment(
            query_id="q02",
            question="gene edit techniques molecular precision genomics",
            relevant_document_ids={
                "doc-para-01": 2,
                "doc-bio-01": 1,
            },
        ),
        # q03: biotech — CRISPR screening immune
        Judgment(
            query_id="q03",
            question="CRISPR screens mapping immune activation pathways",
            relevant_document_ids={
                "doc-bio-01": 2,
                "doc-bio-10": 1,
                "doc-para-01": 1,
            },
        ),
        # q04: energy — solar photovoltaic efficiency
        Judgment(
            query_id="q04",
            question="perovskite solar cell photovoltaic conversion record efficiency",
            relevant_document_ids={
                "doc-ene-01": 2,
                "doc-ene-09": 1,
            },
        ),
        # q05: energy — battery grid storage
        Judgment(
            query_id="q05",
            question="grid-scale battery energy storage renewable stabilisation",
            relevant_document_ids={
                "doc-ene-02": 2,
                "doc-ene-07": 1,
            },
        ),
        # q06: space — Mars rover samples
        Judgment(
            query_id="q06",
            question="Mars rover rock sample ancient river delta geology",
            relevant_document_ids={
                "doc-spa-01": 2,
                "doc-spa-05": 1,
            },
        ),
        # q07: space — gravitational waves neutron star
        Judgment(
            query_id="q07",
            question="gravitational wave detector neutron star merger signal",
            relevant_document_ids={
                "doc-spa-05": 2,
                "doc-spa-01": 1,
            },
        ),
        # q08: finance — central bank interest rate inflation
        Judgment(
            query_id="q08",
            question="central bank interest rate policy inflation monetary",
            relevant_document_ids={
                "doc-fin-01": 2,
                "doc-fin-02": 1,
                "doc-fin-10": 1,
            },
        ),
        # q09: finance — algorithmic trading microstructure
        Judgment(
            query_id="q09",
            question="algorithmic trading limit order book microstructure latency",
            relevant_document_ids={
                "doc-fin-06": 2,
                "doc-fin-03": 1,
            },
        ),
        # q10: medicine — mRNA vaccine immune response
        Judgment(
            query_id="q10",
            question="mRNA vaccine immune protection variant cross-reactive",
            relevant_document_ids={
                "doc-med-01": 2,
                "doc-med-05": 1,
            },
        ),
        # q11: medicine — antibiotic resistance hospital stewardship
        Judgment(
            query_id="q11",
            question="antibiotic resistance stewardship hospital infection treatment",
            relevant_document_ids={
                "doc-med-04": 2,
                "doc-med-07": 1,
            },
        ),
        # q12: ai — transformer scaling language model
        Judgment(
            query_id="q12",
            question="transformer scaling laws language model parameter performance",
            relevant_document_ids={
                "doc-ai-01": 2,
                "doc-ai-06": 1,
            },
        ),
    ]


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #


def main() -> None:
    repo_root = Path(__file__).parent.parent
    out_dir = repo_root / "data" / "sample"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate documents.
    documents = _generate_documents()
    docs_path = out_dir / "documents.jsonl"
    lines = [canonical_dumps(doc.model_dump(mode="json")) + "\n" for doc in documents]
    docs_path.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {len(documents)} documents to {docs_path}")

    # Verify IDs referenced by judgments actually exist.
    doc_ids = {doc.id for doc in documents}
    judgments = _generate_judgments()

    # Check and report any judgment references to non-existent docs.
    for j in judgments:
        missing = [did for did in j.relevant_document_ids if did not in doc_ids]
        if missing:
            print(f"WARNING: judgment {j.query_id!r} references missing ids: {missing}")

    judg_path = out_dir / "judgments.jsonl"
    jlines = [canonical_dumps(j.model_dump(mode="json")) + "\n" for j in judgments]
    judg_path.write_text("".join(jlines), encoding="utf-8")
    print(f"Wrote {len(judgments)} judgments to {judg_path}")


if __name__ == "__main__":
    main()
