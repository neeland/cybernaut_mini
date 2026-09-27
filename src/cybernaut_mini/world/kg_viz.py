"""The knowledge-graph visuals: force-directed ego networks and evolution matrices.

Two figures per (target, year) story, exactly the post's pair: a networkx
``spring_layout`` snapshot where nodes are the entities a company was co-mentioned
with and lines join entities that also co-occur with each other, and a matplotlib
dot-matrix where dot size is that year's association strength and colour is the
change in the entity's share of the target's coverage from the year before.

Blog ref: https://nosible.com/blog/point-in-time-knowledge-graphs-over-named-entities
    — "first its force-directed network for one year, where nodes are the entities
    it was co-mentioned with and lines join entities that also co-occur with each
    other, so related nodes cluster; then the matrix that best tells its story
    over time. The networks colour nodes by category and the matrices colour by
    trend, green where a share grew, white where it held, red where it shrank";
    "Brand green marks products, lighter green peers, mist people, muted green
    organizations, an outline places; size is association strength"; "Dot size is
    that year's association strength; colour is how the product's share of ...
    coverage changed from the year before". Local copy under
    ``docs/blog-archive/``.

Assumptions:
    - Rendering is a side effect over an already-built graph: everything here
      reads :class:`~cybernaut_mini.world.kg.KGEdge` / ``EgoNetwork`` values and
      never recounts events, so a figure can never disagree with the stored bucket.
    - The five-colour palette follows the post's caption by *role* (products,
      peers/GPE, people, organizations, places-as-outline) in the NOSIBLE green
      family; exact hex values are unpublished, so ours are stand-ins keyed by
      NER layer.
    - Share deltas colour through a red-white-green map centred on zero and
      clipped at ``vmax``; an entity's first year has no prior share and draws
      neutral (white), matching "neutral for flat".

Alternatives rejected:
    - Recomputing lift inside the plotting code: the figures must render the
      audited bucket bytes, not a second derivation that could drift.
    - graphviz/plotly backends: matplotlib + networkx are already dependencies
      and the post's figures are static images.
    - Interactive colour scaling per figure: a fixed, documented ``vmax`` keeps
      two years' matrices comparable side by side.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import networkx as nx  # type: ignore[import-untyped]
import numpy as np

from cybernaut_mini.world.kg import EgoNetwork, KGEdge

__all__ = [
    "LAYER_PALETTE",
    "draw_ego_network",
    "draw_evolution_matrix",
    "share_deltas",
]

#: Node colour per NER layer — the post's five categories in the green family
#: ("brand green ... products, lighter green peers, mist people, muted green
#: organizations, an outline places"); hex values are laptop stand-ins.
LAYER_PALETTE: dict[str, str] = {
    "PRODUCT": "#1db954",  # brand green
    "ORG": "#4f7a5c",  # muted green
    "PERSON": "#c8d6cc",  # mist
    "GPE": "#8fd6a8",  # lighter green
    "LOC": "#ffffff",  # outline
}


def draw_ego_network(network: EgoNetwork, path: Path, *, seed: int = 42) -> Path:
    """Force-directed snapshot for one (target, year): spring layout, palette by
    layer, node size by association strength."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    graph = nx.Graph()
    graph.add_node(network.target, layer="TARGET", score=0.0)
    for node in network.nodes:
        graph.add_node(node.entity, layer=node.entity_type, score=node.score)
        graph.add_edge(network.target, node.entity, weight=node.score)
    for left, right, count in network.links:
        graph.add_edge(left, right, weight=float(count))

    layout = nx.spring_layout(graph, seed=seed)
    scores = np.asarray([graph.nodes[name].get("score", 0.0) for name in graph], dtype=float)
    top = float(scores.max()) if scores.size and float(scores.max()) > 0 else 1.0
    sizes = 120.0 + 680.0 * np.clip(scores / top, 0.0, 1.0)
    colors = [
        "#0a5c36" if graph.nodes[name]["layer"] == "TARGET"
        else LAYER_PALETTE.get(graph.nodes[name]["layer"], "#cccccc")
        for name in graph
    ]

    fig, ax = plt.subplots(figsize=(9, 9))
    nx.draw_networkx_edges(graph, layout, ax=ax, alpha=0.25, edge_color="#666666")
    nx.draw_networkx_nodes(
        graph, layout, ax=ax, node_size=sizes, node_color=colors, edgecolors="#333333"
    )
    nx.draw_networkx_labels(graph, layout, ax=ax, font_size=7)
    ax.set_title(f"{network.target} co-mention network, {network.year}")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def share_deltas(graph: Mapping[str, Sequence[KGEdge]]) -> dict[tuple[str, str], float]:
    """Per (entity, year): the change in the entity's share of the target's
    coverage from the year before (``nan`` when there is no prior year)."""
    shares: dict[tuple[str, str], float] = {
        (edge.entity, year): edge.share for year, edges in graph.items() for edge in edges
    }
    deltas: dict[tuple[str, str], float] = {}
    for (entity, year), share in shares.items():
        previous = shares.get((entity, str(int(year) - 1)))
        deltas[(entity, year)] = float("nan") if previous is None else share - previous
    return deltas


def draw_evolution_matrix(
    graph: Mapping[str, Sequence[KGEdge]],
    path: Path,
    *,
    top: int = 40,
    vmax: float = 0.05,
) -> Path:
    """The dot-matrix: rows are entities, columns years, dot size the year's
    normalized lift score, colour the share delta (green grew, white held, red
    shrank)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    years = sorted(graph)
    best: dict[str, float] = {}
    for edges in graph.values():
        for edge in edges:
            best[edge.entity] = max(best.get(edge.entity, 0.0), edge.score)
    entities = [name for name, _ in sorted(best.items(), key=lambda kv: -kv[1])[:top]]
    scores = {
        (edge.entity, year): edge.score for year, edges in graph.items() for edge in edges
    }
    deltas = share_deltas(graph)
    top_score = max(best.values(), default=1.0) or 1.0

    cmap = LinearSegmentedColormap.from_list("RdWhGn", ["#c0392b", "#ffffff", "#1db954"])
    norm = Normalize(vmin=-vmax, vmax=vmax)
    fig, ax = plt.subplots(figsize=(max(6.0, 0.6 * len(years) + 3), max(4.0, 0.28 * len(entities))))
    for row, entity in enumerate(entities):
        for col, year in enumerate(years):
            score = scores.get((entity, year))
            if score is None:
                continue
            delta = deltas.get((entity, year), float("nan"))
            color = cmap(norm(0.0 if np.isnan(delta) else float(np.clip(delta, -vmax, vmax))))
            ax.scatter(col, row, s=20.0 + 380.0 * score / top_score, color=color,
                       edgecolors="#444444", linewidths=0.4)
    ax.set_xticks(range(len(years)), years, rotation=45)
    ax.set_yticks(range(len(entities)), entities, fontsize=7)
    ax.invert_yaxis()
    ax.set_title("Association strength (size) and share change (colour) by year")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
