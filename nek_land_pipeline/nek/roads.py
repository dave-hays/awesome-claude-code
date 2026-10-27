"""Road-network acoustics: steep-grade pieces, dead ends, centrality and access quiet score."""

from collections.abc import Callable

import networkx as nx
import numpy as np
import shapely
from scipy.stats import rankdata
from shapely.geometry import LineString
from shapely.ops import substring

from . import config


def steep_pieces(
    line: LineString,
    sample_z: Callable[[np.ndarray, np.ndarray], np.ndarray],
    step: float = config.GRADE_SAMPLE_M,
    threshold: float = config.GRADE_STEEP,
) -> list[LineString]:
    """Sub-lines whose grade between consecutive samples ``step`` apart exceeds ``threshold``."""
    if line.length < step:
        return []
    s = np.append(np.arange(0.0, line.length, step), line.length)
    pts = shapely.line_interpolate_point(line, s)
    z = sample_z(shapely.get_x(pts), shapely.get_y(pts))
    grade = np.abs(np.diff(z)) / np.diff(s)
    return [substring(line, s[i], s[i + 1]) for i in np.nonzero(grade > threshold)[0]]


def dead_end_edges(g: nx.MultiGraph) -> set:
    """Edges outside the 2-core: every edge on a tree hanging off the looped network.

    Parallel edges count toward degree, so a loop formed by two arcs between the same pair of
    nodes is not a dead end. Self-loops (cul-de-sac circles) are ignored for degree.
    """
    h = nx.MultiGraph(g)
    h.remove_edges_from(list(nx.selfloop_edges(h)))
    queue = [n for n, deg in h.degree() if deg <= 1]
    while queue:
        node = queue.pop()
        if node not in h:
            continue
        neighbours = list(h.neighbors(node))
        h.remove_node(node)
        queue.extend(n for n in neighbours if h.degree(n) <= 1)
    core = {(u, v, k) for u, v, k in h.edges(keys=True)} | {(v, u, k) for u, v, k in h.edges(keys=True)}
    return {(u, v, k) for u, v, k in g.edges(keys=True) if (u, v, k) not in core and u != v}


def centrality_percentile(
    g: nx.MultiGraph, samples: int = config.CENTRALITY_SAMPLES, seed: int = 0
) -> dict[tuple, float]:
    """Edge betweenness (length-weighted, sampled sources) as a percentile rank in (0, 1]."""
    k = min(samples, g.number_of_nodes())
    bc = nx.edge_betweenness_centrality(g, k=k, weight="length", seed=seed, normalized=True)
    keys = list(bc)
    pct = rankdata([bc[e] for e in keys], method="max") / len(keys)
    return dict(zip(keys, pct, strict=True))


def access_quiet_score(
    category: str,
    unpaved: bool,
    dead_end: bool,
    centrality_pct: float,
    feeder_paved: bool = False,
    high_pctl: float = config.CENTRALITY_HIGH_PCTL,
) -> float:
    """0-100 quietness of the fronting road.

    Runbook bands: dead-end Class 4 or private 90-100; unpaved Class 3 with low centrality
    70-89; paved Class 2 or high-centrality gravel 0-40. DECISION: combinations the runbook
    leaves out (paved Class 3, unpaved Class 2, Class 1) score 41-69; numbered routes score
    0-40; a dead-end Class 4 or private road fed by a paved road drops to the 70-89 band.
    """
    p = float(np.clip(centrality_pct, 0.0, 1.0))
    quiet_class = category in ("class4_town", "private")
    if quiet_class and dead_end and not feeder_paved:
        return 100.0 - 10.0 * p
    if quiet_class:
        return 89.0 - 19.0 * p
    if category in ("interstate", "state_highway"):
        return 40.0 * (1.0 - p)
    high = p >= high_pctl
    if category == "class3_town" and unpaved and not high:
        return 89.0 - 19.0 * (p / high_pctl)
    if (category == "class2_town" and not unpaved) or (unpaved and high):
        return 40.0 * (1.0 - p)
    return 69.0 - 28.0 * p


def gate_radii(
    benchmark_nearest_m: dict[str, float],
    defaults: dict[str, float] = config.LOS_GATE_RADIUS_M,
    margin: float = config.BENCHMARK_LOS_MARGIN,
) -> dict[str, float]:
    """Line-of-sight gate radius per tier, capped below the silent benchmark's nearest visible emitter.

    A tier the benchmark cannot see within the computed radius (``inf`` or absent) keeps its default.
    """
    return {tier: min(r, margin * benchmark_nearest_m.get(tier, np.inf)) for tier, r in defaults.items()}
