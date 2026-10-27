"""Drive-time isochrone over a road centerline network.

The graph is undirected: rural Vermont roads are two-way, and the divided sections of I-91
are digitised as parallel carriageways, so ignoring direction changes no reachable set.
Topology comes from arc endpoints only. Noding crossings would join I-91 to every road it
bridges over.
"""

from collections.abc import Iterable

import networkx as nx
import shapely
from shapely.geometry import LineString, Point
from shapely.ops import substring

ORIGIN = "origin"


def _node_key(xy: tuple[float, float], snap_m: float) -> tuple[float, float]:
    return (round(xy[0] / snap_m) * snap_m, round(xy[1] / snap_m) * snap_m)


def build_graph(lines: Iterable[LineString], speeds_mps: Iterable[float], snap_m: float = 1.0) -> nx.MultiGraph:
    """Build a road graph whose edges carry geometry, length (m) and travel time (s)."""
    g = nx.MultiGraph()
    for line, speed in zip(lines, speeds_mps, strict=True):
        if line is None or line.is_empty or speed <= 0:
            continue
        for part in getattr(line, "geoms", [line]):
            coords = part.coords
            a, b = _node_key(coords[0], snap_m), _node_key(coords[-1], snap_m)
            if a == b and part.length < snap_m:
                continue
            g.add_edge(a, b, geometry=part, start=a, length=part.length, time=part.length / speed)
    return g


def add_origin(g: nx.MultiGraph, point: Point):
    """Split the edge nearest to ``point`` at its projection; return the routing source node."""
    edges = list(g.edges(keys=True, data=True))
    tree = shapely.STRtree([d["geometry"] for *_, d in edges])
    u, v, k, d = edges[int(tree.nearest(point))]
    geom, speed = d["geometry"], d["length"] / d["time"]
    at = geom.project(point)
    end = v if d["start"] == u else u
    if at <= 0.0:
        return d["start"]
    if at >= geom.length:
        return end
    g.remove_edge(u, v, k)
    g.add_node(ORIGIN)
    # substring() keeps the original direction: the head piece starts at the old start node,
    # the tail piece starts at the split point.
    head, tail = substring(geom, 0, at), substring(geom, at, geom.length)
    if head.length > 0:
        g.add_edge(d["start"], ORIGIN, geometry=head, start=d["start"], length=head.length, time=head.length / speed)
    if tail.length > 0:
        g.add_edge(ORIGIN, end, geometry=tail, start=ORIGIN, length=tail.length, time=tail.length / speed)
    return ORIGIN


def reachable_lines(g: nx.MultiGraph, source, budget_s: float) -> list[LineString]:
    """Road geometry reachable within ``budget_s``, including partial edges at the frontier."""
    t = nx.single_source_dijkstra_path_length(g, source, cutoff=budget_s, weight="time")
    pieces = []
    for u, v, d in g.edges(data=True):
        start = d["start"]
        end = v if start == u else u
        ts, te = t.get(start), t.get(end)
        if ts is None and te is None:
            continue
        length, time = d["length"], d["time"]
        fs = 0.0 if ts is None else min(1.0, (budget_s - ts) / time)
        fe = 0.0 if te is None else min(1.0, (budget_s - te) / time)
        geom = d["geometry"]
        if fs + fe >= 1.0:
            pieces.append(geom)
            continue
        if fs > 0:
            pieces.append(substring(geom, 0, fs * length))
        if fe > 0:
            pieces.append(substring(geom, length - fe * length, length))
    return [p for p in pieces if p.length > 0]


def isochrone_polygon(lines: list[LineString], buffer_m: float):
    """Dissolved buffer of reachable road geometry."""
    return shapely.union_all(lines).buffer(buffer_m)
