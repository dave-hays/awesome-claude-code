import math

import pytest
from shapely.geometry import LineString, Point

from nek.isochrone import add_origin, build_graph, isochrone_polygon, reachable_lines


def total_length(lines):
    return sum(line.length for line in lines)


def test_origin_mid_edge_reaches_both_directions():
    g = build_graph([LineString([(0, 0), (10_000, 0)])], [10.0])
    src = add_origin(g, Point(5_000, 30))
    lines = reachable_lines(g, src, 200.0)
    assert total_length(lines) == pytest.approx(4_000.0)
    xs = [x for line in lines for x, _ in line.coords]
    assert min(xs) == pytest.approx(3_000.0)
    assert max(xs) == pytest.approx(7_000.0)


def test_origin_at_endpoint_uses_existing_node():
    g = build_graph([LineString([(0, 0), (1_000, 0)])], [10.0])
    assert add_origin(g, Point(-50, 0)) == (0.0, 0.0)


def test_partial_frontier_edge_uses_its_own_speed():
    g = build_graph([LineString([(0, 0), (1_000, 0)]), LineString([(1_000, 0), (2_000, 0)])], [10.0, 5.0])
    lines = reachable_lines(g, (0.0, 0.0), 200.0)
    assert total_length(lines) == pytest.approx(1_500.0)


def test_edge_reached_from_both_ends_is_split_into_two_pieces():
    a, b, c = (0, 0), (1_000, 0), (0, 1_000)
    g = build_graph([LineString([a, b]), LineString([a, c]), LineString([b, (1_000, 1_000), c])], [10.0, 10.0, 2.0])
    lines = reachable_lines(g, (0.0, 0.0), 300.0)
    # 1000 m + 1000 m fully, then 200 s * 2 m/s = 400 m in from each end of the slow 2000 m edge
    assert total_length(lines) == pytest.approx(2_800.0)
    slow_pieces = [line for line in lines if line.length == pytest.approx(400.0)]
    assert len(slow_pieces) == 2


def test_isochrone_polygon_area_matches_stadium():
    poly = isochrone_polygon([LineString([(0, 0), (4_000, 0)])], 1_500.0)
    assert poly.area == pytest.approx(2 * 1_500 * 4_000 + math.pi * 1_500**2, rel=5e-3)
