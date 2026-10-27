import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString

from nek import scoring
from nek.isochrone import build_graph
from nek.roads import access_quiet_score, centrality_percentile, dead_end_edges, gate_radii, steep_pieces


def test_steep_pieces_cover_only_the_steep_half():
    line = LineString([(0, 0), (1_000, 0)])

    def z(x, y):
        return np.where(x <= 500, 0.03 * x, 15 + 0.08 * (x - 500))

    pieces = steep_pieces(line, z)
    assert sum(p.length for p in pieces) == pytest.approx(500.0)
    assert min(p.bounds[0] for p in pieces) == pytest.approx(500.0)


def test_dead_ends_are_trees_off_the_looped_network():
    loop = [((0, 0), (100, 0)), ((100, 0), (100, 100)), ((100, 100), (0, 100)), ((0, 100), (0, 0))]
    spur = [((0, 100), (0, 200)), ((0, 200), (0, 300)), ((0, 200), (50, 250))]
    twin = [((100, 0), (200, 0)), ((200, 0), (300, 0)), ((200, 0), (300, 0))]  # parallel arcs form a loop
    lines = [
        LineString(seg) if i != 9 else LineString([(200, 0), (250, 50), (300, 0)])
        for i, seg in enumerate(loop + spur + twin)
    ]
    g = build_graph(lines, [10.0] * len(lines))
    dead = {frozenset((u, v)) for u, v, _ in dead_end_edges(g)}
    assert dead == {
        frozenset(((0.0, 100.0), (0.0, 200.0))),
        frozenset(((0.0, 200.0), (0.0, 300.0))),
        frozenset(((0.0, 200.0), (50.0, 250.0))),
    }


def test_bridge_between_clusters_has_top_centrality():
    g = nx.MultiGraph(nx.barbell_graph(6, 0))
    nx.set_edge_attributes(g, 1.0, "length")
    pct = centrality_percentile(g, samples=1_000)
    bridge = next(e for e in pct if {e[0], e[1]} == {5, 6})
    assert pct[bridge] == 1.0
    assert all(0 < v <= 1 for v in pct.values())


@pytest.mark.parametrize(
    ("args", "lo", "hi"),
    [
        (("class4_town", True, True, 0.1), 90, 100),
        (("private", True, True, 0.9), 90, 100),
        (("private", True, True, 0.1, True), 70, 89),
        (("class3_town", True, False, 0.3), 70, 89),
        (("class3_town", True, False, 0.9), 0, 40),
        (("class2_town", False, False, 0.2), 0, 40),
        (("state_highway", False, False, 0.5), 0, 40),
        (("class3_town", False, False, 0.4), 41, 69),
    ],
)
def test_access_quiet_score_bands(args, lo, hi):
    assert lo <= access_quiet_score(*args) <= hi


def test_sub_scores_and_composite():
    assert scoring.s_view(110, 120) == pytest.approx(100.0)
    assert scoring.s_view(55, 60) == pytest.approx(50.0)
    assert scoring.s_view(300, 500) == pytest.approx(100.0)
    assert scoring.s_acoustic(100, 8_000) == pytest.approx(100.0)
    assert scoring.s_clearing(0.0) == pytest.approx(100.0)
    assert scoring.s_clearing(7.0) == pytest.approx(0.0)
    assert scoring.s_site(True, True, 250) == pytest.approx(100.0)
    assert scoring.s_site(True, False, 400) == pytest.approx(40.0)
    assert scoring.total_suitability(100, 100, 100, 100) == pytest.approx(100.0)
    assert scoring.total_suitability(100, 0, 0, 0) == pytest.approx(45.0)


def test_uva_compliance():
    assert scoring.uva_compliant(30.0, 1.0)
    assert not scoring.uva_compliant(27.0, 3.0)  # 27 - (2 + 1.5) = 23.5


def test_seller_propensity():
    assert scoring.seller_propensity(False, 10_000, "NH", 12, True) == 100
    assert scoring.seller_propensity(True, 300_000, "VT", 3, False) == 0
    assert scoring.seller_propensity(False, 10_000, None, np.nan, False) == 35
    out = scoring.seller_propensity([False, True], [0, 0], ["MA", "vt"], [10, 9], [False, False])
    assert list(out) == [80, 0]


def test_gate_radii_never_reject_what_the_silent_benchmark_sees():
    defaults = {"aadt_high": 5_000.0, "interstate": 8_000.0, "steep_grade": 8_000.0}
    radii = gate_radii({"aadt_high": 3_000.0, "interstate": np.inf}, defaults, margin=0.9)
    assert radii == {"aadt_high": 2_700.0, "interstate": 8_000.0, "steep_grade": 8_000.0}
