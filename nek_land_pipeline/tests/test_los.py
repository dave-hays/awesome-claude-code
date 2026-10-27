import math

import numpy as np
import pytest

from nek.los import Grid, macro_view, merge_nearest_visible, viewshed

R = 6_371_000.0


def grid_from(fn, half_m: float, cell: float) -> Grid:
    """Grid centred on (0, 0) with z = fn(x, y) at cell centres."""
    n = int(round(half_m / cell))
    idx = np.arange(-n, n + 1)
    x, y = np.meshgrid(idx * cell, -idx * cell)
    return Grid(z=fn(x, y).astype(float), x_left=-(n + 0.5) * cell, y_top=(n + 0.5) * cell, cell=cell)


def east_profile(vs, cell):
    n = vs.visible.shape[0] // 2
    return vs.visible[n, n:], np.arange(n + 1) * cell


def test_flat_earth_flat_plane_everything_visible():
    g = grid_from(lambda x, y: np.zeros_like(x), 2_000, 20)
    vs = viewshed(g, 0, 0, 2.5, 2.0, 2_000, k=1.0)
    inside = vs.distance(20) <= 2_000
    assert vs.visible[inside].all()


def test_curved_plane_matches_two_horizon_distance():
    k, h_obs, h_tgt, cell = 0.13, 2.5, 2.0, 10.0
    r_eff = R / (1 - k)
    expected = math.sqrt(2 * r_eff * h_obs) + math.sqrt(2 * r_eff * h_tgt)  # about 11.5 km
    g = grid_from(lambda x, y: np.zeros_like(x), 14_000, cell)
    vs = viewshed(g, 0, 0, h_obs, h_tgt, 14_000, k=k)
    vis, d = east_profile(vs, cell)
    assert d[vis].max() == pytest.approx(expected, abs=2 * cell)
    assert not vis[d > expected + 2 * cell].any()


def test_wall_casts_shadow_until_rising_ground_clears_it():
    def terrain(x, y):
        z = np.where(x > 1_000, 0.05 * (x - 1_000), 0.0)
        return np.where(np.abs(x - 1_000) < 5, 20.0, z)

    cell = 10.0
    g = grid_from(terrain, 3_000, cell)
    vs = viewshed(g, 0, 0, 2.5, 2.0, 3_000, k=1.0)
    vis, d = east_profile(vs, cell)
    # sightline over the 20 m wall: 0.05 (x - 1000) + 2 - 2.5 >= 17.5 x / 1000  ->  x >= 1553.8 m
    assert not vis[(d > 1_000) & (d < 1_550)].any()
    assert vis[d >= 1_560].all()
    assert vis[d < 1_000].all()


def test_nearest_visible_distance_merges_observers():
    cell = 20.0
    g = grid_from(lambda x, y: np.zeros_like(x), 2_000, cell)
    nearest = np.full(g.z.shape, np.inf)
    for x in (-1_000.0, 1_000.0):
        merge_nearest_visible(nearest, viewshed(g, x, 0, 2.5, 2.0, 1_500, k=1.0), cell)
    n = g.z.shape[0] // 2
    assert nearest[n, n] == pytest.approx(1_000.0)
    assert nearest[n, n + 25] == pytest.approx(500.0)  # x = 500 m
    assert np.isinf(nearest[0, n])  # (0, 2000) is 2236 m from both observers


def test_macro_view_quadrant_plateau_area_arc_and_base_angle():
    d1, d2 = 13_000.0, 20_000.0

    def terrain(x, y):
        r = np.hypot(x, y)
        az = np.degrees(np.arctan2(x, y)) % 360
        return np.where((r >= d1) & (r <= d2) & (az < 90), 700.0, 0.0)

    cell = 50.0
    g = grid_from(terrain, 22_000, cell)
    mv = macro_view(g, 0, 0, h_observer=3.0, z_ground=1_000.0, d_min=12_800, d_max=21_000, k=1.0)
    expected_km2 = math.pi / 4 * (d2**2 - d1**2) / 1e6
    assert mv.distant_peak_sq_km == pytest.approx(expected_km2, rel=0.02)
    assert mv.horizon_arc_deg == 90
    # lowest qualifying terrain is the plateau's front edge; bilinear samples down to 670 m on
    # the edge itself can sit slightly lower
    lo, hi = (math.degrees(math.atan((z - 1_003) / d)) for z, d in ((670, d1 - cell), (700, d1 + cell)))
    assert np.all((mv.bin_base_deg[:90] >= lo) & (mv.bin_base_deg[:90] <= hi))
    assert np.isnan(mv.bin_base_deg[90:]).all()


def test_observer_without_elevation_is_rejected():
    g = grid_from(lambda x, y: np.full_like(x, np.nan, dtype=float), 500, 10)
    with pytest.raises(ValueError):
        viewshed(g, 0, 0, 2.5, 2.0, 400)
