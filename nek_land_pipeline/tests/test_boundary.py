import math

import numpy as np
import pytest

from nek.boundary import boundary_escape, longest_circular_run
from nek.los import Grid, MacroView

THETA = -2.0  # target sightline toward the distant mountain base, degrees
BINS = np.arange(60, 120)


def view() -> MacroView:
    area = np.zeros(360)
    area[BINS] = 1.0
    base = np.full(360, np.nan)
    base[BINS] = THETA
    return MacroView(np.arange(360.0), np.zeros(360), area, base, float(area.sum()), BINS.size)


def east_slope(grade: float, size: int = 400):
    """1 m grid, x from 0 to size, ground falling east at ``grade``; parcel is x < 200."""
    x = np.arange(size) + 0.5
    z = np.tile(600.0 - grade * x, (size, 1))
    parcel = np.tile(x < 200, (size, 1))
    return Grid(z=z, x_left=0.0, y_top=float(size), cell=1.0), parcel


def test_circular_run_wraps_through_north():
    flags = np.zeros(360, dtype=bool)
    flags[350:] = True
    flags[:20] = True
    flags[100:110] = True
    assert longest_circular_run(flags) == (350, 30)
    assert longest_circular_run(np.zeros(5, dtype=bool)) == (None, 0)
    assert longest_circular_run(np.ones(5, dtype=bool)) == (0, 5)


def test_steep_lot_clears_neighbour_forest():
    dem, parcel = east_slope(0.30)
    dsm = dem.z + 20.0
    r = boundary_escape(dem, dsm, parcel, 100.0, 200.0, view())
    assert r.escape_span_deg == 60
    assert r.escape_start_deg == 60
    # clearance at the property line along azimuth a: 13 - 100 tan(2 deg) / sin(a); worst at 60 deg
    worst = 13.0 - 100 * math.tan(math.radians(-THETA)) / math.sin(math.radians(60.5))
    assert r.min_boundary_clearance_ft * 0.3048 == pytest.approx(worst, abs=0.6)
    assert r.existing_open_view_pct == 0.0  # own forest blocks before clearing
    assert r.escape_span_regrowth_deg == 60
    assert r.downslope_grade_pct == pytest.approx(30.0, abs=0.5)


def test_clearing_area_matches_sector_integral():
    dem, parcel = east_slope(0.30)
    dsm = dem.z + 20.0
    r = boundary_escape(dem, dsm, parcel, 100.0, 200.0, view())
    tan_t = math.tan(math.radians(THETA))

    def reach(a):  # own trees block while 20 - 0.3 sin(a) d > 3 + d tan(theta)
        return 17.0 / (0.3 * math.sin(a) + tan_t)

    a = np.radians(np.linspace(60, 120, 2001))
    expected_m2 = np.trapezoid([0.5 * reach(v) ** 2 for v in a], a)
    assert r.clearing_acres_req * 4046.8564224 == pytest.approx(expected_m2, rel=0.05)


def test_gentle_lot_fails_into_neighbour_forest():
    dem, parcel = east_slope(0.10)
    r = boundary_escape(dem, dem.z + 20.0, parcel, 100.0, 200.0, view())
    assert r.escape_span_deg == 0
    assert np.all(r.clearance_m < 0)


def test_view_that_depends_on_neighbours_open_field_is_exposed_by_regrowth():
    dem, parcel = east_slope(0.10)
    dsm = np.where(parcel, dem.z + 20.0, dem.z)  # neighbour is hayfield
    r = boundary_escape(dem, dsm, parcel, 100.0, 200.0, view())
    assert r.escape_span_deg == 60
    assert r.escape_span_regrowth_deg == 0
