import math

import numpy as np
import pytest

from nek.terrain import pad_candidates, plane_fit, slope_aspect, tpi


def plane(n, cell, gx, gy, z0=0.0):
    idx = np.arange(n)
    x, y = np.meshgrid(idx * cell, -idx * cell)  # row 0 is north
    return z0 + gx * x + gy * y


def test_plane_fit_recovers_grade_and_downslope_aspect():
    z = plane(120, 1.0, 0.1, 0.05)
    slope, aspect = plane_fit(z, 1.0, 30.0)
    core = (slice(20, 100), slice(20, 100))
    assert np.allclose(slope[core], 100 * math.hypot(0.1, 0.05))
    # gradient points uphill to the north-east; downslope faces south-west
    assert np.allclose(aspect[core], math.degrees(math.atan2(-0.1, -0.05)) % 360)
    assert np.isnan(slope[0, 0])


def test_slope_aspect_matches_plane_fit_on_a_plane():
    z = plane(50, 2.0, -0.2, 0.0)
    slope, aspect = slope_aspect(z, 2.0)
    assert np.allclose(slope[5:-5, 5:-5], 20.0)
    assert np.allclose(aspect[5:-5, 5:-5], 90.0)  # falls to the east


def test_tpi_is_zero_on_a_plane_and_positive_on_a_spike():
    z = plane(200, 10.0, 0.03, -0.02, z0=500)
    t = tpi(z, 10.0, radius_m=300)
    assert np.allclose(t[40:160, 40:160], 0.0, atol=1e-6)
    spike = np.zeros((101, 101))
    spike[50, 50] = 100.0
    count = (np.hypot(*np.mgrid[-30:31, -30:31]) <= 30).sum()
    assert tpi(spike, 1.0, radius_m=30)[50, 50] == pytest.approx(100.0 - 100.0 / count)


def test_tpi_ignores_nodata_in_the_mean():
    z = np.full((61, 61), 100.0)
    z[:, :20] = np.nan
    t = tpi(z, 1.0, radius_m=15)
    assert np.allclose(t[30, 25:40], 0.0)
    assert np.isnan(t[30, 5])


def test_pad_candidates_respect_slope_footprint_and_thresholds():
    n, cell = 200, 1.0
    gentle, steep = plane(n, cell, 0.10, 0.0, z0=500), plane(n, cell, 0.20, 0.0, z0=500 - 10.0)
    z = np.where(np.arange(n)[None, :] < 100, gentle, steep)
    allowed = np.ones_like(z, dtype=bool)
    allowed[:, :10] = False
    t = np.full_like(z, 100.0)
    road = np.zeros_like(z)
    pads = pad_candidates(z, cell, allowed, t, road, elev_range=(400.0, 700.0))
    cols = set(pads.cols.tolist())
    assert set(range(25, 85)) <= cols  # every footprint wholly on the 10% half and clear of the strip
    assert pads.cols.min() >= 10 + 15  # footprint must clear the excluded strip
    assert pads.cols.max() < 100 + 15  # nothing wholly on the 20% half
    assert np.all(pads.slope_pct <= 15.0)
    far = pad_candidates(z, cell, allowed, t, np.full_like(z, 400.0), elev_range=(400.0, 700.0))
    assert far.rows.size == 0
    low_tpi = pad_candidates(z, cell, allowed, np.full_like(z, 30.0), road, elev_range=(400.0, 700.0))
    assert low_tpi.best() is None
