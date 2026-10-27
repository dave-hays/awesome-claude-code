"""Terrain derivatives and building-pad search on north-up grids (row 0 = north)."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import correlate1d, minimum_filter
from scipy.signal import oaconvolve

from . import config


def slope_aspect(z: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell slope (percent) and downslope aspect (compass degrees)."""
    d_row, d_col = np.gradient(z, cell)
    east, north = d_col, -d_row
    slope_pct = 100.0 * np.hypot(east, north)
    aspect = np.degrees(np.arctan2(-east, -north)) % 360.0
    return slope_pct, aspect


def plane_fit(z: np.ndarray, cell: float, window_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares plane over a moving square window: (slope_pct, downslope aspect).

    The window is odd-sized, ``2m + 1`` cells with ``m = round(window_m / (2 * cell))``. Windows
    touching no-data return NaN.
    """
    m = max(1, int(round(window_m / (2 * cell))))
    ramp = np.arange(-m, m + 1, dtype=float)
    ones = np.ones_like(ramp)
    sxx = ones.size * (ramp**2).sum()
    gx = correlate1d(
        correlate1d(z, ramp, axis=1, mode="constant", cval=np.nan), ones, axis=0, mode="constant", cval=np.nan
    )
    gr = correlate1d(
        correlate1d(z, ramp, axis=0, mode="constant", cval=np.nan), ones, axis=1, mode="constant", cval=np.nan
    )
    east = gx / (sxx * cell)
    north = -gr / (sxx * cell)
    return 100.0 * np.hypot(east, north), np.degrees(np.arctan2(-east, -north)) % 360.0


def disk_kernel(radius_m: float, cell: float, inner_m: float = 0.0) -> np.ndarray:
    n = int(np.floor(radius_m / cell))
    r, c = np.mgrid[-n : n + 1, -n : n + 1]
    dist = np.hypot(r, c) * cell
    return ((dist <= radius_m) & (dist >= inner_m)).astype(float)


def tpi(z: np.ndarray, cell: float, radius_m: float = config.TPI_RADIUS_M, inner_m: float = 0.0) -> np.ndarray:
    """Topographic position index: elevation minus the mean over a disk (or annulus).

    No-data cells are excluded from the mean, not treated as zero.
    """
    kernel = disk_kernel(radius_m, cell, inner_m)
    valid = ~np.isnan(z)
    total = oaconvolve(np.where(valid, z, 0.0), kernel, mode="same")
    count = oaconvolve(valid.astype(float), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0.5, total / count, np.nan)
    return np.where(valid, z - mean, np.nan)


@dataclass(frozen=True)
class PadCandidates:
    rows: np.ndarray
    cols: np.ndarray
    elev: np.ndarray
    slope_pct: np.ndarray
    aspect_deg: np.ndarray
    tpi_m: np.ndarray
    road_dist_m: np.ndarray

    def best(self) -> int | None:
        """Index of the highest-TPI candidate (the runbook's macro-view observer)."""
        return None if self.rows.size == 0 else int(np.argmax(self.tpi_m))


def pad_candidates(
    z: np.ndarray,
    cell: float,
    allowed: np.ndarray,
    tpi_m: np.ndarray,
    road_dist_m: np.ndarray,
    pad_side_m: float = config.PAD_SIDE_M,
    elev_range: tuple[float, float] = (config.PAD_ELEV_MIN_M, config.PAD_ELEV_MAX_M),
    max_slope_pct: float = config.PAD_SLOPE_MAX_PCT,
    min_tpi_m: float = config.TPI_MIN_M,
    max_road_dist_m: float = config.PAD_MAX_ROAD_DIST_M,
) -> PadCandidates:
    """Pad centres whose whole pad footprint lies in ``allowed`` and meets every threshold.

    ``tpi_m`` and ``road_dist_m`` are grids aligned with ``z`` (TPI is typically computed on a
    coarser regional DEM and resampled). Elevation and slope are taken over the pad footprint.
    """
    m = max(1, int(round(pad_side_m / (2 * cell))))
    size = 2 * m + 1
    inside = minimum_filter(allowed.astype(np.uint8), size=size, mode="constant", cval=0).astype(bool)
    slope, aspect = plane_fit(z, cell, pad_side_m)
    ones = np.ones(size)
    mean_z = (
        correlate1d(
            correlate1d(z, ones, axis=0, mode="constant", cval=np.nan), ones, axis=1, mode="constant", cval=np.nan
        )
        / size**2
    )
    ok = (
        inside
        & (mean_z >= elev_range[0])
        & (mean_z <= elev_range[1])
        & (slope <= max_slope_pct)
        & (tpi_m >= min_tpi_m)
        & (road_dist_m <= max_road_dist_m)
    )
    rows, cols = np.nonzero(ok)
    return PadCandidates(rows, cols, mean_z[ok], slope[ok], aspect[ok], tpi_m[ok], road_dist_m[ok])
