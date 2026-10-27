"""Boundary escape: does clearing only the subject parcel open the distant view?

Rays are cast over a composite surface: bare earth inside the parcel (its trees are assumed
cut) and the first-return surface outside it (neighbours' canopy and buildings as they stand).
The runbook samples only 5-30 m past the property line; the composite surface covers that band
and also catches taller obstructions farther out and parcels that re-enter the ray.

For each 1-degree bin with distant mountains, the target sightline runs from the eye toward the
lowest visible mountain terrain in that bin (``MacroView.bin_base_deg``).
"""

from dataclasses import dataclass

import numpy as np

from . import config
from .los import Grid, MacroView, curvature_drop


@dataclass(frozen=True)
class EscapeResult:
    bins: np.ndarray  # 1-degree bins with distant mountains
    clearance_m: np.ndarray  # min vertical clearance of the sightline per bin, own trees cut
    clearance_regrowth_m: np.ndarray  # same, with neighbours' open ground regrown
    clearance_now_m: np.ndarray  # same, nothing cut
    exit_dist_m: np.ndarray  # distance at which the ray first leaves the parcel
    exit_grade_pct: np.ndarray  # mean grade from pad to that exit point (positive = downhill)
    escape_start_deg: int | None
    escape_span_deg: int
    escape_span_regrowth_deg: int
    existing_open_view_pct: float
    min_boundary_clearance_ft: float
    downslope_grade_pct: float
    clearing_acres_req: float


def longest_circular_run(flags: np.ndarray) -> tuple[int | None, int]:
    """(start index, length) of the longest run of True in a circular array."""
    n = flags.size
    if not flags.any():
        return None, 0
    if flags.all():
        return 0, n
    shift = int(np.argmin(flags))  # rotate so the array starts on a False
    rolled = np.append(np.roll(flags, -shift), False)
    best_start, best_len, run_start = 0, 0, None
    for i, f in enumerate(rolled):
        if f and run_start is None:
            run_start = i
        elif not f and run_start is not None:
            if i - run_start > best_len:
                best_start, best_len = run_start, i - run_start
            run_start = None
    return (best_start + shift) % n, best_len


def boundary_escape(
    dem: Grid,
    dsm: np.ndarray,
    parcel: np.ndarray,
    pad_x: float,
    pad_y: float,
    view: MacroView,
    eye_height: float = config.EYE_HEIGHT_M,
    canopy_min: float = config.CANOPY_MIN_M,
    regrowth: float = config.REGROWTH_CANOPY_M,
    min_span: float = config.ESCAPE_SPAN_MIN_DEG,
) -> EscapeResult:
    """Escape test from a pad. ``dsm`` and ``parcel`` (bool) are aligned with ``dem.z``."""
    z_pad = float(dem.sample(pad_x, pad_y))
    z_eye = z_pad + eye_height
    surface_cut = np.where(parcel, dem.z, np.fmax(dem.z, dsm))
    canopy = np.fmax(dsm - dem.z, 0.0)
    surface_regrow = np.where(~parcel & (canopy < canopy_min), dem.z + regrowth, surface_cut)
    surface_now = np.fmax(dem.z, dsm)

    bins = np.nonzero(view.bin_area_m2 > 0)[0]
    nrows, ncols = dem.z.shape
    reach = np.hypot(nrows, ncols) * dem.cell
    d = np.arange(dem.cell, reach, dem.cell)
    a = np.deg2rad(bins + 0.5)[:, None]
    xs, ys = pad_x + np.sin(a) * d, pad_y + np.cos(a) * d
    rr, cc = (np.rint(v).astype(int) for v in dem.rowcol(xs, ys))
    on = (rr >= 0) & (rr < nrows) & (cc >= 0) & (cc < ncols)
    rr, cc = np.clip(rr, 0, nrows - 1), np.clip(cc, 0, ncols - 1)
    sight = z_eye + d[None, :] * np.tan(np.deg2rad(view.bin_base_deg[bins]))[:, None]
    drop = curvature_drop(d)[None, :]

    outside = ~parcel[rr, cc] | ~on

    def clearance(surface: np.ndarray, where: np.ndarray) -> np.ndarray:
        gap = np.where(on & where, sight - (surface[rr, cc] - drop), np.inf)
        return np.nanmin(np.where(np.isnan(gap), np.inf, gap), axis=1)

    # Pass/fail looks at the whole ray (a knoll on one's own land blocks too); the reported
    # boundary clearance is the margin over neighbouring land only.
    everywhere = np.ones_like(outside)
    clear_cut, clear_regrow = clearance(surface_cut, everywhere), clearance(surface_regrow, everywhere)
    clear_now = clearance(surface_now, everywhere)
    clear_neighbour = clearance(surface_cut, outside)
    first_out = np.where(outside.any(axis=1), outside.argmax(axis=1), d.size - 1)
    exit_dist = d[first_out]
    exit_grade = (
        100.0 * (z_pad - dem.z[rr[np.arange(bins.size), first_out], cc[np.arange(bins.size), first_out]]) / exit_dist
    )

    passes = np.zeros(360, dtype=bool)
    passes[bins] = clear_cut >= 0
    start, span = longest_circular_run(passes)
    passes_regrow = np.zeros(360, dtype=bool)
    passes_regrow[bins] = clear_regrow >= 0
    _, span_regrow = longest_circular_run(passes_regrow)

    if span < min_span or start is None:
        return EscapeResult(
            bins,
            clear_cut,
            clear_regrow,
            clear_now,
            exit_dist,
            exit_grade,
            start,
            span,
            span_regrow,
            0.0,
            float("nan"),
            float("nan"),
            float("nan"),
        )

    span_bins = (start + np.arange(span)) % 360
    in_span = np.isin(bins, span_bins)
    centre = span_bins[span // 2]
    clearing = _clearing_m2(dem, dsm, parcel, canopy, pad_x, pad_y, z_eye, view, span_bins, canopy_min)
    return EscapeResult(
        bins=bins,
        clearance_m=clear_cut,
        clearance_regrowth_m=clear_regrow,
        clearance_now_m=clear_now,
        exit_dist_m=exit_dist,
        exit_grade_pct=exit_grade,
        escape_start_deg=int(start),
        escape_span_deg=int(span),
        escape_span_regrowth_deg=int(span_regrow),
        existing_open_view_pct=float(100.0 * (clear_now[in_span] >= 0).mean()),
        min_boundary_clearance_ft=float(clear_neighbour[in_span].min() / config.FT),
        downslope_grade_pct=float(exit_grade[bins == centre][0]),
        clearing_acres_req=clearing / config.ACRE,
    )


def _clearing_m2(dem, dsm, parcel, canopy, pad_x, pad_y, z_eye, view, span_bins, canopy_min) -> float:
    """Area of own canopy (> canopy_min) inside the view sector that rises above the sightline."""
    rows, cols = np.nonzero(parcel & (canopy > canopy_min))
    if rows.size == 0:
        return 0.0
    x, y = dem.xy(rows, cols)
    dx, dy = x - pad_x, y - pad_y
    dist = np.hypot(dx, dy)
    b = np.floor(np.degrees(np.arctan2(dx, dy)) % 360.0).astype(int)
    in_sector = np.isin(b, span_bins) & (dist > 0)
    sight = z_eye + dist * np.tan(np.deg2rad(np.nan_to_num(view.bin_base_deg[b], nan=90.0)))
    blocks = in_sector & (dsm[rows, cols] - curvature_drop(dist) > sight)
    return float(blocks.sum() * dem.cell**2)
