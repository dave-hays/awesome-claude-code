"""Line-of-sight kernels on a north-up elevation grid.

Grid convention: ``z[row, col]`` with row 0 at the north edge; cell centres sit at
``(x_left + (col + 0.5) * cell, y_top - (row + 0.5) * cell)``. Azimuths are compass degrees
(0 = north, clockwise). Earth curvature and refraction lower an object at distance ``d`` by
``d**2 * (1 - k) / (2 * R)``.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates

from . import config


@dataclass(frozen=True)
class Grid:
    z: np.ndarray  # float elevations, NaN = no data
    x_left: float
    y_top: float
    cell: float

    def rowcol(self, x, y):
        """Fractional row/col, cell centres at integers."""
        return (self.y_top - np.asarray(y)) / self.cell - 0.5, (np.asarray(x) - self.x_left) / self.cell - 0.5

    def xy(self, row, col):
        return self.x_left + (np.asarray(col) + 0.5) * self.cell, self.y_top - (np.asarray(row) + 0.5) * self.cell

    def sample(self, x, y):
        """Bilinear elevation at map coordinates; NaN outside the grid."""
        row, col = self.rowcol(x, y)
        shape = np.shape(row)
        out = map_coordinates(self.z, [np.ravel(row), np.ravel(col)], order=1, mode="constant", cval=np.nan)
        return out.reshape(shape)


def curvature_drop(d, k: float = config.REFRACTION_K, radius: float = config.EARTH_RADIUS_M):
    return np.asarray(d) ** 2 * (1.0 - k) / (2.0 * radius)


def ray_visibility(zs: np.ndarray, d: np.ndarray, z_eye: float, h_target: float, k: float, radius: float):
    """Visibility along rays.

    ``zs`` is (n_rays, n_samples) ground elevation at distances ``d`` (n_samples,) or
    (n_rays, n_samples). A sample is visible when a point ``h_target`` above it rises above
    every nearer ground sample as seen from ``z_eye``. Returns (visible, tan_elevation) where
    tan_elevation is the apparent slope of the ground itself. No-data samples never block.
    """
    zc = zs - curvature_drop(d, k, radius)
    ground = (zc - z_eye) / d
    blocking = np.where(np.isnan(ground), -np.inf, ground)
    nearer = np.maximum.accumulate(blocking, axis=1)
    nearer = np.concatenate([np.full((zs.shape[0], 1), -np.inf), nearer[:, :-1]], axis=1)
    target = (zc + h_target - z_eye) / d
    visible = (target >= nearer) & ~np.isnan(target)
    return visible, ground


@dataclass(frozen=True)
class Viewshed:
    visible: np.ndarray  # bool window (2n+1, 2n+1) centred on the observer cell
    row0: int  # grid row of window[0, 0]
    col0: int

    def distance(self, cell: float) -> np.ndarray:
        n = self.visible.shape[0] // 2
        r, c = np.mgrid[-n : n + 1, -n : n + 1]
        return np.hypot(r, c) * cell


def viewshed(
    grid: Grid,
    x: float,
    y: float,
    h_observer: float,
    h_target: float,
    max_radius: float,
    k: float = config.REFRACTION_K,
    radius: float = config.EARTH_RADIUS_M,
    ray_batch: int = 512,
) -> Viewshed:
    """Binary viewshed: radial horizons, then every cell tested against its nearest ray.

    ``8n`` rays (n = radius in cells) are sampled bilinearly at one-cell steps and reduced to a
    running horizon slope. Each cell in the disk is then compared with the horizon of the ray
    nearest its azimuth, over samples strictly nearer than the cell. Every cell gets a verdict;
    marking cells along rays instead leaves gaps where rays fan out.
    """
    n = int(np.ceil(max_radius / grid.cell))
    r_obs, c_obs = (int(round(v)) for v in grid.rowcol(x, y))
    nrows, ncols = grid.z.shape
    z0 = grid.z[r_obs, c_obs] if 0 <= r_obs < nrows and 0 <= c_obs < ncols else np.nan
    if np.isnan(z0):
        raise ValueError("observer has no elevation")
    x0, y0 = grid.xy(r_obs, c_obs)
    z_eye = z0 + h_observer
    n_rays = 8 * n
    d = np.arange(1, n + 1) * grid.cell
    drop = curvature_drop(d, k, radius)
    horizon = np.empty((n_rays, n), dtype=np.float32)
    for i in range(0, n_rays, ray_batch):
        a = (np.arange(i, min(i + ray_batch, n_rays)) * (2 * np.pi / n_rays))[:, None]
        ground = (grid.sample(x0 + np.sin(a) * d, y0 + np.cos(a) * d) - drop - z_eye) / d
        horizon[i : i + ray_batch] = np.maximum.accumulate(np.where(np.isnan(ground), -np.inf, ground), axis=1)

    dr, dc = np.mgrid[-n : n + 1, -n : n + 1]
    dist = np.hypot(dr, dc) * grid.cell
    rows, cols = r_obs + dr, c_obs + dc
    test = (dist > 0) & (dist <= max_radius) & (rows >= 0) & (rows < nrows) & (cols >= 0) & (cols < ncols)
    dist_t = dist[test]
    z_cell = grid.z[rows[test], cols[test]]
    ray = np.rint((np.arctan2(dc[test], -dr[test]) % (2 * np.pi)) / (2 * np.pi / n_rays)).astype(int) % n_rays
    nearer = np.ceil(dist_t / grid.cell).astype(int) - 2  # last sample index strictly nearer than the cell
    blocked = np.where(nearer >= 0, horizon[ray, np.clip(nearer, 0, n - 1)], -np.inf)
    target = (z_cell - curvature_drop(dist_t, k, radius) + h_target - z_eye) / dist_t
    window = np.zeros(dist.shape, dtype=bool)
    window[test] = (target >= blocked) & ~np.isnan(z_cell)
    window[n, n] = True
    return Viewshed(window, r_obs - n, c_obs - n)


def merge_nearest_visible(nearest: np.ndarray, vs: Viewshed, cell: float) -> None:
    """In place: ``nearest`` = min(nearest, distance to this observer) where it is visible."""
    dist = vs.distance(cell)
    n_rows, n_cols = nearest.shape
    h, w = vs.visible.shape
    r0, c0 = max(vs.row0, 0), max(vs.col0, 0)
    r1, c1 = min(vs.row0 + h, n_rows), min(vs.col0 + w, n_cols)
    if r0 >= r1 or c0 >= c1:
        return
    sub_vis = vs.visible[r0 - vs.row0 : r1 - vs.row0, c0 - vs.col0 : c1 - vs.col0]
    sub_dist = dist[r0 - vs.row0 : r1 - vs.row0, c0 - vs.col0 : c1 - vs.col0]
    target = nearest[r0:r1, c0:c1]
    np.minimum(target, np.where(sub_vis, sub_dist, np.inf), out=target)


@dataclass(frozen=True)
class MacroView:
    az_deg: np.ndarray  # ray azimuths
    skyline_deg: np.ndarray  # apparent elevation angle of the skyline per ray
    bin_area_m2: np.ndarray  # (360,) visible qualifying area per 1-degree bin
    bin_base_deg: np.ndarray  # (360,) lowest apparent angle of visible qualifying terrain; NaN if none
    distant_peak_sq_km: float
    horizon_arc_deg: int


def macro_view(
    grid: Grid,
    x: float,
    y: float,
    h_observer: float = config.EYE_HEIGHT_M,
    z_ground: float | None = None,
    d_min: float = config.MACRO_D_MIN_M,
    d_max: float = config.MACRO_D_MAX_M,
    z_min: float = config.MOUNTAIN_ELEV_MIN_M,
    az_step_deg: float = config.MACRO_AZ_STEP_DEG,
    step: float | None = None,
    k: float = config.REFRACTION_K,
    radius: float = config.EARTH_RADIUS_M,
    ray_batch: int = 360,
) -> MacroView:
    """Distant-mountain panorama by polar ray casting.

    Visible area is integrated in polar form (``d * dtheta * dd`` per sample), which avoids the
    near-observer oversampling of cell counting. Terrain qualifies when it is visible, lies at
    ``d_min <= d <= d_max`` and has ground elevation ``>= z_min``.
    """
    step = step or grid.cell
    if z_ground is None:
        z_ground = float(grid.sample(x, y))
    if np.isnan(z_ground):
        raise ValueError("observer has no elevation")
    z_eye = z_ground + h_observer
    az = np.arange(0.0, 360.0, az_step_deg)
    d = np.arange(step, d_max + step / 2, step)
    band = (d >= d_min) & (d <= d_max)
    dtheta = np.deg2rad(az_step_deg)
    skyline = np.empty(az.size)
    ray_area = np.zeros(az.size)
    ray_base = np.full(az.size, np.nan)
    for i in range(0, az.size, ray_batch):
        a = np.deg2rad(az[i : i + ray_batch])[:, None]
        zs = grid.sample(x + np.sin(a) * d, y + np.cos(a) * d)
        vis, ground = ray_visibility(zs, d, z_eye, 0.0, k, radius)
        skyline[i : i + ray_batch] = np.degrees(
            np.arctan(np.nanmax(np.where(np.isnan(ground), -np.inf, ground), axis=1))
        )
        q = vis & band[None, :] & (zs >= z_min)
        ray_area[i : i + ray_batch] = (q * (d * dtheta * step)[None, :]).sum(axis=1)
        base = np.where(q, ground, np.inf).min(axis=1)
        ray_base[i : i + ray_batch] = np.where(np.isfinite(base), np.degrees(np.arctan(base)), np.nan)
    bins = np.floor(az).astype(int) % 360
    bin_area = np.bincount(bins, weights=ray_area, minlength=360)
    bin_base = np.full(360, np.nan)
    for b in np.unique(bins[~np.isnan(ray_base)]):
        bin_base[b] = np.nanmin(ray_base[bins == b])
    return MacroView(
        az_deg=az,
        skyline_deg=skyline,
        bin_area_m2=bin_area,
        bin_base_deg=bin_base,
        distant_peak_sq_km=float(bin_area.sum() / 1e6),
        horizon_arc_deg=int((bin_area > 0).sum()),
    )
