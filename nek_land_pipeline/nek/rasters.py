"""GeoTIFF I/O for north-up grids."""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from . import config
from .los import Grid


def read_grid(path: Path, window=None) -> Grid:
    with rasterio.open(path) as src:
        t = src.window_transform(window) if window is not None else src.transform
        if t.b != 0 or t.d != 0 or abs(t.a) != abs(t.e):
            raise ValueError(f"{path}: expected a north-up square-pixel grid")
        z = src.read(1, window=window, masked=True).astype("float64").filled(np.nan)
    return Grid(z=z, x_left=t.c, y_top=t.f, cell=t.a)


def write_grid(grid: Grid, path: Path, dtype: str = "float32") -> Path:
    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "count": 1,
        "width": grid.z.shape[1],
        "height": grid.z.shape[0],
        "crs": config.CRS,
        "transform": from_origin(grid.x_left, grid.y_top, grid.cell, grid.cell),
        "nodata": np.nan,
        "tiled": True,
        "compress": "deflate",
        "BIGTIFF": "IF_SAFER",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(grid.z.astype(dtype), 1)
    return path
