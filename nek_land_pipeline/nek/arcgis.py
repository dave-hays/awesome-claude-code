"""ArcGIS REST client: feature queries chunked by object id, tiled ImageServer exports.

Feature paging fetches the full object-id list once and then pulls features in id chunks of
``maxRecordCount``. Unlike ``resultOffset`` paging, this cannot skip or repeat records if the
layer changes between pages, and it works on services without pagination support.
"""

import json
import math
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import requests
import shapely
from rasterio.transform import from_origin
from rasterio.windows import Window
from shapely.geometry import LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon

from . import config

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "nek-land-pipeline/0.1"


class ArcGISError(RuntimeError):
    pass


def _request(
    url: str, params: dict, method: str = "POST", retries: int = 4, timeout: float = 180.0
) -> requests.Response:
    """Request with exponential backoff on network errors, HTTP 5xx and ArcGIS 5xx error bodies."""
    for attempt in range(retries + 1):
        try:
            if method == "GET":
                r = SESSION.get(url, params=params, timeout=timeout)
            else:
                r = SESSION.post(url, data=params, timeout=timeout)
            if r.status_code >= 500:
                raise ArcGISError(f"HTTP {r.status_code} from {url}")
            r.raise_for_status()
            if "json" in r.headers.get("Content-Type", "") or params.get("f") == "json":
                body = r.json()
                if isinstance(body, dict) and "error" in body:
                    err = body["error"]
                    if int(err.get("code", 400)) < 500:
                        raise ValueError(f"ArcGIS error from {url}: {err}")
                    raise ArcGISError(f"ArcGIS error from {url}: {err}")
            return r
        except (requests.ConnectionError, requests.Timeout, ArcGISError):
            if attempt == retries:
                raise
            time.sleep(2 ** (attempt + 1))
    raise AssertionError("unreachable")


def service_info(url: str) -> dict:
    return _request(url, {"f": "json"}, method="GET").json()


def coded_domains(info: dict) -> dict[str, dict]:
    """{field name: {code: label}} for every coded-value domain on a layer."""
    out = {}
    for f in info.get("fields", []):
        dom = f.get("domain") or {}
        if dom.get("type") == "codedValue":
            out[f["name"]] = {c["code"]: c["name"] for c in dom["codedValues"]}
    return out


def esri_to_shapely(g: dict | None):
    """Esri JSON geometry to shapely. Polygon exteriors are clockwise, holes counter-clockwise."""
    if not g:
        return None
    if "x" in g:
        return None if g["x"] is None else Point(g["x"], g["y"])
    if "points" in g:
        return MultiPoint(g["points"])
    if "paths" in g:
        lines = [LineString(p) for p in g["paths"] if len(p) >= 2]
        return lines[0] if len(lines) == 1 else MultiLineString(lines)
    if "rings" in g:
        rings = [shapely.LinearRing(r) for r in g["rings"] if len(r) >= 4]
        shells = [r for r in rings if not r.is_ccw] or rings
        holes = [r for r in rings if r.is_ccw] if len(shells) < len(rings) else []
        polys = [Polygon(s) for s in shells]
        owned: list[list] = [[] for _ in polys]
        for h in holes:
            probe = Polygon(h).representative_point()
            owners = [i for i, p in enumerate(polys) if p.contains(probe)]
            if owners:
                owned[min(owners, key=lambda i: polys[i].area)].append(h)
        parts = [Polygon(s, owned[i]) for i, s in enumerate(shells)]
        geom = parts[0] if len(parts) == 1 else MultiPolygon(parts)
        if not geom.is_valid:
            geom = shapely.make_valid(geom)
            geom = shapely.union_all(
                [p for p in getattr(geom, "geoms", [geom]) if p.geom_type in ("Polygon", "MultiPolygon")]
            )
        return geom
    raise ValueError(f"unsupported Esri geometry keys: {sorted(g)}")


def shapely_to_esri(geom) -> tuple[dict, str]:
    """Query-filter geometry in the pipeline CRS."""
    sr = {"wkid": config.EPSG}
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        geom = shapely.orient_polygons(geom, exterior_cw=True)
        polys = getattr(geom, "geoms", [geom])
        rings = [list(p.exterior.coords) for p in polys] + [list(i.coords) for p in polys for i in p.interiors]
        return {"rings": rings, "spatialReference": sr}, "esriGeometryPolygon"
    if geom.geom_type == "Point":
        return {"x": geom.x, "y": geom.y, "spatialReference": sr}, "esriGeometryPoint"
    raise ValueError(f"unsupported filter geometry {geom.geom_type}")


def envelope(xmin: float, ymin: float, xmax: float, ymax: float) -> tuple[dict, str]:
    sr = {"wkid": config.EPSG}
    return {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "spatialReference": sr}, "esriGeometryEnvelope"


def query_features(
    layer_url: str,
    where: str = "1=1",
    geometry: tuple[dict, str] | None = None,
    out_fields: str = "*",
    chunk: int | None = None,
) -> gpd.GeoDataFrame:
    """All features matching ``where`` and an optional spatial filter, in EPSG:32145."""
    info = service_info(layer_url)
    base = {"where": where, "f": "json"}
    if geometry is not None:
        geom, gtype = geometry
        base |= {
            "geometry": json.dumps(geom),
            "geometryType": gtype,
            "spatialRel": "esriSpatialRelIntersects",
            "inSR": config.EPSG,
        }
    ids = _request(f"{layer_url}/query", base | {"returnIdsOnly": "true"}).json().get("objectIds") or []
    ids = sorted(ids)
    size = chunk or int(info.get("maxRecordCount") or 1000)
    records, geoms = [], []
    for i in range(0, len(ids), size):
        params = {
            "objectIds": ",".join(map(str, ids[i : i + size])),
            "outFields": out_fields,
            "outSR": config.EPSG,
            "returnGeometry": "true",
            "f": "json",
        }
        feats = _request(f"{layer_url}/query", params).json().get("features", [])
        for f in feats:
            records.append(f.get("attributes", {}))
            geoms.append(esri_to_shapely(f.get("geometry")))
    gdf = gpd.GeoDataFrame(records, geometry=geoms, crs=config.CRS)
    if len(gdf) != len(ids):
        raise ArcGISError(f"{layer_url}: expected {len(ids)} features, received {len(gdf)}")
    return gdf


def export_image(
    service_url: str,
    bounds: tuple[float, float, float, float],
    cell: float,
    out_path: Path,
    interpolation: str = "RSP_BilinearInterpolation",
) -> Path:
    """Mosaic an ImageServer region to a float32 GeoTIFF on a grid snapped to ``cell``.

    Tiles respect the service's maxImageWidth/maxImageHeight. No-data becomes NaN.
    """
    info = service_info(service_url)
    max_w, max_h = int(info.get("maxImageWidth", 4000)), int(info.get("maxImageHeight", 4000))
    xmin, ymin = math.floor(bounds[0] / cell) * cell, math.floor(bounds[1] / cell) * cell
    xmax, ymax = math.ceil(bounds[2] / cell) * cell, math.ceil(bounds[3] / cell) * cell
    width, height = round((xmax - xmin) / cell), round((ymax - ymin) / cell)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "width": width,
        "height": height,
        "crs": config.CRS,
        "transform": from_origin(xmin, ymax, cell, cell),
        "nodata": np.nan,
        "tiled": True,
        "compress": "deflate",
        "BIGTIFF": "IF_SAFER",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        for row_off in range(0, height, max_h):
            for col_off in range(0, width, max_w):
                w, h = min(max_w, width - col_off), min(max_h, height - row_off)
                x0, y1 = xmin + col_off * cell, ymax - row_off * cell
                params = {
                    "bbox": f"{x0},{y1 - h * cell},{x0 + w * cell},{y1}",
                    "bboxSR": config.EPSG,
                    "imageSR": config.EPSG,
                    "size": f"{w},{h}",
                    "format": "tiff",
                    "pixelType": "F32",
                    "interpolation": interpolation,
                    "f": "image",
                }
                r = _request(f"{service_url}/exportImage", params, method="GET")
                with rasterio.MemoryFile(r.content) as mem, mem.open() as src:
                    band = src.read(1, masked=True).astype("float32").filled(np.nan)
                if band.shape != (h, w):
                    raise ArcGISError(f"tile {row_off},{col_off}: expected {(h, w)}, got {band.shape}")
                dst.write(band, 1, window=Window(col_off, row_off, w, h))
    return out_path
