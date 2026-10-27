import pytest
import shapely
from shapely.geometry import Polygon

from nek import manifest
from nek.arcgis import esri_to_shapely, shapely_to_esri

SHELL_CW = [[0, 0], [0, 10], [10, 10], [10, 0], [0, 0]]
HOLE_CCW = [[2, 2], [4, 2], [4, 4], [2, 4], [2, 2]]
SHELL2_CW = [[20, 0], [20, 5], [25, 5], [25, 0], [20, 0]]


def test_esri_polygon_with_hole_and_second_part():
    g = esri_to_shapely({"rings": [SHELL_CW, HOLE_CCW, SHELL2_CW]})
    assert g.geom_type == "MultiPolygon"
    assert g.area == pytest.approx(100 - 4 + 25)


def test_esri_rings_all_ccw_are_treated_as_shells():
    g = esri_to_shapely({"rings": [SHELL_CW[::-1]]})
    assert g.area == pytest.approx(100)


def test_esri_paths_and_points():
    assert esri_to_shapely({"paths": [[[0, 0], [3, 4]]]}).length == pytest.approx(5)
    assert esri_to_shapely({"paths": [[[0, 0], [1, 0]], [[5, 0], [6, 0]]]}).geom_type == "MultiLineString"
    assert esri_to_shapely({"x": 1.5, "y": 2.5}).coords[0] == (1.5, 2.5)
    assert esri_to_shapely(None) is None


def test_filter_polygon_round_trips_with_esri_orientation():
    poly = Polygon(SHELL_CW[::-1], [HOLE_CCW[::-1]])
    esri, gtype = shapely_to_esri(poly)
    assert gtype == "esriGeometryPolygon"
    assert not shapely.LinearRing(esri["rings"][0]).is_ccw
    assert shapely.LinearRing(esri["rings"][1]).is_ccw
    assert esri_to_shapely(esri).area == pytest.approx(poly.area)


def test_manifest_handoff_detects_failure_and_tampering(tmp_path):
    out = tmp_path / "stage1.gpkg"
    out.write_bytes(b"parcels")
    manifest.write_manifest(1, [out], {"unique_span": True}, {"rows": 1}, workspace=tmp_path)
    assert manifest.require(1, workspace=tmp_path)["status"] == "passed"
    out.write_bytes(b"edited")
    with pytest.raises(manifest.GateError, match="changed"):
        manifest.require(1, workspace=tmp_path)
    with pytest.raises(manifest.GateError, match="crs"):
        manifest.write_manifest(2, [out], {"crs": False}, {}, workspace=tmp_path)
    with pytest.raises(manifest.GateError, match="failed"):
        manifest.require(2, workspace=tmp_path)
