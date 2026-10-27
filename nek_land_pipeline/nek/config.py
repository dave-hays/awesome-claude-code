"""Pipeline constants.

Thresholds are written in the units the runbook states (feet, acres, miles) and converted
to SI at import time, so every comparison inside the pipeline is in metres, square metres
and seconds. Deviations from the runbook are marked DECISION and logged in README.md.
"""

import os
from pathlib import Path

FT = 0.3048
ACRE = 4046.8564224  # m^2
MILE = 1609.344

CRS = "EPSG:32145"  # NAD83 / Vermont, metres
EPSG = 32145

WORKSPACE = Path(os.environ.get("NEK_WORKSPACE", "/workspace/nek_land_pipeline"))
REGISTRY = WORKSPACE / "registry.gpkg"
OUTPUT_DIR = WORKSPACE / "output"

# DECISION: the runbook's projected origin (519995.0, 245050.0) is 81 m from its own WGS84
# coordinate. The WGS84 point is projected at runtime; the reference parcel is located by
# its E911 address point, not by this coordinate.
ORIGIN_LONLAT = (-72.2475, 44.7061)
BENCHMARK_ADDRESS = {"number": "1109", "street": "ANDERSONVILLE", "town": "GLOVER"}

# --- Stage 1: isochrone and cadastral universe -------------------------------------------
DRIVE_BUDGET_S = 45 * 60
ROAD_QUERY_RADIUS_M = 80_000
ISOCHRONE_BUFFER_M = 1_500
NODE_SNAP_M = 1.0
# km/h by road category. AOTCLASS codes are mapped to these categories from the live layer
# domain, never hard-coded (the runbook's "AOTCLASS IN (1, 2) = state highways" conflicts
# with 19 V.S.A. § 302, where classes 1 and 2 are town highways).
SPEED_KMH = {
    "interstate": 105,
    "state_highway": 80,
    "class1_town": 50,  # DECISION: not in runbook; village segments of numbered routes
    "class2_town": 65,
    "class3_town": 45,
    "class4_town": 15,
    "private": 25,  # DECISION: not in runbook
}
MIN_PARCEL_M2 = 3.0 * ACRE
EXCLUDED_CATEGORIES = ("commercial", "industrial", "utility", "municipal", "exempt")

# --- Stage 2: acoustics -------------------------------------------------------------------
INTERSTATE_BUFFER_M = 3.0 * MILE
AADT_HIGH = 1_500
AADT_HIGH_BUFFER_M = 1.0 * MILE
GRADE_SAMPLE_M = 50.0
GRADE_STEEP = 0.05
GRADE_STEEP_BUFFER_M = 2.5 * MILE
QUARRY_BUFFER_M = 1.5 * MILE
EMITTER_SPACING_M = 250.0
EMITTER_HEIGHT_M = 2.5
RECEIVER_HEIGHT_M = 2.0
LOS_COMPUTE_RADIUS_M = 10.0 * MILE
# DECISION: the runbook fails any cell with line of sight to any AADT >= 1500 road within
# 10 miles. The viewshed is computed once out to 10 miles and stores the distance to the
# nearest visible emitter per tier; these radii gate on that raster and can be changed
# without recomputing.
LOS_GATE_RADIUS_M = {
    "aadt_high": 5_000.0,
    "interstate": 8_000.0,
    "steep_grade": 8_000.0,
}
ACOUSTIC_DEM_CELL_M = 10.0
# Owner's ground truth: no road noise is audible at the benchmark on a still night. Each tier's
# gate radius is capped at this fraction of the benchmark's nearest visible emitter distance, so
# the gate never rejects a sightline the benchmark shows to be quiet.
BENCHMARK_SILENT = True
BENCHMARK_LOS_MARGIN = 0.9
MIN_QUIET_M2 = 2.0 * ACRE
LOCAL_AADT_MAX = 400
CENTRALITY_SAMPLES = 500  # sampled sources for edge betweenness; seed fixed for determinism
CENTRALITY_HIGH_PCTL = 0.75  # DECISION: runbook says "high centrality" without a cutoff

# --- Stage 3: topography ------------------------------------------------------------------
PAD_SIDE_M = 30.0
PAD_ELEV_MIN_M = 1_350 * FT
PAD_ELEV_MAX_M = 2_200 * FT
PAD_SLOPE_MAX_PCT = 15.0
TPI_RADIUS_M = 2_000.0
TPI_MIN_M = 200 * FT
PAD_MAX_ROAD_DIST_M = 300.0
EYE_HEIGHT_M = 3.0
REFRACTION_K = 0.13
EARTH_RADIUS_M = 6_371_000.0
MACRO_DEM_CELL_M = 10.0
MACRO_D_MIN_M = 8.0 * MILE
MACRO_D_MAX_M = 35.0 * MILE
MOUNTAIN_ELEV_MIN_M = 2_200 * FT
MACRO_AZ_STEP_DEG = 0.1
HORIZON_ARC_MIN_DEG = 40.0  # DECISION: drop rule; the runbook's ">= 45" belongs to the escape span
DISTANT_PEAK_MIN_KM2 = 25.0
BOUNDARY_WINDOW_BUFFER_M = 150.0
CANOPY_MIN_M = 2.0
ESCAPE_SPAN_MIN_DEG = 45.0
# Canopy assumed on a neighbour's currently open ground (nDSM < 2 m) when testing whether a
# view depends on that ground staying open.
REGROWTH_CANOPY_M = 20.0

# --- Stage 4: due diligence ---------------------------------------------------------------
WETLAND_BUFFER_M = 50 * FT
SEPTIC_RADIUS_M = 60.0
DRIVEWAY_MAX_GRADE = 0.12
UVA_MIN_ENROLLED_ACRES = 25.0
UVA_HOMESTEAD_ACRES = 2.0
UVA_CLEARING_ALLOWANCE_ACRES = 1.5

# --- Stage 5: scoring ---------------------------------------------------------------------
WEIGHTS = {"view": 0.45, "acoustic": 0.30, "clearing": 0.15, "site": 0.10}
VIEW_ARC_FULL_DEG = 110.0
VIEW_AREA_FULL_KM2 = 120.0
HWY_DIST_FULL_M = 8_000.0
CLEARING_FULL_ACRES = 3.5
POWER_NEAR_M = 300.0
REGISTRY_SIZE = 250
DOSSIER_COUNT = 50
BENCHMARK_MIN_SCORE = 88.0
BENCHMARK_MIN_PERCENTILE = 95.0
