# NEK Master Target Registry

Five-stage parcel screen for the Northeast Kingdom: 45-minute drive time from 1109
Andersonville Rd (Glover), acoustic shadow from high-traffic roads, distant mountain views that
survive the neighbours' canopy, regulatory screens, and a ranked acquisition registry.

## Status

| Part | State |
|---|---|
| Computational kernels (`nek/`) | Built. 40 tests pass against synthetic terrain with analytic answers. |
| ArcGIS REST client (`nek/arcgis.py`) | Built. **Not yet run against the live services**: the session's network policy blocks the Vermont hosts. |
| Stage runners 1–5, dossiers | Not built. They need the live layer schemas (field names, AOTCLASS codes). |
| Track A (MLS cross-match) | Not built. No public NNEREN feed; needs credentials or exports. |

Hosts the pipeline needs: `maps.vtrans.vermont.gov`, `maps.vcgi.vermont.gov`,
`services1.arcgis.com`, `anrmaps.vermont.gov`, `geodata.vermont.gov`,
`elevation.nationalmap.gov`, and an NRCan HRDEM host for Quebec terrain.

## Run order

Benchmark first. Stages 1–4 run on 1109 Andersonville Rd alone and report every gate's
measured value against its threshold. Thresholds are set from that report, then the regional
run and Stage 5 follow. Each stage writes `stage_N_manifest.json` with gate results and output
SHA-256 digests; the next stage refuses to start unless that manifest passed and the outputs
are unchanged.

## Kernels

| Module | What it does |
|---|---|
| `isochrone.py` | Road graph from arc endpoints, Dijkstra with partial frontier edges, buffered drive-time polygon |
| `los.py` | Viewshed (radial horizons, every cell tested against its nearest ray), nearest-visible-emitter raster, polar macro panorama with curvature and refraction |
| `terrain.py` | Slope/aspect, 30 m least-squares pad slope, disk TPI with no-data handling, pad search |
| `boundary.py` | Boundary escape test, clearance margins, clearing area, neighbour-regrowth risk |
| `roads.py` | Steep-grade pieces, dead ends (outside the 2-core), sampled edge betweenness, access quiet score |
| `scoring.py` | Composite suitability, Current Use compliance, seller propensity |
| `manifest.py` | Stage handoff gates |

```
cd nek_land_pipeline
pip install -e '.[dev]'
pytest
```

## Decisions that depart from or fill gaps in the runbook

1. **Acoustic gate is distance-tiered.** The runbook drops any cell with line of sight to any
   AADT ≥ 1500 road within 10 miles, which conflicts with Stage 3's requirement of wide distant
   views. The viewshed runs once to 10 miles and stores the distance to the nearest visible
   emitter per tier. Defaults gate at 5 km (AADT ≥ 1500) and 8 km (I-91, grades over 5%);
   changing them needs no recompute. The owner reports no audible road noise at the benchmark,
   so each tier's radius is capped at 0.9 × the benchmark's own nearest visible emitter
   distance (`roads.gate_radii`): the gate never rejects a sightline the benchmark shows is quiet.
2. **Calibration halts and reports.** If the benchmark scores below 88 or the 95th percentile,
   the run stops with its sub-scores. Weights are not re-tuned automatically: with four
   weights almost any parcel can be forced to the top.
3. **Terrain outside Vermont.** The 56 km panorama from northern parcels reaches Owl's Head
   and the Sutton range in Quebec, beyond the VCGI lidar. Fill with USGS 3DEP (NH) and NRCan
   HRDEM (Quebec).
4. **Stage 3 gates.** `horizon_arc_deg` (count of 1° bins with distant mountains) drops below
   40. The ≥ 45 assertion applies to `escape_span_deg`, the longest continuous run of bins that
   pass the boundary test.
5. **Formulas lost from the runbook text**, reconstructed. Boundary obstruction: the highest
   surface point beyond the pad along each ray. Composite:
   `0.45·S_view + 0.30·S_acoustic + 0.15·S_clearing + 0.10·S_site`.
6. **Boundary test surface.** Rays run over bare earth inside the parcel and the first-return
   surface outside it, across the whole parcel-plus-150 m window rather than only 5–30 m past
   the line. This catches taller obstructions farther out and parcels that re-enter the ray.
   `min_boundary_clearance_ft` is the margin over neighbouring land only.
7. **Target sightline** per 1° bin is the lowest visible terrain at ≥ 670 m in the 8–35 mile
   band: the base of the visible mountains.
8. **Regrowth risk.** `escape_span_regrowth_deg` re-runs the test with the neighbours' open
   ground (canopy < 2 m) grown to 20 m. A view that depends on someone else's hayfield shows up
   as a large drop.
9. **AOTCLASS codes** are mapped to speed categories from the live layer domain. The runbook's
   "AOTCLASS IN (1, 2) = state highways" conflicts with 19 V.S.A. § 302 (classes 1 and 2 are
   town highways).
10. **Speeds not in the runbook:** Class 1 town highway 50 km/h, private roads 25 km/h.
11. **Undirected graph, endpoint topology.** Rural roads are two-way and I-91's carriageways are
    digitised separately; noding crossings would join I-91 to every road it bridges.
12. **Object-id paging** replaces `resultOffset` paging: same records, no skips or repeats if
    the layer changes mid-pull, works where pagination is unsupported.
13. **Access score gaps.** "High centrality" is the 75th percentile of sampled edge
    betweenness. Paved Class 3, unpaved Class 2 and Class 1 frontage score 41–69; numbered
    routes 0–40; a dead-end Class 4 or private road fed by a paved road drops to 70–89.
14. **Origin point.** The runbook's projected origin is 81 m from its WGS84 coordinate; the WGS84
    point is projected at runtime and the benchmark parcel is found by its E911 address point.
15. **Lidar resolution.** The VCGI DEM service reports a 0.7 m statewide mosaic, not 0.35 m.
16. **Manifests** carry SHA-256 digests (tamper-evident), not keyed signatures.
17. **osmnx and pandana dropped.** The VTrans graph and networkx cover routing.
