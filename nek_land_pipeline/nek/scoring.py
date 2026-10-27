"""Composite suitability, Current Use compliance and seller propensity."""

import numpy as np

from . import config


def _pct(value, full):
    return np.minimum(100.0, np.asarray(value, dtype=float) / full * 100.0)


def s_view(horizon_arc_deg, distant_peak_sq_km):
    return 0.5 * _pct(horizon_arc_deg, config.VIEW_ARC_FULL_DEG) + 0.5 * _pct(
        distant_peak_sq_km, config.VIEW_AREA_FULL_KM2
    )


def s_acoustic(access_quiet_score, nearest_hwy_dist_m):
    return 0.6 * np.asarray(access_quiet_score, dtype=float) + 0.4 * _pct(nearest_hwy_dist_m, config.HWY_DIST_FULL_M)


def s_clearing(clearing_acres_req):
    return 100.0 - _pct(clearing_acres_req, config.CLEARING_FULL_ACRES)


def s_site(soil_suitable, year_round_frontage, dist_nearest_power_m):
    return (
        40.0 * np.asarray(soil_suitable, dtype=float)
        + 40.0 * np.asarray(year_round_frontage, dtype=float)
        + 20.0 * (np.asarray(dist_nearest_power_m, dtype=float) <= config.POWER_NEAR_M)
    )


def total_suitability(view, acoustic, clearing, site, weights=config.WEIGHTS):
    return (
        weights["view"] * view
        + weights["acoustic"] * acoustic
        + weights["clearing"] * clearing
        + weights["site"] * site
    )


def uva_compliant(total_acres, clearing_acres_req):
    """Enrolled land left after the homestead exclusion and view clearing stays >= 25 acres."""
    excluded = config.UVA_HOMESTEAD_ACRES + np.maximum(
        0.0, np.asarray(clearing_acres_req) - config.UVA_CLEARING_ALLOWANCE_ACRES
    )
    return np.asarray(total_acres) - excluded >= config.UVA_MIN_ENROLLED_ACRES


def seller_propensity(hs_decl, bldg_flv, mail_state, years_owned, severable_bench):
    """Off-market selling propensity, 0-100. Missing mailing state scores as in-state."""
    hs_decl, bldg_flv = np.asarray(hs_decl, dtype=bool), np.asarray(bldg_flv, dtype=float)
    states = np.atleast_1d(np.asarray(mail_state, dtype=object))
    out_of_state = np.array([isinstance(s, str) and s.strip().upper() not in ("", "VT") for s in states])
    out_of_state = out_of_state.reshape(np.shape(mail_state))
    return (
        35 * (~hs_decl & (bldg_flv < 25_000))
        + 25 * out_of_state
        + 20 * (np.asarray(years_owned, dtype=float) >= 10)
        + 20 * np.asarray(severable_bench, dtype=bool)
    )
