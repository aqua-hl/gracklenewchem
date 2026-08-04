########################################################################
#
# Interactive viewer for the CGM coolant phase cube.
#
# Reads the NPZ written by cgm_coolant_phase_cube.py and emits one
# dependency-free, self-contained HTML page.  The primary view is a set
# of linked orthogonal slices through (nH, T, Z), accompanied by paired
# cooling/heating budget bars and the selected constant-pressure track.
#
# The reader understands both the current phase-cube schema and the
# earlier dominant-coolant NPZ files.  Legacy files remain viewable, but
# their raw solver rates are deliberately not labelled as physical
# cooling coefficients.
#
#   python cgm_cube_interactive.py cgm_coolant_phase_cube.npz \
#       --output cgm_coolant_phase_cube.html
#
########################################################################

import argparse
import json
import os
import sys

import numpy as np


# Field index, rather than prevalence in a particular run, determines the
# colour.  Keeping this table here also makes the Python figures and the
# standalone HTML easy to align without an external plotting dependency.
_CHANNEL_COLORS = (
    "#3277c5", "#68a7dc", "#d95f45", "#ef9b55", "#1aa982",
    "#70c6a8", "#4b9b55", "#7959b6", "#2f8a3d", "#d7a21b",
    "#e6bd3e", "#e47da5", "#c96937", "#a66bb0", "#6c8ec5",
    "#6045b5", "#bd6d95", "#df5670", "#9b6a3f", "#7d8d33",
    "#43a2a5", "#2d7c89", "#8b7540", "#9974c3", "#3a6f46",
    "#ca4b87", "#7762a6", "#4b8f77", "#bd634f", "#647fc2",
    "#aa8130", "#b95f9a", "#5f9d3e", "#d17831", "#5270a7",
    "#a86e59", "#5b938e", "#8b6cae", "#b27746", "#6f8750",
)


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Build the offline interactive phase-cube viewer.")
    parser.add_argument("npz", help="NPZ from cgm_coolant_phase_cube.py")
    parser.add_argument("--output", default=None,
                        help="output HTML path (default: NPZ name + .html)")
    parser.add_argument("--body-only", default=None,
                        help="also write the page without the document "
                             "wrapper to this path")
    return parser.parse_args(args)


def clean_label(label):
    return (str(label).replace("$_2$", "₂").replace("$_3$", "₃")
            .replace("$", ""))


def _string_value(data, name, default=""):
    if name not in data.files:
        return default
    value = np.asarray(data[name])
    if value.size == 0:
        return default
    item = value.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8", errors="replace")
    return str(item)


def _scalar_value(data, name, default=None):
    if name not in data.files:
        return default
    value = np.asarray(data[name])
    if value.size == 0:
        return default
    item = value.reshape(-1)[0]
    if isinstance(item, np.generic):
        item = item.item()
    return item


def _cube_array(data, name, shape, default, dtype=float):
    """Read an array and broadcast common scalar/2-D cube representations."""
    if name not in data.files:
        return np.full(shape, default, dtype=dtype)
    arr = np.asarray(data[name], dtype=dtype)
    if arr.shape == shape:
        return np.array(arr, copy=True)
    if arr.size == 1:
        return np.full(shape, arr.reshape(-1)[0], dtype=dtype)
    if arr.size == int(np.prod(shape)):
        return np.array(arr.reshape(shape), copy=True)
    try:
        return np.array(np.broadcast_to(arr, shape), dtype=dtype, copy=True)
    except ValueError:
        return np.full(shape, default, dtype=dtype)


def _finite_float(value, digits=7):
    value = float(value)
    if not np.isfinite(value):
        return None
    if value == 0.0:
        return 0.0
    return float(f"{value:.{digits}g}")


def _float_list(values, digits=7):
    return [_finite_float(v, digits) for v in np.asarray(values).reshape(-1)]


def _metadata_array(data, name):
    if name not in data.files:
        return None
    arr = np.asarray(data[name])
    values = []
    for value in arr.reshape(-1):
        if isinstance(value, bytes):
            values.append(value.decode("utf-8", errors="replace"))
        elif isinstance(value, (str, np.str_)):
            values.append(str(value))
        elif isinstance(value, (bool, np.bool_)):
            values.append(bool(value))
        elif isinstance(value, (int, np.integer)):
            values.append(int(value))
        else:
            values.append(_finite_float(value))
    return {"shape": list(arr.shape), "values": values}


def _native_track_payload(data, nz):
    """Return compact native isobaric-track arrays when the NPZ has them."""
    required = ("track_nH", "track_gas_temperature")
    if not all(name in data.files for name in required):
        return {"available": False}

    track_nh = np.asarray(data["track_nH"], dtype=float)
    track_tgas = np.asarray(data["track_gas_temperature"], dtype=float)
    if track_nh.shape != track_tgas.shape or track_nh.ndim not in (2, 3):
        return {"available": False}
    if track_nh.ndim == 2:
        track_nh = np.broadcast_to(track_nh[None, ...],
                                   (nz,) + track_nh.shape)
        track_tgas = np.broadcast_to(track_tgas[None, ...],
                                     (nz,) + track_tgas.shape)
    elif track_nh.shape[0] == 1 and nz > 1:
        track_nh = np.broadcast_to(track_nh, (nz,) + track_nh.shape[1:])
        track_tgas = np.broadcast_to(track_tgas,
                                     (nz,) + track_tgas.shape[1:])
    if track_nh.shape[0] != nz:
        return {"available": False}

    shape = track_nh.shape
    if shape[1] == 0:
        return {"available": False}

    def optional(name, default, dtype=float):
        if name not in data.files:
            return np.full(shape, default, dtype=dtype)
        arr = np.asarray(data[name], dtype=dtype)
        if arr.shape == shape:
            return np.array(arr, copy=True)
        if arr.shape == shape[1:]:
            return np.array(np.broadcast_to(arr[None, ...], shape),
                            copy=True)
        if arr.shape[0:1] == (1,) and arr.shape[1:] == shape[1:]:
            return np.array(np.broadcast_to(arr, shape), copy=True)
        return np.full(shape, default, dtype=dtype)

    dust = optional("track_dust_temperature", np.nan)
    sample_time = optional("track_time_myr", np.nan)
    valid = optional("track_valid", True, bool)
    valid &= (np.isfinite(track_nh) & (track_nh > 0.0) &
              np.isfinite(track_tgas) & (track_tgas > 0.0))

    pressure = np.asarray(data["track_pressure"], dtype=float).reshape(-1) \
        if "track_pressure" in data.files else np.asarray([])
    ntrack = shape[1]
    if pressure.size != ntrack:
        pressure = np.full(ntrack, np.nan)
        for track in range(ntrack):
            state_pressure = track_nh[:, track] * track_tgas[:, track]
            good = valid[:, track] & np.isfinite(state_pressure)
            if np.any(good):
                pressure[track] = float(np.nanmedian(state_pressure[good]))

    return {
        "available": True,
        "shape": [int(v) for v in shape],
        "pressure": _float_list(pressure),
        "nH": _float_list(track_nh),
        "gasT": _float_list(track_tgas),
        "dustT": _float_list(dust),
        "time": _float_list(sample_time),
        "valid": [bool(v) for v in valid.reshape(-1)],
    }


def _derive_dominance(rates, threshold, cooling=True):
    side = np.maximum(rates if cooling else -rates, 0.0)
    total = side.sum(axis=-1)
    shares = np.divide(
        side, total[..., None],
        out=np.zeros_like(side, dtype=float),
        where=total[..., None] > 0.0,
    )
    top = np.argmax(shares, axis=-1).astype(np.int32)
    top_share = np.max(shares, axis=-1)
    top[total <= 0.0] = -1
    ranked = np.sort(shares, axis=-1)[..., ::-1]
    cumulative = np.cumsum(ranked, axis=-1)
    needed = np.minimum(
        rates.shape[-1],
        1 + np.count_nonzero(cumulative < threshold, axis=-1),
    )
    needed[total <= 0.0] = 0
    return top, top_share, needed.astype(np.int16), total


def _ranked_budget(values, denominator, threshold):
    positive = np.maximum(np.asarray(values, dtype=float), 0.0)
    if not np.isfinite(denominator) or denominator <= 0.0:
        denominator = float(positive.sum())
    if denominator <= 0.0:
        return []
    fractions = positive / denominator
    order = np.argsort(fractions)[::-1]
    rows = []
    cumulative = 0.0
    for channel in order:
        fraction = float(fractions[channel])
        if fraction <= 0.0:
            break
        rows.append([int(channel), int(round(10000.0 * fraction))])
        cumulative += fraction
        if cumulative >= threshold:
            break
    return rows


def _initial_cell(valid, pressure_index, reference_pressure_index):
    nz, nx, ny = valid.shape
    target = np.array([nz // 2, nx // 2, ny // 2])
    candidates = np.argwhere(valid)
    if reference_pressure_index is not None:
        try:
            ref = int(reference_pressure_index)
        except (TypeError, ValueError):
            ref = None
        if ref is not None:
            matched = np.argwhere(valid & (pressure_index == ref))
            if matched.size:
                candidates = matched
    if not candidates.size:
        return [int(v) for v in target]
    distance = ((candidates - target) ** 2).sum(axis=1)
    k, i, j = candidates[np.argmin(distance)]
    return [int(i), int(j), int(k)]


def build_payload(npz):
    with np.load(npz, allow_pickle=False) as data:
        nH = np.asarray(data["nH"], dtype=float)
        temperature = np.asarray(data["T"], dtype=float)
        metallicity = np.asarray(data["Z"], dtype=float)
        rates = np.asarray(data["cooling_rates"], dtype=float)
        expected_shape = (metallicity.size, nH.size, temperature.size)
        if rates.ndim != 4 or rates.shape[:3] != expected_shape:
            raise ValueError(
                "cooling_rates must have shape (nZ, nH, nT, nchannel)")

        fields = [str(v) for v in np.asarray(data["channel_fields"])]
        labels = [clean_label(v)
                  for v in np.asarray(data["channel_labels"])]
        nchannel = rates.shape[-1]
        if len(fields) != nchannel or len(labels) != nchannel:
            raise ValueError("channel metadata does not match cooling_rates")
        colors = [
            _CHANNEL_COLORS[i % len(_CHANNEL_COLORS)]
            for i in range(nchannel)
        ]
        if "channel_colors" in data.files:
            stored_colors = np.asarray(data["channel_colors"]).reshape(-1)
            if stored_colors.size == nchannel:
                colors = [
                    value.decode("utf-8") if isinstance(value, bytes)
                    else str(value)
                    for value in stored_colors
                ]

        threshold = float(_scalar_value(
            data, "budget_threshold", 0.999))
        threshold = float(np.clip(threshold, 0.0, 1.0))

        derived_cdom, derived_cshare, derived_cn, derived_ctotal = \
            _derive_dominance(rates, threshold, cooling=True)
        derived_hdom, derived_hshare, derived_hn, derived_htotal = \
            _derive_dominance(rates, threshold, cooling=False)

        # Legacy aliases are intentionally accepted here.
        if "cooling_dominant_index" in data.files:
            cdom = _cube_array(data, "cooling_dominant_index",
                               expected_shape, -1, int)
        else:
            cdom = _cube_array(data, "dominant_index", expected_shape,
                               -1, int)
            if "dominant_index" not in data.files:
                cdom = derived_cdom
        if "cooling_dominant_share" in data.files:
            cshare = _cube_array(data, "cooling_dominant_share",
                                 expected_shape, 0.0)
        else:
            cshare = _cube_array(data, "dominant_share", expected_shape,
                                 0.0)
            if "dominant_share" not in data.files:
                cshare = derived_cshare
        if "cooling_n_needed" in data.files:
            cn = _cube_array(data, "cooling_n_needed",
                             expected_shape, 0, int)
        else:
            cn = _cube_array(data, "n_needed", expected_shape, 0, int)
            if "n_needed" not in data.files:
                cn = derived_cn

        hdom = _cube_array(data, "heating_dominant_index",
                           expected_shape, -1, int)
        hshare = _cube_array(data, "heating_dominant_share",
                             expected_shape, 0.0)
        hn = _cube_array(data, "heating_n_needed",
                         expected_shape, 0, int)
        if "heating_dominant_index" not in data.files:
            hdom, hshare, hn = derived_hdom, derived_hshare, derived_hn

        cooling_total = _cube_array(
            data, "cooling_budget_total", expected_shape, np.nan)
        if "cooling_budget_total" not in data.files:
            cooling_total = derived_ctotal
        heating_total = _cube_array(
            data, "heating_budget_total", expected_shape, np.nan)
        if "heating_budget_total" not in data.files:
            heating_total = derived_htotal
        total_rate = _cube_array(
            data, "total_rate", expected_shape, np.nan)
        if "total_rate" not in data.files:
            total_rate = cooling_total - heating_total
        residual_rate = _cube_array(
            data, "residual_rate", expected_shape, 0.0)

        valid = _cube_array(
            data, "valid_mask", expected_shape, True, bool)
        finite_state = (np.isfinite(rates).all(axis=-1) &
                        np.isfinite(cooling_total) &
                        np.isfinite(heating_total) &
                        np.isfinite(total_rate) &
                        (cooling_total >= 0.0) &
                        (heating_total >= 0.0))
        valid &= finite_state
        if "valid_mask" not in data.files and \
                "dominant_index" in data.files:
            # Earlier files used -2 for an all-zero/failed block.
            valid &= cdom != -2
        cooling_present = valid & (cooling_total > 0.0)
        heating_present = valid & (heating_total > 0.0)
        cdom = np.where(
            cooling_present & (cdom >= 0) & (cdom < nchannel), cdom, -1)
        hdom = np.where(
            heating_present & (hdom >= 0) & (hdom < nchannel), hdom, -1)
        cshare = np.nan_to_num(
            np.clip(cshare, 0.0, 1.0), nan=0.0, posinf=0.0, neginf=0.0)
        hshare = np.nan_to_num(
            np.clip(hshare, 0.0, 1.0), nan=0.0, posinf=0.0, neginf=0.0)
        cshare = np.where(cooling_present, cshare, 0.0)
        hshare = np.where(heating_present, hshare, 0.0)
        cn = np.where(cooling_present, np.maximum(cn, 0), 0)
        hn = np.where(heating_present, np.maximum(hn, 0), 0)

        extrapolated = _cube_array(
            data, "extrapolated_mask", expected_shape, False, bool)
        extrapolated &= valid
        dust_temperature = _cube_array(
            data, "dust_temperature", expected_shape, np.nan)
        nominal_nh = np.broadcast_to(
            nH[None, :, None], expected_shape)
        actual_nh = _cube_array(
            data, "actual_nH", expected_shape, np.nan)
        if "actual_nH" not in data.files:
            actual_nh = np.array(nominal_nh, copy=True)
        sample_time = _cube_array(
            data, "sample_time_myr", expected_shape, np.nan)
        pressure_index = _cube_array(
            data, "pressure_index", expected_shape, -1, int)

        denominator = cooling_total + heating_total
        balance = _cube_array(
            data, "thermal_balance", expected_shape, np.nan)
        if "thermal_balance" not in data.files:
            balance = np.divide(
                total_rate, denominator,
                out=np.zeros_like(total_rate),
                where=denominator > 0.0,
            )
        balance = np.clip(balance, -1.0, 1.0)

        cooling_budgets = []
        heating_budgets = []
        for k in range(expected_shape[0]):
            for i in range(expected_shape[1]):
                for j in range(expected_shape[2]):
                    if not valid[k, i, j]:
                        cooling_budgets.append([])
                        heating_budgets.append([])
                        continue
                    cooling_budgets.append(_ranked_budget(
                        rates[k, i, j], cooling_total[k, i, j],
                        threshold))
                    heating_budgets.append(_ranked_budget(
                        -rates[k, i, j], heating_total[k, i, j],
                        threshold))

        status = np.zeros(expected_shape, dtype=np.int8)
        status[extrapolated] = 1
        status[~valid] = 2

        new_schema = all(
            name in data.files for name in (
                "valid_mask", "cooling_budget_total",
                "heating_budget_total", "total_rate",
            )
        )
        rate_units = _string_value(
            data, "rate_units",
            "erg cm³ s⁻¹" if new_schema else "legacy stored solver units",
        )

        pressure = np.asarray(data["pressure"]) \
            if "pressure" in data.files else np.asarray([])
        if pressure.shape == expected_shape:
            pressure_values = []
            pressure_cell = _float_list(pressure)
        else:
            pressure_values = _float_list(pressure)
            pressure_cell = []

        reference_pressure_index = _scalar_value(
            data, "reference_pressure_index", None)
        initial = _initial_cell(
            valid, pressure_index, reference_pressure_index)

        trajectory_meta = {}
        for name in (
            "trajectory_stop_code", "trajectory_stop_temperature",
            "trajectory_stop_time_myr", "trajectory_steps",
            "trajectory_solver_retries", "trajectory_safety_factor",
        ):
            value = _metadata_array(data, name)
            if value is not None:
                trajectory_meta[name] = value

        pressure_line_meta = {}
        for name in (
            "pressure_stop_code", "pressure_stop_temperature",
            "pressure_stop_time_myr", "pressure_steps",
            "pressure_solver_retries", "pressure_safety_factor",
        ):
            value = _metadata_array(data, name)
            if value is not None:
                pressure_line_meta[name] = value

        stop_labels = []
        if "stop_labels" in data.files:
            for value in np.asarray(data["stop_labels"]).reshape(-1):
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="replace")
                stop_labels.append(str(value))

        table_bounds = _metadata_array(data, "table_nH_bounds")
        native_tracks = _native_track_payload(data, metallicity.size)

        return {
            "nH": _float_list(nH, 12),
            "T": _float_list(temperature, 12),
            "Z": _float_list(metallicity, 12),
            "dims": [int(nH.size), int(temperature.size),
                     int(metallicity.size)],
            "fields": fields,
            "labels": labels,
            "colors": colors,
            "coolDom": [int(v) for v in cdom.reshape(-1)],
            "coolShare": [
                int(round(10000.0 * float(v)))
                for v in cshare.reshape(-1)
            ],
            "coolNeeded": [int(v) for v in cn.reshape(-1)],
            "heatDom": [int(v) for v in hdom.reshape(-1)],
            "heatShare": [
                int(round(10000.0 * float(v)))
                for v in hshare.reshape(-1)
            ],
            "heatNeeded": [int(v) for v in hn.reshape(-1)],
            "coolTotal": _float_list(cooling_total),
            "heatTotal": _float_list(heating_total),
            "totalRate": _float_list(total_rate),
            "residualRate": _float_list(residual_rate),
            "balance": _float_list(balance, 6),
            "status": [int(v) for v in status.reshape(-1)],
            "dustT": _float_list(dust_temperature),
            "actualNH": _float_list(actual_nh),
            "sampleTime": _float_list(sample_time),
            "pressureIndex": [
                int(v) for v in pressure_index.reshape(-1)
            ],
            "pressureValues": pressure_values,
            "pressureCell": pressure_cell,
            "coolBudget": cooling_budgets,
            "heatBudget": heating_budgets,
            "threshold": threshold,
            "initial": initial,
            "referencePressureIndex": (
                int(reference_pressure_index)
                if reference_pressure_index is not None else None
            ),
            "trajectory": trajectory_meta,
            "pressureLine": pressure_line_meta,
            "stopLabels": stop_labels,
            "tracks": native_tracks,
            "tableNHBounds": table_bounds,
            "samplingMode": _string_value(
                data, "sampling_mode",
                "legacy phase cube" if not new_schema else "history cube"),
            "pressureLaw": _string_value(
                data, "pressure_law", "not recorded"),
            "initialCondition": _string_value(
                data, "initial_condition", "not recorded"),
            "chemistryConfiguration": _string_value(
                data, "chemistry_configuration", "not recorded"),
            "rateNormalization": _string_value(
                data, "rate_normalization", "not recorded"),
            "uvb": os.path.basename(_string_value(
                data, "uvb_data_file", "")),
            "rateUnits": rate_units,
            "physicalRates": bool(new_schema),
            "legacy": not new_schema,
        }


TEMPLATE = r'''<title>CGM coolant isobaric-history cube</title>
<style>
:root {
  --surface: #f5f7fa; --panel: #ffffff; --ink: #18212d; --muted: #5d6877;
  --hair: #dce2e9; --accent: #1c66b1; --accent-soft: #e4effa;
  --invalid: #c73572; --extrap: #e39b25; --empty: #b6bcc3;
  --shadow: 0 1px 4px rgba(18,30,48,.09);
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface: #10161e; --panel: #171f2a; --ink: #e8edf4; --muted: #9aa6b6;
    --hair: #2a3544; --accent: #83b8ee; --accent-soft: #1d3045;
    --invalid: #ed6a9f; --extrap: #efb14c; --empty: #647080;
    --shadow: 0 1px 4px rgba(0,0,0,.38);
  }
}
* { box-sizing: border-box; }
html, body { margin: 0; min-height: 100%; }
body {
  background: var(--surface); color: var(--ink);
  font: 14px/1.42 system-ui, -apple-system, "Segoe UI", sans-serif;
}
.mono { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
        font-variant-numeric: tabular-nums; }
header { padding: 14px 18px 9px; }
header h1 { margin: 0; font-size: 18px; font-weight: 680; }
header p { margin: 3px 0 0; color: var(--muted); font-size: 12.5px; }
.toolbar {
  margin: 0 14px 12px; padding: 9px 11px; display: flex; flex-wrap: wrap;
  gap: 8px 16px; align-items: center; background: var(--panel);
  border: 1px solid var(--hair); border-radius: 8px; box-shadow: var(--shadow);
}
.toolbar label { color: var(--muted); font-size: 12px; }
select, input {
  color: var(--ink); background: var(--panel); border: 1px solid var(--hair);
}
select { border-radius: 5px; padding: 4px 7px; margin-left: 5px; }
.slider { display: inline-flex; align-items: center; gap: 6px; }
.slider input { width: 105px; accent-color: var(--accent); }
.slider output { min-width: 78px; color: var(--ink); font-size: 11.5px; }
.layout { display: grid; grid-template-columns: minmax(0, 1fr) 350px;
          gap: 12px; padding: 0 14px 14px; align-items: start; }
.main { min-width: 0; }
.slices { display: grid; grid-template-columns: repeat(3, minmax(250px, 1fr));
          gap: 10px; }
.panel, .card {
  background: var(--panel); border: 1px solid var(--hair); border-radius: 8px;
  box-shadow: var(--shadow);
}
.panel { position: relative; min-width: 0; padding: 8px 8px 5px; }
.panel h2 { margin: 0 4px 2px; font-size: 12.5px; font-weight: 650; }
.panel h2 span { color: var(--muted); font-weight: 450; }
.slice { display: block; width: 100%; height: 300px; outline: none; }
.slice:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
#tip {
  position: fixed; z-index: 8; display: none; pointer-events: none;
  padding: 5px 8px; max-width: 250px; background: var(--panel);
  border: 1px solid var(--hair); border-radius: 6px; box-shadow: var(--shadow);
  font-size: 11.5px;
}
.legend {
  margin-top: 10px; padding: 9px 11px; min-height: 42px;
  background: var(--panel); border: 1px solid var(--hair); border-radius: 8px;
  display: flex; flex-wrap: wrap; gap: 6px 15px; align-items: center;
}
.legrow { display: inline-flex; align-items: center; gap: 5px;
          font-size: 11.5px; white-space: nowrap; }
.swatch { width: 11px; height: 11px; display: inline-block; border-radius: 2px;
          border: 1px solid rgba(80,90,105,.28); }
.gradient { width: 170px; height: 11px; border-radius: 5px;
            border: 1px solid var(--hair); }
.aside { display: flex; flex-direction: column; gap: 10px; min-width: 0; }
.card { padding: 11px 13px; }
.card h2 { margin: 0 0 8px; color: var(--muted); font-size: 11px;
           letter-spacing: .09em; text-transform: uppercase; }
.state-head { display: flex; gap: 8px; align-items: center; }
.state-head b { font-size: 15px; }
.badge { margin-left: auto; border-radius: 10px; padding: 2px 7px;
         font-size: 10.5px; font-weight: 650; }
.where { color: var(--muted); font-size: 11.5px; margin: 3px 0 9px; }
.dompair { display: grid; grid-template-columns: 1fr 1fr; gap: 8px;
           margin-bottom: 9px; }
.domcell { border: 1px solid var(--hair); border-radius: 6px; padding: 6px 7px; }
.domcell small { display: block; color: var(--muted); }
.domname { display: flex; align-items: center; gap: 5px; margin-top: 2px;
           font-size: 12px; }
.domname strong { margin-left: auto; }
.budgets { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.budget h3 { margin: 0 0 5px; font-size: 11.5px; }
.brow { display: grid; grid-template-columns: 9px minmax(0,1fr) 39px;
        gap: 5px; align-items: center; margin: 4px 0; font-size: 10.5px; }
.brow .nm { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.bar { grid-column: 2 / 4; height: 4px; margin-top: -3px;
       border-radius: 3px; background: var(--surface); overflow: hidden; }
.bar i { display: block; height: 100%; border-radius: 3px; }
.metrics { margin-top: 9px; padding-top: 7px; border-top: 1px solid var(--hair);
           display: grid; grid-template-columns: 1fr auto; gap: 3px 8px;
           font-size: 11px; }
.metrics span:nth-child(odd) { color: var(--muted); }
.notice { color: var(--muted); font-size: 12px; }
.notice strong { color: var(--ink); }
#track { display: block; width: 100%; height: 190px; }
#track-title { color: var(--muted); font-size: 11px; margin: -3px 0 3px; }
.track-key { display: flex; gap: 14px; font-size: 10.5px; color: var(--muted); }
.linekey { display: inline-block; width: 22px; height: 0;
           border-top: 2px solid var(--ink); vertical-align: 3px; }
.linekey.dust { border-top: 2px dashed #818791; }
.method { color: var(--muted); font-size: 11.5px; }
.method b { color: var(--ink); }
.legacy { color: #a94825; }
@media (max-width: 1250px) {
  .slices { grid-template-columns: repeat(2, minmax(260px, 1fr)); }
}
@media (max-width: 900px) {
  .layout { grid-template-columns: 1fr; }
  .aside { display: grid; grid-template-columns: repeat(2, minmax(250px, 1fr)); }
}
@media (max-width: 650px) {
  .slices, .aside { grid-template-columns: 1fr; }
  .slice { height: 280px; }
}
</style>

<header>
  <h1>CGM coolant isobaric-history cube</h1>
  <p>@@SUBTITLE@@</p>
</header>

<div class="toolbar">
  <label>Quantity
    <select id="quantity">
      <option value="coolDom">dominant cooling channel</option>
      <option value="heatDom">dominant heating channel</option>
      <option value="balance">thermal balance</option>
      <option value="coolNeeded">cooling channels for budget</option>
      <option value="heatNeeded">heating channels for budget</option>
      <option value="status">sampling status</option>
    </select>
  </label>
  <label class="slider">n<sub>H</sub>
    <input id="si" type="range"><output id="oi" class="mono"></output>
  </label>
  <label class="slider">T
    <input id="sj" type="range"><output id="oj" class="mono"></output>
  </label>
  <label class="slider">Z/Z⊙
    <input id="sk" type="range"><output id="ok" class="mono"></output>
  </label>
</div>

<div class="layout">
  <main class="main">
    <div class="slices">
      <section class="panel">
        <h2>n<sub>H</sub>–T <span id="title-nt"></span></h2>
        <canvas class="slice" id="nt" tabindex="0"
          aria-label="Density-temperature slice"></canvas>
      </section>
      <section class="panel">
        <h2>n<sub>H</sub>–Z <span id="title-nz"></span></h2>
        <canvas class="slice" id="nz" tabindex="0"
          aria-label="Density-metallicity slice"></canvas>
      </section>
      <section class="panel">
        <h2>T–Z <span id="title-tz"></span></h2>
        <canvas class="slice" id="tz" tabindex="0"
          aria-label="Temperature-metallicity slice"></canvas>
      </section>
    </div>
    <div class="legend" id="legend"></div>
  </main>

  <aside class="aside">
    <section class="card">
      <h2>Selected state</h2>
      <div id="inspector"></div>
    </section>
    <section class="card">
      <h2>Selected isobaric cooling track</h2>
      <div id="track-title"></div>
      <canvas id="track"></canvas>
      <div class="track-key">
        <span><i class="linekey"></i> gas T</span>
        <span><i class="linekey dust"></i> dust T</span>
      </div>
    </section>
    <section class="card method">
      <h2>Dataset</h2>
      <div id="method"></div>
    </section>
  </aside>
</div>
<div id="tip"></div>

<script>
"use strict";
const DATA = @@DATA@@;
const NX = DATA.dims[0], NY = DATA.dims[1], NZ = DATA.dims[2];
const idx = (i,j,k) => (k*NX + i)*NY + j;
const logn = DATA.nH.map(Math.log10);
const logt = DATA.T.map(Math.log10);
const logz = DATA.Z.map(Math.log10);
const state = { i: DATA.initial[0], j: DATA.initial[1], k: DATA.initial[2],
                quantity: "coolDom" };
const css = () => getComputedStyle(document.documentElement);
const token = name => css().getPropertyValue(name).trim();
const tip = document.getElementById("tip");
const canvases = {
  nt: document.getElementById("nt"),
  nz: document.getElementById("nz"),
  tz: document.getElementById("tz"),
};
const plotInfo = new Map();
const SUP = {"-":"⁻","0":"⁰","1":"¹","2":"²","3":"³","4":"⁴",
             "5":"⁵","6":"⁶","7":"⁷","8":"⁸","9":"⁹"};
const pow10 = e => "10" + String(e).split("").map(c => SUP[c] || c).join("");
const esc = value => String(value).replace(/[&<>"']/g, c => (
  {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

function edges(values) {
  if (values.length === 1) return [values[0]-0.5, values[0]+0.5];
  const out = [values[0] - (values[1]-values[0])/2];
  for (let n=1; n<values.length; n++)
    out.push((values[n-1]+values[n])/2);
  out.push(values[values.length-1] +
           (values[values.length-1]-values[values.length-2])/2);
  return out;
}
const en = edges(logn), et = edges(logt), ez = edges(logz);

function fmtNumber(value) {
  if (!Number.isFinite(value)) return "—";
  if (value === 0) return "0";
  const a = Math.abs(value), e = Math.floor(Math.log10(a));
  if (e >= -2 && e <= 3) return Number(value.toPrecision(4)).toString();
  return (value/Math.pow(10,e)).toFixed(2) + "×" + pow10(e);
}
function fmtRate(value) {
  if (!Number.isFinite(value)) return "—";
  const sign = value > 0 ? "+" : value < 0 ? "−" : "";
  return sign + fmtNumber(Math.abs(value));
}
function hexRgb(hex) {
  return [parseInt(hex.slice(1,3),16), parseInt(hex.slice(3,5),16),
          parseInt(hex.slice(5,7),16)];
}
function mix(a,b,t) {
  const x=hexRgb(a), y=hexRgb(b);
  return "#" + x.map((v,n) =>
    Math.round(v+(y[n]-v)*t).toString(16).padStart(2,"0")).join("");
}
function balanceColor(value) {
  if (!Number.isFinite(value)) return token("--empty");
  return value < 0
    ? mix("#c85a3c", "#f1f1ec", value+1)
    : mix("#f1f1ec", "#2a72bb", value);
}
const NEED = ["#edf3fb","#c9def3","#8dbbe6","#4b91cf","#1f619e","#123b68"];
function neededColor(value) {
  if (!value) return token("--empty");
  return NEED[Math.min(NEED.length-1, Math.max(0,value-1))];
}
function statusName(code) {
  return code === 2 ? "invalid" : code === 1 ? "extrapolated" : "sampled";
}
function statusColor(code) {
  return code === 2 ? token("--invalid")
       : code === 1 ? token("--extrap") : "#42a678";
}
function channelName(channel) {
  return channel >= 0 && channel < DATA.labels.length
    ? DATA.labels[channel] : "no positive contribution";
}
function channelColor(channel) {
  return channel >= 0 && channel < DATA.colors.length
    ? DATA.colors[channel] : token("--empty");
}

function cellVisual(cell) {
  const status = DATA.status[cell];
  if (status === 2) return { color: token("--invalid"), label: "invalid" };
  if (state.quantity === "status")
    return { color: statusColor(status), label: statusName(status) };
  if (state.quantity === "coolDom") {
    const c=DATA.coolDom[cell];
    return { color:channelColor(c), label:channelName(c),
             value:(DATA.coolShare[cell]/100).toFixed(1)+"%" };
  }
  if (state.quantity === "heatDom") {
    const c=DATA.heatDom[cell];
    return { color:channelColor(c), label:channelName(c),
             value:(DATA.heatShare[cell]/100).toFixed(1)+"%" };
  }
  if (state.quantity === "balance") {
    const v=DATA.balance[cell];
    return { color:balanceColor(v), label:v < 0 ? "net heating" : "net cooling",
             value:Number.isFinite(v) ? v.toFixed(3) : "—" };
  }
  const value = state.quantity === "coolNeeded"
    ? DATA.coolNeeded[cell] : DATA.heatNeeded[cell];
  return { color:neededColor(value), label:value ? value+" channels" : "none",
           value:String(value) };
}

function setupCanvas(canvas) {
  const rect=canvas.getBoundingClientRect(), dpr=window.devicePixelRatio || 1;
  const w=Math.max(260,rect.width), h=Math.max(160,rect.height);
  const W=Math.round(w*dpr), H=Math.round(h*dpr);
  if (canvas.width !== W || canvas.height !== H) {
    canvas.width=W; canvas.height=H;
  }
  const ctx=canvas.getContext("2d");
  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,w,h);
  return {ctx,w,h,p:{l:51,r:w-12,t:10,b:h-43}};
}
function xmap(v,lo,hi,p) { return p.l+(v-lo)/(hi-lo)*(p.r-p.l); }
function ymap(v,lo,hi,p) { return p.b-(v-lo)/(hi-lo)*(p.b-p.t); }
function ticks(lo,hi) {
  const span=hi-lo;
  let step=span > 7 ? 2 : span > 3 ? 1 : span > 1.4 ? .5 : .25;
  const out=[];
  for (let v=Math.ceil(lo/step)*step; v<=hi+1e-8; v+=step)
    out.push(+v.toFixed(6));
  return out;
}
function drawAxes(ctx,p,xrange,yrange,xlabel,ylabel) {
  ctx.strokeStyle=token("--hair"); ctx.lineWidth=1;
  ctx.strokeRect(p.l,p.t,p.r-p.l,p.b-p.t);
  ctx.fillStyle=token("--muted"); ctx.font="10.5px system-ui";
  ctx.textAlign="center"; ctx.textBaseline="top";
  for (const v of ticks(xrange[0],xrange[1])) {
    const x=xmap(v,xrange[0],xrange[1],p);
    ctx.beginPath(); ctx.moveTo(x,p.b); ctx.lineTo(x,p.b+4); ctx.stroke();
    ctx.fillText(pow10(v),x,p.b+6);
  }
  ctx.textAlign="right"; ctx.textBaseline="middle";
  for (const v of ticks(yrange[0],yrange[1])) {
    const y=ymap(v,yrange[0],yrange[1],p);
    ctx.beginPath(); ctx.moveTo(p.l-4,y); ctx.lineTo(p.l,y); ctx.stroke();
    ctx.fillText(pow10(v),p.l-6,y);
  }
  ctx.fillStyle=token("--ink"); ctx.font="11px system-ui";
  ctx.textAlign="center"; ctx.textBaseline="bottom";
  ctx.fillText(xlabel,(p.l+p.r)/2,p.b+40);
  ctx.save(); ctx.translate(12,(p.t+p.b)/2); ctx.rotate(-Math.PI/2);
  ctx.fillText(ylabel,0,0); ctx.restore();
}
function hatch(ctx,x0,y0,x1,y1) {
  ctx.save(); ctx.beginPath(); ctx.rect(x0,y0,x1-x0,y1-y0); ctx.clip();
  ctx.strokeStyle="rgba(25,30,38,.48)"; ctx.lineWidth=.7;
  for (let q=x0-(y1-y0); q<x1+(y1-y0); q+=7) {
    ctx.beginPath(); ctx.moveTo(q,y1); ctx.lineTo(q+(y1-y0),y0); ctx.stroke();
  }
  ctx.restore();
}
function drawCell(ctx,cell,x0,y0,x1,y1) {
  const vis=cellVisual(cell);
  ctx.fillStyle=vis.color;
  ctx.fillRect(Math.min(x0,x1),Math.min(y0,y1),
               Math.abs(x1-x0)+.35,Math.abs(y1-y0)+.35);
  if (DATA.status[cell] === 1 && state.quantity !== "status")
    hatch(ctx,Math.min(x0,x1),Math.min(y0,y1),
          Math.max(x0,x1),Math.max(y0,y1));
  if (DATA.status[cell] === 2) {
    ctx.strokeStyle="rgba(255,255,255,.75)"; ctx.lineWidth=.8;
    ctx.beginPath(); ctx.moveTo(x0,y0); ctx.lineTo(x1,y1);
    ctx.moveTo(x0,y1); ctx.lineTo(x1,y0); ctx.stroke();
  }
}
function crosshair(ctx,p,x,y) {
  ctx.save(); ctx.strokeStyle=token("--accent"); ctx.lineWidth=1.25;
  ctx.setLineDash([3,2]);
  ctx.beginPath(); ctx.moveTo(x,p.t); ctx.lineTo(x,p.b);
  ctx.moveTo(p.l,y); ctx.lineTo(p.r,y); ctx.stroke();
  ctx.setLineDash([]); ctx.fillStyle=token("--panel");
  ctx.beginPath(); ctx.arc(x,y,4,0,2*Math.PI); ctx.fill();
  ctx.strokeStyle=token("--accent"); ctx.stroke(); ctx.restore();
}

function drawSlice(type) {
  const canvas=canvases[type], setup=setupCanvas(canvas);
  const {ctx,p}=setup;
  let xv,yv,xe,ye,xlabel,ylabel;
  if (type === "nt") {
    xv=logn; yv=logt; xe=en; ye=et;
    xlabel="nH [cm⁻³]"; ylabel="gas T [K]";
    for (let i=0;i<NX;i++) for (let j=0;j<NY;j++) {
      drawCell(ctx,idx(i,j,state.k),
        xmap(xe[i],xe[0],xe[xe.length-1],p),
        ymap(ye[j],ye[0],ye[ye.length-1],p),
        xmap(xe[i+1],xe[0],xe[xe.length-1],p),
        ymap(ye[j+1],ye[0],ye[ye.length-1],p));
    }
  } else if (type === "nz") {
    xv=logn; yv=logz; xe=en; ye=ez;
    xlabel="nH [cm⁻³]"; ylabel="Z/Z⊙";
    for (let i=0;i<NX;i++) for (let k=0;k<NZ;k++) {
      drawCell(ctx,idx(i,state.j,k),
        xmap(xe[i],xe[0],xe[xe.length-1],p),
        ymap(ye[k],ye[0],ye[ye.length-1],p),
        xmap(xe[i+1],xe[0],xe[xe.length-1],p),
        ymap(ye[k+1],ye[0],ye[ye.length-1],p));
    }
  } else {
    xv=logt; yv=logz; xe=et; ye=ez;
    xlabel="gas T [K]"; ylabel="Z/Z⊙";
    for (let j=0;j<NY;j++) for (let k=0;k<NZ;k++) {
      drawCell(ctx,idx(state.i,j,k),
        xmap(xe[j],xe[0],xe[xe.length-1],p),
        ymap(ye[k],ye[0],ye[ye.length-1],p),
        xmap(xe[j+1],xe[0],xe[xe.length-1],p),
        ymap(ye[k+1],ye[0],ye[ye.length-1],p));
    }
  }
  const xr=[xe[0],xe[xe.length-1]], yr=[ye[0],ye[ye.length-1]];
  drawAxes(ctx,p,xr,yr,xlabel,ylabel);
  const xi=type==="tz" ? state.j : state.i;
  const yi=type==="nt" ? state.j : state.k;
  crosshair(ctx,p,xmap(xv[xi],xr[0],xr[1],p),
                 ymap(yv[yi],yr[0],yr[1],p));
  plotInfo.set(canvas,{type,p,xr,yr,xv,yv});
}

function nearest(values,value) {
  let best=0, dist=Infinity;
  for (let n=0;n<values.length;n++) {
    const d=Math.abs(values[n]-value);
    if (d<dist) {dist=d; best=n;}
  }
  return best;
}
function pointCell(canvas,event,commit) {
  const info=plotInfo.get(canvas);
  if (!info) return -1;
  const rect=canvas.getBoundingClientRect();
  const x=event.clientX-rect.left, y=event.clientY-rect.top, p=info.p;
  if (x<p.l || x>p.r || y<p.t || y>p.b) return -1;
  const xv=info.xr[0]+(x-p.l)/(p.r-p.l)*(info.xr[1]-info.xr[0]);
  const yv=info.yr[0]+(p.b-y)/(p.b-p.t)*(info.yr[1]-info.yr[0]);
  const a=nearest(info.xv,xv), b=nearest(info.yv,yv);
  let i=state.i,j=state.j,k=state.k;
  if (info.type==="nt") {i=a;j=b;} else if(info.type==="nz"){i=a;k=b;}
  else {j=a;k=b;}
  if (commit) { state.i=i;state.j=j;state.k=k; renderAll(); }
  return idx(i,j,k);
}
for (const canvas of Object.values(canvases)) {
  canvas.addEventListener("pointermove",event => {
    const cell=pointCell(canvas,event,false);
    if (cell<0) {tip.style.display="none"; return;}
    const v=cellVisual(cell);
    const k=Math.floor(cell/(NX*NY)), i=Math.floor(cell/NY)%NX, j=cell%NY;
    tip.innerHTML="<b>"+esc(v.label)+"</b>"+
      (v.value ? " <span class=mono>"+esc(v.value)+"</span>" : "")+
      "<br><span class=mono>nH="+fmtNumber(DATA.nH[i])+
      " · T="+fmtNumber(DATA.T[j])+" · Z="+fmtNumber(DATA.Z[k])+"</span>";
    tip.style.display="block"; tip.style.left=(event.clientX+12)+"px";
    tip.style.top=(event.clientY+10)+"px";
  });
  canvas.addEventListener("pointerleave",()=>tip.style.display="none");
  canvas.addEventListener("click",event=>pointCell(canvas,event,true));
  canvas.addEventListener("keydown",event => {
    let used=true;
    if (canvas===canvases.nt) {
      if(event.key==="ArrowLeft")state.i--; else if(event.key==="ArrowRight")state.i++;
      else if(event.key==="ArrowDown")state.j--; else if(event.key==="ArrowUp")state.j++;
      else used=false;
    } else if (canvas===canvases.nz) {
      if(event.key==="ArrowLeft")state.i--; else if(event.key==="ArrowRight")state.i++;
      else if(event.key==="ArrowDown")state.k--; else if(event.key==="ArrowUp")state.k++;
      else used=false;
    } else {
      if(event.key==="ArrowLeft")state.j--; else if(event.key==="ArrowRight")state.j++;
      else if(event.key==="ArrowDown")state.k--; else if(event.key==="ArrowUp")state.k++;
      else used=false;
    }
    if (used) {
      event.preventDefault();
      state.i=Math.max(0,Math.min(NX-1,state.i));
      state.j=Math.max(0,Math.min(NY-1,state.j));
      state.k=Math.max(0,Math.min(NZ-1,state.k));
      renderAll();
    }
  });
}

function buildLegend() {
  const host=document.getElementById("legend");
  let html="";
  if (state.quantity==="coolDom" || state.quantity==="heatDom") {
    const source=state.quantity==="coolDom" ? DATA.coolDom : DATA.heatDom;
    const counts=new Map();
    for(let q=0;q<source.length;q++) if(DATA.status[q]!==2 && source[q]>=0)
      counts.set(source[q],(counts.get(source[q])||0)+1);
    for(const c of [...counts.keys()].sort((a,b)=>a-b))
      html+='<span class="legrow"><i class="swatch" style="background:'+
        DATA.colors[c]+'"></i>'+esc(DATA.labels[c])+
        ' <span class="mono">'+counts.get(c)+'</span></span>';
    html+='<span class="legrow"><i class="swatch" style="background:var(--empty)"></i>'+
      'no positive contribution</span>';
  } else if (state.quantity==="balance") {
    html='<span class="legrow">heating</span><i class="gradient" '+
      'style="background:linear-gradient(90deg,#c85a3c,#f1f1ec,#2a72bb)"></i>'+
      '<span class="legrow">cooling</span>'+
      '<span class="legrow mono">η = Λnet/(C+H), −1 … +1</span>';
  } else if (state.quantity==="status") {
    for(const [code,name] of [[0,"sampled"],[1,"extrapolated"],[2,"invalid"]])
      html+='<span class="legrow"><i class="swatch" style="background:'+
        statusColor(code)+'"></i>'+name+'</span>';
  } else {
    for(let n=1;n<=5;n++) html+='<span class="legrow"><i class="swatch" '+
      'style="background:'+NEED[n-1]+'"></i>'+n+'</span>';
    html+='<span class="legrow"><i class="swatch" style="background:'+
      NEED[5]+'"></i>≥6 channels for '+(DATA.threshold*100).toFixed(2)+'%</span>';
  }
  html+='<span class="legrow"><i class="swatch" style="background:var(--extrap)"></i>'+
    'diagonal hatch = extrapolated</span>'+
    '<span class="legrow"><i class="swatch" style="background:var(--invalid)"></i>'+
    'cross = invalid</span>';
  host.innerHTML=html;
}

function budgetHTML(entries) {
  if (!entries || !entries.length) return '<p class="notice">none</p>';
  return entries.map(([c,bp]) => {
    const pct=bp/100;
    return '<div class="brow"><i class="swatch" style="background:'+
      DATA.colors[c]+'"></i><span class="nm" title="'+esc(DATA.labels[c])+'">'+
      esc(DATA.labels[c])+'</span><span class="mono">'+pct.toFixed(2)+
      '%</span><span class="bar"><i style="width:'+Math.min(100,pct)+
      '%;background:'+DATA.colors[c]+'"></i></span></div>';
  }).join("");
}
function recordedPressure(cell) {
  if (DATA.pressureCell.length) return DATA.pressureCell[cell];
  const p=DATA.pressureIndex[cell];
  if (p>=0 && p<DATA.pressureValues.length) return DATA.pressureValues[p];
  if (DATA.pressureValues.length===1) return DATA.pressureValues[0];
  return null;
}
function selectedNativeTrack(cell) {
  if (!DATA.tracks.available) return -1;
  const ntrack=DATA.tracks.shape[1];
  let target=recordedPressure(cell);
  if(!Number.isFinite(target)) {
    const k=Math.floor(cell/(NX*NY)), i=Math.floor(cell/NY)%NX, j=cell%NY;
    target=DATA.nH[i]*DATA.T[j];
  }
  let best=-1, distance=Infinity;
  for(let q=0;q<ntrack;q++) {
    const value=DATA.tracks.pressure[q];
    if(!Number.isFinite(value) || value<=0 || target<=0) continue;
    const d=Math.abs(Math.log10(value)-Math.log10(target));
    if(d<distance){distance=d;best=q;}
  }
  return best;
}
function pressureAt(cell) {
  const recorded=recordedPressure(cell);
  if(Number.isFinite(recorded)) return recorded;
  const track=selectedNativeTrack(cell);
  if(track>=0 && track<DATA.tracks.pressure.length)
    return DATA.tracks.pressure[track];
  const i=Math.floor(cell/NY)%NX, j=cell%NY;
  return DATA.nH[i]*DATA.T[j];
}
function showInspector() {
  const cell=idx(state.i,state.j,state.k), status=DATA.status[cell];
  const c=DATA.coolDom[cell], h=DATA.heatDom[cell];
  const badgeColor=statusColor(status);
  const td=DATA.dustT[cell], time=DATA.sampleTime[cell], pressure=pressureAt(cell);
  const actualNH=DATA.actualNH[cell], pressureIndex=DATA.pressureIndex[cell];
  const pressureStop=metadataAt(
    DATA.pressureLine.pressure_stop_code,pressureIndex,state.k);
  const pressureStopTemperature=metadataAt(
    DATA.pressureLine.pressure_stop_temperature,pressureIndex,state.k);
  const pressureStopTime=metadataAt(
    DATA.pressureLine.pressure_stop_time_myr,pressureIndex,state.k);
  const pressureSteps=metadataAt(
    DATA.pressureLine.pressure_steps,pressureIndex,state.k);
  const pressureRetries=metadataAt(
    DATA.pressureLine.pressure_solver_retries,pressureIndex,state.k);
  const pressureSafety=metadataAt(
    DATA.pressureLine.pressure_safety_factor,pressureIndex,state.k);
  let html='<div class="state-head"><b>nH–T–Z sample</b>'+
    '<span class="badge" style="color:'+badgeColor+
    ';background:'+badgeColor+'22">'+statusName(status)+'</span></div>'+
    '<div class="where mono">nH = '+fmtNumber(DATA.nH[state.i])+
    ' cm⁻³ · T = '+fmtNumber(DATA.T[state.j])+' K · Z = '+
    fmtNumber(DATA.Z[state.k])+' Z⊙</div>';
  if(status===2) {
    html+='<p class="notice"><strong>No physical result is available for this '+
      'sample.</strong> Budget values are intentionally hidden.</p>';
  } else {
    html+='<div class="dompair"><div class="domcell"><small>dominant cooling</small>'+
      '<div class="domname"><i class="swatch" style="background:'+channelColor(c)+
      '"></i><span>'+esc(channelName(c))+'</span><strong class="mono">'+
      (DATA.coolShare[cell]/100).toFixed(1)+'%</strong></div></div>'+
      '<div class="domcell"><small>dominant heating</small>'+
      '<div class="domname"><i class="swatch" style="background:'+channelColor(h)+
      '"></i><span>'+esc(channelName(h))+'</span><strong class="mono">'+
      (DATA.heatShare[cell]/100).toFixed(1)+'%</strong></div></div></div>'+
      '<div class="budgets"><div class="budget"><h3>Cooling budget</h3>'+
      budgetHTML(DATA.coolBudget[cell])+'</div><div class="budget">'+
      '<h3>Heating budget</h3>'+budgetHTML(DATA.heatBudget[cell])+'</div></div>';
  }
  html+='<div class="metrics"><span>cooling total C</span><b class="mono">'+
    fmtRate(DATA.coolTotal[cell])+'</b><span>heating total H</span>'+
    '<b class="mono">'+fmtRate(DATA.heatTotal[cell])+'</b>'+
    '<span>Λnet (+ means cooling)</span><b class="mono">'+
    fmtRate(DATA.totalRate[cell])+'</b><span>budget residual</span>'+
    '<b class="mono">'+fmtRate(DATA.residualRate[cell])+'</b>'+
    '<span>rate units</span><b>'+esc(DATA.rateUnits)+'</b>'+
    '<span>dust temperature</span><b class="mono">'+
    (Number.isFinite(td)?fmtNumber(td)+' K':"—")+'</b>'+
    '<span>sample time</span><b class="mono">'+
    (Number.isFinite(time)?fmtNumber(time)+' Myr':"—")+'</b>'+
    '<span>sampled nH</span><b class="mono">'+
    (Number.isFinite(actualNH)?fmtNumber(actualNH)+' cm⁻³':"—")+'</b>'+
    '<span>P/kB</span><b class="mono">'+
    (Number.isFinite(pressure)?fmtNumber(pressure)+' K cm⁻³':"—")+
    '</b>'+
    (pressureStop!==null
      ? '<span>pressure-history stop</span><b>'+
        esc(stopLabel(pressureStop))+
        (Number.isFinite(pressureStopTemperature)
          ? ' · T='+fmtNumber(pressureStopTemperature)+' K' : '')+
        (Number.isFinite(pressureStopTime)
          ? ' · '+fmtNumber(pressureStopTime)+' Myr' : '')+
        (Number.isFinite(pressureSteps)
          ? ' · '+pressureSteps+' steps' : '')+
        (Number.isFinite(pressureRetries) && pressureRetries>0
          ? ' · '+pressureRetries+' retries' : '')+
        (Number.isFinite(pressureSafety)
          ? ' · safety='+fmtNumber(pressureSafety) : '')+'</b>' : '')+
    '</div>';
  document.getElementById("inspector").innerHTML=html;
}

function metadataAt(meta,p,k) {
  if(!meta || !meta.values || !meta.values.length) return null;
  const s=meta.shape, v=meta.values;
  if(s.length===0 || v.length===1) return v[0];
  if(s.length===1) return p>=0 && p<s[0] ? v[p] : null;
  if(s.length===2) {
    if(s[0]===NZ && p>=0 && p<s[1]) return v[k*s[1]+p];
    if(s[1]===NZ && p>=0 && p<s[0]) return v[p*s[1]+k];
  }
  return null;
}
function stopLabel(code) {
  const value=Number(code);
  return Number.isInteger(value) && value>=0 && value<DATA.stopLabels.length
    ? DATA.stopLabels[value] : String(code);
}
function drawTrack() {
  const canvas=document.getElementById("track"), setup=setupCanvas(canvas);
  const {ctx,w,h}=setup, p={l:46,r:w-10,t:9,b:h-36};
  const selected=idx(state.i,state.j,state.k);
  let pressure=DATA.pressureIndex[selected], track=-1;
  let source="native history-dependent track";
  const rows=[];
  if(DATA.tracks.available) {
    track=selectedNativeTrack(selected);
    const shape=DATA.tracks.shape, ntrack=shape[1], ntarget=shape[2];
    if(track>=0 && track<ntrack) {
      for(let q=0;q<ntarget;q++) {
        const at=(state.k*ntrack+track)*ntarget+q;
        if(!DATA.tracks.valid[at]) continue;
        const nh=DATA.tracks.nH[at], tg=DATA.tracks.gasT[at];
        if(!Number.isFinite(nh) || nh<=0 || !Number.isFinite(tg) || tg<=0)
          continue;
        rows.push({n:Math.log10(nh),t:Math.log10(tg),
                   td:DATA.tracks.dustT[at],time:DATA.tracks.time[at]});
      }
      pressure=track;
    }
  } else if(pressure>=0) {
    source="atlas samples sharing pressure index";
    for(let i=0;i<NX;i++) for(let j=0;j<NY;j++) {
      const cell=idx(i,j,state.k);
      if(DATA.status[cell]!==2 && DATA.pressureIndex[cell]===pressure)
        rows.push({i,j,cell,n:logn[i],t:logt[j],td:DATA.dustT[cell],
                   time:DATA.sampleTime[cell]});
    }
  }
  if(rows.length<2) {
    // A visual isobar through the selected history-cube state.  It is an
    // analytic guide only, not a replacement for a time-dependent track.
    source="analytic nH Tgas = constant guide";
    const guidePressure=DATA.nH[state.i]*DATA.T[state.j];
    rows.length=0;
    for(let i=0;i<NX;i++) {
      const tg=guidePressure/DATA.nH[i], lt=Math.log10(tg);
      if(lt<et[0] || lt>et[et.length-1]) continue;
      const j=nearest(logt,lt), cell=idx(i,j,state.k);
      if(DATA.status[cell]===2) continue;
      rows.push({i,j,cell,n:logn[i],t:lt,td:DATA.dustT[cell],time:null});
    }
    track=-1;
    pressure=-1;
  }
  const title=document.getElementById("track-title");
  if(rows.length<2) {
    ctx.fillStyle=token("--muted"); ctx.font="12px system-ui";
    ctx.textAlign="center"; ctx.fillText(
      "no isobaric track or in-range analytic guide",w/2,h/2);
    title.textContent="track unavailable";
    return;
  }
  const timed=rows.filter(r=>Number.isFinite(r.time)).length>=2;
  rows.sort(timed ? ((a,b)=>a.time-b.time) : ((a,b)=>b.t-a.t || a.n-b.n));
  const xr=[en[0],en[en.length-1]], yr=[et[0],et[et.length-1]];
  drawAxes(ctx,p,xr,yr,"nH [cm⁻³]","temperature [K]");
  function path(key,color,dash) {
    ctx.save(); ctx.beginPath(); ctx.rect(p.l,p.t,p.r-p.l,p.b-p.t); ctx.clip();
    ctx.strokeStyle=color; ctx.lineWidth=1.8; ctx.setLineDash(dash);
    ctx.beginPath(); let started=false;
    for(const r of rows) {
      const yv=key==="t" ? r.t : (Number.isFinite(r.td)?Math.log10(r.td):NaN);
      if(!Number.isFinite(yv)) {started=false; continue;}
      const x=xmap(r.n,xr[0],xr[1],p), y=ymap(yv,yr[0],yr[1],p);
      if(!started){ctx.moveTo(x,y);started=true;}else ctx.lineTo(x,y);
    }
    ctx.stroke(); ctx.restore();
  }
  path("t",token("--ink"),[]);
  path("td","#818791",[5,4]);
  let nearestRow=rows[0], nearestDistance=Infinity;
  for(const row of rows) {
    const distance=(row.n-logn[state.i])**2+(row.t-logt[state.j])**2;
    if(distance<nearestDistance){nearestDistance=distance;nearestRow=row;}
  }
  ctx.fillStyle=token("--accent"); ctx.strokeStyle=token("--panel");
  ctx.lineWidth=1.2; ctx.beginPath();
  ctx.arc(xmap(nearestRow.n,xr[0],xr[1],p),
          ymap(nearestRow.t,yr[0],yr[1],p),4,0,2*Math.PI);
  ctx.fill();ctx.stroke();
  const pv=DATA.tracks.available && track>=0
    ? DATA.tracks.pressure[track] : pressureAt(selected);
  let text=source;
  if(track>=0) text+=" "+track;
  else if(pressure>=0) text+=" "+pressure;
  if(Number.isFinite(pv)) text+=" · P/kB = "+fmtNumber(pv)+" K cm⁻³";
  const stop=metadataAt(DATA.trajectory.trajectory_stop_code,
                        track>=0 ? track : pressure,state.k);
  const stopTime=metadataAt(DATA.trajectory.trajectory_stop_time_myr,
                            track>=0 ? track : pressure,state.k);
  const stopTemperature=metadataAt(
    DATA.trajectory.trajectory_stop_temperature,
    track>=0 ? track : pressure,state.k);
  const steps=metadataAt(DATA.trajectory.trajectory_steps,
                         track>=0 ? track : pressure,state.k);
  const retries=metadataAt(DATA.trajectory.trajectory_solver_retries,
                           track>=0 ? track : pressure,state.k);
  const safety=metadataAt(DATA.trajectory.trajectory_safety_factor,
                          track>=0 ? track : pressure,state.k);
  if(stop!==null) text+=" · stop: "+stopLabel(stop);
  if(Number.isFinite(stopTemperature))
    text+=" · T="+fmtNumber(stopTemperature)+" K";
  if(Number.isFinite(stopTime)) text+=" · "+fmtNumber(stopTime)+" Myr";
  if(Number.isFinite(steps)) text+=" · "+steps+" steps";
  if(Number.isFinite(retries) && retries>0) text+=" · "+retries+" retries";
  if(Number.isFinite(safety)) text+=" · safety="+fmtNumber(safety);
  title.textContent=text;
}

function syncControls() {
  const controls=[["si","oi",state.i,NX,DATA.nH[state.i]," cm⁻³"],
                  ["sj","oj",state.j,NY,DATA.T[state.j]," K"],
                  ["sk","ok",state.k,NZ,DATA.Z[state.k]," Z⊙"]];
  for(const [sid,oid,value,n,val,unit] of controls) {
    const s=document.getElementById(sid); s.min=0;s.max=n-1;s.value=value;
    document.getElementById(oid).textContent=fmtNumber(val)+unit;
  }
  document.getElementById("title-nt").textContent=
    "at Z="+fmtNumber(DATA.Z[state.k])+" Z⊙";
  document.getElementById("title-nz").textContent=
    "at T="+fmtNumber(DATA.T[state.j])+" K";
  document.getElementById("title-tz").textContent=
    "at nH="+fmtNumber(DATA.nH[state.i])+" cm⁻³";
}
for(const [id,key] of [["si","i"],["sj","j"],["sk","k"]]) {
  document.getElementById(id).addEventListener("input",event=>{
    state[key]=+event.target.value; renderAll();
  });
}
document.getElementById("quantity").addEventListener("change",event=>{
  state.quantity=event.target.value; renderVisuals();
});

function renderVisuals() {
  drawSlice("nt");drawSlice("nz");drawSlice("tz");buildLegend();
}
function renderAll() {
  syncControls();renderVisuals();showInspector();drawTrack();
}
document.getElementById("method").innerHTML=
  '<b>cube:</b> history-dependent nH Tgas = constant cooling samples<br>'+
  '<b>sampling:</b> '+esc(DATA.samplingMode)+'<br>'+
  '<b>pressure law:</b> '+esc(DATA.pressureLaw)+'<br>'+
  '<b>initial state:</b> '+esc(DATA.initialCondition)+'<br>'+
  '<b>chemistry:</b> '+esc(DATA.chemistryConfiguration)+'<br>'+
  '<b>UVB table:</b> '+esc(DATA.uvb || "not recorded")+'<br>'+
  '<b>grid:</b> '+NX+' × '+NY+' × '+NZ+' (nH × T × Z)<br>'+
  '<b>budget threshold:</b> '+(DATA.threshold*100).toFixed(3)+'%<br>'+
  (DATA.physicalRates
    ? '<b>rates:</b> '+esc(DATA.rateNormalization)+
      '; positive = cooling'
    : '<span class="legacy"><b>legacy rates:</b> raw stored solver values; '+
      'absolute physical units were not recorded</span>')+
  '<br>Diagonal hatching marks table extrapolation. Invalid samples are never '+
  'silently interpreted as physical cooling states. Sampled status follows '+
  '<b>valid_mask</b>; <b>converged_mask</b> is intentionally ignored for '+
  'history states.<br><b>Reference track:</b> the '+
  (DATA.tracks.available ? 'native isobaric history is shown as a 1-D path '+
    'through the same history-cube coordinates.'
    : 'solid line is an analytic nH Tgas = constant guide when no native '+
      'history-dependent trajectory is stored; dashed dust T is sampled from '+
      'the nearest cube states.');

let resizeTimer=0;
window.addEventListener("resize",()=>{
  clearTimeout(resizeTimer); resizeTimer=setTimeout(renderVisuals,80);
});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change",renderAll);
renderAll();
</script>
'''


def main(argv=None):
    args = parse_args(argv)
    payload = build_payload(args.npz)
    nx, ny, nz = payload["dims"]
    subtitle = (
        f"{payload['samplingMode']} · history-dependent "
        f"nH Tgas = constant sampling · UV background: "
        f"{payload['uvb'] or 'not recorded'} · grid {nx} × {ny} × {nz} "
        f"(nH × T × Z) · click a slice or use arrow keys to inspect a state"
    )
    html = (TEMPLATE
            .replace("@@SUBTITLE@@", subtitle)
            .replace("@@DATA@@", json.dumps(
                payload, separators=(",", ":"), ensure_ascii=False)))

    output = args.output or os.path.splitext(args.npz)[0] + ".html"
    with open(output, "w", encoding="utf-8") as handle:
        handle.write("<!doctype html>\n<html lang=\"en\">\n<head>\n")
        handle.write("<meta charset=\"utf-8\">\n")
        handle.write("<meta name=\"viewport\" "
                     "content=\"width=device-width, initial-scale=1\">\n")
        handle.write("</head>\n<body>\n")
        handle.write(html)
        handle.write("\n</body>\n</html>\n")
    print(f"Saved {output}")

    if args.body_only:
        with open(args.body_only, "w", encoding="utf-8") as handle:
            handle.write(html)
        print(f"Saved {args.body_only}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
