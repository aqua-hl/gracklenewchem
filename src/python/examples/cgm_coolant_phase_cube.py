########################################################################
#
# CGM cooling-contribution phase cube with isobaric cooling tracks.
#
# The cube is assembled from history-dependent, non-equilibrium cooling
# parcels.  One hot, ionized parcel is evolved for every unique n_H*T diagonal
# in the grid.  The parcels follow the simplified pressure law requested for
# this experiment, n_H T = constant, by compressing every density field after
# each native Grackle cooling step.  Cells below a UV-heated equilibrium or
# beyond a failed/limited trajectory remain explicitly unsampled.
#
# Cooling contributions are taken from the experimental
# calculate_cooling_rate_contributions hook.  The hook returns raw code edot
# values; this script converts them to effective coefficients,
#
#       Lambda_i = -edot_i * cooling_units / n_H**2,
#
# so positive values are cooling and negative values are heating in
# erg cm^3 s^-1.  The full primitive budget includes both the normal cooling
# channels and the individual chemistry-reaction energy terms used by
# dust_model0_cooling_contributions.py.
#
########################################################################

import argparse
import contextlib
import csv
import ctypes
import hashlib
import json
import multiprocessing
import os
import sys
import time
from dataclasses import asdict, dataclass

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dust_model0_cooling_contributions as base

import gracklepy
from gracklepy import FluidContainer, setup_fluid_container
from gracklepy import grackle_wrapper
from gracklepy.utilities.physical_constants import mass_hydrogen_cgs

_MODEL_NAME = os.path.basename(__file__[:-3])
_FORMAT_VERSION = 11
_SAMPLING_MODE = "isobaric_history_cube"
_LIBC = ctypes.CDLL(None)
_INVALID_COLOR = "#ece9e2"
_EXTRAPOLATED_COLOR = "#bd35bd"
_STOP_LABELS = np.asarray([
    "target_reached",
    "net_heating",
    "maximum_steps",
    "maximum_time",
    "solver_error",
    "invalid_state",
    "negligible_temperature_change",
    "not_run",
])


@dataclass(frozen=True)
class WorkerConfig:
    uvb_data_file: str
    dust_species: int
    nH: tuple
    temperature: tuple
    metallicity: float
    pressure: tuple
    pressure_index: tuple
    safety_factor: float
    max_timestep_growth: float
    max_track_steps: int
    max_track_time_myr: float
    stall_steps: int
    stall_tolerance: float
    show_solver_output: bool
    track_pressures: tuple


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description=(
            "Build a CGM cooling-contribution phase cube from a family of "
            "history-dependent n_H*T=constant cooling tracks."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--uvb-data-file", type=os.path.abspath, required=True,
        help="UV-background HDF5 table, e.g. HM2012_shielded_metal_test.h5.",
    )
    parser.add_argument(
        "--nH-range", type=float, nargs=2, default=(1.0, 1.0e8),
        metavar=("MIN", "MAX"),
        help="hydrogen-nuclei number-density range in cm^-3.",
    )
    parser.add_argument(
        "--T-range", type=float, nargs=2, default=(10.0, 1.0e6),
        metavar=("MIN", "MAX"), help="gas-temperature range in K.",
    )
    parser.add_argument(
        "--Z-values", type=float, nargs="*",
        default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0],
        help="metallicities in solar units.",
    )
    parser.add_argument(
        "--dust-species", type=int, choices=(0, 1, 2, 3), default=0,
        help=(
            "dust_species selector for dust_model=0. The stable bulk-dust "
            "phase-cube default is 0; use 3 (usually with "
            "--track-safety-factor 0.015) to mirror the reference "
            "freefall script's grain-species setup."
        ),
    )
    parser.add_argument(
        "--points-per-decade", type=int, default=4,
        help="logarithmic grid resolution in n_H and T.",
    )
    parser.add_argument(
        "--budget-threshold", type=float, default=0.999,
        help="same-sign cooling/heating budget fraction to cover.",
    )
    parser.add_argument(
        "--track-pressures", type=float, nargs="*", default=None,
        metavar="P_OVER_K",
        help=(
            "n_H*T values in K cm^-3 for non-equilibrium isobaric tracks. "
            "The default is one track through the log-centre of the cube; "
            "pass no values after the option to disable tracks."
        ),
    )
    parser.add_argument(
        "--track-safety-factor", type=float, default=0.03,
        help="fraction of the native cooling time used per track step.",
    )
    parser.add_argument(
        "--max-timestep-growth", type=float, default=2.0,
        help=(
            "maximum growth of consecutive accepted timesteps; this avoids "
            "a divergent step when the cooling time blows up near equilibrium."
        ),
    )
    parser.add_argument(
        "--max-track-steps", type=int, default=5000,
        help="hard step limit for each isobaric track.",
    )
    parser.add_argument(
        "--max-track-time-myr", type=float, default=0.0,
        help="track time limit in Myr; zero means no time limit.",
    )
    parser.add_argument(
        "--stall-steps", type=int, default=8,
        help="consecutive negligible/non-cooling steps before a track stops.",
    )
    parser.add_argument(
        "--stall-tolerance", type=float, default=1.0e-3,
        help="relative temperature decrease considered negligible.",
    )
    parser.add_argument(
        "--show-solver-output", action="store_true",
        help=(
            "show native solver diagnostics; by default recovered subcycle "
            "errors are summarized without printing the C-level trace."
        ),
    )
    parser.add_argument(
        "--jobs", type=int, default=min(4, os.cpu_count() or 1),
        help="supervised worker processes, capped at the number of Z slices.",
    )
    parser.add_argument(
        "--worker-timeout", type=float, default=1800.0, metavar="SECONDS",
        help=(
            "wall-clock limit for one metallicity worker; zero disables the "
            "limit (default: 1800 seconds)."
        ),
    )
    parser.add_argument(
        "--output", default=_MODEL_NAME, help="output prefix.",
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help="per-metallicity checkpoint directory (default: <output>_cache).",
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help="reuse matching per-metallicity checkpoints.",
    )
    parser.add_argument(
        "--html", action=argparse.BooleanOptionalAction, default=True,
        help="write the self-contained interactive HTML viewer.",
    )
    parser.add_argument(
        "--plots", nargs="*", choices=("slices", "balance", "tracks", "cube3d"),
        default=["slices", "balance", "tracks", "cube3d"],
        help="static figures to write.",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="use 2 points/decade and 3 metallicities.",
    )
    return parser.parse_args(args)


def validate_args(args):
    if not os.path.isfile(args.uvb_data_file):
        raise FileNotFoundError(args.uvb_data_file)
    for name, bounds in (("nH", args.nH_range), ("T", args.T_range)):
        if (
            not np.isfinite(bounds).all() or
            bounds[0] <= 0 or bounds[1] <= bounds[0]
        ):
            raise ValueError(f"--{name}-range must contain positive MIN < MAX")
    metallicities = np.asarray(args.Z_values, dtype=float)
    if (
        metallicities.size == 0 or not np.isfinite(metallicities).all() or
        np.any(metallicities <= 0)
    ):
        raise ValueError("--Z-values must contain positive values")
    if np.any(np.diff(metallicities) <= 0):
        raise ValueError(
            "--Z-values must be strictly increasing without duplicates"
        )
    if args.points_per_decade < 1:
        raise ValueError("--points-per-decade must be positive")
    if not 0 < args.budget_threshold <= 1:
        raise ValueError("--budget-threshold must be in (0, 1]")
    if args.max_track_steps < 1:
        raise ValueError("track step count must be positive")
    if not 0 < args.track_safety_factor < 1:
        raise ValueError("--track-safety-factor must be in (0, 1)")
    if (
        not np.isfinite(args.max_timestep_growth) or
        args.max_timestep_growth <= 1
    ):
        raise ValueError("--max-timestep-growth must be greater than 1")
    if (
        not np.isfinite(args.max_track_time_myr) or
        args.max_track_time_myr < 0
    ):
        raise ValueError(
            "--max-track-time-myr must be finite and non-negative"
        )
    if args.stall_steps < 1:
        raise ValueError("--stall-steps must be positive")
    if not 0 <= args.stall_tolerance < 1:
        raise ValueError("--stall-tolerance must be in [0, 1)")
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    if not np.isfinite(args.worker_timeout) or args.worker_timeout < 0:
        raise ValueError("--worker-timeout must be finite and non-negative")


def log_grid(lo, hi, points_per_decade):
    n_points = max(
        2, int(round(np.log10(hi / lo) * points_per_decade)) + 1
    )
    return np.logspace(np.log10(lo), np.log10(hi), n_points)


def channel_fields():
    return [field for field, _ in base.combined_stack_order()]


def channel_labels():
    return dict(base.combined_stack_order())


def channel_colors(n_channels):
    maps = (plt.cm.tab20, plt.cm.tab20b, plt.cm.tab20c)
    rgba = np.vstack([cmap(np.linspace(0, 1, 20)) for cmap in maps])
    if n_channels > rgba.shape[0]:
        rgba = plt.cm.hsv(np.linspace(0, 1, n_channels, endpoint=False))
    return rgba[:n_channels]


def short_channel_label(field, fallback):
    """Use compact reaction labels where the reference script provides them."""
    return base._CHEMISTRY_TEXT_LABELS.get(str(field), str(fallback))


def raw_name_for_field(field):
    name = field.removeprefix("cooling_")
    for raw_name, alias in base._CONTRIBUTION_KEY_OVERRIDES.items():
        if alias == name:
            return raw_name
    return name


def make_opts(uvb_data_file, dust_species=0):
    return base.parse_args([
        "--uvb-data-file", uvb_data_file,
        "--isrf", "0",
        "--photoelectric-heating", "0",
        "--dust-species", str(dust_species),
    ])


def make_chemistry(uvb_data_file, dust_species=0):
    # Raising on an exhausted internal subcycle lets Python reject the entire
    # pressure-line attempt rather than accepting a partial solver state.
    return base.make_chemistry(
        make_opts(uvb_data_file, dust_species),
        overrides={"exit_after_iterations_exceeded": 1},
    )


def read_table_bounds(path):
    """Return tabulated (n_H, T) bounds from the Cloudy HDF5 metadata."""
    with h5py.File(path, "r") as handle:
        candidates = (
            "/CoolingRates/Metals/Cooling",
            "/CoolingRates/Primordial/Cooling",
        )
        for candidate in candidates:
            if candidate not in handle:
                continue
            dataset = handle[candidate]
            p1_name = dataset.attrs.get("Parameter1_Name", "")
            if isinstance(p1_name, bytes):
                p1_name = p1_name.decode()
            if str(p1_name).lower() != "hden":
                continue
            log_nH = np.asarray(dataset.attrs["Parameter1"], dtype=float)
            temperature = np.asarray(dataset.attrs["Temperature"], dtype=float)
            return (
                float(10.0 ** log_nH.min()),
                float(10.0 ** log_nH.max()),
                float(temperature.min()),
                float(temperature.max()),
            )
    return (np.nan, np.nan, np.nan, np.nan)


def pressure_index_grid(nH, temperature):
    """Stable integer map from every (n_H,T) cell to a unique log(n_H*T)."""
    log_pressure = (
        np.log10(np.asarray(nH))[:, None] +
        np.log10(np.asarray(temperature))[None, :]
    )
    quantized = np.round(log_pressure, decimals=12)
    unique = np.unique(quantized)
    indices = np.searchsorted(unique, quantized)
    return 10.0 ** unique, indices.astype(np.int32)


def _state_is_valid(fc):
    for field in fc.input_fields:
        values = np.asarray(fc[field])
        if not np.isfinite(values).all():
            return False
        if field in fc.density_fields and np.any(values < 0):
            return False
    return True


def _restore_inputs(fc, saved):
    for field, values in saved.items():
        fc[field][:] = values


@contextlib.contextmanager
def _native_output(show):
    """Optionally silence C-level stdout/stderr for one native solver call."""
    if show:
        yield
        return
    sys.stdout.flush()
    sys.stderr.flush()
    _LIBC.fflush(None)
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    null_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        _LIBC.fflush(None)
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(null_fd)
        os.close(saved_stdout)
        os.close(saved_stderr)


def _template_for_state(chem, metallicity, temperature, state):
    metal_fraction = metallicity * chem.SolarMetalFractionByMass
    dust_ratio = metallicity * chem.local_dust_to_gas_ratio
    fc = setup_fluid_container(
        chem,
        density=mass_hydrogen_cgs,
        temperature=np.asarray(temperature, dtype=float),
        state=state,
        metal_mass_fraction=metal_fraction,
        dust_to_gas_ratio=dust_ratio,
        converge=False,
    )
    base.seed_model0_grain_species(fc)
    fc.calculate_nuclei_density("H")
    actual_nH = np.asarray(fc["H_nuclei_density"], dtype=float)
    density_per_nH = {
        field: np.asarray(fc[field], dtype=float) / actual_nH
        for field in fc.density_fields
    }
    extras = {
        field: np.asarray(fc[field], dtype=float).copy()
        for field in fc.input_fields
        if field not in fc.density_fields
    }
    return density_per_nH, extras


def _extract_contributions(fc):
    """Return physical n_H-normalized coefficients and dust temperature."""
    fc.calculate_nuclei_density("H")
    nH_actual = np.asarray(fc["H_nuclei_density"], dtype=float)
    conversion = fc.chemistry_data.cooling_units / np.maximum(
        nH_actual, 1.0e-300
    ) ** 2
    raw = fc.calculate_cooling_rate_contributions(0.0)
    fields = channel_fields()
    rates = np.full((fc.n_vals, len(fields)), np.nan)
    for index, field in enumerate(fields):
        raw_name = raw_name_for_field(field)
        if raw_name in raw:
            rates[:, index] = -np.asarray(raw[raw_name], dtype=float) * conversion
    total = -np.asarray(raw["total"], dtype=float) * conversion
    residual = -np.asarray(raw["residual"], dtype=float) * conversion
    fc.calculate_dust_temperature()
    dust_temperature = np.asarray(fc["dust_temperature"], dtype=float).copy()
    return rates, total, residual, dust_temperature, nH_actual


def _capture_track_state(fc, pressure, current_temperature):
    nH = pressure / current_temperature
    return {
        field: float(fc[field][0]) / nH
        for field in fc.density_fields
    }


def _interpolate_track_state(
        previous, current, previous_temperature, current_temperature,
        target_temperature, target_nH):
    log_previous = np.log(previous_temperature)
    log_current = np.log(current_temperature)
    if np.isclose(log_previous, log_current):
        weight = 1.0
    else:
        weight = (
            (log_previous - np.log(target_temperature)) /
            (log_previous - log_current)
        )
    weight = float(np.clip(weight, 0.0, 1.0))
    return {
        field: (
            (1.0 - weight) * previous[field] + weight * current[field]
        ) * target_nH
        for field in previous
    }, weight


def _diagnose_track_samples(chem, samples):
    if not samples:
        return None
    fc = FluidContainer(chem, len(samples))
    density_fields = set(fc.density_fields)
    for field in fc.input_fields:
        if field in density_fields:
            fc[field][:] = [sample["densities"][field] for sample in samples]
        elif field == "internal_energy":
            continue
        else:
            fc[field][:] = 0.0
    # Interpolation and the experimental metal network can introduce tiny
    # elemental-normalization drift.  Project every density field together so
    # the diagnostic state lies exactly on the requested n_H coordinate.
    fc.calculate_nuclei_density("H")
    target_nH = np.asarray([sample["nH"] for sample in samples])
    density_ratio = target_nH / np.asarray(fc["H_nuclei_density"])
    for field in fc.density_fields:
        fc[field] *= density_ratio
    fc.calculate_mean_molecular_weight()
    target_temperature = np.asarray(
        [sample["temperature"] for sample in samples]
    )
    fc["internal_energy"][:] = (
        target_temperature / chem.temperature_units /
        fc["mean_molecular_weight"] / (chem.Gamma - 1.0)
    )
    return _extract_contributions(fc)


def _run_track(
        chem, density_per_nH, extras, template_index, temperature_grid,
        pressure, safety_factor, max_timestep_growth, max_steps, max_time_myr,
        stall_steps, stall_tolerance, target_indices=None):
    """Evolve one scalar parcel with the requested n_H*T pressure proxy."""
    temperature_grid = np.asarray(temperature_grid, dtype=float)
    if target_indices is None:
        target_indices = np.arange(temperature_grid.size)
    else:
        target_indices = np.asarray(target_indices, dtype=int)
    target_order = target_indices[
        np.argsort(temperature_grid[target_indices])[::-1]
    ]
    targets = np.asarray(temperature_grid)[target_order]
    initial_temperature = float(np.max(temperature_grid))
    initial_nH = pressure / initial_temperature

    fc = FluidContainer(chem, 1)
    for field in fc.input_fields:
        if field in fc.density_fields:
            fc[field][0] = density_per_nH[field][template_index] * initial_nH
        else:
            fc[field][0] = extras[field][template_index]

    samples = []
    next_target = 0
    if targets.size and np.isclose(targets[0], initial_temperature):
        samples.append({
            "target_index": int(target_order[0]),
            "temperature": initial_temperature,
            "nH": initial_nH,
            "time_myr": 0.0,
            "densities": {
                field: float(fc[field][0]) for field in fc.density_fields
            },
        })
        next_target = 1
    current_time = 0.0
    stall_count = 0
    stop_code = 2
    accepted_steps = 0
    solver_retries = 0
    previous_dt = None

    old_temperature = initial_temperature
    last_temperature = initial_temperature
    for _ in range(max_steps):
        fc.calculate_temperature()
        old_temperature = float(fc["temperature"][0])
        if next_target >= targets.size:
            stop_code = 0
            break
        fc.calculate_cooling_time()
        cooling_time = float(fc["cooling_time"][0])
        if not np.isfinite(cooling_time):
            stop_code = 5
            break
        # Grackle's cooling-rate convention is negative for energy loss, so
        # its energy/edot cooling time is negative while net heating is
        # positive.
        if cooling_time >= 0:
            stop_code = 1
            break
        dt = -safety_factor * cooling_time
        if previous_dt is not None:
            dt = min(dt, max_timestep_growth * previous_dt)
        if max_time_myr > 0:
            remaining = max_time_myr - current_time
            if remaining <= 0:
                stop_code = 3
                break
            dt = min(dt, remaining)

        previous_fractions = _capture_track_state(
            fc, pressure, old_temperature
        )
        saved = {field: fc[field].copy() for field in fc.input_fields}
        trial_dt = dt
        try:
            fc.solve_chemistry(trial_dt)
            fc.calculate_temperature()
        except RuntimeError:
            _restore_inputs(fc, saved)
            solver_retries += 1
            stop_code = 4
            break
        new_temperature = float(fc["temperature"][0])
        if (
            not _state_is_valid(fc) or
            not np.isfinite(new_temperature) or new_temperature <= 0
        ):
            _restore_inputs(fc, saved)
            stop_code = 5
            break

        # The intentionally simplified pressure projection requested for this
        # experiment.  It omits mu, gamma, and compressional PdV work.
        density_ratio = old_temperature / new_temperature
        for field in fc.density_fields:
            fc[field] *= density_ratio
        current_time += trial_dt
        accepted_steps += 1
        previous_dt = trial_dt
        last_temperature = new_temperature

        current_fractions = _capture_track_state(
            fc, pressure, new_temperature
        )
        while (
            next_target < targets.size and
            new_temperature <= targets[next_target] <= old_temperature
        ):
            target_temperature = float(targets[next_target])
            target_nH = pressure / target_temperature
            densities, weight = _interpolate_track_state(
                previous_fractions, current_fractions,
                old_temperature, new_temperature,
                target_temperature, target_nH,
            )
            samples.append({
                "target_index": int(target_order[next_target]),
                "temperature": target_temperature,
                "nH": target_nH,
                "time_myr": (
                    current_time - trial_dt + weight * trial_dt
                ),
                "densities": densities,
            })
            next_target += 1

        if next_target >= targets.size:
            stop_code = 0
            break

        relative_drop = (old_temperature - new_temperature) / old_temperature
        if relative_drop <= stall_tolerance:
            stall_count += 1
        else:
            stall_count = 0
        if stall_count >= stall_steps:
            stop_code = 6
            break
        if max_time_myr > 0 and current_time >= max_time_myr:
            stop_code = 3
            break

    # Do not call back into a chemistry object after its native subcycle
    # raised.  The caller rebuilds the object and reruns this pressure line
    # from the hot initial state with a smaller safety factor.
    diagnosed = None if stop_code == 4 else _diagnose_track_samples(
        chem, samples
    )
    nT = len(temperature_grid)
    n_channels = len(channel_fields())
    result = {
        "nH": np.full(nT, np.nan),
        "actual_nH": np.full(nT, np.nan),
        "gas_temperature": np.full(nT, np.nan),
        "dust_temperature": np.full(nT, np.nan),
        "time_myr": np.full(nT, np.nan),
        "valid": np.zeros(nT, dtype=bool),
        "rates": np.full((nT, n_channels), np.nan),
        "total": np.full(nT, np.nan),
        "residual": np.full(nT, np.nan),
        "stop_code": np.int16(stop_code),
        "stop_temperature": np.float64(last_temperature),
        "stop_time_myr": np.float64(current_time),
        "steps": np.int32(accepted_steps),
        "solver_retries": np.int32(solver_retries),
        "safety_factor": np.float64(safety_factor),
    }
    if diagnosed is None:
        return result
    rates, total, residual, dust_temperature, actual_nH = diagnosed
    for sample_index, sample in enumerate(samples):
        if not (
            np.isfinite(rates[sample_index]).all() and
            np.isfinite(total[sample_index]) and
            np.isfinite(residual[sample_index]) and
            np.isfinite(dust_temperature[sample_index]) and
            np.isfinite(actual_nH[sample_index]) and
            actual_nH[sample_index] > 0
        ):
            continue
        target_index = sample["target_index"]
        result["nH"][target_index] = sample["nH"]
        result["actual_nH"][target_index] = actual_nH[sample_index]
        result["gas_temperature"][target_index] = sample["temperature"]
        result["dust_temperature"][target_index] = dust_temperature[sample_index]
        result["time_myr"][target_index] = sample["time_myr"]
        result["valid"][target_index] = True
        result["rates"][target_index] = rates[sample_index]
        result["total"][target_index] = total[sample_index]
        result["residual"][target_index] = residual[sample_index]
    return result


def _metallicity_worker(config_dict):
    config = WorkerConfig(**config_dict)
    nH = np.asarray(config.nH, dtype=float)
    temperature = np.asarray(config.temperature, dtype=float)
    pressure = np.asarray(config.pressure, dtype=float)
    pressure_index = np.asarray(config.pressure_index, dtype=np.int32)
    n_channels = len(channel_fields())
    def fresh_chemistry():
        new_chem = make_chemistry(
            config.uvb_data_file, config.dust_species
        )
        new_template = _template_for_state(
            new_chem, config.metallicity, temperature, "ionized"
        )
        return new_chem, new_template

    chem, ionized = fresh_chemistry()

    def run_pressure_line(track_pressure, target_indices=None):
        nonlocal chem, ionized
        total_retries = 0
        safety_factor = config.safety_factor
        one = None
        for _attempt in range(3):
            with _native_output(config.show_solver_output):
                one = _run_track(
                    chem, ionized[0], ionized[1], hot_index, temperature,
                    float(track_pressure), safety_factor,
                    config.max_timestep_growth,
                    config.max_track_steps, config.max_track_time_myr,
                    config.stall_steps, config.stall_tolerance,
                    target_indices=target_indices,
                )
            total_retries += int(one["solver_retries"])
            if int(one["stop_code"]) != 4:
                break
            chem, ionized = fresh_chemistry()
            safety_factor *= 0.5
        one["solver_retries"] = np.int32(total_retries)
        return one

    shape = (nH.size, temperature.size)
    rates = np.full(shape + (n_channels,), np.nan)
    total = np.full(shape, np.nan)
    residual = np.full(shape, np.nan)
    dust_temperature = np.full(shape, np.nan)
    actual_nH = np.full(shape, np.nan)
    sample_time = np.full(shape, np.nan)
    valid = np.zeros(shape, dtype=bool)
    pressure_stop_code = np.full(pressure.size, 7, dtype=np.int16)
    pressure_stop_temperature = np.full(pressure.size, np.nan)
    pressure_stop_time_myr = np.full(pressure.size, np.nan)
    pressure_steps = np.zeros(pressure.size, dtype=np.int32)
    pressure_solver_retries = np.zeros(pressure.size, dtype=np.int32)
    pressure_safety_factor = np.full(pressure.size, np.nan)

    hot_index = int(np.argmax(temperature))
    for track_index, track_pressure in enumerate(pressure):
        cells = np.argwhere(pressure_index == track_index)
        target_indices = np.unique(cells[:, 1])
        one = run_pressure_line(
            track_pressure,
            target_indices=target_indices,
        )
        pressure_stop_code[track_index] = one["stop_code"]
        pressure_stop_temperature[track_index] = one["stop_temperature"]
        pressure_stop_time_myr[track_index] = one["stop_time_myr"]
        pressure_steps[track_index] = one["steps"]
        pressure_solver_retries[track_index] = one["solver_retries"]
        pressure_safety_factor[track_index] = one["safety_factor"]
        for density_index, temperature_index in cells:
            if not one["valid"][temperature_index]:
                continue
            rates[density_index, temperature_index] = (
                one["rates"][temperature_index]
            )
            total[density_index, temperature_index] = (
                one["total"][temperature_index]
            )
            residual[density_index, temperature_index] = (
                one["residual"][temperature_index]
            )
            dust_temperature[density_index, temperature_index] = (
                one["dust_temperature"][temperature_index]
            )
            actual_nH[density_index, temperature_index] = (
                one["actual_nH"][temperature_index]
            )
            sample_time[density_index, temperature_index] = (
                one["time_myr"][temperature_index]
            )
            valid[density_index, temperature_index] = True
    n_tracks = len(config.track_pressures)
    track_shape = (n_tracks, temperature.size)
    track = {
        "track_nH": np.full(track_shape, np.nan),
        "track_actual_nH": np.full(track_shape, np.nan),
        "track_gas_temperature": np.full(track_shape, np.nan),
        "track_dust_temperature": np.full(track_shape, np.nan),
        "track_time_myr": np.full(track_shape, np.nan),
        "track_valid": np.zeros(track_shape, dtype=bool),
        "track_cooling_rates": np.full(
            track_shape + (n_channels,), np.nan
        ),
        "track_total_rate": np.full(track_shape, np.nan),
        "track_residual_rate": np.full(track_shape, np.nan),
        "trajectory_stop_code": np.full(n_tracks, 7, dtype=np.int16),
        "trajectory_stop_temperature": np.full(n_tracks, np.nan),
        "trajectory_stop_time_myr": np.full(n_tracks, np.nan),
        "trajectory_steps": np.zeros(n_tracks, dtype=np.int32),
        "trajectory_solver_retries": np.zeros(n_tracks, dtype=np.int32),
        "trajectory_safety_factor": np.full(n_tracks, np.nan),
    }
    for track_index, reference_pressure in enumerate(config.track_pressures):
        # Cube pressure lines only sample temperatures that land on the
        # requested density grid.  Reference tracks are rerun against the
        # complete temperature axis so their curves are not silently truncated.
        one = run_pressure_line(reference_pressure)
        for key in (
            "nH", "actual_nH", "gas_temperature", "dust_temperature",
            "time_myr", "valid", "rates", "total", "residual",
        ):
            output_key = {
                "nH": "track_nH",
                "actual_nH": "track_actual_nH",
                "gas_temperature": "track_gas_temperature",
                "dust_temperature": "track_dust_temperature",
                "time_myr": "track_time_myr",
                "valid": "track_valid",
                "rates": "track_cooling_rates",
                "total": "track_total_rate",
                "residual": "track_residual_rate",
            }[key]
            track[output_key][track_index] = one[key]
        track["trajectory_stop_code"][track_index] = one["stop_code"]
        track["trajectory_stop_temperature"][track_index] = (
            one["stop_temperature"]
        )
        track["trajectory_stop_time_myr"][track_index] = one["stop_time_myr"]
        track["trajectory_steps"][track_index] = one["steps"]
        track["trajectory_solver_retries"][track_index] = (
            one["solver_retries"]
        )
        track["trajectory_safety_factor"][track_index] = one["safety_factor"]

    return {
        "cooling_rates": rates,
        "total_rate": total,
        "residual_rate": residual,
        "dust_temperature": dust_temperature,
        "actual_nH": actual_nH,
        "sample_time_myr": sample_time,
        "valid_mask": valid,
        "pressure_stop_code": pressure_stop_code,
        "pressure_stop_temperature": pressure_stop_temperature,
        "pressure_stop_time_myr": pressure_stop_time_myr,
        "pressure_steps": pressure_steps,
        "pressure_solver_retries": pressure_solver_retries,
        "pressure_safety_factor": pressure_safety_factor,
        **track,
    }


def _metallicity_process_entry(config, cache_path):
    """Run one Z slice in a supervised child and atomically checkpoint it."""
    _save_cache(cache_path, _metallicity_worker(config))


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_fingerprint(args, nH, temperature, Z, track_pressures):
    payload = {
        "format": _FORMAT_VERSION,
        "sampling_mode": _SAMPLING_MODE,
        "gracklepy": gracklepy.__version__,
        "gracklepy_native_sha256": _sha256(grackle_wrapper.__file__),
        "chemistry_source_sha256": _sha256(base.__file__),
        "uvb_sha256": _sha256(args.uvb_data_file),
        "dust_species": args.dust_species,
        "nH": np.asarray(nH).tolist(),
        "temperature": np.asarray(temperature).tolist(),
        "Z": np.asarray(Z).tolist(),
        "track_pressures": np.asarray(track_pressures).tolist(),
        "track_safety_factor": args.track_safety_factor,
        "max_timestep_growth": args.max_timestep_growth,
        "max_track_steps": args.max_track_steps,
        "max_track_time_myr": args.max_track_time_myr,
        "stall_steps": args.stall_steps,
        "stall_tolerance": args.stall_tolerance,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]


def _cache_file(cache_dir, fingerprint, metallicity_index):
    return os.path.join(
        cache_dir, f"{fingerprint}_Z{metallicity_index:03d}.npz"
    )


def _save_cache(path, result):
    temporary = f"{path}.tmp.npz"
    np.savez_compressed(temporary, **result)
    os.replace(temporary, path)


def _load_cache(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def build_atlas(args):
    nH = log_grid(*args.nH_range, args.points_per_decade)
    temperature = log_grid(*args.T_range, args.points_per_decade)
    Z = np.asarray(args.Z_values, dtype=float)
    pressures, pressure_index = pressure_index_grid(nH, temperature)
    if args.track_pressures is None:
        centre = np.sqrt(
            args.nH_range[0] * args.T_range[0] *
            args.nH_range[1] * args.T_range[1]
        )
        track_pressures = np.asarray([pressures[np.argmin(abs(
            np.log10(pressures / centre)
        ))]])
    else:
        track_pressures = np.asarray(args.track_pressures, dtype=float)
        if (
            not np.isfinite(track_pressures).all() or
            np.any(track_pressures <= 0)
        ):
            raise ValueError("--track-pressures must be finite and positive")

    cache_dir = args.cache_dir or f"{args.output}_cache"
    os.makedirs(cache_dir, exist_ok=True)
    fingerprint = cache_fingerprint(
        args, nH, temperature, Z, track_pressures
    )
    results = [None] * Z.size
    pending = []
    for metallicity_index, metallicity in enumerate(Z):
        cache_path = _cache_file(cache_dir, fingerprint, metallicity_index)
        if args.resume and os.path.exists(cache_path):
            results[metallicity_index] = _load_cache(cache_path)
            print(
                f"  Z={metallicity:g}: loaded checkpoint "
                f"{os.path.basename(cache_path)}",
                flush=True,
            )
            continue
        config = WorkerConfig(
            uvb_data_file=args.uvb_data_file,
            dust_species=args.dust_species,
            nH=tuple(float(value) for value in nH),
            temperature=tuple(float(value) for value in temperature),
            metallicity=float(metallicity),
            pressure=tuple(float(value) for value in pressures),
            pressure_index=tuple(
                tuple(int(value) for value in row) for row in pressure_index
            ),
            safety_factor=args.track_safety_factor,
            max_timestep_growth=args.max_timestep_growth,
            max_track_steps=args.max_track_steps,
            max_track_time_myr=args.max_track_time_myr,
            stall_steps=args.stall_steps,
            stall_tolerance=args.stall_tolerance,
            show_solver_output=args.show_solver_output,
            track_pressures=tuple(float(value) for value in track_pressures),
        )
        pending.append((metallicity_index, cache_path, asdict(config)))

    start = time.perf_counter()
    if pending:
        # OpenMP is disabled in the supported Python build.  One supervised
        # process per metallicity gives useful parallelism without the old
        # thousands of interpreter startups.  Explicit supervision matters:
        # a native solver segfault now produces a finite error instead of a
        # multiprocessing pool silently replacing the worker and hanging.
        context = multiprocessing.get_context("spawn")
        waiting = list(pending)
        active = []
        worker_limit = min(args.jobs, len(pending))
        try:
            while waiting or active:
                while waiting and len(active) < worker_limit:
                    metallicity_index, cache_path, config = waiting.pop(0)
                    process = context.Process(
                        target=_metallicity_process_entry,
                        args=(config, cache_path),
                    )
                    process.start()
                    active.append(
                        (
                            process, metallicity_index, cache_path,
                            time.perf_counter(),
                        )
                    )

                completed = False
                for item in list(active):
                    (
                        process, metallicity_index, cache_path, worker_start,
                    ) = item
                    process.join(timeout=0)
                    if process.exitcode is None:
                        if (
                            args.worker_timeout > 0 and
                            time.perf_counter() - worker_start >
                            args.worker_timeout
                        ):
                            raise TimeoutError(
                                "metallicity worker exceeded "
                                f"--worker-timeout={args.worker_timeout:g} s "
                                f"for Z={Z[metallicity_index]:g}; its "
                                "checkpoint was not accepted"
                            )
                        continue
                    completed = True
                    active.remove(item)
                    if process.exitcode != 0:
                        exit_code = process.exitcode
                        process.close()
                        raise RuntimeError(
                            "native worker failed for "
                            f"Z={Z[metallicity_index]:g} "
                            f"(exit code {exit_code}); its checkpoint "
                            "was not accepted. Try a smaller "
                            "--track-safety-factor."
                        )
                    results[metallicity_index] = _load_cache(cache_path)
                    process.close()
                    print(
                        f"  Z={Z[metallicity_index]:g}: "
                        "computed and checkpointed",
                        flush=True,
                    )
                if active and not completed:
                    time.sleep(0.05)
        except BaseException:
            for (
                process, _metallicity_index, _cache_path, _worker_start,
            ) in active:
                if process.is_alive():
                    process.terminate()
                process.join()
                process.close()
            raise
    print(f"  atlas/track computation: {time.perf_counter() - start:.2f} s")

    stacked = {}
    for key in results[0]:
        stacked[key] = np.stack([result[key] for result in results], axis=0)
    return (
        nH, temperature, Z, pressures, pressure_index, track_pressures, stacked
    )


def budget_statistics(rates, valid_mask, threshold):
    """Cooling/heating dominance, totals, and same-sign budget complexity."""
    rates = np.asarray(rates)
    valid = np.asarray(valid_mask, dtype=bool)
    positive = np.where(np.isfinite(rates) & (rates > 0), rates, 0.0)
    negative = np.where(np.isfinite(rates) & (rates < 0), -rates, 0.0)

    def side(values):
        total = values.sum(axis=-1)
        fractions = np.divide(
            values, total[..., None],
            out=np.zeros_like(values), where=total[..., None] > 0,
        )
        dominant = np.argmax(fractions, axis=-1).astype(np.int16)
        share = np.max(fractions, axis=-1)
        ranked = np.sort(fractions, axis=-1)[..., ::-1]
        cumulative = np.cumsum(ranked, axis=-1)
        needed = np.minimum(
            values.shape[-1],
            1 + np.count_nonzero(cumulative < threshold, axis=-1),
        )
        present = valid & (total > 0)
        dominant = np.where(present, dominant, -1)
        share = np.where(present, share, np.nan)
        needed = np.where(present, needed, 0).astype(np.int16)
        return total, dominant, share, needed

    cooling = side(positive)
    heating = side(negative)
    denominator = cooling[0] + heating[0]
    balance = np.divide(
        cooling[0] - heating[0], denominator,
        out=np.full_like(denominator, np.nan),
        where=valid & (denominator > 0),
    )
    return {
        "cooling_budget_total": cooling[0],
        "cooling_dominant_index": cooling[1],
        "cooling_dominant_share": cooling[2],
        "cooling_n_needed": cooling[3],
        "heating_budget_total": heating[0],
        "heating_dominant_index": heating[1],
        "heating_dominant_share": heating[2],
        "heating_n_needed": heating[3],
        "thermal_balance": balance,
        # Compatibility aliases for older readers.
        "dominant_index": cooling[1],
        "dominant_share": cooling[2],
        "n_needed": cooling[3],
    }


def _dominance_rgba(dominant, share, valid, extrapolated, colors):
    rgba = np.empty(dominant.shape + (4,), dtype=float)
    rgba[...] = (*plt.matplotlib.colors.to_rgb(_INVALID_COLOR), 1.0)
    for channel in np.unique(dominant[dominant >= 0]):
        mask = valid & (dominant == channel)
        rgba[mask, :3] = colors[channel, :3]
        rgba[mask, 3] = 0.45 + 0.55 * np.nan_to_num(share[mask])
    rgba[extrapolated & valid, :3] = (
        0.68 * rgba[extrapolated & valid, :3] +
        0.32 * np.asarray(plt.matplotlib.colors.to_rgb(_EXTRAPOLATED_COLOR))
    )
    return rgba


def _overlay_track(ax, data, metallicity_index, track_index=0):
    if data["track_pressure"].size == 0:
        return
    valid = (
        data["track_valid"][metallicity_index, track_index].copy() &
        (data["track_nH"][metallicity_index, track_index] >= data["nH"][0]) &
        (data["track_nH"][metallicity_index, track_index] <= data["nH"][-1])
    )
    if not np.any(valid):
        return
    x = np.log10(data["track_nH"][metallicity_index, track_index, valid])
    gas = np.log10(
        data["track_gas_temperature"][metallicity_index, track_index, valid]
    )
    dust = np.log10(np.maximum(
        data["track_dust_temperature"][metallicity_index, track_index, valid],
        1.0e-300,
    ))
    ax.plot(x, gas, color="black", linewidth=1.25, zorder=20)
    ax.plot(x, dust, color="0.35", linestyle="--", linewidth=1.05, zorder=20)


def _overlay_tracks(ax, data, metallicity_index):
    for track_index in range(data["track_pressure"].size):
        _overlay_track(ax, data, metallicity_index, track_index)


def plot_slices(data, args):
    nH, temperature, Z = data["nH"], data["T"], data["Z"]
    dominant = data["cooling_dominant_index"]
    share = data["cooling_dominant_share"]
    valid = data["valid_mask"]
    extrapolated = data["extrapolated_mask"]
    colors = channel_colors(len(data["channel_fields"]))
    ncols = min(3, Z.size)
    nrows = int(np.ceil(Z.size / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.0 * ncols, 3.25 * nrows),
        sharex=True, sharey=True, constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()
    extent = (
        np.log10(nH[0]), np.log10(nH[-1]),
        np.log10(temperature[0]), np.log10(temperature[-1]),
    )
    present = set()
    for metallicity_index, ax in enumerate(axes):
        if metallicity_index >= Z.size:
            ax.set_visible(False)
            continue
        rgba = _dominance_rgba(
            dominant[metallicity_index], share[metallicity_index],
            valid[metallicity_index], extrapolated[metallicity_index], colors,
        )
        ax.imshow(
            np.transpose(rgba, (1, 0, 2)), origin="lower", extent=extent,
            aspect="auto", interpolation="nearest",
        )
        net_heating = (
            valid[metallicity_index] &
            (data["thermal_balance"][metallicity_index] < 0)
        )
        ax.contourf(
            np.log10(nH), np.log10(temperature), net_heating.T.astype(float),
            levels=[0.5, 1.5], colors="none", hatches=["...."],
        )
        ax.contour(
            np.log10(nH), np.log10(temperature),
            data["thermal_balance"][metallicity_index].T,
            levels=[0], colors="black", linestyles=":", linewidths=0.8,
        )
        _overlay_tracks(ax, data, metallicity_index)
        ax.set_title(rf"$Z/Z_\odot={Z[metallicity_index]:g}$")
        ax.set_xlabel(r"$\log_{10} n_H$ [cm$^{-3}$]")
        ax.set_ylabel(r"$\log_{10} T_{\rm gas}$ [K]")
        present.update(np.unique(
            dominant[metallicity_index][dominant[metallicity_index] >= 0]
        ).tolist())
    handles = [
        Patch(
            facecolor=colors[index],
            label=short_channel_label(
                data["channel_fields"][index],
                data["channel_labels"][index],
            ),
        )
        for index in sorted(present)
    ]
    handles += [
        Patch(facecolor=_INVALID_COLOR, label="unreached history"),
        Patch(facecolor="white", hatch="....", label="net heating"),
        Patch(facecolor=_EXTRAPOLATED_COLOR, alpha=0.32,
              label="outside Cloudy n_H table"),
        Line2D([], [], color="black", label="reference gas T"),
        Line2D([], [], color="0.35", linestyle="--", label="reference dust T"),
    ]
    fig.legend(
        handles=handles, loc="center left", bbox_to_anchor=(1.005, 0.5),
        frameon=False, fontsize=8,
    )
    fig.suptitle(
        "Dominant contribution along the isobaric-history cube; "
        "opacity shows dominant share\n"
        "dotted = net heating, solid/dashed = reference gas/dust tracks",
        fontsize=11,
    )
    filename = f"{args.output}_slices.png"
    fig.savefig(filename, bbox_inches="tight", dpi=250)
    plt.close(fig)
    print(f"Saved {filename}")


def plot_balance(data, args):
    nH, temperature, Z = data["nH"], data["T"], data["Z"]
    ncols = min(3, Z.size)
    nrows = int(np.ceil(Z.size / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.8 * ncols, 3.05 * nrows),
        sharex=True, sharey=True, constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()
    image = None
    for metallicity_index, ax in enumerate(axes):
        if metallicity_index >= Z.size:
            ax.set_visible(False)
            continue
        values = np.ma.masked_where(
            ~data["valid_mask"][metallicity_index],
            data["thermal_balance"][metallicity_index],
        )
        ax.set_facecolor(_INVALID_COLOR)
        image = ax.pcolormesh(
            np.log10(nH), np.log10(temperature), values.T,
            shading="nearest", cmap="RdBu", norm=Normalize(-1, 1),
        )
        ax.contour(
            np.log10(nH), np.log10(temperature), values.T,
            levels=[0], colors="black", linestyles=":", linewidths=0.8,
        )
        _overlay_tracks(ax, data, metallicity_index)
        table_max = data["table_nH_bounds"][1]
        if np.isfinite(table_max) and nH[-1] > table_max:
            ax.axvspan(
                np.log10(table_max), np.log10(nH[-1]),
                facecolor="none", edgecolor=_EXTRAPOLATED_COLOR,
                hatch="///", linewidth=0,
            )
        ax.set_title(rf"$Z/Z_\odot={Z[metallicity_index]:g}$")
        ax.set_xlabel(r"$\log_{10} n_H$ [cm$^{-3}$]")
        ax.set_ylabel(r"$\log_{10} T_{\rm gas}$ [K]")
    colorbar = fig.colorbar(image, ax=axes.tolist(), shrink=0.82)
    colorbar.set_label(r"$\eta=(C-H)/(C+H)$")
    fig.suptitle(
        "Thermal balance of sampled isobaric histories "
        "(blue=cooling, red=heating, grey=unreached)"
    )
    filename = f"{args.output}_thermal_balance.png"
    fig.savefig(filename, bbox_inches="tight", dpi=250)
    plt.close(fig)
    print(f"Saved {filename}")


def _selected_stack_fields(rates, side, threshold):
    magnitude = np.where(rates > 0, rates, 0) if side == "cooling" \
        else np.where(rates < 0, -rates, 0)
    total = magnitude.sum(axis=-1)
    fractions = np.divide(
        magnitude, total[:, None], out=np.zeros_like(magnitude),
        where=total[:, None] > 0,
    )
    selected = set()
    for row in fractions:
        cumulative = 0.0
        for index in np.argsort(row)[::-1]:
            if row[index] <= 0:
                break
            selected.add(int(index))
            cumulative += row[index]
            if cumulative >= threshold:
                break
    return sorted(selected), fractions


def plot_tracks(data, args):
    if data["track_pressure"].size == 0:
        print("No isobaric tracks requested; skipping track plot")
        return
    Z = data["Z"]
    colors = channel_colors(len(data["channel_fields"]))
    for track_index, pressure in enumerate(data["track_pressure"]):
        fig, axes = plt.subplots(
            Z.size, 2, figsize=(10.0, max(3.0, 2.45 * Z.size)),
            sharex=True, constrained_layout=True, squeeze=False,
        )
        legend_fields = set()
        for metallicity_index, metallicity in enumerate(Z):
            valid = data["track_valid"][metallicity_index, track_index]
            x = data["track_nH"][metallicity_index, track_index, valid]
            order = np.argsort(x)
            x = x[order]
            rates = data["track_cooling_rates"][
                metallicity_index, track_index, valid
            ][order]
            gas_temperature = data["track_gas_temperature"][
                metallicity_index, track_index, valid
            ][order]
            dust_temperature = data["track_dust_temperature"][
                metallicity_index, track_index, valid
            ][order]
            for column, side in enumerate(("cooling", "heating")):
                ax = axes[metallicity_index, column]
                selected, fractions = _selected_stack_fields(
                    rates, side, args.budget_threshold
                )
                legend_fields.update(selected)
                if selected:
                    ax.stackplot(
                        x, *[fractions[:, index] for index in selected],
                        colors=[colors[index] for index in selected],
                        linewidth=0,
                    )
                ax.set_xscale("log")
                ax.set_ylim(0, 1)
                ax.grid(True, which="major", color="white", alpha=0.55)
                tax = ax.twinx()
                tax.patch.set_alpha(0.0)
                tax.plot(x, gas_temperature, color="black", linewidth=1.15)
                tax.plot(
                    x, np.maximum(dust_temperature, 1.0e-300),
                    color="0.35", linestyle="--", linewidth=1.0,
                )
                tax.set_yscale("log")
                if column == 1:
                    tax.set_ylabel(r"$T$ [K]")
                else:
                    tax.set_yticklabels([])
                if column == 0:
                    ax.set_ylabel(
                        rf"$Z/Z_\odot={metallicity:g}$" +
                        "\nbudget fraction"
                    )
                if metallicity_index == 0:
                    ax.set_title(side.capitalize())
                if metallicity_index == Z.size - 1:
                    ax.set_xlabel(r"$n_H$ [cm$^{-3}$]")
        handles = [
            Patch(
                facecolor=colors[index],
                label=short_channel_label(
                    data["channel_fields"][index],
                    data["channel_labels"][index],
                ),
            )
            for index in sorted(legend_fields)
        ]
        handles += [
            Line2D([], [], color="black", label="gas T"),
            Line2D([], [], color="0.35", linestyle="--", label="dust T"),
        ]
        fig.legend(
            handles=handles, loc="upper center",
            bbox_to_anchor=(0.5, -0.015), frameon=False, fontsize=7,
            ncol=min(5, max(1, len(handles))),
        )
        fig.suptitle(
            rf"Non-equilibrium $n_HT={pressure:.3g}$ K cm$^{{-3}}$ track; "
            rf"same-sign {args.budget_threshold:.1%} contribution budget"
        )
        suffix = "" if track_index == 0 else f"_P{track_index:03d}"
        filename = f"{args.output}_isobaric_tracks{suffix}.png"
        fig.savefig(filename, bbox_inches="tight", dpi=250)
        plt.close(fig)
        print(f"Saved {filename}")


def plot_cube3d(data, args):
    nH, temperature, Z = data["nH"], data["T"], data["Z"]
    dominant = data["cooling_dominant_index"]
    share = data["cooling_dominant_share"]
    valid = data["valid_mask"]
    extrapolated = data["extrapolated_mask"]
    colors = channel_colors(len(data["channel_fields"]))
    ix, iy, iz = len(nH) // 2, len(temperature) // 2, len(Z) // 2
    x, y, z = np.log10(nH), np.log10(temperature), np.log10(Z)
    fig = plt.figure(figsize=(8.2, 6.4))
    ax = fig.add_subplot(projection="3d")

    rgba_z = _dominance_rgba(
        dominant[iz], share[iz], valid[iz], extrapolated[iz], colors,
    )
    X, Y = np.meshgrid(x, y, indexing="ij")
    ax.plot_surface(
        X, Y, np.full_like(X, z[iz]), facecolors=rgba_z,
        shade=False, linewidth=0, antialiased=False,
    )
    rgba_y = _dominance_rgba(
        dominant[:, :, iy], share[:, :, iy], valid[:, :, iy],
        extrapolated[:, :, iy], colors,
    )
    X2, Z2 = np.meshgrid(x, z, indexing="ij")
    ax.plot_surface(
        X2, np.full_like(X2, y[iy]), Z2,
        facecolors=np.transpose(rgba_y, (1, 0, 2)),
        shade=False, linewidth=0, antialiased=False,
    )
    rgba_x = _dominance_rgba(
        dominant[:, ix, :], share[:, ix, :], valid[:, ix, :],
        extrapolated[:, ix, :], colors,
    )
    Y3, Z3 = np.meshgrid(y, z, indexing="ij")
    ax.plot_surface(
        np.full_like(Y3, x[ix]), Y3, Z3,
        facecolors=np.transpose(rgba_x, (1, 0, 2)),
        shade=False, linewidth=0, antialiased=False,
    )
    if data["track_pressure"].size:
        for metallicity_index in range(Z.size):
            for track_index in range(data["track_pressure"].size):
                mask = data["track_valid"][metallicity_index, track_index]
                ax.plot(
                    np.log10(data["track_nH"][
                        metallicity_index, track_index, mask
                    ]),
                    np.log10(data["track_gas_temperature"][
                        metallicity_index, track_index, mask
                    ]),
                    np.full(mask.sum(), z[metallicity_index]),
                    color="black", linewidth=0.9,
                )
    ax.set_xlabel(r"$\log_{10} n_H$ [cm$^{-3}$]")
    ax.set_ylabel(r"$\log_{10} T$ [K]")
    ax.set_zlabel(r"$\log_{10} Z/Z_\odot$")
    ax.set_title("Dominant-cooling orthogonal cuts (3-D context)")
    ax.view_init(elev=24, azim=-126)
    ax.set_box_aspect((1.15, 0.9, 0.6))
    filename = f"{args.output}_cube3d.png"
    fig.savefig(filename, bbox_inches="tight", dpi=250)
    plt.close(fig)
    print(f"Saved {filename}")


def save_outputs(data, args):
    filename = f"{args.output}.npz"
    np.savez_compressed(filename, **data)
    print(f"Saved {filename}")

    csv_name = f"{args.output}_summary.csv"
    labels = data["channel_labels"]
    with open(csv_name, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "Z_zsun", "requested_nH_cm-3", "actual_nH_cm-3", "T_K",
            "P_over_kB_K_cm-3", "sampled", "outside_table",
            "sample_time_myr", "dust_temperature_K",
            "dominant_cooling", "cooling_share", "dominant_heating",
            "heating_share", "cooling_budget_erg_cm3_s",
            "heating_budget_erg_cm3_s", "net_total_erg_cm3_s",
            "residual_erg_cm3_s", "pressure_history_stop",
            "pressure_history_steps", "pressure_history_solver_retries",
            "pressure_history_safety_factor",
        ])
        for iz, metallicity in enumerate(data["Z"]):
            for ix, density in enumerate(data["nH"]):
                for iy, temperature in enumerate(data["T"]):
                    cool = data["cooling_dominant_index"][iz, ix, iy]
                    heat = data["heating_dominant_index"][iz, ix, iy]
                    pressure_index = data["pressure_index"][iz, ix, iy]
                    stop_code = data["pressure_stop_code"][
                        iz, pressure_index
                    ]
                    writer.writerow([
                        metallicity, density,
                        data["actual_nH"][iz, ix, iy],
                        temperature,
                        data["pressure"][pressure_index],
                        int(data["valid_mask"][iz, ix, iy]),
                        int(data["extrapolated_mask"][iz, ix, iy]),
                        data["sample_time_myr"][iz, ix, iy],
                        data["dust_temperature"][iz, ix, iy],
                        labels[cool] if cool >= 0 else "",
                        data["cooling_dominant_share"][iz, ix, iy],
                        labels[heat] if heat >= 0 else "",
                        data["heating_dominant_share"][iz, ix, iy],
                        data["cooling_budget_total"][iz, ix, iy],
                        data["heating_budget_total"][iz, ix, iy],
                        data["total_rate"][iz, ix, iy],
                        data["residual_rate"][iz, ix, iy],
                        data["stop_labels"][stop_code],
                        data["pressure_steps"][iz, pressure_index],
                        data["pressure_solver_retries"][iz, pressure_index],
                        data["pressure_safety_factor"][iz, pressure_index],
                    ])
    print(f"Saved {csv_name}")


def build_output_data(args):
    (
        nH, temperature, Z, pressure, pressure_index, track_pressures, result
    ) = build_atlas(args)
    statistics = budget_statistics(
        result["cooling_rates"], result["valid_mask"], args.budget_threshold
    )
    table_bounds = np.asarray(read_table_bounds(args.uvb_data_file))
    extrapolated_2d = (
        (nH[:, None] < table_bounds[0]) |
        (nH[:, None] > table_bounds[1]) |
        (temperature[None, :] < table_bounds[2]) |
        (temperature[None, :] > table_bounds[3])
    )
    extrapolated = np.broadcast_to(
        extrapolated_2d, (Z.size,) + extrapolated_2d.shape
    ).copy()
    closure = (
        np.nansum(result["cooling_rates"], axis=-1) +
        result["residual_rate"] -
        result["total_rate"]
    )
    reference_pressure_index = -1
    if track_pressures.size:
        reference_pressure_index = int(np.argmin(abs(
            np.log10(pressure / track_pressures[0])
        )))
    data = {
        "format_version": np.int16(_FORMAT_VERSION),
        "sampling_mode": np.asarray(_SAMPLING_MODE),
        "pressure_law": np.asarray(
            "n_H*T=constant; composition, gamma, and PdV corrections omitted"
        ),
        "initial_condition": np.asarray(
            "ionized parcel at the maximum cube temperature"
        ),
        "chemistry_configuration": np.asarray(
            f"dust_model=0; dust_species={args.dust_species}; "
            "ISRF=0; photoelectric_heating=0"
        ),
        "dust_model": np.int16(0),
        "dust_species": np.int16(args.dust_species),
        "isrf_habing": np.float64(0.0),
        "photoelectric_heating": np.int16(0),
        "rate_units": np.asarray("erg cm^3 s^-1"),
        "rate_normalization": np.asarray(
            "-edot * cooling_units / n_H^2"
        ),
        "nH": nH,
        "T": temperature,
        "Z": Z,
        "pressure": pressure,
        "pressure_index": np.broadcast_to(
            pressure_index, (Z.size,) + pressure_index.shape
        ).copy(),
        "track_pressure": track_pressures,
        "reference_pressure_index": np.int32(reference_pressure_index),
        "channel_fields": np.asarray(channel_fields()),
        "channel_labels": np.asarray(
            [channel_labels()[field] for field in channel_fields()]
        ),
        "channel_colors": np.asarray([
            plt.matplotlib.colors.to_hex(color)
            for color in channel_colors(len(channel_fields()))
        ]),
        "budget_threshold": np.float64(args.budget_threshold),
        "uvb_data_file": np.asarray([args.uvb_data_file]),
        "uvb_sha256": np.asarray([_sha256(args.uvb_data_file)]),
        "table_nH_bounds": table_bounds[:2],
        "table_temperature_bounds": table_bounds[2:],
        "extrapolated_mask": extrapolated,
        # Kept for schema compatibility.  These are sampled non-equilibrium
        # states, not fixed-state chemistry equilibria.
        "converged_mask": np.zeros_like(result["valid_mask"], dtype=bool),
        "closure_rate": closure,
        "stop_labels": _STOP_LABELS,
        **result,
        **statistics,
    }
    return data


def main(argv=None):
    args = parse_args(argv)
    if args.quick:
        args.points_per_decade = 2
        args.Z_values = [1e-4, 1e-2, 1.0]
    validate_args(args)
    print(f"=== {args.output} ===", flush=True)
    print(
        f"  history cube: n_H={args.nH_range[0]:g}.."
        f"{args.nH_range[1]:g} cm^-3, "
        f"T={args.T_range[0]:g}..{args.T_range[1]:g} K, "
        f"Z={args.Z_values} Zsun",
        flush=True,
    )
    print(
        f"  resolution={args.points_per_decade}/decade, "
        f"workers={min(args.jobs, len(args.Z_values))}, "
        f"UVB={args.uvb_data_file}",
        flush=True,
    )
    print(
        "  pressure projection: n_H*T=constant "
        "(composition, gamma, and PdV corrections intentionally omitted)",
        flush=True,
    )
    print(
        f"  dust setup: dust_model=0, dust_species={args.dust_species}, "
        "ISRF=0, photoelectric_heating=0",
        flush=True,
    )
    if args.dust_species > 0 and args.track_safety_factor > 0.02:
        print(
            "  WARNING: grain-species chemistry is less tolerant of large "
            "steps; use --track-safety-factor 0.015 if a native worker fails.",
            flush=True,
        )
    table_bounds = read_table_bounds(args.uvb_data_file)
    print(
        f"  Cloudy table domain: n_H={table_bounds[0]:.3g}.."
        f"{table_bounds[1]:.3g} cm^-3, T={table_bounds[2]:.3g}.."
        f"{table_bounds[3]:.3g} K",
        flush=True,
    )
    if args.nH_range[1] > table_bounds[1]:
        print(
            "  WARNING: requested densities above the Cloudy table are "
            "marked as extrapolated in every output.",
            flush=True,
        )

    data = build_output_data(args)
    save_outputs(data, args)
    if "slices" in args.plots:
        plot_slices(data, args)
    if "balance" in args.plots:
        plot_balance(data, args)
    if "tracks" in args.plots:
        plot_tracks(data, args)
    if "cube3d" in args.plots:
        plot_cube3d(data, args)

    if args.html:
        import cgm_cube_interactive
        cgm_cube_interactive.main([
            f"{args.output}.npz", "--output", f"{args.output}.html"
        ])

    valid = int(np.count_nonzero(data["valid_mask"]))
    total = int(data["valid_mask"].size)
    print(f"  sampled cells: {valid}/{total}")
    recovered = int(np.count_nonzero(data["pressure_solver_retries"]))
    if recovered:
        used = data["pressure_safety_factor"][
            data["pressure_solver_retries"] > 0
        ]
        print(
            f"  recovered pressure lines: {recovered} "
            f"(minimum accepted safety factor={np.nanmin(used):g})"
        )
    if data["track_pressure"].size:
        for iz, metallicity in enumerate(data["Z"]):
            code = int(data["trajectory_stop_code"][iz, 0])
            print(
                f"  Z={metallicity:g} track: {_STOP_LABELS[code]}, "
                f"T_stop={data['trajectory_stop_temperature'][iz, 0]:.4g} K, "
                f"steps={data['trajectory_steps'][iz, 0]}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
