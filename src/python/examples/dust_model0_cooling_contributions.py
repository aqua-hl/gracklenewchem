########################################################################
#
# Dust model 0 temperature sweep and cooling contribution diagnostics.
#
# This follows the dust_model1 metallicity-temperature freefall example, but
# runs the default dust_model=0 path. It then calls an experimental diagnostic
# hook that records the additive edot terms assembled by cool1d_multi_g and
# rate_timestep_g. The saved cooling_* arrays are signed as:
#   positive = net cooling, negative = net heating.
#
########################################################################

import argparse
import contextlib
import csv
import io
import json
import os
import sys

import numpy as np
from matplotlib import pyplot as plt

from gracklepy import (
    FluidContainer,
    chemistry_data,
    evolve_constant_density,
    evolve_constant_pressure,
    evolve_freefall,
    setup_fluid_container,
)
from gracklepy.utilities.data_path import grackle_data_dir
from gracklepy.utilities.physical_constants import (
    cm_per_mpc,
    gravitational_constant_cgs,
    mass_hydrogen_cgs,
    sec_per_Myr,
)

_MODEL_NAME = os.path.basename(__file__[:-3])
_DEFAULT_METALLICITIES = [1.0]
_DEFAULT_CONTRIBUTION_METALLICITIES = [1.0]

_SETTING_ALIASES = {
    "dust_temperature_multi": "use_multiple_dust_temperatures",
    "radiative_transfer_HDI_diss": "radiative_transfer_HDI_dissociation",
    "radiative_transfer_metal_diss": "radiative_transfer_metal_dissociation",
    # This legacy/local name is not a chemistry_data flag in this checkout.
    # H2 dissociation is read from RT_H2_dissociation_rate when RT is enabled.
    "radiative_transfer_H2II_diss": None,
}
_WARNED_SKIPPED_SETTINGS = set()

_GRAIN_FIELDS_BY_DUST_SPECIES = {
    0: [],
    1: ["MgSiO3_dust_density", "AC_dust_density"],
    2: [
        "MgSiO3_dust_density",
        "AC_dust_density",
        "SiM_dust_density",
        "FeM_dust_density",
        "Mg2SiO4_dust_density",
        "Fe3O4_dust_density",
        "SiO2_dust_density",
        "MgO_dust_density",
        "FeS_dust_density",
        "Al2O3_dust_density",
    ],
    3: [
        "MgSiO3_dust_density",
        "AC_dust_density",
        "SiM_dust_density",
        "FeM_dust_density",
        "Mg2SiO4_dust_density",
        "Fe3O4_dust_density",
        "SiO2_dust_density",
        "MgO_dust_density",
        "FeS_dust_density",
        "Al2O3_dust_density",
        "ref_org_dust_density",
        "vol_org_dust_density",
        "H2O_ice_dust_density",
    ],
}

_COOLING_SPECIES_FIELDS = [
    "CI_density",
    "CII_density",
    "OI_density",
    "CO_density",
    "OH_density",
    "H2O_density",
]

# Raw C contribution field name -> diagnostic key (without the "cooling_"
# prefix) for the few channels whose plot key differs from the C field name.
# Every other channel keeps its C name. See calculate_contributions().
_CONTRIBUTION_KEY_OVERRIDES = {
    "recombination_cooling": "recombination",
    "ci": "species_CI",
    "cii": "species_CII",
    "oi": "species_OI",
    "co": "species_CO",
    "oh": "species_OH",
    "h2o": "species_H2O",
}

# Aggregate chemistry channels that summarise their per-reaction children.
# Dropped from the diagnostics so the per-reaction terms are not double-counted.
_CONTRIBUTION_AGGREGATE_FIELDS = frozenset({
    "chemistry_hi_hei",
    "chemistry_heii",
    "chemistry_h2_gas",
})

_CHANNEL_PLOT_ORDER = [
    ("cooling_collisional_excitation", r"H/He exc."),
    ("cooling_collisional_ionisation", r"H/He ion."),
    ("cooling_recombination", r"H/He recomb."),
    ("cooling_bremsstrahlung", "free-free"),
    ("cooling_h2_line", r"H$_2$ line"),
    ("cooling_h2_cie", r"H$_2$ CIE"),
    ("cooling_hd", "HD"),
    ("cooling_cloudy_primordial", "Cloudy primordial"),
    ("cooling_cloudy_metal", "Cloudy metals"),
    ("cooling_species_CI", "CI"),
    ("cooling_species_CII", "CII"),
    ("cooling_species_OI", "OI"),
    ("cooling_species_CO", "CO"),
    ("cooling_species_OH", "OH"),
    ("cooling_species_H2O", r"H$_2$O"),
    ("cooling_dust_gas_grain", "gas-grain"),
    ("cooling_dust_recombination", "grain recomb."),
    ("cooling_photoelectric", "photoelectric"),
    ("cooling_photoionization_heating", "photoion."),
    ("cooling_rt_photoheating", "RT photoheat"),
    ("cooling_volumetric_heating", "volumetric heat"),
    ("cooling_specific_heating", "specific heat"),
    ("cooling_chemistry_h2_dust", r"H$_2$ formation"),
    ("cooling_compton", "Compton"),
]

_CHEMISTRY_DETAIL_PLOT_ORDER = [
    ("cooling_chemistry_hi_collisional_ionization",
     r"H I + e$^-$ $\rightarrow$ H II + 2e$^-$"),
    ("cooling_chemistry_hii_recombination",
     r"H II + e$^-$ $\rightarrow$ H I"),
    ("cooling_chemistry_hei_collisional_ionization",
     r"He I + e$^-$ $\rightarrow$ He II + 2e$^-$"),
    ("cooling_chemistry_heii_recombination",
     r"He II + e$^-$ $\rightarrow$ He I"),
    ("cooling_chemistry_heii_collisional_ionization",
     r"He II + e$^-$ $\rightarrow$ He III + 2e$^-$"),
    ("cooling_chemistry_heiii_recombination",
     r"He III + e$^-$ $\rightarrow$ He II"),
    ("cooling_chemistry_h2_hminus_formation",
     r"H + H$^-$ $\rightarrow$ H$_2$ + e$^-$"),
    ("cooling_chemistry_h2_threebody_formation",
     r"H + H + H $\rightarrow$ H$_2$ + H"),
    ("cooling_chemistry_h2_collisional_dissociation",
     r"H$_2$ + H $\rightarrow$ H + H + H"),
    ("cooling_chemistry_h2_dust", r"H$_2$ formation"),
]

_THERMAL_BALANCE_PLOT_ORDER = [
    ("heating_compressional", "compression"),
    ("heating_microphysical_net", "microphysics"),
    ("heating_thermal_balance", "net thermal source"),
]

_COMBINED_STACK_DUPLICATE_FIELDS = {"cooling_chemistry_h2_dust"}

_CHEMISTRY_REACTION_FORMULAE = {
    "cooling_chemistry_hi_collisional_ionization":
        "HI + e -> HII + 2e",
    "cooling_chemistry_hii_recombination":
        "HII + e -> HI",
    "cooling_chemistry_hei_collisional_ionization":
        "HeI + e -> HeII + 2e",
    "cooling_chemistry_heii_recombination":
        "HeII + e -> HeI",
    "cooling_chemistry_heii_collisional_ionization":
        "HeII + e -> HeIII + 2e",
    "cooling_chemistry_heiii_recombination":
        "HeIII + e -> HeII",
    "cooling_chemistry_h2_hminus_formation":
        "H + H- -> H2 + e",
    "cooling_chemistry_h2_threebody_formation":
        "H + H + H -> H2 + H",
    "cooling_chemistry_h2_collisional_dissociation":
        "H2 + H -> H + H + H",
    "cooling_chemistry_h2_dust":
        "H + H on dust -> H2",
}

_CHEMISTRY_PROCESS_DESCRIPTIONS = {
    "cooling_chemistry_hi_collisional_ionization":
        "collisional ionization energy sink",
    "cooling_chemistry_hii_recombination":
        "recombination chemical heating term",
    "cooling_chemistry_hei_collisional_ionization":
        "collisional ionization energy sink",
    "cooling_chemistry_heii_recombination":
        "recombination chemical heating term",
    "cooling_chemistry_heii_collisional_ionization":
        "collisional ionization energy sink",
    "cooling_chemistry_heiii_recombination":
        "recombination chemical heating term",
    "cooling_chemistry_h2_hminus_formation":
        "H2 formation heating through the H- channel",
    "cooling_chemistry_h2_threebody_formation":
        "three-body H2 formation heating",
    "cooling_chemistry_h2_collisional_dissociation":
        "collisional dissociation energy sink",
    "cooling_chemistry_h2_dust":
        "dust-catalyzed H2 formation heating",
}

_CHEMISTRY_TEXT_LABELS = {
    "cooling_chemistry_hi_collisional_ionization": "H I ion.",
    "cooling_chemistry_hii_recombination": "H II recomb.",
    "cooling_chemistry_hei_collisional_ionization": "He I ion.",
    "cooling_chemistry_heii_recombination": "He II recomb.",
    "cooling_chemistry_heii_collisional_ionization": "He II ion.",
    "cooling_chemistry_heiii_recombination": "He III recomb.",
    "cooling_chemistry_h2_hminus_formation": "H- channel",
    "cooling_chemistry_h2_threebody_formation": "3-body H2",
    "cooling_chemistry_h2_collisional_dissociation": "H2 dissoc.",
    "cooling_chemistry_h2_dust": "H2 formation",
}


def configure_matplotlib():
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "axes.linewidth": 0.8,
        "legend.fontsize": 7,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "lines.linewidth": 1.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    })


configure_matplotlib()


# Names of the figures main() can produce, matching their PNG suffixes. Used by
# the --plots selector. "temperature_density" is the only one that does not need
# the cooling-contribution diagnostics.
_PLOT_NAMES = (
    "temperature_density",
    "cooling_channel_rates",
    "cooling_channel_fractions",
    "cooling_channel_rates_signed",
    "cooling_channel_fractions_signed",
    "thermal_balance",
    "dominant_budget_stack",
    "chemistry_reaction_fractions",
    "chemistry_reaction_fractions_signed",
)


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run a dust_model=0 freefall metallicity sweep and estimate "
            "instantaneous cooling contributions by channel."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--metallicities",
        type=float,
        nargs="*",
        default=_DEFAULT_METALLICITIES,
        help=(
            "Metallicities in solar units. Default is a single solar-metallicity "
            "run for cleaner diagnostic figures."
        ),
    )
    parser.add_argument(
        "--contribution-metallicities",
        type=float,
        nargs="*",
        default=_DEFAULT_CONTRIBUTION_METALLICITIES,
        help="Metallicities to include in the diagnostic contribution panels.",
    )
    parser.add_argument("--initial-nH", type=float, default=1e-1,
                        help="Initial gas density in cm^-3.")
    parser.add_argument("--final-nH", type=float, default=1e10,
                        help="Final gas density in cm^-3.")
    parser.add_argument("--initial-temperature", type=float, default=5e6,
                        help="Initial gas temperature in K.")
    parser.add_argument("--cooling-temperature", type=float, default=1e6,
                        help="Constant-density pre-cooling target in K.")
    parser.add_argument("--isrf", type=float, default=1,
                        help="Interstellar radiation field in Habing units.")
    parser.add_argument(
        "--uvb-data-file",
        type=os.path.abspath,
        default=None,
        help=(
            "UV-background HDF5 table. Supplying this option enables the "
            "UV background; by default the no-UVB table is used."
        ),
    )
    parser.add_argument("--three-body-rate", type=int, default=1,
                        help="Grackle three_body_rate selector.")
    parser.add_argument(
        "--dust-species",
        type=int,
        choices=sorted(_GRAIN_FIELDS_BY_DUST_SPECIES),
        default=3,
        help="dust_species selector for the dust_model=0 grain-species path.",
    )
    parser.add_argument(
        "--use-multiple-dust-temperatures",
        action="store_true",
        help="Track a separate dust temperature field for each grain species.",
    )
    parser.add_argument("--dust-sublimation", action=argparse.BooleanOptionalAction,
                        default=True, help="Enable dust sublimation.")
    parser.add_argument("--grain-growth", action=argparse.BooleanOptionalAction,
                        default=False, help="Enable dust_model=0 grain growth.")
    parser.add_argument("--freefall-safety", type=float, default=0.001,
                        help="Safety factor for the freefall integrator.")
    parser.add_argument("--precool-safety", type=float, default=0.001,
                        help="Safety factor for the pre-cooling integrator.")
    parser.add_argument(
        "--trajectory",
        choices=("freefall", "isobaric"),
        default="freefall",
        help=(
            "Thermodynamic path the parcel follows. 'freefall' pre-cools then "
            "collapses to --final-nH (ISM/collapse). 'isobaric' cools at "
            "constant pressure from --initial-temperature down to "
            "--isobaric-final-temperature (pressure-confined, e.g. CGM)."
        ),
    )
    parser.add_argument("--isobaric-final-temperature", type=float, default=1e4,
                        help="Final gas temperature for the isobaric path in K.")
    parser.add_argument("--isobaric-safety", type=float, default=0.02,
                        help="Safety factor for the constant-pressure integrator.")
    parser.add_argument(
        "--photoelectric-heating", type=int, default=2,
        help=(
            "Grackle photoelectric_heating mode. Set 0 to disable photoelectric "
            "heating (with --isrf 0) for a cooling-only study."
        ),
    )
    parser.add_argument("--output", default=_MODEL_NAME,
                        help="Output prefix for PNG and NPZ files.")
    parser.add_argument("--skip-diagnostics", action="store_true",
                        help="Only make the temperature-density plot.")
    parser.add_argument(
        "--diagnostic-dt",
        type=float,
        default=0.0,
        help=(
            "dt passed to the non-equilibrium diagnostic rate lookup. Most "
            "cooling terms are instantaneous; dust growth lookups can use dt."
        ),
    )
    parser.add_argument(
        "--plot-dynamic-range",
        type=float,
        default=1.0e8,
        help=(
            "Maximum plotted dynamic range per panel. Values smaller than "
            "panel_max / plot_dynamic_range are hidden from figures but still "
            "saved in the data files. Set <= 0 to disable this masking."
        ),
    )
    parser.add_argument(
        "--plot-min-fraction",
        type=float,
        default=1.0e-8,
        help=(
            "Minimum absolute fractional contribution shown in fraction plots. "
            "Use 0 to show all finite nonzero fractions."
        ),
    )
    parser.add_argument(
        "--plot-min-rate",
        type=float,
        default=0.0,
        help=(
            "Minimum absolute rate shown in rate plots, in erg cm^3 s^-1. "
            "The dynamic-range mask is applied in addition to this."
        ),
    )
    parser.add_argument(
        "--fraction-y-scale",
        choices=("log", "linear"),
        default="log",
        help=(
            "Y-axis scaling for the split cooling/heating fraction plots. "
            "Log scale reveals subdominant channels; signed fraction plots "
            "remain symlog."
        ),
    )
    parser.add_argument(
        "--stacked-budget-threshold",
        type=float,
        default=0.9999,
        help=(
            "For the stacked dominance summary, show the union of channels "
            "needed to explain this same-sign budget fraction at each density."
        ),
    )
    parser.add_argument("--show-progress", action="store_true",
                        help="Show per-step output from the evolve helpers.")
    parser.add_argument(
        "--plots", nargs="*", choices=("all",) + _PLOT_NAMES, default=["all"],
        metavar="PLOT",
        help=(
            "Which figures to produce (default: all). Names match the PNG "
            "suffixes, e.g. '--plots dominant_budget_stack' makes only that one. "
            "Data outputs (npz/csv) are still written."
        ),
    )
    return parser.parse_args(args)


def apply_common_chemistry_options(my_chemistry, opts, overrides=None):
    values = {
        "use_grackle": 1,
        "with_radiative_cooling": 1,
        "primordial_chemistry": 4,
        "metal_cooling": 1,
        "metal_chemistry": 1,
        "multi_metals": 0,
        "metal_abundances": 0,
        "dust_chemistry": 1,
        "UVbackground": int(opts.uvb_data_file is not None),
        "cmb_temperature_floor": 1,
        "Gamma": 1.66667,
        "h2_on_dust": 1,
        "use_dust_density_field": 1,
        "dust_model": 0,
        "dust_species_track": 0,
        "dust_species": opts.dust_species,
        "dust_temperature_multi": int(opts.use_multiple_dust_temperatures),
        "grain_growth": int(opts.grain_growth),
        "dust_sublimation": int(opts.dust_sublimation),
        "dust_recombination_cooling": 1,
        "photoelectric_heating": opts.photoelectric_heating,
        "photoelectric_heating_rate": 0.0,
        "use_sne_field": 0,
        "use_isrf_field": 0,
        "interstellar_radiation_field": opts.isrf,
        "solver_method": 1,
        "three_body_rate": opts.three_body_rate,
        "cie_cooling": 1,
        "h2_optical_depth_approximation": 1,
        "CaseBRecombination": 1,
        "ih2co": 1,
        "ipiht": 1,
        "HydrogenFractionByMass": 0.76,
        "DeuteriumToHydrogenRatio": 6.8e-5,
        "SolarMetalFractionByMass": 0.01295,
        "local_dust_to_gas_ratio": 0.009387,
        "NumberOfTemperatureBins": 600,
        "TemperatureStart": 1.0,
        "TemperatureEnd": 1.0e9,
        "NumberOfDustTemperatureBins": 250,
        "DustTemperatureStart": 1.0,
        "DustTemperatureEnd": 1500.0,
        "Compton_xray_heating": 0,
        "LWbackground_sawtooth_suppression": 0,
        "LWbackground_intensity": 0.0,
        "UVbackground_redshift_on": -99999.0,
        "UVbackground_redshift_off": -99999.0,
        "UVbackground_redshift_fullon": -99999.0,
        "UVbackground_redshift_drop": -99999.0,
        "cloudy_electron_fraction_factor": 0.00915396,
        "use_radiative_transfer": 0,
        "radiative_transfer_coupled_rate_solver": 0,
        "radiative_transfer_intermediate_step": 0,
        "radiative_transfer_hydrogen_only": 0,
        "self_shielding_method": 3,
        "H2_custom_shielding": 0,
        "H2_self_shielding": 0,
        "radiative_transfer_H2II_diss": 1,
        "radiative_transfer_HDI_diss": 1,
        "radiative_transfer_metal_ionization": 0,
        "radiative_transfer_metal_diss": 0,
        "grackle_data_file": (
            opts.uvb_data_file
            if opts.uvb_data_file is not None
            else os.path.join(grackle_data_dir, "HM2012_metal_test.h5")
        ),
    }
    if overrides:
        values.update(overrides)

    skipped = []
    for key, value in values.items():
        target = _SETTING_ALIASES.get(key, key)
        if target is None:
            skipped.append((key, "not a chemistry_data flag in this checkout"))
            continue
        if hasattr(my_chemistry, target):
            setattr(my_chemistry, target, value)
        else:
            skipped.append((key, "not found"))

    new_skipped = [
        item for item in skipped
        if item[0] not in _WARNED_SKIPPED_SETTINGS
    ]
    if new_skipped:
        for key, reason in new_skipped:
            print(f"# skipped chemistry setting {key!r}: {reason}")
            _WARNED_SKIPPED_SETTINGS.add(key)

    redshift = 0.0
    my_chemistry.comoving_coordinates = 0
    my_chemistry.a_units = 1.0
    my_chemistry.a_value = 1.0 / (1.0 + redshift) / my_chemistry.a_units
    my_chemistry.density_units = mass_hydrogen_cgs
    my_chemistry.length_units = cm_per_mpc
    my_chemistry.time_units = sec_per_Myr
    my_chemistry.set_velocity_units()


def make_chemistry(opts, overrides=None):
    my_chemistry = chemistry_data()
    apply_common_chemistry_options(my_chemistry, opts, overrides=overrides)
    return my_chemistry


def maybe_quiet(show_progress):
    if show_progress:
        return contextlib.nullcontext()
    return contextlib.redirect_stdout(io.StringIO())


def metallicity_label(metallicity):
    if metallicity == 0:
        return "0"
    exponent = np.log10(metallicity)
    if np.isclose(exponent, round(exponent)) and not np.isclose(metallicity, 1):
        return rf"$10^{{{int(round(exponent))}}}$"
    return f"{metallicity:g}"


def metallicity_tag(metallicity):
    if metallicity == 0:
        return "Z0"
    exponent = np.log10(metallicity)
    if np.isclose(exponent, round(exponent)):
        return f"Z1e{int(round(exponent)):+d}".replace("+", "p").replace("-", "m")
    return f"Z{metallicity:g}".replace(".", "p").replace("-", "m")


def array(data, field):
    return np.atleast_1d(np.asarray(data[field], dtype=float))


def save_plot(fig, fname):
    fig.savefig(fname, bbox_inches="tight", pad_inches=0.04)
    print(f"Saved {fname}")
    plt.close(fig)


def channel_colors(n_colors):
    maps = (plt.cm.tab20, plt.cm.tab20b, plt.cm.tab20c)
    colors = np.vstack([cmap(np.linspace(0, 1, 20)) for cmap in maps])
    if n_colors <= len(colors):
        return colors[:n_colors]
    return plt.cm.hsv(np.linspace(0, 1, n_colors, endpoint=False))


def format_axis(ax):
    ax.minorticks_on()
    ax.grid(True, which="major", color="0.86", linewidth=0.55)
    ax.grid(True, which="minor", color="0.92", linewidth=0.35)


def annotate_metallicity(ax, metallicity, loc="upper left"):
    x = 0.04 if loc.endswith("left") else 0.96
    y = 0.94 if loc.startswith("upper") else 0.06
    ha = "left" if loc.endswith("left") else "right"
    va = "top" if loc.startswith("upper") else "bottom"
    ax.text(
        x, y, rf"$Z/Z_\odot = {metallicity:g}$",
        transform=ax.transAxes, ha=ha, va=va,
        bbox={"facecolor": "white", "edgecolor": "0.75",
              "boxstyle": "square,pad=0.25", "alpha": 0.9},
    )


def add_shared_legend(fig, axes, ncol=4):
    handles, labels = [], []
    for ax in np.atleast_1d(axes):
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(
            handles, labels, loc="center left", ncol=ncol,
            frameon=False, bbox_to_anchor=(1.01, 0.5),
            columnspacing=1.0, handlelength=2.2,
        )


def finite_abs_values(values):
    arrays = []
    for value in values:
        arr = np.abs(np.asarray(value, dtype=float))
        arr = arr[np.isfinite(arr) & (arr > 0)]
        if arr.size:
            arrays.append(arr)
    if not arrays:
        return np.array([], dtype=float)
    return np.concatenate(arrays)


def plot_threshold(values, opts, fractions=False):
    abs_values = finite_abs_values(values)
    threshold = 0.0
    if abs_values.size and opts.plot_dynamic_range > 0:
        threshold = np.nanmax(abs_values) / opts.plot_dynamic_range
    if fractions:
        threshold = max(threshold, opts.plot_min_fraction)
    else:
        threshold = max(threshold, opts.plot_min_rate)
    return threshold


def same_sign_budget_fractions(diag, fields):
    valid_fields = [field for field in fields if field in diag]
    if not valid_fields:
        return {}, None, None

    template = np.asarray(diag[valid_fields[0]], dtype=float)
    cooling_total = np.zeros_like(template)
    heating_total = np.zeros_like(template)
    for field in valid_fields:
        values = np.asarray(diag[field], dtype=float)
        cooling_total += np.where(values > 0, values, 0.0)
        heating_total += np.where(values < 0, -values, 0.0)

    shares = {}
    for field in valid_fields:
        values = np.asarray(diag[field], dtype=float)
        share = np.zeros_like(values)
        cooling = values > 0
        heating = values < 0
        np.divide(
            values, cooling_total, out=share,
            where=cooling & (cooling_total > 0),
        )
        np.divide(
            values, heating_total, out=share,
            where=heating & (heating_total > 0),
        )
        shares[field] = share
    return shares, cooling_total, heating_total


def combined_stack_order():
    return [
        (field, label) for field, label in _CHANNEL_PLOT_ORDER
        if field not in _COMBINED_STACK_DUPLICATE_FIELDS
    ] + _CHEMISTRY_DETAIL_PLOT_ORDER


def dominant_budget_fields(shares, fields, side, threshold):
    selected = set()
    threshold = np.clip(threshold, 0.0, 1.0)
    if not shares:
        return []

    first = next(iter(shares.values()))
    n_values = np.asarray(first).size
    for index in range(n_values):
        ranked = []
        for field in fields:
            if field not in shares:
                continue
            value = float(shares[field][index])
            if side == "cooling" and value > 0:
                ranked.append((value, field))
            elif side == "heating" and value < 0:
                ranked.append((-value, field))
        ranked.sort(reverse=True)

        cumulative = 0.0
        for value, field in ranked:
            selected.add(field)
            cumulative += value
            if cumulative >= threshold:
                break

    return [field for field in fields if field in selected]


def hydrogen_mass_density(data):
    rho_H = array(data, "HI_density") + array(data, "HII_density")
    for field in ("HM_density", "H2I_density", "H2II_density"):
        if field in data:
            rho_H = rho_H + array(data, field)
    for field in ("HDI_density", "HDII_density"):
        if field in data:
            rho_H = rho_H + array(data, field) / 3.0
    if "HeHII_density" in data:
        rho_H = rho_H + array(data, "HeHII_density") / 5.0
    return np.maximum(rho_H, 1e-300)


def seed_model0_grain_species(fc):
    chem = fc.chemistry_data
    fields = [
        field for field in _GRAIN_FIELDS_BY_DUST_SPECIES[chem.dust_species]
        if field in fc
    ]
    if not fields:
        return

    try:
        yield_map = chem._experimental_grain_inj_path_yields()
    except Exception:
        yield_map = {}

    weights = []
    for field in fields:
        grain_name = field[:-8]
        grain_yield = yield_map.get(grain_name)
        if grain_yield is None or len(grain_yield) == 0:
            weights.append(0.0)
        else:
            weights.append(float(np.asarray(grain_yield, dtype=float)[0]))

    weights = np.asarray(weights, dtype=float)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.ones(len(fields), dtype=float)
    weights = weights / weights.sum()

    for field, weight in zip(fields, weights):
        fc[field][:] = fc["dust_density"] * weight


def run_case(metallicity, opts):
    my_chemistry = make_chemistry(opts)
    metal_mass_fraction = metallicity * my_chemistry.SolarMetalFractionByMass
    dust_to_gas_ratio = metallicity * my_chemistry.local_dust_to_gas_ratio

    fc = setup_fluid_container(
        my_chemistry,
        density=opts.initial_nH * mass_hydrogen_cgs,
        temperature=opts.initial_temperature,
        metal_mass_fraction=metal_mass_fraction,
        dust_to_gas_ratio=dust_to_gas_ratio,
        state="ionized",
        converge=False,
    )
    seed_model0_grain_species(fc)

    with maybe_quiet(opts.show_progress):
        if opts.trajectory == "isobaric":
            # Pressure-confined cooling from the hot initial state: record the
            # contributions as the parcel cools at constant pressure.
            data = evolve_constant_pressure(
                fc,
                final_temperature=opts.isobaric_final_temperature,
                safety_factor=opts.isobaric_safety,
            )
        else:
            evolve_constant_density(
                fc,
                final_temperature=opts.cooling_temperature,
                safety_factor=opts.precool_safety,
            )
            data = evolve_freefall(
                fc,
                opts.final_nH * mass_hydrogen_cgs,
                safety_factor=opts.freefall_safety,
                include_pressure=True,
            )

    return data


def field_conversion_to_cgs(my_chemistry, fc, field):
    if field in fc.density_fields:
        return my_chemistry.density_units
    if field == "internal_energy":
        return my_chemistry.energy_units
    if field in ("x_velocity", "y_velocity", "z_velocity"):
        return my_chemistry.velocity_units
    if field == "pressure":
        return my_chemistry.pressure_units
    if field == "cooling_time":
        return my_chemistry.time_units
    return 1.0


def build_snapshot_container(source_data, opts, overrides=None):
    chem = make_chemistry(opts, overrides=overrides)
    if chem.initialize() == 0:
        raise RuntimeError("Failed to initialize diagnostic chemistry_data.")

    n_values = np.atleast_1d(array(source_data, "density")).size
    fc = FluidContainer(chem, n_values)

    for field in fc.input_fields:
        if field not in source_data:
            continue
        conversion = field_conversion_to_cgs(chem, fc, field)
        fc[field][:] = np.atleast_1d(array(source_data, field)) / conversion

    return fc


def net_cooling_rate(source_data, opts, overrides=None, floor_fields=None):
    fc = build_snapshot_container(source_data, opts, overrides=overrides)
    if floor_fields:
        for field in floor_fields:
            if field in fc:
                fc[field][:] = np.maximum(1.0e-40 * fc["density"], 1.0e-300)

    fc.calculate_cooling_rate()
    return -np.asarray(fc["cooling_rate"], dtype=float)


def cooling_delta(source_data, opts, baseline, overrides=None, floor_fields=None):
    without = net_cooling_rate(
        source_data, opts, overrides=overrides, floor_fields=floor_fields
    )
    return baseline - without


def compressional_heating_rate(source_data, opts):
    fc = build_snapshot_container(source_data, opts)
    chem = fc.chemistry_data
    gravitational_constant = (
        4.0 * np.pi * gravitational_constant_cgs *
        chem.density_units * chem.time_units**2
    )
    freefall_time_constant = np.sqrt(
        (32.0 * gravitational_constant) / (3.0 * np.pi)
    )
    du_dt = (
        (chem.Gamma - 1.0) * fc["internal_energy"] *
        freefall_time_constant * np.sqrt(fc["density"])
    )
    density_proper = fc["density"] / (
        chem.a_units * chem.a_value
    )**(3 * chem.comoving_coordinates)
    return chem.cooling_units * du_dt / density_proper


def calculate_contributions(data, opts):
    fc = build_snapshot_container(data, opts)
    if not hasattr(fc, "calculate_cooling_rate_contributions"):
        raise RuntimeError(
            "This checkout needs to be rebuilt so gracklepy exposes "
            "calculate_cooling_rate_contributions."
        )

    raw = fc.calculate_cooling_rate_contributions(opts.diagnostic_dt)
    # Raw edot terms follow the solver convention (negative = cooling). Flip the
    # sign so positive = cooling, rename the few channels whose plot key differs
    # from the C name, and drop the aggregate chemistry channels (their
    # per-reaction children are kept, so including the aggregates double-counts).
    diag = {
        f"cooling_{_CONTRIBUTION_KEY_OVERRIDES.get(name, name)}":
            -np.asarray(values, dtype=float)
        for name, values in raw.items()
        if name not in _CONTRIBUTION_AGGREGATE_FIELDS
    }
    diag["heating_compressional"] = compressional_heating_rate(data, opts)
    diag["heating_microphysical_net"] = -diag["cooling_total"]
    diag["heating_thermal_balance"] = (
        diag["heating_compressional"] + diag["heating_microphysical_net"]
    )

    primitive_fields = [
        key for key in diag
        if key.startswith("cooling_") and key not in {
            "cooling_total",
            "cooling_residual",
        }
    ]
    budget_shares, budget_cooling, budget_heating = same_sign_budget_fractions(
        diag, primitive_fields
    )
    if budget_cooling is not None:
        diag["budget_cooling_total"] = budget_cooling
        diag["budget_heating_total"] = budget_heating
        for field, values in budget_shares.items():
            diag[f"budget_fraction_{field[len('cooling_'):]}"] = values

    denom = np.where(np.abs(diag["cooling_total"]) > 1.0e-300,
                     diag["cooling_total"], np.nan)
    for key, values in list(diag.items()):
        if key == "cooling_total" or not key.startswith("cooling_"):
            continue
        diag[f"fraction_{key[len('cooling_'):]}"] = values / denom

    tracked = np.zeros_like(diag["cooling_total"])
    for key, values in diag.items():
        if key.startswith("cooling_") and key not in {
            "cooling_total",
            "cooling_residual",
        }:
            tracked = tracked + values
    diag["cooling_tracked_sum"] = tracked
    diag["fraction_tracked_sum"] = tracked / denom
    diag["fraction_residual"] = diag["cooling_residual"] / denom

    max_residual = np.nanmax(
        np.abs(diag["cooling_residual"]) /
        np.maximum(np.abs(diag["cooling_total"]), 1.0e-300)
    )
    print(f"    max |residual / total| = {max_residual:.3e}")

    return diag


def plot_temperature_sweep(results, opts):
    fig, ax = plt.subplots(figsize=(3.55, 2.8), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

    for color, (metallicity, data, _) in zip(colors, results):
        nH = hydrogen_mass_density(data) / mass_hydrogen_cgs
        label = metallicity_label(metallicity)

        ax.loglog(
            nH,
            np.maximum(array(data, "temperature"), 1e-300),
            color=color,
            linewidth=1.8,
            label=label,
        )

        tdust = array(data, "dust_temperature")
        if metallicity > 0 and np.nanmax(tdust) > 0:
            ax.loglog(
                nH,
                np.maximum(tdust, 1e-300),
                color=color,
                linestyle="--",
                linewidth=1.8,
            )

    ax.set_xlabel(r"$n_H$ [cm$^{-3}$]")
    ax.set_ylabel(r"$T$ [K]")
    format_axis(ax)
    style_handles = [
        plt.Line2D([0], [0], color="0.2", linewidth=1.8, label="gas"),
        plt.Line2D([0], [0], color="0.2", linestyle="--",
                   linewidth=1.8, label="dust"),
    ]
    style_legend = ax.legend(handles=style_handles, loc="lower right",
                             frameon=False)
    ax.add_artist(style_legend)
    ax.legend(title=r"$Z/Z_\odot$", loc="upper right", frameon=False)
    ax.text(
        0.04, 0.06,
        rf"dust model 0; ISRF={opts.isrf:g}",
        transform=ax.transAxes, ha="left", va="bottom",
    )

    save_plot(fig, f"{opts.output}_temperature_density.png")


def plot_signed_channel(ax, nH, values, label, color, min_abs=0.0):
    values = np.asarray(values, dtype=float)
    cooling = values > min_abs
    heating = values < -min_abs
    label_used = False
    if np.any(cooling):
        ax.loglog(nH[cooling], values[cooling],
                  color=color, linewidth=1.7, label=label)
        label_used = True
    if np.any(heating):
        ax.loglog(nH[heating], -values[heating],
                  color=color, linestyle="--", linewidth=1.4,
                  label=f"_{label}" if label_used else label)


def add_magnitude_sign_legend(ax):
    style_handles = [
        plt.Line2D([0], [0], color="0.25", linewidth=1.7,
                   label="solid: cooling"),
        plt.Line2D([0], [0], color="0.25", linestyle="--",
                   linewidth=1.4, label="dashed: heating"),
    ]
    legend = ax.legend(handles=style_handles, loc="upper right",
                       frameon=False)
    ax.add_artist(legend)


def symlog_linthresh(values, min_abs=0.0):
    finite_abs = []
    for value in values:
        arr = np.abs(np.asarray(value, dtype=float))
        arr = arr[np.isfinite(arr) & (arr > min_abs)]
        if arr.size:
            finite_abs.append(np.nanmin(arr))
    if not finite_abs:
        return 1.0
    return max(min(finite_abs) * 0.5, min_abs)


def plot_symlog_channel(ax, nH, values, label, color, min_abs=0.0):
    values = np.asarray(values, dtype=float)
    plotted = np.where(np.abs(values) > min_abs, values, np.nan)
    if np.any(np.isfinite(plotted)):
        ax.plot(nH, plotted, color=color, linewidth=1.5, label=label)


def selected_contribution_results(results, opts):
    selected = []
    requested = opts.contribution_metallicities or []
    for target in requested:
        for metallicity, data, diag in results:
            if diag is not None and np.isclose(metallicity, target):
                selected.append((metallicity, data, diag))
                break
    if not selected:
        for metallicity, data, diag in reversed(results):
            if diag is not None:
                selected.append((metallicity, data, diag))
                break
    return selected


def plot_contributions(results, opts, channel_order=None,
                       suffix="cooling_contributions",
                       title_note="magnitudes only"):
    if channel_order is None:
        channel_order = _CHANNEL_PLOT_ORDER
    selected = selected_contribution_results(results, opts)
    if not selected:
        return

    fig, axes = plt.subplots(
        len(selected), 1,
        figsize=(5.8, 2.25 * len(selected) + 0.75),
        sharex=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    colors = channel_colors(len(channel_order))

    for ax, (metallicity, data, diag) in zip(axes, selected):
        nH = hydrogen_mass_density(data) / mass_hydrogen_cgs
        plot_values = [
            diag[field] for field, _ in channel_order
            if field in diag
        ]
        min_abs = plot_threshold(plot_values, opts, fractions=False)
        for color, (field, label) in zip(colors, channel_order):
            if field in diag:
                plot_signed_channel(
                    ax, nH, diag[field], label, color, min_abs=min_abs
                )

        add_magnitude_sign_legend(ax)
        ax.set_ylabel(r"$|\Lambda_i|$ [erg cm$^3$ s$^{-1}$]")
        annotate_metallicity(ax, metallicity)
        format_axis(ax)

    axes[-1].set_xlabel(r"$n_H$ [cm$^{-3}$]")
    add_shared_legend(fig, axes, ncol=1)
    fname = f"{opts.output}_{suffix}.png"
    save_plot(fig, fname)


def plot_fractional_contributions(results, opts, channel_order=None,
                                  suffix="cooling_fractions",
                                  title_note="magnitudes only"):
    if channel_order is None:
        channel_order = _CHANNEL_PLOT_ORDER
    selected = selected_contribution_results(results, opts)
    if not selected:
        return

    fig, axes = plt.subplots(
        len(selected), 2,
        figsize=(8.2, 2.45 * len(selected) + 0.75),
        sharex=True, constrained_layout=True
    )
    axes = np.asarray(axes).reshape(len(selected), 2)
    colors = channel_colors(len(channel_order))

    for row, (metallicity, data, diag) in zip(axes, selected):
        ax_cooling, ax_heating = row
        nH = hydrogen_mass_density(data) / mass_hydrogen_cgs
        fields = [field for field, _ in channel_order if field in diag]
        shares, _, _ = same_sign_budget_fractions(diag, fields)
        plot_values = list(shares.values())
        min_abs = plot_threshold(plot_values, opts, fractions=True)
        for color, (field, label) in zip(colors, channel_order):
            if field in shares:
                values = np.asarray(shares[field], dtype=float)
                cooling = values > min_abs
                heating = values < -min_abs
                if np.any(cooling):
                    ax_cooling.semilogx(
                        nH[cooling], values[cooling],
                        color=color, linewidth=1.5, label=label,
                    )
                if np.any(heating):
                    ax_heating.semilogx(
                        nH[heating], -values[heating],
                        color=color, linewidth=1.5, label=label,
                    )

        ax_cooling.set_ylabel("budget fraction")
        annotate_metallicity(ax_cooling, metallicity)
        for ax in row:
            if opts.fraction_y_scale == "log":
                ymin = max(min_abs, 1.0e-12)
                ax.set_yscale("log")
                ax.set_ylim(ymin, 1.05)
            else:
                ax.set_ylim(0.0, 1.05)
            ax.axhline(1.0, color="0.65", linewidth=0.6, linestyle=":")
            format_axis(ax)

    axes[0, 0].set_title("Cooling")
    axes[0, 1].set_title("Heating")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$n_H$ [cm$^{-3}$]")
    add_shared_legend(fig, axes.ravel(), ncol=1)
    fname = f"{opts.output}_{suffix}.png"
    save_plot(fig, fname)


def plot_signed_symlog_contributions(results, opts, fractions=False,
                                     channel_order=None, suffix=None):
    if channel_order is None:
        channel_order = _CHANNEL_PLOT_ORDER
    selected = selected_contribution_results(results, opts)
    if not selected:
        return

    fig, axes = plt.subplots(
        len(selected), 1,
        figsize=(5.8, 2.25 * len(selected) + 0.75),
        sharex=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    colors = channel_colors(len(channel_order))

    for ax, (metallicity, data, diag) in zip(axes, selected):
        nH = hydrogen_mass_density(data) / mass_hydrogen_cgs
        plot_values = []
        plot_fields = []
        share_values = {}
        if fractions:
            source_fields = [field for field, _ in channel_order if field in diag]
            share_values, _, _ = same_sign_budget_fractions(diag, source_fields)
        for field, label in channel_order:
            if fractions:
                if field in share_values:
                    plot_fields.append((field, label))
                    plot_values.append(share_values[field])
            elif field in diag:
                plot_fields.append((field, label))
                plot_values.append(diag[field])
        min_abs = plot_threshold(plot_values, opts, fractions=fractions)

        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=symlog_linthresh(plot_values, min_abs))
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)

        for color, (field, label) in zip(colors, plot_fields):
            values = share_values[field] if fractions else diag[field]
            plot_symlog_channel(
                ax, nH, values, label, color, min_abs=min_abs
            )

        if fractions:
            ax.set_ylabel("same-sign budget fraction")
            ax.set_ylim(-1.05, 1.05)
            ax.axhline(1.0, color="0.65", linewidth=0.6, linestyle=":")
            ax.axhline(-1.0, color="0.65", linewidth=0.6, linestyle=":")
        else:
            ax.set_ylabel(r"$\Lambda_i$ [erg cm$^3$ s$^{-1}$]")
        annotate_metallicity(ax, metallicity)
        ax.text(
            0.04, 0.06, "positive: cooling",
            transform=ax.transAxes, ha="left", va="bottom",
        )
        format_axis(ax)

    axes[-1].set_xlabel(r"$n_H$ [cm$^{-3}$]")
    if suffix is None:
        suffix = "cooling_fraction_signed" if fractions else "cooling_signed"
    add_shared_legend(fig, axes, ncol=1)
    fname = f"{opts.output}_{suffix}.png"
    save_plot(fig, fname)


def plot_thermal_balance(results, opts):
    selected = selected_contribution_results(results, opts)
    if not selected:
        return

    fig, axes = plt.subplots(
        len(selected), 1,
        figsize=(5.8, 2.25 * len(selected) + 0.55),
        sharex=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    colors = ["0.15", "#4C78A8", "#D62728"]
    linestyles = ["-", "--", "-"]
    linewidths = [1.6, 1.6, 2.0]
    zero_crossing_field = "heating_thermal_balance"

    for ax, (metallicity, data, diag) in zip(axes, selected):
        nH = hydrogen_mass_density(data) / mass_hydrogen_cgs
        plot_values = [
            diag[field] for field, _ in _THERMAL_BALANCE_PLOT_ORDER
            if field in diag
        ]
        min_abs = plot_threshold(plot_values, opts, fractions=False)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=symlog_linthresh(plot_values, min_abs))
        ax.axhline(0.0, color="0.35", linewidth=0.8)

        for color, linestyle, linewidth, (field, label) in zip(
            colors, linestyles, linewidths, _THERMAL_BALANCE_PLOT_ORDER
        ):
            if field in diag:
                if field == zero_crossing_field:
                    plotted = np.asarray(diag[field], dtype=float)
                else:
                    plotted = np.where(
                        np.abs(diag[field]) > min_abs, diag[field], np.nan
                    )
                ax.plot(
                    nH, plotted, color=color, linestyle=linestyle,
                    linewidth=linewidth, label=label,
                )

        if zero_crossing_field in diag:
            balance = np.asarray(diag[zero_crossing_field], dtype=float)
            finite = np.isfinite(balance)
            crossing = np.flatnonzero(
                finite[:-1] & finite[1:] &
                (np.signbit(balance[:-1]) != np.signbit(balance[1:]))
            )
            for index in crossing:
                y0, y1 = balance[index], balance[index + 1]
                if y1 == y0:
                    x_cross = nH[index]
                else:
                    logx0, logx1 = np.log10(nH[index]), np.log10(nH[index + 1])
                    frac = -y0 / (y1 - y0)
                    x_cross = 10.0**(logx0 + frac * (logx1 - logx0))
                ax.axvline(
                    x_cross, color="#D62728", linewidth=0.8,
                    linestyle=":", alpha=0.75,
                )

        ax.set_ylabel(r"$\mathcal{H}$ [erg cm$^3$ s$^{-1}$]")
        annotate_metallicity(ax, metallicity)
        ax.text(
            0.04, 0.06, "positive: heating",
            transform=ax.transAxes, ha="left", va="bottom",
        )
        format_axis(ax)

    axes[-1].set_xlabel(r"$n_H$ [cm$^{-3}$]")
    add_shared_legend(fig, axes, ncol=1)
    save_plot(fig, f"{opts.output}_thermal_balance.png")


def plot_stacked_budget_panel(
    ax, nH, diag, channel_order, side, opts, legend_anchor=(1.01, 0.5)
):
    fields = [field for field, _ in channel_order if field in diag]
    labels_by_field = dict(channel_order)
    shares, _, _ = same_sign_budget_fractions(diag, fields)
    selected_fields = dominant_budget_fields(
        shares, fields, side, opts.stacked_budget_threshold
    )
    if not selected_fields:
        ax.text(
            0.5, 0.5, f"no {side} contribution",
            transform=ax.transAxes, ha="center", va="center",
        )
        return

    stacked = []
    labels = []
    for field in selected_fields:
        values = np.asarray(shares[field], dtype=float)
        if side == "cooling":
            values = np.where(values > 0, values, 0.0)
        else:
            values = np.where(values < 0, -values, 0.0)
        stacked.append(values)
        labels.append(labels_by_field[field])

    colors = channel_colors(len(selected_fields))
    ax.stackplot(
        nH, stacked, labels=labels, colors=colors,
        linewidth=0.0, alpha=0.92,
    )
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.0)
    ax.axhline(1.0, color="0.55", linewidth=0.5, linestyle=":")
    format_axis(ax)
    ax.legend(
        loc="center left", bbox_to_anchor=legend_anchor,
        frameon=False, fontsize=6.5, handlelength=1.2,
    )


def overplot_temperature_axis(ax, nH, data, show_ylabel=False, show_legend=False):
    tax = ax.twinx()
    tax.patch.set_alpha(0.0)
    gas_line, = tax.plot(
        nH, np.maximum(array(data, "temperature"), 1.0e-300),
        color="black", linewidth=1.1, alpha=0.8, label="gas T",
        zorder=10,
    )
    handles = [gas_line]
    if "dust_temperature" in data:
        tdust = array(data, "dust_temperature")
        if np.nanmax(tdust) > 0:
            dust_line, = tax.plot(
                nH, np.maximum(tdust, 1.0e-300),
                color="0.35", linestyle="--", linewidth=1.0, alpha=0.75,
                label="dust T", zorder=10,
            )
            handles.append(dust_line)
    tax.set_yscale("log")
    tax.tick_params(axis="y", which="both", direction="in")
    if show_ylabel:
        tax.set_ylabel(r"$T$ [K]")
    else:
        tax.set_yticklabels([])
    if show_legend:
        tax.legend(handles=handles, loc="upper right", frameon=False)
    return tax


def plot_dominant_budget_stack(results, opts):
    selected = selected_contribution_results(results, opts)
    if not selected:
        return

    for metallicity, data, diag in selected:
        nH = hydrogen_mass_density(data) / mass_hydrogen_cgs
        fig = plt.figure(figsize=(9.6, 8.2), constrained_layout=True)
        grid = fig.add_gridspec(3, 2, height_ratios=[1.05, 1.0, 1.0])

        ax_combined_cool = fig.add_subplot(grid[0, 0])
        ax_combined_heat = fig.add_subplot(grid[0, 1], sharex=ax_combined_cool)
        ax_channel_cool = fig.add_subplot(grid[1, 0], sharex=ax_combined_cool)
        ax_channel_heat = fig.add_subplot(grid[1, 1], sharex=ax_combined_cool)
        ax_reaction_cool = fig.add_subplot(grid[2, 0], sharex=ax_combined_cool)
        ax_reaction_heat = fig.add_subplot(grid[2, 1], sharex=ax_combined_cool)

        plot_stacked_budget_panel(
            ax_combined_cool, nH, diag, combined_stack_order(), "cooling", opts,
            legend_anchor=(1.01, 0.5),
        )
        plot_stacked_budget_panel(
            ax_combined_heat, nH, diag, combined_stack_order(), "heating", opts,
            legend_anchor=(1.34, 0.5),
        )
        plot_stacked_budget_panel(
            ax_channel_cool, nH, diag, _CHANNEL_PLOT_ORDER, "cooling", opts,
            legend_anchor=(1.01, 0.5),
        )
        plot_stacked_budget_panel(
            ax_channel_heat, nH, diag, _CHANNEL_PLOT_ORDER, "heating", opts,
            legend_anchor=(1.34, 0.5),
        )
        plot_stacked_budget_panel(
            ax_reaction_cool, nH, diag, _CHEMISTRY_DETAIL_PLOT_ORDER,
            "cooling", opts, legend_anchor=(1.01, 0.5),
        )
        plot_stacked_budget_panel(
            ax_reaction_heat, nH, diag, _CHEMISTRY_DETAIL_PLOT_ORDER,
            "heating", opts, legend_anchor=(1.34, 0.5),
        )
        overplot_temperature_axis(
            ax_combined_cool, nH, data, show_ylabel=False, show_legend=False
        )
        overplot_temperature_axis(
            ax_combined_heat, nH, data, show_ylabel=True, show_legend=True
        )
        overplot_temperature_axis(
            ax_channel_cool, nH, data, show_ylabel=False, show_legend=False
        )
        overplot_temperature_axis(
            ax_channel_heat, nH, data, show_ylabel=True, show_legend=True
        )
        overplot_temperature_axis(
            ax_reaction_cool, nH, data, show_ylabel=False, show_legend=False
        )
        overplot_temperature_axis(
            ax_reaction_heat, nH, data, show_ylabel=True, show_legend=False
        )

        ax_combined_cool.set_title("Cooling: channels + reactions")
        ax_combined_heat.set_title("Heating: channels + reactions")
        ax_channel_cool.set_title("Cooling channels")
        ax_channel_heat.set_title("Heating channels")
        ax_reaction_cool.set_title("Cooling reactions")
        ax_reaction_heat.set_title("Heating reactions")
        ax_combined_cool.set_ylabel("budget fraction")
        ax_channel_cool.set_ylabel("budget fraction")
        ax_reaction_cool.set_ylabel("budget fraction")
        ax_reaction_cool.set_xlabel(r"$n_H$ [cm$^{-3}$]")
        ax_reaction_heat.set_xlabel(r"$n_H$ [cm$^{-3}$]")
        annotate_metallicity(ax_combined_cool, metallicity)

        for ax in (
            ax_combined_cool, ax_combined_heat, ax_channel_cool,
            ax_channel_heat
        ):
            plt.setp(ax.get_xticklabels(), visible=False)

        suffix = "dominant_budget_stack"
        if len(selected) > 1:
            suffix = f"{suffix}_{metallicity_tag(metallicity)}"
        save_plot(fig, f"{opts.output}_{suffix}.png")


def print_chemistry_detail_summary(results, opts):
    selected = selected_contribution_results(results, opts)
    if not selected:
        return

    for metallicity, data, diag in selected:
        nH = hydrogen_mass_density(data) / mass_hydrogen_cgs
        rows = []
        fields = [field for field, _ in _CHEMISTRY_DETAIL_PLOT_ORDER]
        budget_shares, _, _ = same_sign_budget_fractions(diag, fields)
        for field, label in _CHEMISTRY_DETAIL_PLOT_ORDER:
            fraction_field = f"fraction_{field[len('cooling_'):]}"
            if field not in diag or fraction_field not in diag:
                continue
            fractions = np.asarray(diag[fraction_field], dtype=float)
            finite = np.isfinite(fractions)
            if not np.any(finite):
                continue
            score = np.where(finite, np.abs(fractions), -1.0)
            index = int(np.nanargmax(score))
            budget_fraction = budget_shares.get(
                field, np.zeros_like(fractions)
            )[index]
            formula = _CHEMISTRY_REACTION_FORMULAE.get(field, "")
            rows.append((
                score[index], fractions[index], budget_fraction, nH[index],
                diag[field][index], _CHEMISTRY_TEXT_LABELS.get(field, label),
                formula, _CHEMISTRY_PROCESS_DESCRIPTIONS.get(field, label)
            ))

        if not rows:
            continue
        rows.sort(reverse=True, key=lambda row: row[0])
        print(f"  Chemistry process detail, Z={metallicity:g} Zsun:")
        for (
            _, fraction, budget_fraction, density, rate, label, formula,
            description
        ) in rows[:6]:
            print(
                f"    {label:17s} net={fraction:+.3e} "
                f"budget={budget_fraction:+.3e} "
                f"rate={rate:+.3e} at nH={density:.3e} cm^-3; "
                f"reaction: {formula}; process: {description}"
            )


def save_chemistry_detail_csv(results, opts):
    fname = f"{opts.output}_chemistry_reaction_fractions.csv"
    with open(fname, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "metallicity_zsun",
            "nH_cm-3",
            "field",
            "label",
            "reaction_formula",
            "process",
            "cooling_rate_erg_cm3_s",
            "fraction_of_net_cooling",
            "same_sign_budget_fraction",
        ])
        for metallicity, data, diag in results:
            if diag is None:
                continue
            nH = hydrogen_mass_density(data) / mass_hydrogen_cgs
            fields = [field for field, _ in _CHEMISTRY_DETAIL_PLOT_ORDER]
            budget_shares, _, _ = same_sign_budget_fractions(diag, fields)
            for field, label in _CHEMISTRY_DETAIL_PLOT_ORDER:
                fraction_field = f"fraction_{field[len('cooling_'):]}"
                if field not in diag or fraction_field not in diag:
                    continue
                description = _CHEMISTRY_PROCESS_DESCRIPTIONS.get(field, label)
                formula = _CHEMISTRY_REACTION_FORMULAE.get(field, "")
                text_label = _CHEMISTRY_TEXT_LABELS.get(field, label)
                budget_fraction = budget_shares.get(
                    field, np.zeros_like(diag[field])
                )
                for density, rate, fraction, budget_fraction_value in zip(
                    nH, diag[field], diag[fraction_field], budget_fraction
                ):
                    writer.writerow([
                        metallicity,
                        density,
                        field,
                        text_label,
                        formula,
                        description,
                        rate,
                        fraction,
                        budget_fraction_value,
                    ])
    print(f"Saved {fname}")


def save_contribution_data(results, opts):
    payload = {}
    metadata = {
        "model": "dust_model=0",
        "trajectory": opts.trajectory,
        "photoelectric_heating": opts.photoelectric_heating,
        "dust_species": opts.dust_species,
        "dust_chemistry": True,
        "use_multiple_dust_temperatures": bool(opts.use_multiple_dust_temperatures),
        "grain_growth": bool(opts.grain_growth),
        "isrf_habing": opts.isrf,
        "uvb_data_file": opts.uvb_data_file,
        "diagnostic_dt": opts.diagnostic_dt,
        "note": (
            "Cooling contributions are additive terms recorded from "
            "cool1d_multi_g and rate_timestep_g. Positive means cooling; "
            "negative means heating. The residual is total minus the sum of "
            "primitive tracked additive channels. Aggregate '*_total' "
            "convenience fields are intentionally omitted from the plots and "
            "saved diagnostics, except for cooling_total as the net rate. "
            "Thermal-balance heating fields use the opposite sign convention: "
            "positive means heating. Plotted fraction figures use same-sign "
            "budget fractions: cooling terms are normalized by the sum of "
            "positive cooling terms, while heating terms are normalized by the "
            "sum of absolute heating terms and plotted negative."
        ),
        "chemistry_reactions": {
            field: _CHEMISTRY_PROCESS_DESCRIPTIONS[field]
            for field, _ in _CHEMISTRY_DETAIL_PLOT_ORDER
        },
        "chemistry_reaction_formulae": {
            field: _CHEMISTRY_REACTION_FORMULAE[field]
            for field, _ in _CHEMISTRY_DETAIL_PLOT_ORDER
        },
    }
    payload["metadata_json"] = np.asarray([json.dumps(metadata, sort_keys=True)])

    for metallicity, data, diag in results:
        tag = metallicity_tag(metallicity)
        payload[f"{tag}__nH"] = hydrogen_mass_density(data) / mass_hydrogen_cgs
        payload[f"{tag}__temperature"] = array(data, "temperature")
        if "dust_temperature" in data:
            payload[f"{tag}__dust_temperature"] = array(data, "dust_temperature")
        if diag is None:
            continue
        for field, values in diag.items():
            payload[f"{tag}__{field}"] = np.asarray(values, dtype=float)

    fname = f"{opts.output}_cooling_contributions.npz"
    np.savez(fname, **payload)
    print(f"Saved {fname}")


def main(args=None):
    opts = parse_args(args)
    if opts.use_multiple_dust_temperatures and opts.dust_species == 0:
        raise ValueError("--use-multiple-dust-temperatures requires --dust-species > 0")

    results = []

    print(f"=== {opts.output} ===")
    print(f"  dust_model          = 0")
    print(f"  dust_chemistry      = 1")
    print(f"  dust_species        = {opts.dust_species}")
    print(f"  metallicities       = {opts.metallicities}")
    print(f"  trajectory          = {opts.trajectory}")
    print(f"  initial nH          = {opts.initial_nH:.3e} cm^-3")
    if opts.trajectory == "isobaric":
        print(f"  initial T           = {opts.initial_temperature:.3e} K")
        print(f"  isobaric target     = {opts.isobaric_final_temperature:.3g} K")
    else:
        print(f"  final nH            = {opts.final_nH:.3e} cm^-3")
        print(f"  cooling target      = {opts.cooling_temperature:.3g} K")
    print(f"  ISRF                = {opts.isrf:g} Habing")
    print(f"  UV background       = {opts.uvb_data_file or 'off'}")

    for metallicity in opts.metallicities:
        print(f"Running Z={metallicity:g} Zsun...")
        data = run_case(metallicity, opts)
        diag = None
        if not opts.skip_diagnostics:
            print(f"  Calculating cooling contribution diagnostics for Z={metallicity:g}...")
            diag = calculate_contributions(data, opts)
        results.append((metallicity, data, diag))

    want = (lambda name: True) if "all" in opts.plots \
        else (lambda name: name in opts.plots)

    if want("temperature_density"):
        plot_temperature_sweep(results, opts)
    if not opts.skip_diagnostics:
        # if want("cooling_channel_rates"):
        #     plot_contributions(
        #         results, opts, suffix="cooling_channel_rates",
        #     )
        # if want("cooling_channel_fractions"):
        #     plot_fractional_contributions(
        #         results, opts, suffix="cooling_channel_fractions",
        #     )
        # if want("cooling_channel_rates_signed"):
        #     plot_signed_symlog_contributions(
        #         results, opts, fractions=False,
        #         suffix="cooling_channel_rates_signed",
        #     )
        # if want("cooling_channel_fractions_signed"):
        #     plot_signed_symlog_contributions(
        #         results, opts, fractions=True,
        #         suffix="cooling_channel_fractions_signed",
        #     )
        # if want("thermal_balance"):
        #     plot_thermal_balance(results, opts)
        if want("dominant_budget_stack"):
            plot_dominant_budget_stack(results, opts)
        # if want("chemistry_reaction_fractions"):
        #     plot_fractional_contributions(
        #         results, opts, channel_order=_CHEMISTRY_DETAIL_PLOT_ORDER,
        #         suffix="chemistry_reaction_fractions",
        #         title_note="chemistry reaction magnitudes only",
        #     )
        # if want("chemistry_reaction_fractions_signed"):
        #     plot_signed_symlog_contributions(
        #         results, opts, fractions=True,
        #         channel_order=_CHEMISTRY_DETAIL_PLOT_ORDER,
        #         suffix="chemistry_reaction_fractions_signed",
        #     )
        print_chemistry_detail_summary(results, opts)
        # save_chemistry_detail_csv(results, opts)
        # save_contribution_data(results, opts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
