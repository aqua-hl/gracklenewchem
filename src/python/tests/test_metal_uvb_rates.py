########################################################################
#
# Tests for the UV-background photo-ionization/photo-dissociation rates
# of metal species (kphCI/kphOI/kdissCO datasets in the UV background
# table, applied to C I, O I and CO when metal_chemistry > 0).
#
# Copyright (c) 2013, Enzo/Grackle Development Team.
#
# Distributed under the terms of the Enzo Public Licence.
#
# The full license is in the file LICENSE, distributed with this
# software.
########################################################################

import os
import shutil

import h5py
import numpy as np

from gracklepy import chemistry_data, FluidContainer
from gracklepy.utilities.data_path import grackle_data_dir
from gracklepy.utilities.physical_constants import (
    mass_hydrogen_cgs,
    sec_per_Myr,
    cm_per_mpc,
)

STOCK_TABLE = os.path.join(grackle_data_dir, "CloudyData_UVB=HM2012.h5")
SHIELDED_TABLE = os.path.join(
    grackle_data_dir, "CloudyData_UVB=HM2012_shielded.h5"
)

# dummy redshift-independent photo rates (in 1/s) written into copies of
# the UV background table by make_table_with_rates
K_UVB_CI = 1.0e-12
K_UVB_OI = 3.0e-13
K_UVB_CO = 2.0e-12

# dummy radiative-transfer photo rates (in 1/s)
K_RT_CI = 2.0e-12
K_RT_OI = 5.0e-13
K_RT_CO = 2.0e-12

# density scale used for the CO tests: CO couples to the rest of the
# network through fast two-body reactions, which all scale with density
# while the photo rates do not, so a low density isolates the photo terms
LOW_DENSITY = 1.0e-6

# time step (in s) for the single backward-Euler step taken by each test;
# chosen so that k*dt ~ 1e-2 (well above the comparison tolerance, but
# small enough that a single step is well converged)
DT_SECONDS = 1.0e10

# relative tolerance when comparing measured species ratios against the
# analytic single-step estimate (residual coupling through the electron
# density perturbs the ratios at the ~1e-4 level)
RATIO_ATOL = 2.0e-3

# CO drifts slightly between calls while the metal-network repartition
# settles, so its ratio comparisons get a little more headroom
CO_RATIO_ATOL = 5.0e-3

# Native and Python helpers retain slightly different historical hydrogen-mass
# constants, so physical flux comparisons need per-mille relative tolerance.
FLUX_RTOL = 2.0e-3


def make_table_with_rates(tmp_path, k_ci, k_oi, k_co=0.0, base=STOCK_TABLE):
    """Copy a stock UV background table and add constant kphCI/kphOI/
    kdissCO datasets with the given values (in 1/s)."""
    table_copy = str(tmp_path / "HM2012_with_metal_rates.h5")
    shutil.copy(base, table_copy)
    with h5py.File(table_copy, "r+") as f:
        z = f["/UVBRates/z"][:]
        f["/UVBRates/Chemistry/kphCI"] = np.full_like(z, k_ci)
        f["/UVBRates/Chemistry/kphOI"] = np.full_like(z, k_oi)
        f["/UVBRates/Chemistry/kdissCO"] = np.full_like(z, k_co)
    return table_copy


def setup_fluid_container(data_file, use_rt=False, rt_ci=0.0, rt_oi=0.0,
                          rt_co=0.0, self_shielding=0, density_factor=1.0,
                          n_vals=1):
    """Return an initialized (chemistry_data, FluidContainer) pair with
    metal_chemistry enabled, the UV background on, and (optionally) the
    radiative-transfer metal rates set (in 1/s)."""
    my_chemistry = chemistry_data()
    my_chemistry.use_grackle = 1
    my_chemistry.with_radiative_cooling = 0
    my_chemistry.primordial_chemistry = 2
    my_chemistry.metal_cooling = 1
    my_chemistry.UVbackground = 1
    my_chemistry.grackle_data_file = data_file
    my_chemistry.metal_chemistry = 1
    my_chemistry.dust_chemistry = 0
    my_chemistry.dust_species = 0
    my_chemistry.multi_metals = 0
    my_chemistry.self_shielding_method = self_shielding
    if use_rt:
        my_chemistry.use_radiative_transfer = 1
        my_chemistry.use_photo_fields = 1
        my_chemistry.use_CI_ionization = 3
        my_chemistry.use_OI_ionization = 3
        my_chemistry.use_COI_dissociation = 3
        my_chemistry.use_OHI_dissociation = 3
        my_chemistry.use_H2OI_dissociation = 3

    # Set units (proper frame, z = 0)
    my_chemistry.comoving_coordinates = 0
    my_chemistry.a_units = 1.0
    my_chemistry.a_value = 1.0
    my_chemistry.density_units = mass_hydrogen_cgs
    my_chemistry.length_units = cm_per_mpc
    my_chemistry.time_units = sec_per_Myr
    my_chemistry.set_velocity_units()

    assert my_chemistry.initialize() == 1

    fc = FluidContainer(my_chemistry, n_vals=n_vals)
    fc["density"][:] = density_factor
    fc["HI_density"][:] = 1.0e-4 * density_factor
    fc["HII_density"][:] = (0.759 - 1.0e-4) * density_factor
    fc["HeI_density"][:] = 1.0e-5 * density_factor
    fc["HeII_density"][:] = 1.0e-5 * density_factor
    fc["HeIII_density"][:] = (0.24 - 2.0e-5) * density_factor
    fc["e_density"][:] = fc["HII_density"] + \
        fc["HeII_density"] / 4.0 + fc["HeIII_density"] / 2.0
    for name in fc.density_fields:
        if fc[name][0] == 0.0:
            fc[name][:] = 1.0e-20 * density_factor
    fc["metal_density"][:] = 1.0e-3 * density_factor
    fc["CI_density"][:] = 1.0e-6 * density_factor
    fc["OI_density"][:] = 1.0e-6 * density_factor
    fc["COI_density"][:] = 1.0e-6 * density_factor
    fc["internal_energy"][:] = \
        1.0e13 / my_chemistry.get_velocity_units() ** 2 * 1.0e4
    if use_rt:
        for name in ("RT_heating_rate", "HI_ionization_rate",
                     "HeI_ionization_rate", "HeII_ionization_rate",
                     "H2I_dissociation_rate", "OHI_dissociation_rate",
                     "H2OI_dissociation_rate"):
            fc[name][:] = 0.0
        # the RT rate fields are in code units (1 / time_units)
        fc["CI_ionization_rate"][:] = rt_ci * my_chemistry.time_units
        fc["OI_ionization_rate"][:] = rt_oi * my_chemistry.time_units
        fc["COI_dissociation_rate"][:] = rt_co * my_chemistry.time_units
    return my_chemistry, fc


def evolve_ratios(data_file, species=("CI", "OI"), spin_up=False, **kwargs):
    """Take a single chemistry step of DT_SECONDS and return the
    final/initial density ratio for each requested species.

    With spin_up=True an extra unmeasured step is taken first, so that the
    injection-pathway repartition of the metal pool (which rescales the
    species on the first call) has settled before the measurement.
    """
    my_chemistry, fc = setup_fluid_container(data_file, **kwargs)
    dt = DT_SECONDS / my_chemistry.time_units
    if spin_up:
        fc.solve_chemistry(dt)
    initial = {s: fc[f"{s}_density"][0].item() for s in species}
    fc.solve_chemistry(dt)
    return tuple(
        fc[f"{s}_density"][0].item() / initial[s] for s in species
    )


def evolve_differentials(base_config, perturbed_config, species):
    """Return, for each requested species, the density change over one
    (spun-up) step in a perturbed run minus the same change in a baseline
    run. This isolates the effect of the perturbation (extra photo rates)
    from the background drift of the metal network.

    Each config is a (data_file, kwargs) pair for setup_fluid_container.
    """
    changes = []
    for data_file, kwargs in (base_config, perturbed_config):
        my_chemistry, fc = setup_fluid_container(data_file, **kwargs)
        dt = DT_SECONDS / my_chemistry.time_units
        fc.solve_chemistry(dt)
        before = {s: fc[f"{s}_density"][0].item() for s in species}
        fc.solve_chemistry(dt)
        changes.append(tuple(
            fc[f"{s}_density"][0].item() - before[s] for s in species
        ))
    return tuple(p - b for p, b in zip(changes[1], changes[0]))


def expected_ratio(k_total):
    # the solver takes a single backward-Euler step:
    #   n_final = n_initial / (1 + k * dt)
    return 1.0 / (1.0 + k_total * DT_SECONDS)


def test_metal_reaction_flux_diagnostic(tmp_path):
    """The diagnostic reports post-lookup UVB, RT, and binary fluxes."""
    metal_table = make_table_with_rates(
        tmp_path, K_UVB_CI, K_UVB_OI, k_co=K_UVB_CO
    )
    my_chemistry, fc = setup_fluid_container(
        metal_table, use_rt=True, rt_ci=K_RT_CI, rt_oi=K_RT_OI,
        rt_co=K_RT_CO,
    )
    fc["OHI_density"][:] = 2.0e-6
    fluxes = fc.calculate_metal_reaction_fluxes()

    number_unit = my_chemistry.density_units / mass_hydrogen_cgs
    n_ci = fc["CI_density"][0] * number_unit / 12.0
    n_oi = fc["OI_density"][0] * number_unit / 16.0
    n_co = fc["COI_density"][0] * number_unit / 28.0

    assert fluxes["active"].dtype == np.bool_
    assert fluxes["active"][0]
    # Since the UVB/RT summation refactor the solver applies one total photo
    # rate per channel; the *_bg diagnostic reports that applied total
    # (UV background plus any local RT), while *_rt still reports the RT
    # field contribution on its own.
    assert np.isclose(
        fluxes["kphCI_bg"][0], (K_UVB_CI + K_RT_CI) * n_ci,
        rtol=FLUX_RTOL, atol=0.0,
    )
    assert np.isclose(
        fluxes["kphOI_bg"][0], (K_UVB_OI + K_RT_OI) * n_oi,
        rtol=FLUX_RTOL, atol=0.0,
    )
    assert np.isclose(
        fluxes["kdissCO_bg"][0], (K_UVB_CO + K_RT_CO) * n_co,
        rtol=FLUX_RTOL, atol=0.0,
    )
    assert np.isclose(
        fluxes["kphCI_rt"][0], K_RT_CI * n_ci,
        rtol=FLUX_RTOL, atol=0.0,
    )
    assert np.isclose(
        fluxes["kphOI_rt"][0], K_RT_OI * n_oi,
        rtol=FLUX_RTOL, atol=0.0,
    )
    assert np.isclose(
        fluxes["kdissCO_rt"][0], K_RT_CO * n_co,
        rtol=FLUX_RTOL, atol=0.0,
    )

    # kz15 is temperature independent. This directly checks the conversion
    # from code mass densities to a physical gross event flux.
    n_hi = fc["HI_density"][0] * number_unit
    n_ch = fc["CHI_density"][0] * number_unit / 13.0
    assert np.isclose(
        fluxes["kz15"][0], 4.98e-11 * n_hi * n_ch,
        rtol=FLUX_RTOL, atol=0.0,
    )

    # The solver convention for two identical reactants is k*n**2, without a
    # combinatorial 1/2. Reconstruct its exact log-temperature interpolation.
    fc.calculate_temperature()
    log_temperature_grid = np.linspace(
        np.log(my_chemistry.TemperatureStart),
        np.log(my_chemistry.TemperatureEnd),
        my_chemistry.NumberOfTemperatureBins,
    )
    kz25_code = np.interp(
        np.log(fc["temperature"][0]), log_temperature_grid,
        np.asarray(my_chemistry.kz25),
    )
    k_unit = mass_hydrogen_cgs / (
        my_chemistry.density_units * my_chemistry.time_units
    )
    n_oh = fc["OHI_density"][0] * number_unit / 17.0
    assert np.isclose(
        fluxes["kz25"][0], kz25_code * k_unit * n_oh**2,
        rtol=FLUX_RTOL, atol=0.0,
    )

    # The activity mask follows the solver's minimum-metallicity cut, and no
    # stale values may leak through for excluded cells.
    fc["metal_density"][:] = 0.0
    inactive_fluxes = fc.calculate_metal_reaction_fluxes()
    assert not inactive_fluxes["active"][0]
    assert all(
        inactive_fluxes[channel][0] == 0.0
        for channel in inactive_fluxes if channel != "active"
    )


def test_metal_reaction_flux_vector_layout(tmp_path):
    metal_table = make_table_with_rates(tmp_path, K_UVB_CI, 0.0)
    my_chemistry, fc = setup_fluid_container(metal_table, n_vals=3)
    fc["CI_density"][:] = (1.2e-6, 2.4e-6, 3.6e-6)
    fc["metal_density"][:] = (1.0e-3, 0.0, 1.0e-3)

    fluxes = fc.calculate_metal_reaction_fluxes()
    number_unit = my_chemistry.density_units / mass_hydrogen_cgs
    expected = K_UVB_CI * fc["CI_density"] * number_unit / 12.0

    assert np.array_equal(fluxes["active"], (True, False, True))
    assert np.allclose(
        fluxes["kphCI_bg"][[0, 2]], expected[[0, 2]],
        rtol=FLUX_RTOL, atol=0.0,
    )
    assert fluxes["kphCI_bg"][1] == 0.0


def test_uvb_metal_photoionization(tmp_path):
    # a table with kphCI/kphOI datasets must ionize C I and O I at the
    # tabulated rates; normalizing by an otherwise-identical run without
    # the datasets isolates the photo-ionization terms from the rest of
    # the metal network (collisional terms, renormalization).
    metal_table = make_table_with_rates(tmp_path, K_UVB_CI, K_UVB_OI)

    ci_base, oi_base = evolve_ratios(STOCK_TABLE)
    ci_uvb, oi_uvb = evolve_ratios(metal_table)

    assert np.isclose(ci_uvb / ci_base, expected_ratio(K_UVB_CI),
                      rtol=0.0, atol=RATIO_ATOL)
    assert np.isclose(oi_uvb / oi_base, expected_ratio(K_UVB_OI),
                      rtol=0.0, atol=RATIO_ATOL)


def test_uvb_co_photodissociation(tmp_path):
    # a table with a kdissCO dataset must destroy CO at the tabulated rate
    # and hand the fragments to the same network terms as the established
    # RT dissociation path
    co_table = make_table_with_rates(tmp_path, 0.0, 0.0, k_co=K_UVB_CO)
    species = ("COI", "CI", "OI")

    base = evolve_ratios(STOCK_TABLE, species=("COI",), spin_up=True,
                         density_factor=LOW_DENSITY)
    uvb = evolve_ratios(co_table, species=("COI",), spin_up=True,
                        density_factor=LOW_DENSITY)
    assert np.isclose(uvb[0] / base[0], expected_ratio(K_UVB_CO),
                      rtol=0.0, atol=CO_RATIO_ATOL)

    d_uvb = evolve_differentials(
        (STOCK_TABLE, dict(density_factor=LOW_DENSITY)),
        (co_table, dict(density_factor=LOW_DENSITY)),
        species,
    )
    # C I receives its 12/28 mass share of the destroyed CO. (The O I gain
    # falls short of 16/28 -- a pre-existing property of the metal-network
    # element repartition, shared exactly by the RT path below.)
    assert np.isclose(d_uvb[1], -12.0 / 28.0 * d_uvb[0], rtol=0.02, atol=0.0)

    # the UV background path must be numerically equivalent to driving the
    # pre-existing RT dissociation path at the same rate
    d_rt = evolve_differentials(
        (STOCK_TABLE, dict(use_rt=True, density_factor=LOW_DENSITY)),
        (STOCK_TABLE, dict(use_rt=True, rt_co=K_UVB_CO,
                           density_factor=LOW_DENSITY)),
        species,
    )
    assert np.allclose(d_uvb, d_rt, rtol=1.0e-6, atol=0.0)


def test_rt_and_uvb_rates_are_summed(tmp_path):
    # the UV background rates must be added on top of the local
    # radiative-transfer rates, never replace them.
    # C I/O I and CO are checked in separate runs: destroying CO feeds its
    # fragments to C I, which would contaminate the C I rate measurement.
    ion_table = make_table_with_rates(tmp_path, K_UVB_CI, K_UVB_OI)

    species = ("CI", "OI")
    base = evolve_ratios(STOCK_TABLE, species=species, use_rt=True,
                         spin_up=True, density_factor=LOW_DENSITY)
    rt = evolve_ratios(STOCK_TABLE, species=species, use_rt=True,
                       rt_ci=K_RT_CI, rt_oi=K_RT_OI,
                       spin_up=True, density_factor=LOW_DENSITY)
    both = evolve_ratios(ion_table, species=species, use_rt=True,
                         rt_ci=K_RT_CI, rt_oi=K_RT_OI,
                         spin_up=True, density_factor=LOW_DENSITY)

    # RT alone still works
    for i, k_rt in enumerate((K_RT_CI, K_RT_OI)):
        assert np.isclose(rt[i] / base[i], expected_ratio(k_rt),
                          rtol=0.0, atol=RATIO_ATOL)
    # RT + UV background destroy the species at the summed rate
    for i, (k_rt, k_uvb) in enumerate(
        ((K_RT_CI, K_UVB_CI), (K_RT_OI, K_UVB_OI))
    ):
        assert np.isclose(both[i] / base[i], expected_ratio(k_rt + k_uvb),
                          rtol=0.0, atol=RATIO_ATOL)

    # same for CO
    co_table = make_table_with_rates(tmp_path, 0.0, 0.0, k_co=K_UVB_CO)
    co_base = evolve_ratios(STOCK_TABLE, species=("COI",), use_rt=True,
                            spin_up=True, density_factor=LOW_DENSITY)
    co_rt = evolve_ratios(STOCK_TABLE, species=("COI",), use_rt=True,
                          rt_co=K_RT_CO,
                          spin_up=True, density_factor=LOW_DENSITY)
    co_both = evolve_ratios(co_table, species=("COI",), use_rt=True,
                            rt_co=K_RT_CO,
                            spin_up=True, density_factor=LOW_DENSITY)
    assert np.isclose(co_rt[0] / co_base[0], expected_ratio(K_RT_CO),
                      rtol=0.0, atol=CO_RATIO_ATOL)
    assert np.isclose(co_both[0] / co_base[0],
                      expected_ratio(K_RT_CO + K_UVB_CO),
                      rtol=0.0, atol=CO_RATIO_ATOL)


def test_old_table_backward_compatible(tmp_path, capfd):
    # a UV background table without the metal-rate datasets (all tables
    # predating their introduction) must initialize successfully, warn
    # about the missing datasets, and behave exactly like a table with
    # all-zero rates
    zero_table = make_table_with_rates(tmp_path, 0.0, 0.0, k_co=0.0)

    ci_old, oi_old = evolve_ratios(STOCK_TABLE)
    captured = capfd.readouterr()
    assert "kphCI" in captured.err
    assert "kphOI" in captured.err
    assert "kdissCO" in captured.err

    ci_zero, oi_zero = evolve_ratios(zero_table)
    captured = capfd.readouterr()
    assert "kphCI" not in captured.err
    assert "kphOI" not in captured.err
    assert "kdissCO" not in captured.err

    assert np.isclose(ci_old, ci_zero, rtol=1.0e-12, atol=0.0)
    assert np.isclose(oi_old, oi_zero, rtol=1.0e-12, atol=0.0)


def test_oi_shielded_ci_optically_thin(tmp_path):
    # with self_shielding_method > 0, the O I rate (13.62 eV, same photon
    # band as H I) must be attenuated by the Rahmati H I factor while the
    # C I rate (11.26 eV, below the Lyman limit) stays optically thin.
    # The base table must carry the CrossSections datasets.
    metal_table = make_table_with_rates(
        tmp_path, K_UVB_CI, K_UVB_OI, base=SHIELDED_TABLE)

    ci_base, oi_base = evolve_ratios(SHIELDED_TABLE, self_shielding=3)
    ci_uvb, oi_uvb = evolve_ratios(metal_table, self_shielding=3)

    # C I: unaffected by shielding
    assert np.isclose(ci_uvb / ci_base, expected_ratio(K_UVB_CI),
                      rtol=0.0, atol=RATIO_ATOL)
    # O I: at nH ~ 0.76 cm^-3 the Rahmati factor is well below unity, so
    # O I must be destroyed significantly more slowly than optically thin
    oi_thin = expected_ratio(K_UVB_OI)
    measured = oi_uvb / oi_base
    suppression = (1.0 - measured) / (1.0 - oi_thin)
    assert measured > oi_thin
    assert suppression < 0.5

    # The diagnostic must expose the same post-shielding buffer used above,
    # rather than the raw table scalar.
    my_chemistry, fc = setup_fluid_container(
        metal_table, self_shielding=3
    )
    fluxes = fc.calculate_metal_reaction_fluxes()
    number_unit = my_chemistry.density_units / mass_hydrogen_cgs
    thin_ci_flux = K_UVB_CI * fc["CI_density"][0] * number_unit / 12.0
    thin_oi_flux = K_UVB_OI * fc["OI_density"][0] * number_unit / 16.0
    assert np.isclose(
        fluxes["kphCI_bg"][0], thin_ci_flux,
        rtol=FLUX_RTOL, atol=0.0,
    )
    assert fluxes["kphOI_bg"][0] < 0.5 * thin_oi_flux
