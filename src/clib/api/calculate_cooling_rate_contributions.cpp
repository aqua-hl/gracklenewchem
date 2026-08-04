/***********************************************************************
/
/ Calculate additive cooling/heating contribution fields.
/
/ This is an experimental diagnostic API for inspecting the terms that are
/ accumulated into edot by cool1d_multi_g and rate_timestep_g.
/
************************************************************************/

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#include "cool1d_multi_g.hpp"
#include "cooling_rate_contributions.hpp"
#include "field_adaptor.hpp"
#include "full_rxn_rate_buf.hpp"
#include "gas_props.hpp"
#include "grackle.h"
#include "inject_model/grain_metal_inject_pathways.hpp"
#include "inject_model/misc.hpp"
#include "internal_types.hpp"
#include "internal_units.hpp"
#include "lookup_cool_rates1d.hpp"
#include "mask.hpp"
#include "rate_timestep_g.hpp"
#include "scale_fields.hpp"
#include "self_shielding_err_check.hpp"
#include "support/index_helper.hpp"
#include "update_UVbackground_rates.hpp"
#include "utils-cpp.hpp"

#include "ceiling_species.hpp"

namespace {

using grackle::impl::CoolingContributionFieldCount;
using grackle::impl::CoolingContributionResidual;
using grackle::impl::CoolingContributionScratch;
using grackle::impl::CoolingContributionTotal;

// Built from the same GRACKLE_COOLING_CONTRIBUTION_CHANNELS X-macro as the
// enum, so the pointer at index CoolingContribution<X> is always data-><X's
// field>; the two cannot fall out of order.
std::array<gr_float*, CoolingContributionFieldCount>
contribution_output_fields(grackle_cooling_rate_contribution_data* data) {
  return {
#define GRACKLE_COOLING_CONTRIBUTION_PTR(enum_suffix, field) data->field,
      GRACKLE_COOLING_CONTRIBUTION_CHANNELS(GRACKLE_COOLING_CONTRIBUTION_PTR)
#undef GRACKLE_COOLING_CONTRIBUTION_PTR
  };
}

photo_rate_storage contribution_uvb_rates(chemistry_data* my_chemistry,
                                          chemistry_data_storage* my_rates,
                                          code_units* my_units,
                                          int* ierr) {
  photo_rate_storage my_uvb_rates{};

  my_uvb_rates.k24 = my_uvb_rates.k25 = my_uvb_rates.k26 =
      my_uvb_rates.k27 = my_uvb_rates.k28 = my_uvb_rates.k29 =
          my_uvb_rates.k30 = my_uvb_rates.k31 =
              my_uvb_rates.kphCI_bg = my_uvb_rates.kphOI_bg =
                  my_uvb_rates.kdissCO_bg =
                  my_uvb_rates.piHI = my_uvb_rates.piHeI =
                      my_uvb_rates.piHeII = my_uvb_rates.crsHI =
                          my_uvb_rates.crsHeI = my_uvb_rates.crsHeII =
                              my_uvb_rates.comp_xray =
                                  my_uvb_rates.temp_xray = 0.;

  if (my_chemistry->UVbackground == 1) {
    if (grackle::impl::update_UVbackground_rates(
            my_chemistry, my_rates, &my_uvb_rates, my_units) != GR_SUCCESS) {
      std::fprintf(stderr, "Error in update_UVbackground_rates.\n");
      *ierr = GR_FAIL;
    }
  } else {
    my_uvb_rates.k24 = my_rates->k24;
    my_uvb_rates.k25 = my_rates->k25;
    my_uvb_rates.k26 = my_rates->k26;
    my_uvb_rates.k27 = my_rates->k27;
    my_uvb_rates.k28 = my_rates->k28;
    my_uvb_rates.k29 = my_rates->k29;
    my_uvb_rates.k30 = my_rates->k30;
    my_uvb_rates.k31 = my_rates->k31;
    my_uvb_rates.kphCI_bg = my_rates->kphCI_bg;
    my_uvb_rates.kphOI_bg = my_rates->kphOI_bg;
    my_uvb_rates.kdissCO_bg = my_rates->kdissCO_bg;
    my_uvb_rates.kdissOH_bg = my_rates->kdissOH_bg;
    my_uvb_rates.kdissH2O_bg = my_rates->kdissH2O_bg;
    my_uvb_rates.piHI = my_rates->piHI;
    my_uvb_rates.piHeI = my_rates->piHeI;
    my_uvb_rates.piHeII = my_rates->piHeII;
    my_uvb_rates.crsHI = my_rates->crsHI;
    my_uvb_rates.crsHeI = my_rates->crsHeI;
    my_uvb_rates.crsHeII = my_rates->crsHeII;
    my_uvb_rates.comp_xray = my_rates->comp_xray;
    my_uvb_rates.temp_xray = my_rates->temp_xray;
  }

  return my_uvb_rates;
}

}  // namespace

extern "C" int local_calculate_cooling_rate_contributions(
    chemistry_data* my_chemistry, chemistry_data_storage* my_rates,
    code_units* my_units, grackle_field_data* my_fields, double dt_value,
    grackle_cooling_rate_contribution_data* contributions) {
  if (!my_chemistry->use_grackle) {
    return GR_SUCCESS;
  }

  if (contributions == nullptr) {
    std::fprintf(stderr, "Cooling contribution output struct is null.\n");
    return GR_FAIL;
  }

  int ierr = GR_SUCCESS;
  photo_rate_storage my_uvb_rates =
      contribution_uvb_rates(my_chemistry, my_rates, my_units, &ierr);
  if (ierr == GR_FAIL) {
    return GR_FAIL;
  }

  if (grackle::impl::self_shielding_err_check(my_chemistry, my_fields,
                                              "local_calculate_cooling_rate_"
                                              "contributions") !=
      GR_SUCCESS) {
    return GR_FAIL;
  }

  GRIMPL_NS::InternalGrUnits internalu = GRIMPL_NS::new_internalu_(my_units);

  const gr_mask_type anydust =
      ((my_chemistry->h2_on_dust > 0) || (my_chemistry->dust_chemistry > 0))
          ? MASK_TRUE
          : MASK_FALSE;
  const double dom = GRIMPL_NS::internalu_calc_dom_(internalu);
  const double chunit = GRIMPL_NS::internalu_get_chunit_(internalu);
  const double dx_cgs = my_fields->grid_dx * internalu.xbase1;
  const double c_ljeans =
      GRIMPL_NS::internalu_calc_coef_ljeans_(internalu, my_chemistry->Gamma);

  const auto output_fields = contribution_output_fields(contributions);

  if (internalu.extfields_in_comoving == 1) {
    const gr_float factor = (gr_float)(std::pow(internalu.a_value, -3));
    GRIMPL_NS::scale_fields(
        factor, my_chemistry, my_fields,
        GRIMPL_NS::get_n_inject_pathway_density_ptrs(my_rates));
  }

  GRIMPL_NS::ceiling_species(my_chemistry, my_fields);

  const GRIMPL_NS::IndexHelper idx_helper =
      GRIMPL_NS::build_index_helper_(my_fields);

  OMP_PRAGMA("omp parallel")
  {
    GRIMPL_NS::FieldAdaptorManager field_adaptor_mgr(my_fields);
    GRIMPL_NS::GrainSpeciesCollection grain_temperatures =
        GRIMPL_NS::new_GrainSpeciesCollection(my_fields->grid_dimension[0]);
    GRIMPL_NS::LnTLinInterpBuf logTlininterp_buf =
        GRIMPL_NS::new_LnTLinInterpBuf(my_fields->grid_dimension[0]);
    GRIMPL_NS::CoolHeatScratchBuf coolingheating_buf =
        GRIMPL_NS::new_CoolHeatScratchBuf(my_fields->grid_dimension[0]);
    std::vector<double> dedot(my_fields->grid_dimension[0]);
    std::vector<double> HIdot(my_fields->grid_dimension[0]);
    GRIMPL_NS::FullRxnRateBuf rxn_rate_buf =
        GRIMPL_NS::new_FullRxnRateBuf(my_fields->grid_dimension[0]);
    GRIMPL_NS::ChemHeatingRates chemheatrates_buf =
        GRIMPL_NS::new_ChemHeatingRates(my_fields->grid_dimension[0]);

    std::vector<double> tgas(my_fields->grid_dimension[0]);
    std::vector<double> mmw(my_fields->grid_dimension[0]);
    std::vector<double> nelec_times_mH(my_fields->grid_dimension[0]);
    std::vector<double> tdust(my_fields->grid_dimension[0]);
    std::vector<double> metallicity(my_fields->grid_dimension[0]);
    std::vector<double> dust2gas(my_fields->grid_dimension[0]);
    std::vector<double> rhoH(my_fields->grid_dimension[0]);
    std::vector<double> edot(my_fields->grid_dimension[0]);
    std::vector<double> alpha_continuum(my_fields->grid_dimension[0]);

    std::vector<gr_mask_type> itmask(my_fields->grid_dimension[0]);
    std::vector<gr_mask_type> itmask_metal(my_fields->grid_dimension[0]);

    std::array<std::vector<double>, CoolingContributionFieldCount>
        contribution_storage;
    CoolingContributionScratch contribution_scratch;
    for (int channel = 0; channel < CoolingContributionFieldCount; channel++) {
      contribution_storage[channel].resize(my_fields->grid_dimension[0]);
      contribution_scratch.fields[channel] =
          contribution_storage[channel].data();
    }

    OMP_PRAGMA("omp for")
    for (int t = 0; t < idx_helper.outer_ind_size; t++) {
      const GRIMPL_NS::IndexRange idx_range =
          GRIMPL_NS::make_idx_range_(t, &idx_helper);
      const int k = idx_range.k;
      const int j = idx_range.j;
      const auto species_densities =
          field_adaptor_mgr.get_species_data(idx_range);

      for (int channel = 0; channel < CoolingContributionFieldCount; channel++) {
        std::fill(contribution_storage[channel].begin() + idx_range.i_start,
                  contribution_storage[channel].begin() + idx_range.i_stop,
                  0.0);
      }

      for (int i = idx_range.i_start; i < idx_range.i_stop; i++) {
        itmask[i] = MASK_TRUE;
      }

      GRIMPL_NS::extended_gas_props(
          tgas.data(), mmw.data(), rhoH.data(), metallicity.data(),
          nelec_times_mH.data(), logTlininterp_buf, itmask.data(),
          my_chemistry, &my_rates->cloudy_primordial, my_fields, internalu,
          idx_range, nullptr);

      GRIMPL_NS::mask::adjust_from_Tfloor(
          itmask.data(), tgas.data(), idx_range, my_chemistry, my_fields);
      GRIMPL_NS::mask::fill_itmask_metal(
          itmask_metal.data(), itmask.data(), metallicity.data(), idx_range,
          my_chemistry);
      for (int i = idx_range.i_start; i < idx_range.i_stop; i++) {
        edot[i] = (itmask[i] == MASK_FALSE) ? tiny_fortran_val : 0.0;
        alpha_continuum[i] = 0.0;
      }

      my_rates->opaque_storage->dust_solver.calc_Tdust_and_chem_contrib(
          edot.data(), alpha_continuum.data(), &rxn_rate_buf,
          dust2gas.data(), tdust.data(), grain_temperatures, tgas.data(),
          rhoH.data(), nelec_times_mH.data(), metallicity.data(),
          itmask.data(), itmask_metal.data(), my_chemistry, my_rates,
          my_fields, species_densities, internalu, idx_range,
          logTlininterp_buf, &contribution_scratch);

      GRIMPL_NS::cool1d_multi_g(
          edot.data(), alpha_continuum.data(), tgas.data(), mmw.data(),
          metallicity.data(), rhoH.data(),
          nelec_times_mH.data(), itmask.data(), itmask_metal.data(),
          my_chemistry, my_rates, my_fields, species_densities,
          my_uvb_rates, internalu, idx_range, logTlininterp_buf,
          coolingheating_buf, &contribution_scratch);

      if (my_chemistry->primordial_chemistry > 0) {
        GRIMPL_NS::lookup_cool_rates1d(
            idx_range, tgas.data(), mmw.data(), dom, dx_cgs, c_ljeans,
            itmask.data(), itmask_metal.data(), my_chemistry, my_rates,
            my_fields, species_densities, my_uvb_rates, internalu,
            logTlininterp_buf, rxn_rate_buf, chemheatrates_buf);

        GRIMPL_NS::rate_timestep_g(
            dedot.data(), HIdot.data(), anydust, rhoH.data(),
            itmask.data(), edot.data(), chunit, dom, my_chemistry, my_fields,
            idx_range, chemheatrates_buf, rxn_rate_buf,
            &contribution_scratch);
      }

      for (int i = idx_range.i_start; i < idx_range.i_stop; i++) {
        contribution_scratch.fields[CoolingContributionTotal][i] = edot[i];
        contribution_scratch.fields[CoolingContributionResidual][i] =
            edot[i] -
            GRIMPL_NS::cooling_contribution_sum_active(&contribution_scratch,
                                                       i);

        for (int channel = 0; channel < CoolingContributionFieldCount;
             channel++) {
          if (output_fields[channel] != nullptr) {
            GRIMPL_NS::FortranView<gr_float***> output(
                output_fields[channel], my_fields->grid_dimension[0],
                my_fields->grid_dimension[1], my_fields->grid_dimension[2]);
            output(i, j, k) =
                (gr_float)contribution_scratch.fields[channel][i];
          }
        }
      }
    }

    GRIMPL_NS::drop_GrainSpeciesCollection(&grain_temperatures);
    GRIMPL_NS::drop_LnTLinInterpBuf(&logTlininterp_buf);
    GRIMPL_NS::drop_CoolHeatScratchBuf(&coolingheating_buf);
    GRIMPL_NS::drop_FullRxnRateBuffer(&rxn_rate_buf);
    GRIMPL_NS::drop_ChemHeatingRates(&chemheatrates_buf);
  }

  if (internalu.extfields_in_comoving == 1) {
    const gr_float factor = (gr_float)(std::pow(internalu.a_value, 3));
    GRIMPL_NS::scale_fields(
        factor, my_chemistry, my_fields,
        GRIMPL_NS::get_n_inject_pathway_density_ptrs(my_rates));
  }

  return GR_SUCCESS;
}
