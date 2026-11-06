//===----------------------------------------------------------------------===//
//
// See the LICENSE file for license and copyright information
// SPDX-License-Identifier: NCSA AND BSD-3-Clause
//
//===----------------------------------------------------------------------===//
///
/// @file
/// Defines/declares the step_rate_newton_raphson function
///
//===----------------------------------------------------------------------===//

// This file was initially generated automatically during conversion of the
// step_rate_newton_raphson function from FORTRAN to C++

#ifndef STEP_RATE_NEWTON_RAPHSON_HPP
#define STEP_RATE_NEWTON_RAPHSON_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "grackle.h"             // gr_float
#include "field_adaptor.hpp"
#include "fortran_func_decls.h"  // gr_mask_int
#include "inject_model/grain_metal_inject_pathways.hpp"
#include "internal_types.hpp"
#include "internal_units.hpp"
#include "math/adaptive_integrate.hpp"
#include "opaque_storage.hpp"
#include "support/View.hpp"
#include "support/config.hpp"
#include "support/index_helper.hpp"
#include "time_deriv_0d.hpp"
#include "utils-cpp.hpp"

namespace GRIMPL_NAMESPACE_DECL {

int inline sanity_check_() {
  GRIMPL_REQUIRE(((1+SpLUT::e)== 1), "index of e is %d", SpLUT::e);
  GRIMPL_REQUIRE(((1+SpLUT::HI)== 2), "index of HI is %d", SpLUT::HI);
  GRIMPL_REQUIRE(((1+SpLUT::HII)== 3), "index of HII is %d", SpLUT::HII);
  GRIMPL_REQUIRE(((1+SpLUT::HeI)== 4), "index of HeI is %d", SpLUT::HeI);
  GRIMPL_REQUIRE(((1+SpLUT::HeII)== 5), "index of HeII is %d", SpLUT::HeII);
  GRIMPL_REQUIRE(((1+SpLUT::HeIII)== 6), "index of HeIII is %d", SpLUT::HeIII);
  GRIMPL_REQUIRE(((1+SpLUT::HM)== 7), "index of HM is %d", SpLUT::HM);
  GRIMPL_REQUIRE(((1+SpLUT::H2I)== 8), "index of H2I is %d", SpLUT::H2I);
  GRIMPL_REQUIRE(((1+SpLUT::H2II)== 9), "index of H2II is %d", SpLUT::H2II);
  GRIMPL_REQUIRE(((1+SpLUT::DI)==10), "index of DI is %d", SpLUT::DI);
  GRIMPL_REQUIRE(((1+SpLUT::DII)==11), "index of DII is %d", SpLUT::DII);
  GRIMPL_REQUIRE(((1+SpLUT::HDI)==12), "index of HDI is %d", SpLUT::HDI);
  GRIMPL_REQUIRE(((1+SpLUT::DM)==13), "index of DM is %d", SpLUT::DM);
  GRIMPL_REQUIRE(((1+SpLUT::HDII)==14), "index of HDII is %d", SpLUT::HDII);
  GRIMPL_REQUIRE(((1+SpLUT::HeHII)==15), "index of HeHII is %d", SpLUT::HeHII);
  GRIMPL_REQUIRE(((1+SpLUT::CI)==16), "index of CI is %d", SpLUT::CI);
  GRIMPL_REQUIRE(((1+SpLUT::CII)==17), "index of CII is %d", SpLUT::CII);
  GRIMPL_REQUIRE(((1+SpLUT::COI)==18), "index of COI is %d", SpLUT::COI);
  GRIMPL_REQUIRE(((1+SpLUT::CO2I)==19), "index of CO2I is %d", SpLUT::CO2I);
  GRIMPL_REQUIRE(((1+SpLUT::OI)==20), "index of OI is %d", SpLUT::OI);
  GRIMPL_REQUIRE(((1+SpLUT::OHI)==21), "index of OHI is %d", SpLUT::OHI);
  GRIMPL_REQUIRE(((1+SpLUT::H2OI)==22), "index of H2OI is %d", SpLUT::H2OI);
  GRIMPL_REQUIRE(((1+SpLUT::O2I)==23), "index of O2I is %d", SpLUT::O2I);
  GRIMPL_REQUIRE(((1+SpLUT::SiI)==24), "index of SiI is %d", SpLUT::SiI);
  GRIMPL_REQUIRE(((1+SpLUT::SiOI)==25), "index of SiOI is %d", SpLUT::SiOI);
  GRIMPL_REQUIRE(((1+SpLUT::SiO2I)==26), "index of SiO2I is %d", SpLUT::SiO2I);
  GRIMPL_REQUIRE(((1+SpLUT::CHI)==27), "index of CHI is %d", SpLUT::CHI);
  GRIMPL_REQUIRE(((1+SpLUT::CH2I)==28), "index of CH2I is %d", SpLUT::CH2I);
  GRIMPL_REQUIRE(((1+SpLUT::COII)==29), "index of COII is %d", SpLUT::COII);
  GRIMPL_REQUIRE(((1+SpLUT::OII)==30), "index of OII is %d", SpLUT::OII);
  GRIMPL_REQUIRE(((1+SpLUT::OHII)==31), "index of OHII is %d", SpLUT::OHII);
  GRIMPL_REQUIRE(((1+SpLUT::H2OII)==32), "index of H2OII is %d", SpLUT::H2OII);
  GRIMPL_REQUIRE(((1+SpLUT::H3OII)==33), "index of H3OII is %d", SpLUT::H3OII);
  GRIMPL_REQUIRE(((1+SpLUT::O2II)==34), "index of O2II is %d", SpLUT::O2II);
  GRIMPL_REQUIRE(((1+SpLUT::Mg)==35), "index of Mg is %d", SpLUT::Mg);
  GRIMPL_REQUIRE(((1+SpLUT::Al)==36), "index of Al is %d", SpLUT::Al);
  GRIMPL_REQUIRE(((1+SpLUT::S)==37), "index of S is %d", SpLUT::S);
  GRIMPL_REQUIRE(((1+SpLUT::Fe)==38), "index of Fe is %d", SpLUT::Fe);
  GRIMPL_REQUIRE(((1+SpLUT::MgSiO3_dust)==39), "index of MgSiO3 is %d", SpLUT::MgSiO3_dust);
  GRIMPL_REQUIRE(((1+SpLUT::AC_dust)==40), "index of AC is %d", SpLUT::AC_dust);
  GRIMPL_REQUIRE(((1+SpLUT::SiM_dust)==41), "index of SiM is %d", SpLUT::SiM_dust);
  GRIMPL_REQUIRE(((1+SpLUT::FeM_dust)==42), "index of FeM is %d", SpLUT::FeM_dust);
  GRIMPL_REQUIRE(((1+SpLUT::Mg2SiO4_dust)==43), "index of Mg2SiO4 is %d", SpLUT::Mg2SiO4_dust);
  GRIMPL_REQUIRE(((1+SpLUT::Fe3O4_dust)==44), "index of Fe3O4 is %d", SpLUT::Fe3O4_dust);
  GRIMPL_REQUIRE(((1+SpLUT::SiO2_dust)==45), "index of SiO2_dust is %d", SpLUT::SiO2_dust);
  GRIMPL_REQUIRE(((1+SpLUT::MgO_dust)==46), "index of MgO is %d", SpLUT::MgO_dust);
  GRIMPL_REQUIRE(((1+SpLUT::FeS_dust)==47), "index of FeS is %d", SpLUT::FeS_dust);
  GRIMPL_REQUIRE(((1+SpLUT::Al2O3_dust)==48), "index of Al2O3 is %d", SpLUT::Al2O3_dust);
  GRIMPL_REQUIRE(((1+SpLUT::ref_org_dust)==49), "index of reforg is %d", SpLUT::ref_org_dust);
  GRIMPL_REQUIRE(((1+SpLUT::vol_org_dust)==50), "index of volorg is %d", SpLUT::vol_org_dust);
  GRIMPL_REQUIRE(((1+SpLUT::H2O_ice_dust)==51), "index of H2Oice is %d", SpLUT::H2O_ice_dust);

  //GRIMPL_REQUIRE(false, "early abort");

  return 1;
}

/// acts as a temporary adaptor layer for time_deriv_0d::derivatives:
///
/// @note
/// Some additional context
/// - during transcription, time_deriv_0d::derivatives (formerly called
///   lookup_cool_rates0d) needed to be adapted to make use of data structures
///   already adopted in other parts of the transcribed code
///   - (recall that these data structures were introduced to remove HUNDREDS
///     of function arguments and HUNDREDS of local variables)
/// - since the newton-raphson solver already does a bunch of "translation"
///   between data representations, it should directly map into these preferred
///   data structures
/// - as a short-term (and extremely temporary) solution, we this function
///   to copy between formats & wrap the time_deriv_0d::derivatives function.
///   This allows to finish polishing time_deriv_0d::derivatives without
///   refactoring the newton-raphson scheme)
inline void wrapped_calc_derivatives(
  double dt,
  const double* dsp,
  double* dspdot,
  grackle::impl::time_deriv_0d::ContextPack& pack,
  std::vector<gr_float>& rhosp_grflt,         // <- preallocated temp buffer
  grackle::impl::SpeciesCollection& rhosp_dot // <- preallocated temp buffer
) {

  gr_float local_eint[1];
  double eint_dot_specific[1];

  // 1. copy values out of dsp into rhosp_grflt and local_eint
  //    - at some point, we need to add some logic to deal with edge cases
  //      that can arise when `sizeof(gr_float) < sizeof(double)
  for (int sp_idx = 0; sp_idx < MAX_EVOLVED_SPECIES_FIELDS; sp_idx++) {
    rhosp_grflt[sp_idx] = static_cast<gr_float>(dsp[sp_idx]);
  }
  local_eint[0] = static_cast<gr_float>(dsp[MAX_EVOLVED_SPECIES_FIELDS]);

  // 2. call the wrapped function
  grackle::impl::time_deriv_0d::derivatives(
    dt, rhosp_grflt.data(), rhosp_dot, local_eint, eint_dot_specific, pack
  );

  // 3. copy the computed derivatives into dspdot
  for (int sp_idx = 0; sp_idx < MAX_EVOLVED_SPECIES_FIELDS; sp_idx++) {
    dspdot[sp_idx] = rhosp_dot.data[sp_idx][0];
  }
  dspdot[MAX_EVOLVED_SPECIES_FIELDS] = eint_dot_specific[0];
}


/// An alternative to step_rate_g for evolving the species rate equations that
/// employs the Newton-Raphson scheme rather than Gauss-Seidel
///
/// Whereas the Gauss-Seidel scheme always operator-splits evolution of the
/// energy equation, this function internally evolves the energy equation
///
/// Each selected cell advances one accepted substep, whose duration is written
/// to dtit. Scratch buffers are refreshed at the accepted state.
///
/// @param minimum_subcycle_dt The smallest interval that advances each cell's
/// accumulated elapsed time in floating-point arithmetic.
/// @return false on bounded integration failure. The failing cell retains its
/// original fields; earlier accepted cells in this row remain advanced.
inline bool step_rate_newton_raphson(
    IndexRange idx_range, double dom, double chunit, double dx_cgs,
    double c_ljeans, double* dtit, const double* minimum_subcycle_dt,
    double* next_nr_dt,
    double* tgas, double* tdust,
    double* metallicity, double* dust2gas, double* rhoH, double* mmw,
    double* nelec_times_mH, double* edot, gr_mask_type anydust,
    const gr_mask_type* itmask_nr, const gr_mask_type* itmask_metal,
    const int* imp_eng, chemistry_data* my_chemistry,
    chemistry_data_storage* my_rates, grackle_field_data* my_fields,
    photo_rate_storage my_uvb_rates, InternalGrUnits internalu,
    grackle::impl::GrainSpeciesCollection grain_temperatures,
    grackle::impl::LnTLinInterpBuf logTlininterp_buf,
    grackle::impl::CoolHeatScratchBuf coolingheating_buf,
    grackle::impl::ChemHeatingRates chemheatrates_buf) {
  // shorten `grackle::impl::time_deriv_0d` to `t_deriv` within this function
  namespace t_deriv = ::grackle::impl::time_deriv_0d;

  // The Gauss-Seidel fast path must not allocate Newton workspaces.
  bool any_nr_cells = false;
  for (int i = idx_range.i_start; i < idx_range.i_stop; ++i) {
    any_nr_cells = any_nr_cells || (itmask_nr[i] != MASK_FALSE);
  }
  if (!any_nr_cells) return true;

  // the following is a quick sanity check
  // -> we can remove this once we finish transcribing lookup_cool_rate0d
  sanity_check_();

  // Density, energy and velocity fields fields

  FortranView<gr_float***> d(my_fields->density, my_fields->grid_dimension[0], my_fields->grid_dimension[1], my_fields->grid_dimension[2]);
  FortranView<gr_float***> e(my_fields->internal_energy, my_fields->grid_dimension[0], my_fields->grid_dimension[1], my_fields->grid_dimension[2]);

  // Local variable
  int nsp, isp, id;

  // flag for if Gen Chiaki's dust model is enabled
  // -> to simplify bookkeeping, we ignore whether the grain_growth flag was
  //    enabled, (if it's not enabled, the relevant reaction rates are
  //    explicitly set to 0)
  const bool chiaki_model_dust_evolution = my_chemistry->dust_chemistry == 2;

  // the following specifies the historical 1-based index that we would use to
  // hold energy
  const int i_eng = MAX_EVOLVED_SPECIES_FIELDS + 1;
  // There may be an argument for allocating the following at a higher
  // level function, but we will leave that for after transcription
  std::vector<double> dsp(i_eng);
  std::vector<double> reduced_dsp(i_eng);
  std::vector<double> full_dsp_buf(i_eng);
  std::vector<double> full_dspdot_buf(i_eng);
  std::vector<double> full_dspdot_buf1(i_eng);
  std::vector<double> absolute_tolerance(i_eng);

  // Error control is independent of the nonlinear-solve tolerance. A
  // successful step uses two backward-Euler half steps and advances only
  // the accepted interval, with bounded retries and thermal stability guards.
  constexpr double relative_tolerance = 1.e-3;
  // The RHS evaluates gr_float fields, even though Newton iterates in double.
  // Its convergence tolerance must therefore accommodate field roundoff. It
  // only needs to be small next to the 1e-3 error tolerance.
  constexpr double nonlinear_tolerance = std::max(
    1.e-6, 32.*std::numeric_limits<gr_float>::epsilon());
  // Resolve finite differences above the precision of the RHS fields.
  constexpr double eps = (std::numeric_limits<gr_float>::digits <
                          std::numeric_limits<double>::digits) ? 1.e-2 : 1.e-4;
  integrate::AdaptiveBackwardEuler ode_solver(
    i_eng, relative_tolerance, 0.1, 40, nonlinear_tolerance, eps);

  std::vector<double> ode_scratch_buf(ode_solver.num_scratch_buf_elements());

  // (In the future, we may want to reconsider when/how we allocate
  // the following 3 variables)
  std::vector<int> idsp;

  // allocate some scratch buffers. In the future, when we refactor to support
  // calculations of the time derivatives of multiple values at once, we should
  // forward the buffers passed into this routine as arguments rather than
  // allocating separate buffers
  t_deriv::MainScratchBuf main_scratch_buf =
    t_deriv::new_MainScratchBuf(
        grackle::impl::GrainMetalInjectPathways_get_n_log10Tdust_vals(
            my_rates->opaque_storage->inject_pathway_props
        )
    );

  // collect args that are forwarded to the time-derivative calculation and are
  // effectively frozen between various calls
  t_deriv::FrozenSimpleArgs frozen_tderiv_args = {
    dom, chunit, dx_cgs, c_ljeans, anydust, my_chemistry,
    my_rates, my_uvb_rates, internalu
  };

  // partially initialize the struct we will use for the time derivative calc
  t_deriv::ContextPack pack = t_deriv::new_ContextPack(
    frozen_tderiv_args, main_scratch_buf
  );

  std::vector<gr_float> rhosp_grflt(MAX_EVOLVED_SPECIES_FIELDS);
  grackle::impl::SpeciesCollection rhosp_dot =
    grackle::impl::new_SpeciesCollection(1);

  auto release_scratch = [&]() {
    t_deriv::drop_MainScratchBuf(&main_scratch_buf);
    grackle::impl::drop_SpeciesCollection(&rhosp_dot);
  };

  // The following was extracted from another subroutine
  for (int i = idx_range.i_start; i < idx_range.i_stop; i++) {
    const int j = idx_range.j;
    const int k = idx_range.k;

    // remap the 3D index to a 1D index
    const int field_idx1d = layoutleft_3D_index_to_1D_(
        my_fields->grid_dimension, i, j, k
    );

    if (itmask_nr[i] != MASK_FALSE)  {

      // Every trial starts from the original state. In particular, a
      // rejected Newton step must not leave a speculative energy update.
      std::fill(dsp.begin(), dsp.end(), 0.0);
      const double initial_energy = e(i,j,k);
      const double initial_energy_derivative = edot[i] / d(i,j,k);

      // initialize arrays
      nsp = (my_chemistry->primordial_chemistry > 0) ? 6 : 0;
      if (my_chemistry->primordial_chemistry > 1) { nsp = nsp + 3; }
      if (my_chemistry->primordial_chemistry > 2) { nsp = nsp + 3; }
      if (my_chemistry->primordial_chemistry > 3) { nsp = nsp + 3; }
      if (itmask_metal[i] != MASK_FALSE)  {
        if (my_chemistry->metal_chemistry == 1)  {
          nsp = nsp + 19;
          if (chiaki_model_dust_evolution)  {
            if (my_chemistry->dust_species > 0) { nsp = nsp + 1; }
            if (my_chemistry->dust_species > 1) { nsp = nsp + 3; }
          }
        }
        if (chiaki_model_dust_evolution)  {
          if (my_chemistry->dust_species > 0) { nsp = nsp + 2; }
          if (my_chemistry->dust_species > 1) { nsp = nsp + 8; }
          if (my_chemistry->dust_species > 2) { nsp = nsp + 3; }
        }
      }
      nsp = nsp + imp_eng[i];
      if (nsp == 0) continue; // no chemistry and no energy evolution
      idsp.resize(nsp);

      // copy values into dsp from my_fields
      // -> in the future, we will be able write the next ~80 lines as
      //    a concise 4-line for-loop when we start to use the
      //    SpeciesMultiView type

      if ( my_chemistry->primordial_chemistry > 0 )  {
        dsp[SpLUT::e] = my_fields->e_density[field_idx1d];
        dsp[SpLUT::HI] = my_fields->HI_density[field_idx1d];
        dsp[SpLUT::HII] = my_fields->HII_density[field_idx1d];
        dsp[SpLUT::HeI] = my_fields->HeI_density[field_idx1d];
        dsp[SpLUT::HeII] = my_fields->HeII_density[field_idx1d];
        dsp[SpLUT::HeIII] = my_fields->HeIII_density[field_idx1d];
      }
      if ( my_chemistry->primordial_chemistry > 1 )  {
        dsp[SpLUT::HM] = my_fields->HM_density[field_idx1d];
        dsp[SpLUT::H2I] = my_fields->H2I_density[field_idx1d];
        dsp[SpLUT::H2II] = my_fields->H2II_density[field_idx1d];
      }
      if ( my_chemistry->primordial_chemistry > 2 )  {
        dsp[SpLUT::DI] = my_fields->DI_density[field_idx1d];
        dsp[SpLUT::DII] = my_fields->DII_density[field_idx1d];
        dsp[SpLUT::HDI] = my_fields->HDI_density[field_idx1d];
      }
      if ( my_chemistry->primordial_chemistry > 3 )  {
        dsp[SpLUT::DM] = my_fields->DM_density[field_idx1d];
        dsp[SpLUT::HDII] = my_fields->HDII_density[field_idx1d];
        dsp[SpLUT::HeHII] = my_fields->HeHII_density[field_idx1d];
      }
      if ( itmask_metal[i] != MASK_FALSE )  {
        if ( my_chemistry->metal_chemistry == 1 )  {
          dsp[SpLUT::CI] = my_fields->CI_density[field_idx1d];
          dsp[SpLUT::CII] = my_fields->CII_density[field_idx1d];
          dsp[SpLUT::COI] = my_fields->COI_density[field_idx1d];
          dsp[SpLUT::CO2I] = my_fields->CO2I_density[field_idx1d];
          dsp[SpLUT::OI] = my_fields->OI_density[field_idx1d];
          dsp[SpLUT::OHI] = my_fields->OHI_density[field_idx1d];
          dsp[SpLUT::H2OI] = my_fields->H2OI_density[field_idx1d];
          dsp[SpLUT::O2I] = my_fields->O2I_density[field_idx1d];
          dsp[SpLUT::SiI] = my_fields->SiI_density[field_idx1d];
          dsp[SpLUT::SiOI] = my_fields->SiOI_density[field_idx1d];
          dsp[SpLUT::SiO2I] = my_fields->SiO2I_density[field_idx1d];
          dsp[SpLUT::CHI] = my_fields->CHI_density[field_idx1d];
          dsp[SpLUT::CH2I] = my_fields->CH2I_density[field_idx1d];
          dsp[SpLUT::COII] = my_fields->COII_density[field_idx1d];
          dsp[SpLUT::OII] = my_fields->OII_density[field_idx1d];
          dsp[SpLUT::OHII] = my_fields->OHII_density[field_idx1d];
          dsp[SpLUT::H2OII] = my_fields->H2OII_density[field_idx1d];
          dsp[SpLUT::H3OII] = my_fields->H3OII_density[field_idx1d];
          dsp[SpLUT::O2II] = my_fields->O2II_density[field_idx1d];
          if (chiaki_model_dust_evolution) {
            if (my_chemistry->dust_species > 0)  {
              dsp[SpLUT::Mg] = my_fields->Mg_density[field_idx1d];
            }
            if (my_chemistry->dust_species > 1)  {
              dsp[SpLUT::Al] = my_fields->Al_density[field_idx1d];
              dsp[SpLUT::S] = my_fields->S_density[field_idx1d];
              dsp[SpLUT::Fe] = my_fields->Fe_density[field_idx1d];
            }
          }
        }
        if (chiaki_model_dust_evolution) {
          if (my_chemistry->dust_species > 0)  {
            dsp[SpLUT::MgSiO3_dust] = my_fields->MgSiO3_dust_density[field_idx1d];
            dsp[SpLUT::AC_dust] = my_fields->AC_dust_density[field_idx1d];
          }
          if (my_chemistry->dust_species > 1)  {
            dsp[SpLUT::SiM_dust] = my_fields->SiM_dust_density[field_idx1d];
            dsp[SpLUT::FeM_dust] = my_fields->FeM_dust_density[field_idx1d];
            dsp[SpLUT::Mg2SiO4_dust] = my_fields->Mg2SiO4_dust_density[field_idx1d];
            dsp[SpLUT::Fe3O4_dust] = my_fields->Fe3O4_dust_density[field_idx1d];
            dsp[SpLUT::SiO2_dust] = my_fields->SiO2_dust_density[field_idx1d];
            dsp[SpLUT::MgO_dust] = my_fields->MgO_dust_density[field_idx1d];
            dsp[SpLUT::FeS_dust] = my_fields->FeS_dust_density[field_idx1d];
            dsp[SpLUT::Al2O3_dust] = my_fields->Al2O3_dust_density[field_idx1d];
          }
          if (my_chemistry->dust_species > 2)  {
            dsp[SpLUT::ref_org_dust] = my_fields->ref_org_dust_density[field_idx1d];
            dsp[SpLUT::vol_org_dust] = my_fields->vol_org_dust_density[field_idx1d];
            dsp[SpLUT::H2O_ice_dust] = my_fields->H2O_ice_dust_density[field_idx1d];
          }
        }
      }
      dsp[i_eng-1] = e(i,j,k);

      // here, we fill in the idsp array
      // -> the idsp array is used for mapping between the compressed vector
      //    form (that only has nsp elements) and dsp (that always has
      //    `MAX_EVOLVED_SPECIES_FIELDS + 1` entries)
      // -> in the future, we should construct this array first (before doing
      //    anything else):
      //    -> it gives us the value of nsp for free!
      //    -> using this array with SpeciesMultiView lets us cut out
      //       (duplicated) logic when we copy values from my_fields to dsp and
      //       the reverse (we can cut ~70 lines in each place)
      id = 0;
      if ( my_chemistry->primordial_chemistry > 0 )  {
        idsp[id++] = SpLUT::e;
        idsp[id++] = SpLUT::HI;
        idsp[id++] = SpLUT::HII;
        idsp[id++] = SpLUT::HeI;
        idsp[id++] = SpLUT::HeII;
        idsp[id++] = SpLUT::HeIII;
      }
      if ( my_chemistry->primordial_chemistry > 1 )  {
        idsp[id++] = SpLUT::HM;
        idsp[id++] = SpLUT::H2I;
        idsp[id++] = SpLUT::H2II;
      }
      if ( my_chemistry->primordial_chemistry > 2 )  {
        idsp[id++] = SpLUT::DI;
        idsp[id++] = SpLUT::DII;
        idsp[id++] = SpLUT::HDI;
      }
      if ( my_chemistry->primordial_chemistry > 3 )  {
        idsp[id++] = SpLUT::DM;
        idsp[id++] = SpLUT::HDII;
        idsp[id++] = SpLUT::HeHII;
      }
      if ( itmask_metal[i] != MASK_FALSE )  {
        if ( my_chemistry->metal_chemistry == 1 )  {
          idsp[id++] = SpLUT::CI;
          idsp[id++] = SpLUT::CII;
          idsp[id++] = SpLUT::COI;
          idsp[id++] = SpLUT::CO2I;
          idsp[id++] = SpLUT::OI;
          idsp[id++] = SpLUT::OHI;
          idsp[id++] = SpLUT::H2OI;
          idsp[id++] = SpLUT::O2I;
          idsp[id++] = SpLUT::SiI;
          idsp[id++] = SpLUT::SiOI;
          idsp[id++] = SpLUT::SiO2I;
          idsp[id++] = SpLUT::CHI;
          idsp[id++] = SpLUT::CH2I;
          idsp[id++] = SpLUT::COII;
          idsp[id++] = SpLUT::OII;
          idsp[id++] = SpLUT::OHII;
          idsp[id++] = SpLUT::H2OII;
          idsp[id++] = SpLUT::H3OII;
          idsp[id++] = SpLUT::O2II;
          if (chiaki_model_dust_evolution)  {
            if (my_chemistry->dust_species > 0)  {
              idsp[id++] = SpLUT::Mg;
            }
            if (my_chemistry->dust_species > 1)  {
              idsp[id++] = SpLUT::Al;
              idsp[id++] = SpLUT::S;
              idsp[id++] = SpLUT::Fe;
            }
          }
        }
        if (chiaki_model_dust_evolution)  {
          if (my_chemistry->dust_species > 0)  {
            idsp[id++] = SpLUT::MgSiO3_dust;
            idsp[id++] = SpLUT::AC_dust;
          }
          if (my_chemistry->dust_species > 1)  {
            idsp[id++] = SpLUT::SiM_dust;
            idsp[id++] = SpLUT::FeM_dust;
            idsp[id++] = SpLUT::Mg2SiO4_dust;
            idsp[id++] = SpLUT::Fe3O4_dust;
            idsp[id++] = SpLUT::SiO2_dust;
            idsp[id++] = SpLUT::MgO_dust;
            idsp[id++] = SpLUT::FeS_dust;
            idsp[id++] = SpLUT::Al2O3_dust;
          }
          if (my_chemistry->dust_species > 2)  {
            idsp[id++] = SpLUT::ref_org_dust;
            idsp[id++] = SpLUT::vol_org_dust;
            idsp[id++] = SpLUT::H2O_ice_dust;
          }
        }
      }

      if ( imp_eng[i] ==1 )  {
        idsp[id++] = i_eng-1;
      }

      GRIMPL_REQUIRE((id == nsp), "SANITY CHECK FAILED");

      // The legacy derivative API still accepts a timestep, although the
      // current rates are functions of state alone. Keep a reference so this
      // compatibility argument reflects the accepted duration after adaptation.
      double& dt_FIXME = dtit[i];

      // configure pack for use during the current index
      t_deriv::configure_ContextPack(
        &pack, my_fields, field_idx1d, itmask_metal[i], imp_eng[i]
      );

      // copy values at the current index from the external scratch buffers
      // passed into the current function as args into the corresponding
      // buffers managed by pack
      // -> see the called function's docstring for an explanation as to why
      //    this is not currently combined with the prior function call
      // -> in a lot of cases, the need to copy values is dictated by the value
      //    of imp_eng
      // -> in the future, we should really refactor so that this isn't
      //    necessary
      t_deriv::scratchbufs_copy_into_pack(
        i, &pack, tgas, tdust, metallicity, dust2gas, rhoH, mmw, nelec_times_mH,
        edot, grain_temperatures, logTlininterp_buf,
        coolingheating_buf, chemheatrates_buf
      );

      // we want to integrate a system of coupled stiff ODEs
      //
      // From a purely mathematical perspective:
      // - let `y(t)` specifies the (vector-valued) state of this system at a
      //   time `t`
      // - let `dy/dt` be equivalent to a (vector-valued) function `f(y)`.
      // - If the current time is t₀, then we want to compute y(t₀ + Δt) from
      //   y(t₀) for a timestep Δt.
      // - To do this, we must define a callback function that computes
      //   f(y) and the Jacobian matrix for f(y).
      //
      // From a more practical perspective:
      // - the `dsp` buffer holds all variables actively evolved by the current
      //   grackle configuration.
      // - for the current timestep, only a subset of the variables may be
      //    evolved. The selection of evolved variables is tracked by the
      //   `idsp` buffer.
      // - this reduced set of variables corresponds to `y(t)`

      // initialize full_dsp_buf with copies of all entries in dsp
      // -> this buffer is used internally by the callback function. It has
      //    space for every variable that the current grackle configuration
      //    evolves.
      // -> the basic idea, is that the callback will overwrite the elements
      //    corresponding to the subset of variables being evolved in the
      //    current timestep, won't mutate other elements (if any), and will
      //    pass this to the buffer that does the heavy lifting of computing
      //    derivatives.
      std::memcpy(full_dsp_buf.data(), dsp.data(), sizeof(double)*i_eng);

      // actually define the callback function:
      // - reduced_dsp corresponds to the current value of the variable y
      // - reduced_dspdot is an output buffer for holding f(y)
      // - jacobian is an output buffer for holding the Jacobian of f
      //
      // Important: index `i` in these buffers correspond to the `i`th variable
      //            being evolved in the current timestep. The corresponding
      //            entry in `dsp` is at the index `idsp[isp]`.
      const auto calc_f_and_jacobian = [&](
          const double* reduced_dsp, double* reduced_dspdot,
          View<double**>& jacobian) -> void {

        // copy values out of reduced_dsp into full_dsp_buf
        for (int isp = 0; isp < nsp; isp++) {
          full_dsp_buf[idsp[isp]] = reduced_dsp[isp];
        }

        wrapped_calc_derivatives(dt_FIXME, full_dsp_buf.data(),
                                 full_dspdot_buf.data(), pack,
                                 rhosp_grflt, rhosp_dot);

        // copy the reduced set of values that we care about from
        // full_dspdot_buf into dspdot
        for (int isp = 0; isp < nsp; isp++) {
          reduced_dspdot[isp] = full_dspdot_buf[idsp[isp]];
        }

        // fill in the jacobian matrix for the time derivative
        // -> to accomplish this, we use finite differences to estimate
        //    partial derivative for each evolved variable (i.e. the species
        //    densities and possibly the total energy)
        for (int jsp = 0; jsp < nsp; jsp++) {
          // Zero and trace abundances need a nonzero, density-scaled
          // perturbation. Use the actual representable difference below.
          const bool energy_column = (idsp[jsp] == i_eng-1);
          const double scale_floor = energy_column
            ? std::max(std::fabs(initial_energy)*1.e-12, double{tiny8})
            : std::max(double{d(i,j,k)}*1.e-10, double{tiny8});
          const double base = static_cast<double>(
            static_cast<gr_float>(reduced_dsp[jsp]));
          double perturbed = base +
            eps*std::max(std::fabs(base), scale_floor);
          perturbed = static_cast<double>(static_cast<gr_float>(perturbed));
          if (perturbed == base) {
            perturbed = std::nextafter(
              static_cast<gr_float>(base),
              std::numeric_limits<gr_float>::infinity());
          }
          const double dspj = perturbed - base;
          for (int isp = 0; isp < nsp; isp++) {
            if (isp == jsp) {
              full_dsp_buf[idsp[isp]] = perturbed;
            } else {
              full_dsp_buf[idsp[isp]] = reduced_dsp[isp];
            }
          }

          wrapped_calc_derivatives(dt_FIXME, full_dsp_buf.data(),
                                   full_dspdot_buf1.data(),
                                   pack, rhosp_grflt, rhosp_dot);

          for (int isp = 0; isp < nsp; isp++) {
            double tderiv = full_dspdot_buf[idsp[isp]];
            double offset_tderiv = full_dspdot_buf1[idsp[isp]];
            jacobian(isp, jsp) = (offset_tderiv - tderiv) / dspj;
          }
        }
      };

      // f(y) alone: one RHS evaluation (Newton reuses the Jacobian)
      const auto calc_f = [&](const double* reduced_dsp_in,
                              double* reduced_dspdot) -> void {
        for (int isp = 0; isp < nsp; isp++) {
          full_dsp_buf[idsp[isp]] = reduced_dsp_in[isp];
        }
        wrapped_calc_derivatives(dt_FIXME, full_dsp_buf.data(),
                                 full_dspdot_buf.data(), pack,
                                 rhosp_grflt, rhosp_dot);
        for (int isp = 0; isp < nsp; isp++) {
          reduced_dspdot[isp] = full_dspdot_buf[idsp[isp]];
        }
      };

      for (int isp = 0; isp < nsp; ++isp) {
        reduced_dsp[isp] = dsp[idsp[isp]];
        // The helper's fractional-change scale is atol / relative_tolerance.
        absolute_tolerance[isp] = (idsp[isp] == i_eng-1)
          ? relative_tolerance*std::max(std::fabs(initial_energy)*1.e-12,
                                        double{tiny8})
          : relative_tolerance*std::max(double{d(i,j,k)}*1.e-10,
                                        double{tiny8});
      }
      const int energy_index = (imp_eng[i] == 1) ? nsp-1 : -1;
      const auto result = ode_solver.step(
        reduced_dsp.data(), dtit[i], minimum_subcycle_dt[i],
        calc_f_and_jacobian, calc_f, nsp, energy_index,
        absolute_tolerance.data(),
        ode_scratch_buf.data(), d(i,j,k));
      if (!result.converged() || !std::isfinite(result.accepted_dt) ||
          result.accepted_dt < minimum_subcycle_dt[i]) {
        release_scratch();
        return false;
      }
      dtit[i] = result.accepted_dt;

      // copy values from reduced_dsp back to dsp
      for (int isp = 0; isp < nsp; isp++) {
        dsp[idsp[isp]] = reduced_dsp[isp];
      } 


      // Preserve the explicitly split mode without advancing energy by a
      // rejected trial's longer interval. The hybrid normally couples energy.
      if ((imp_eng[i] == 0) && (my_chemistry->with_radiative_cooling == 1)) {
        dsp[i_eng-1] = initial_energy + initial_energy_derivative*dtit[i];
      }
      const gr_float accepted_energy = static_cast<gr_float>(dsp[i_eng-1]);
      if (!std::isfinite(accepted_energy) || accepted_energy <= 0.) {
        release_scratch();
        return false;
      }

      for (int isp = 0; isp < nsp; ++isp) {
        const double value = dsp[idsp[isp]];
        if (!std::isfinite(value) || value < 0. ||
            !std::isfinite(static_cast<gr_float>(value))) {
          release_scratch();
          return false;
        }
      }

      // Finite-difference probes leave scratch buffers at a perturbed trial.
      // Refresh them at the accepted state before returning them to the caller.
      wrapped_calc_derivatives(dtit[i], dsp.data(), full_dspdot_buf.data(),
                               pack, rhosp_grflt, rhosp_dot);
      if (!std::isfinite(pack.other_scratch_buf.tgas[0]) ||
          pack.other_scratch_buf.tgas[0] <= 0. ||
          !std::isfinite(pack.other_scratch_buf.edot[0])) {
        release_scratch();
        return false;
      }
      t_deriv::scratchbufs_copy_from_pack(
        i, &pack, tgas, tdust, metallicity, dust2gas, rhoH, mmw, nelec_times_mH,
        edot, grain_temperatures, logTlininterp_buf,
        coolingheating_buf, chemheatrates_buf
      );

      // overwrite the fields tracked by my_fields with the evolved values
      // -> in the future, we will be able write the next ~80 lines as
      //    a concise 4-line for-loop when we start to use the
      //    SpeciesMultiView type

      if ( my_chemistry->primordial_chemistry > 0 )  {
        my_fields->e_density[field_idx1d]       = dsp[SpLUT::e];
        my_fields->HI_density[field_idx1d]      = dsp[SpLUT::HI];
        my_fields->HII_density[field_idx1d]     = dsp[SpLUT::HII];
        my_fields->HeI_density[field_idx1d]     = dsp[SpLUT::HeI];
        my_fields->HeII_density[field_idx1d]    = dsp[SpLUT::HeII];
        my_fields->HeIII_density[field_idx1d]   = dsp[SpLUT::HeIII];
      }
      if ( my_chemistry->primordial_chemistry > 1 )  {
        my_fields->HM_density[field_idx1d]      = dsp[SpLUT::HM];
        my_fields->H2I_density[field_idx1d]     = dsp[SpLUT::H2I];
        my_fields->H2II_density[field_idx1d]    = dsp[SpLUT::H2II];
      }
      if ( my_chemistry->primordial_chemistry > 2 )  {
        my_fields->DI_density[field_idx1d]      = dsp[SpLUT::DI];
        my_fields->DII_density[field_idx1d]     = dsp[SpLUT::DII];
        my_fields->HDI_density[field_idx1d]     = dsp[SpLUT::HDI];
      }
      if ( my_chemistry->primordial_chemistry > 3 )  {
        my_fields->DM_density[field_idx1d]      = dsp[SpLUT::DM];
        my_fields->HDII_density[field_idx1d]    = dsp[SpLUT::HDII];
        my_fields->HeHII_density[field_idx1d]   = dsp[SpLUT::HeHII];
      }
      if ( itmask_metal[i] != MASK_FALSE )  {
        if ( my_chemistry->metal_chemistry == 1 )  {
          my_fields->CI_density[field_idx1d]      = dsp[SpLUT::CI];
          my_fields->CII_density[field_idx1d]     = dsp[SpLUT::CII];
          my_fields->COI_density[field_idx1d]     = dsp[SpLUT::COI];
          my_fields->CO2I_density[field_idx1d]    = dsp[SpLUT::CO2I];
          my_fields->OI_density[field_idx1d]      = dsp[SpLUT::OI];
          my_fields->OHI_density[field_idx1d]     = dsp[SpLUT::OHI];
          my_fields->H2OI_density[field_idx1d]    = dsp[SpLUT::H2OI];
          my_fields->O2I_density[field_idx1d]     = dsp[SpLUT::O2I];
          my_fields->SiI_density[field_idx1d]     = dsp[SpLUT::SiI];
          my_fields->SiOI_density[field_idx1d]    = dsp[SpLUT::SiOI];
          my_fields->SiO2I_density[field_idx1d]   = dsp[SpLUT::SiO2I];
          my_fields->CHI_density[field_idx1d]     = dsp[SpLUT::CHI];
          my_fields->CH2I_density[field_idx1d]    = dsp[SpLUT::CH2I];
          my_fields->COII_density[field_idx1d]    = dsp[SpLUT::COII];
          my_fields->OII_density[field_idx1d]     = dsp[SpLUT::OII];
          my_fields->OHII_density[field_idx1d]    = dsp[SpLUT::OHII];
          my_fields->H2OII_density[field_idx1d]   = dsp[SpLUT::H2OII];
          my_fields->H3OII_density[field_idx1d]   = dsp[SpLUT::H3OII];
          my_fields->O2II_density[field_idx1d]    = dsp[SpLUT::O2II];
          if (chiaki_model_dust_evolution)  {
            if (my_chemistry->dust_species > 0)  {
              my_fields->Mg_density[field_idx1d]      = dsp[SpLUT::Mg];
            }
            if (my_chemistry->dust_species > 1)  {
              my_fields->Al_density[field_idx1d]      = dsp[SpLUT::Al];
              my_fields->S_density[field_idx1d]       = dsp[SpLUT::S];
              my_fields->Fe_density[field_idx1d]      = dsp[SpLUT::Fe];
            }
          }
        }
        if (chiaki_model_dust_evolution)  {
          if (my_chemistry->dust_species > 0)  {
            my_fields->MgSiO3_dust_density[field_idx1d]  = dsp[SpLUT::MgSiO3_dust];
            my_fields->AC_dust_density[field_idx1d]      = dsp[SpLUT::AC_dust];
          }
          if (my_chemistry->dust_species > 1)  {
            my_fields->SiM_dust_density[field_idx1d]     = dsp[SpLUT::SiM_dust];
            my_fields->FeM_dust_density[field_idx1d]     = dsp[SpLUT::FeM_dust];
            my_fields->Mg2SiO4_dust_density[field_idx1d] = dsp[SpLUT::Mg2SiO4_dust];
            my_fields->Fe3O4_dust_density[field_idx1d]   = dsp[SpLUT::Fe3O4_dust];
            my_fields->SiO2_dust_density[field_idx1d]   = dsp[SpLUT::SiO2_dust];
            my_fields->MgO_dust_density[field_idx1d]     = dsp[SpLUT::MgO_dust];
            my_fields->FeS_dust_density[field_idx1d]     = dsp[SpLUT::FeS_dust];
            my_fields->Al2O3_dust_density[field_idx1d]   = dsp[SpLUT::Al2O3_dust];
          }
          if (my_chemistry->dust_species > 2)  {
            my_fields->ref_org_dust_density[field_idx1d]  = dsp[SpLUT::ref_org_dust];
            my_fields->vol_org_dust_density[field_idx1d]  = dsp[SpLUT::vol_org_dust];
            my_fields->H2O_ice_dust_density[field_idx1d]  = dsp[SpLUT::H2O_ice_dust];
          }
        }
      }

      e(i,j,k)     = dsp[i_eng-1];
      next_nr_dt[i] = result.suggested_dt;

      idsp.clear();
    }
  }

  release_scratch();
  return true;
}

} // namespace GRIMPL_NAMESPACE_DECL


#endif /* STEP_RATE_NEWTON_RAPHSON_HPP */
