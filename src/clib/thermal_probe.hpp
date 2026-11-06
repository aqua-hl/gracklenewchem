//===----------------------------------------------------------------------===//
// See the LICENSE file for license and copyright information
// SPDX-License-Identifier: NCSA AND BSD-3-Clause
//===----------------------------------------------------------------------===//
#ifndef GRACKLE_THERMAL_PROBE_HPP
#define GRACKLE_THERMAL_PROBE_HPP

#include <cmath>
#include <vector>
#include "time_deriv_0d.hpp"
#include "opaque_storage.hpp"

namespace GRIMPL_NAMESPACE_DECL {

/// Fixed-composition thermal slopes from two row-wide RHS evaluations.
///
/// Each OpenMP worker owns this workspace. A row view points at the original
/// species and auxiliary input fields, while both perturbed energy arrays and
/// every derivative/interpolation buffer are private. No user field or outer
/// solver scratch buffer is modified. Batched probes amortize cooling/dust
/// temporaries across a row instead of allocating them twice for every cell.
class ThermalSlopeProbe {
  time_deriv_0d::FrozenSimpleArgs args_;
  int row_size_;
  grackle_field_data fields_;
  int dimensions_[3];
  int starts_[3] = {0, 0, 0};
  int ends_[3] = {0, 0, 0};
  GrainSpeciesCollection grain_temperatures_;
  LnTLinInterpBuf logT_;
  CoolHeatScratchBuf cooling_;
  ChemHeatingRates chemical_heating_;
  FullRxnRateBuf rates_;
  std::vector<gr_float> energy_lo_, energy_hi_;
  std::vector<gr_mask_type> mask_, metal_mask_;
  std::vector<double> temperature_, dust_temperature_, metallicity_, dust2gas_;
  std::vector<double> rhoH_, mmw_, electron_density_, edot_, alpha_, low_rhs_;
  std::vector<double> electron_dot_, HI_dot_;

  void evaluate_rhs_(IndexRange row, const gr_mask_type* active,
                     gr_float* perturbed_energy) {
    fields_.internal_energy = perturbed_energy;
    for (int i = row.i_start; i < row.i_stop; ++i) { mask_[i] = active[i]; }
    FieldAdaptorManager adaptor(&fields_);
    const SpeciesMultiView<const gr_float> species = adaptor.get_species_data(row);
    chemistry_data* chemistry = args_.my_chemistry;
    chemistry_data_storage* storage = args_.my_rates;

    // Do not reuse the outer gas properties: its temperature interpolation
    // can be damped using an earlier subcycle, which is not a partial
    // derivative at fixed composition and current energy.
    extended_gas_props(temperature_.data(), mmw_.data(), rhoH_.data(),
      metallicity_.data(), electron_density_.data(), logT_, mask_.data(),
      chemistry, &storage->cloudy_primordial, &fields_, args_.internalu, row,
      nullptr);
    mask::adjust_from_Tfloor(mask_.data(), temperature_.data(), row,
                            chemistry, &fields_);
    mask::fill_itmask_metal(metal_mask_.data(), mask_.data(), metallicity_.data(),
                            row, chemistry);
    for (int i = row.i_start; i < row.i_stop; ++i) {
      edot_[i] = mask_[i] == MASK_FALSE ? tiny_fortran_val : 0.;
      alpha_[i] = 0.;
    }
    storage->opaque_storage->dust_solver.calc_Tdust_and_chem_contrib(
      edot_.data(), alpha_.data(), &rates_, dust2gas_.data(),
      dust_temperature_.data(), grain_temperatures_, temperature_.data(),
      rhoH_.data(), electron_density_.data(), metallicity_.data(),
      mask_.data(), metal_mask_.data(), chemistry, storage, &fields_, species,
      args_.internalu, row, logT_);
    cool1d_multi_g(edot_.data(), alpha_.data(), temperature_.data(), mmw_.data(),
      metallicity_.data(), rhoH_.data(), electron_density_.data(), mask_.data(),
      metal_mask_.data(), chemistry, storage, &fields_, species,
      args_.my_uvb_rates, args_.internalu, row, logT_, cooling_);
    if (chemistry->primordial_chemistry > 0) {
      lookup_cool_rates1d(row, temperature_.data(), mmw_.data(), args_.dom,
        args_.dx_cgs, args_.c_ljeans, mask_.data(), metal_mask_.data(),
        chemistry, storage, &fields_, species, args_.my_uvb_rates,
        args_.internalu, logT_, rates_, chemical_heating_);
      rate_timestep_g(electron_dot_.data(), HI_dot_.data(), args_.anydust,
        rhoH_.data(), mask_.data(), edot_.data(), args_.chunit, args_.dom,
        chemistry, &fields_, row, chemical_heating_, rates_);
    }
    // Species derivatives are unnecessary for a fixed-composition energy
    // probe. rate_timestep_g above supplies the reaction-heating terms.
  }

public:
  ThermalSlopeProbe(time_deriv_0d::FrozenSimpleArgs args, int row_size)
      : args_(args), row_size_(row_size), dimensions_{row_size, 1, 1},
        grain_temperatures_(new_GrainSpeciesCollection(row_size)),
        logT_(new_LnTLinInterpBuf(row_size)),
        cooling_(new_CoolHeatScratchBuf(row_size)),
        chemical_heating_(new_ChemHeatingRates(row_size)),
        rates_(new_FullRxnRateBuf(row_size)),
        energy_lo_(row_size), energy_hi_(row_size),
        mask_(row_size), metal_mask_(row_size),
        temperature_(row_size), dust_temperature_(row_size),
        metallicity_(row_size), dust2gas_(row_size), rhoH_(row_size),
        mmw_(row_size), electron_density_(row_size), edot_(row_size),
        alpha_(row_size), low_rhs_(row_size), electron_dot_(row_size),
        HI_dot_(row_size) {
    gr_initialize_field_data(&fields_);
    fields_.grid_rank = 3;
    fields_.grid_dimension = dimensions_;
    fields_.grid_start = starts_;
    fields_.grid_end = ends_;
  }

  ThermalSlopeProbe(const ThermalSlopeProbe&) = delete;
  ThermalSlopeProbe& operator=(const ThermalSlopeProbe&) = delete;
  ~ThermalSlopeProbe() {
    drop_GrainSpeciesCollection(&grain_temperatures_);
    drop_LnTLinInterpBuf(&logT_);
    drop_CoolHeatScratchBuf(&cooling_);
    drop_ChemHeatingRates(&chemical_heating_);
    drop_FullRxnRateBuffer(&rates_);
  }

private:
  /// Point the private row view at row (j, k) of `fields`; returns the 1D
  /// offset of that row and rewrites `row` to index the private view.
  int bind_row_(const grackle_field_data* fields, IndexRange& row) {
    const int offset = layoutleft_3D_index_to_1D_(
      fields->grid_dimension, 0, row.j, row.k);
    copy_offset_fieldmember_ptrs_(&fields_, fields, offset);
    fields_.grid_dx = fields->grid_dx;
    starts_[0] = row.i_start;
    ends_[0] = row.i_end;
    row.j = row.k = 0;
    row.jp1 = row.kp1 = 1;
    return offset;
  }

  static constexpr double epsilon_ =
    (sizeof(gr_float) == sizeof(float)) ? 1.e-2 : 1.e-4;

public:
  /// One-sided slope from a caller-supplied f(e) = edot/rho evaluated at the
  /// current state without temperature damping: one RHS evaluation per row
  /// instead of two.
  bool evaluate_one_sided(const grackle_field_data* fields, IndexRange row,
                          const gr_mask_type* active, const double* base_rhs,
                          double* slopes) {
    if (fields->grid_dimension[0] != row_size_ || row.i_start < 0 ||
        row.i_stop > row_size_) return false;
    const int offset = bind_row_(fields, row);
    bool any_active = false;
    for (int i = row.i_start; i < row.i_stop; ++i) {
      if (active[i] == MASK_FALSE) continue;
      any_active = true;
      const double energy = fields->internal_energy[offset + i];
      const double density = fields->density[offset + i];
      energy_hi_[i] = static_cast<gr_float>(energy * (1. + epsilon_));
      if (!(std::isfinite(energy) && energy > 0. && std::isfinite(density) &&
            density > 0. && std::isfinite(energy_hi_[i]) &&
            energy_hi_[i] > energy && std::isfinite(base_rhs[i]))) return false;
    }
    if (!any_active) return true;
    evaluate_rhs_(row, active, energy_hi_.data());
    for (int i = row.i_start; i < row.i_stop; ++i) {
      if (active[i] == MASK_FALSE) continue;
      const double high_rhs = edot_[i] / fields_.density[i];
      slopes[i] = (high_rhs - base_rhs[i]) /
        (double(energy_hi_[i]) - double(fields->internal_energy[offset + i]));
      if (!(std::isfinite(high_rhs) && std::isfinite(slopes[i]))) return false;
    }
    return true;
  }

  bool evaluate(const grackle_field_data* fields, IndexRange row,
                const gr_mask_type* active, double* slopes) {
    if (fields->grid_dimension[0] != row_size_ || row.i_start < 0 ||
        row.i_stop > row_size_) return false;
    const int offset = bind_row_(fields, row);
    const double epsilon = epsilon_;
    bool any_active = false;
    for (int i = row.i_start; i < row.i_stop; ++i) {
      if (active[i] == MASK_FALSE) continue;
      any_active = true;
      const double energy = fields->internal_energy[offset + i];
      const double density = fields->density[offset + i];
      energy_lo_[i] = static_cast<gr_float>(energy * (1. - epsilon));
      energy_hi_[i] = static_cast<gr_float>(energy * (1. + epsilon));
      if (!(std::isfinite(energy) && std::isfinite(density) && density > 0. &&
            std::isfinite(energy_lo_[i]) && std::isfinite(energy_hi_[i]) &&
            energy_lo_[i] > 0. && energy_hi_[i] > energy_lo_[i])) return false;
    }
    if (!any_active) return true;

    evaluate_rhs_(row, active, energy_lo_.data());
    for (int i = row.i_start; i < row.i_stop; ++i) {
      if (active[i] == MASK_FALSE) continue;
      low_rhs_[i] = edot_[i] / fields_.density[i];
      if (!std::isfinite(low_rhs_[i])) return false;
    }
    evaluate_rhs_(row, active, energy_hi_.data());
    for (int i = row.i_start; i < row.i_stop; ++i) {
      if (active[i] == MASK_FALSE) continue;
      const double high_rhs = edot_[i] / fields_.density[i];
      slopes[i] = (high_rhs - low_rhs_[i]) /
        (double(energy_hi_[i]) - double(energy_lo_[i]));
      if (!(std::isfinite(high_rhs) && std::isfinite(slopes[i]))) return false;
    }
    return true;
  }
};

} // namespace GRIMPL_NAMESPACE_DECL
#endif
