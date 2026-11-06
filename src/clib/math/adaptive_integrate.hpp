//===----------------------------------------------------------------------===//
// See the LICENSE file for license and copyright information
// SPDX-License-Identifier: NCSA AND BSD-3-Clause
//===----------------------------------------------------------------------===//

#ifndef MATH_ADAPTIVE_INTEGRATE_HPP
#define MATH_ADAPTIVE_INTEGRATE_HPP

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>

#include "integrate.hpp"

namespace GRIMPL_NAMESPACE_DECL {
namespace integrate {

/// Result of one accepted adaptive substep, which may be shorter than requested.
struct AdaptiveIntegrateResult {
  TerminateFlag flag;
  double accepted_dt;
  double suggested_dt;
  int retries;
  int newton_iterations;
  double error_norm;

  bool converged() const noexcept { return flag == TerminateFlag::CONVERGED; }
};

/// Backward Euler with step doubling, bounded retries, and a thermal guard.
///
/// The accepted state is obtained from two half steps. We do not extrapolate
/// or clip components: both operations can sacrifice positivity or linear
/// reaction invariants. The difference from one full step estimates error.
///
/// An energy index enables an additional safeguard: a positive diagonal
/// energy derivative means the frozen-composition thermal mode is unstable.
/// We require h*J_ee <= 0.1 at the start, midpoint, and endpoint. This prevents
/// large backward-Euler steps from spuriously damping that growing mode.
/// It does not assert stability of every eigenmode of a coupled network.
class AdaptiveBackwardEuler {
  int max_variables_;
  double relative_tolerance_;
  double max_fractional_change_;
  int max_retries_;
  StiffNewtonRaphson newton_;

  static bool valid_state_(const double* y, int n, int energy_index) {
    for (int i = 0; i < n; ++i) {
      if (!std::isfinite(y[i]) || y[i] < 0. ||
          (i == energy_index && y[i] == 0.)) {
        return false;
      }
    }
    return true;
  }

  static bool finite_derivatives_(const double* f, View<double**> jac, int n) {
    for (int i = 0; i < n; ++i) {
      if (!std::isfinite(f[i])) { return false; }
      for (int j = 0; j < n; ++j) {
        if (!std::isfinite(jac(i, j))) { return false; }
      }
    }
    return true;
  }

public:
  explicit AdaptiveBackwardEuler(int max_variables,
                                 double relative_tolerance = 1.e-3,
                                 double max_fractional_change = 0.1,
                                 int max_retries = 40,
                                 double nonlinear_tolerance = 1.e-8)
      : max_variables_(max_variables),
        relative_tolerance_(relative_tolerance),
        max_fractional_change_(max_fractional_change),
        max_retries_(max_retries),
        newton_(max_variables, nonlinear_tolerance, 20) {}

  int num_scratch_buf_elements() const noexcept {
    // Six vectors and one matrix, plus the Newton solver's scratch space.
    return 6 * max_variables_ + max_variables_ * max_variables_
      + newton_.num_scratch_buf_elements();
  }

  /// Advance one substep; leave y completely unchanged on failure.
  ///
  /// absolute_tolerance[i] is the absolute error tolerance in component i's
  /// own units. The fractional-change limiter uses a component scale of
  /// max(abs(y_start[i]), absolute_tolerance[i]/relative_tolerance).
  /// This makes trace species harmless to timestep selection while resolving
  /// relative changes in abundant species. The caller should choose the
  /// abundance scale from its conserved elemental densities.
  ///
  /// minimum_dt should include the caller's floating-point progress floor
  /// (for example, nextafter(elapsed, +infinity)-elapsed). If a safe accurate
  /// step cannot exceed that floor within max_retries, failure is explicit.
  /// Set energy_index=-1 for a species-only system. calc_f_and_jacobian has
  /// the same interface as StiffNewtonRaphson::step.
  template <typename Fn>
  AdaptiveIntegrateResult step(
      double* y, double proposed_dt, double minimum_dt,
      const Fn& calc_f_and_jacobian, int n, int energy_index,
      const double* absolute_tolerance, double* scratch,
      double local_density = 1.0) const {
    const double infinity = std::numeric_limits<double>::infinity();
    AdaptiveIntegrateResult result{
      TerminateFlag::OTHER_ERR, 0., proposed_dt, 0, 0, infinity};
    if (n <= 0 || n > max_variables_ || energy_index < -1 ||
        energy_index >= n || absolute_tolerance == nullptr ||
        !std::isfinite(proposed_dt) || proposed_dt <= 0. ||
        !std::isfinite(minimum_dt) || minimum_dt < 0. ||
        !std::isfinite(local_density) || local_density <= 0. ||
        !std::isfinite(relative_tolerance_) || relative_tolerance_ <= 0. ||
        !std::isfinite(max_fractional_change_) || max_fractional_change_ <= 0. ||
        max_retries_ < 0 || !valid_state_(y, n, energy_index)) {
      return result;
    }

    double* initial = scratch;
    double* full = initial + n;
    double* fine = full + n;
    double* middle = fine + n;
    double* f = middle + n;
    double* newton_atol = f + n;
    View<double**> jac(newton_atol + n, n, n);
    double* newton_scratch = jac.data() + n * n;

    for (int i = 0; i < n; ++i) {
      if (!std::isfinite(absolute_tolerance[i]) || absolute_tolerance[i] < 0.) {
        return result;
      }
      initial[i] = y[i];
      // Newton errors must remain small compared with the error estimator.
      newton_atol[i] = 1.e-4 * absolute_tolerance[i];
    }
    calc_f_and_jacobian(initial, f, jac);
    if (!finite_derivatives_(f, jac, n)) { return result; }

    double h = proposed_dt;
    const double initial_thermal_j =
      (energy_index >= 0) ? jac(energy_index, energy_index) : 0.;
    if (initial_thermal_j > 0.) {
      h = std::min(h, 0.1 / initial_thermal_j);
    }

    // A diagonal linearized backward-Euler predictor avoids repeatedly
    // halving a cosmological step just to resolve an initial transient.
    // Acceptance still uses the full coupled solutions below; this predictor
    // makes no claim that off-diagonal couplings can be neglected.
    for (int i = 0; i < n; ++i) {
      const double scale = std::max(std::fabs(initial[i]),
                                   absolute_tolerance[i] / relative_tolerance_);
      const double limit = max_fractional_change_ * scale;
      const double excess = std::fabs(f[i])
        - limit * std::max(0., -jac(i, i));
      if (excess > 0.) { h = std::min(h, limit / excess); }
    }

    // A strictly positive floor also avoids underflow of the two half steps.
    const double h_floor = std::max(minimum_dt,
      2. * std::numeric_limits<double>::min());
    result.flag = TerminateFlag::CONVERGE_ERR;
    for (int retry = 0; retry <= max_retries_; ++retry) {
      result.retries = retry;
      result.suggested_dt = h;
      if (!std::isfinite(h) || h < h_floor || h <= 0.) { return result; }

      std::copy(initial, initial + n, full);
      std::copy(initial, initial + n, fine);
      auto run_newton = [&](double* state, double step_dt) {
        IntegrateResult nr = newton_.step(
          state, step_dt, calc_f_and_jacobian, n, newton_scratch, true,
          local_density, newton_atol, true);
        result.newton_iterations += nr.iterations;
        return nr.converged() && valid_state_(state, n, energy_index);
      };

      if (!run_newton(full, h) || !run_newton(fine, 0.5 * h)) {
        h *= 0.5;
        continue;
      }
      std::copy(fine, fine + n, middle);
      if (!run_newton(fine, 0.5 * h)) {
        h *= 0.5;
        continue;
      }

      // Check the entire candidate path, not only the initial derivative.
      double max_positive_j = std::max(0., initial_thermal_j);
      bool finite_candidate = true;
      if (energy_index >= 0) {
        for (const double* candidate : {middle, fine}) {
          calc_f_and_jacobian(candidate, f, jac);
          finite_candidate = finite_candidate && finite_derivatives_(f, jac, n);
          if (finite_candidate) {
            max_positive_j = std::max(max_positive_j,
                                      jac(energy_index, energy_index));
          }
        }
      }
      if (!finite_candidate) {
        h *= 0.5;
        continue;
      }
      if (max_positive_j > 0. && h > 0.1 / max_positive_j) {
        h = std::min(0.5 * h, 0.09 / max_positive_j);
        continue;
      }

      double error = 0., change = 0.;
      for (int i = 0; i < n; ++i) {
        const double error_scale = absolute_tolerance[i]
          + relative_tolerance_ * std::max(std::fabs(initial[i]),
                                          std::fabs(fine[i]));
        const double error_delta = std::fabs(fine[i] - full[i]);
        const double err_i = (error_scale > 0.) ? error_delta / error_scale
                              : (error_delta == 0. ? 0. : infinity);
        const double change_scale = max_fractional_change_
          * std::max(std::fabs(initial[i]),
                     absolute_tolerance[i] / relative_tolerance_);
        const double change_delta = std::fabs(fine[i] - initial[i]);
        const double change_i = (change_scale > 0.) ? change_delta / change_scale
                               : (change_delta == 0. ? 0. : infinity);
        error = std::max(error, err_i);
        change = std::max(change, change_i);
      }
      result.error_norm = error;
      if (error <= 1. && change <= 1.) {
        std::copy(fine, fine + n, y);
        result.flag = TerminateFlag::CONVERGED;
        result.accepted_dt = h;
        const double growth = std::min(2.,
          0.9 / std::sqrt(std::max(error, 1.e-16)));
        // The next outer subcycle caps this proposal by its remaining
        // time. Capping by this trial's proposal would prevent growth when
        // a caller reuses suggested_dt after resolving a transient.
        result.suggested_dt = std::min(std::numeric_limits<double>::max(),
                                       h * std::max(1., growth));
        return result;
      }

      // The BE step-doubling difference is second order in a small step.
      // The additional state-change bound is first order in that step.
      double factor = 0.5;
      if (error > 1.) { factor = std::min(factor, 0.8 / std::sqrt(error)); }
      if (change > 1.) { factor = std::min(factor, 0.8 / change); }
      h *= std::max(0.1, factor);
    }
    result.suggested_dt = h;
    return result;
  }
};

}  // namespace integrate
}  // namespace GRIMPL_NAMESPACE_DECL

#endif  // MATH_ADAPTIVE_INTEGRATE_HPP
