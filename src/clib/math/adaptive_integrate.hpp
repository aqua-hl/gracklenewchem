//===----------------------------------------------------------------------===//
// See the LICENSE file for license and copyright information
// SPDX-License-Identifier: NCSA AND BSD-3-Clause
//===----------------------------------------------------------------------===//

#ifndef MATH_ADAPTIVE_INTEGRATE_HPP
#define MATH_ADAPTIVE_INTEGRATE_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <vector>

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
///
/// Cost: the Jacobian at the start of a substep is reused by the backward-Euler
/// solves (modified Newton). It is rebuilt when the iteration contracts slowly
/// or endpoint residual validation fails. Before convergence is accepted,
/// the RHS is always checked at the returned state. The first half step's
/// actual endpoint Jacobian starts the second half. At the final endpoint only
/// the thermal diagonal is needed, so the guard evaluates it directly.
class AdaptiveBackwardEuler {
  int max_variables_;
  double relative_tolerance_;
  double max_fractional_change_;
  int max_retries_;
  double jacobian_step_;  // relative energy perturbation for the J_ee guard
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
                                 double nonlinear_tolerance = 1.e-8,
                                 double jacobian_step = 1.e-4,
                                 double rhs_relative_roundoff = 0.,
                                 double rhs_absolute_roundoff = 0.)
      : max_variables_(max_variables),
        relative_tolerance_(relative_tolerance),
        max_fractional_change_(max_fractional_change),
        max_retries_(max_retries),
        jacobian_step_(jacobian_step),
        newton_(max_variables, nonlinear_tolerance, 20,
                rhs_relative_roundoff, rhs_absolute_roundoff) {}

  int num_scratch_buf_elements() const noexcept {
    // Nine vectors and two matrices, plus the Newton solver's scratch space.
    return 9 * max_variables_ + 2 * max_variables_ * max_variables_
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
  /// Components that are stiff and stable at the trial step
  /// (h * -J_ii > 10 at the start state) are exempt from the fractional-change
  /// limiter: backward Euler holds them at their quasi-equilibrium with the
  /// slower components, and the step-doubling error estimate still controls
  /// them. Without this, fast trace species (H-, H2+) force ~1e-5 tau steps.
  ///
  /// minimum_dt should include the caller's floating-point progress floor
  /// (for example, nextafter(elapsed, +infinity)-elapsed). If a safe accurate
  /// step cannot exceed that floor within max_retries, failure is explicit.
  /// Set energy_index=-1 for a species-only system. calc_f_and_jacobian has
  /// the same interface as StiffNewtonRaphson::step; calc_f(y, f) evaluates
  /// only f(y).
  template <typename Fn, typename FnF>
  AdaptiveIntegrateResult step(
      double* y, double proposed_dt, double minimum_dt,
      const Fn& calc_f_and_jacobian, const FnF& calc_f, int n,
      int energy_index, const double* absolute_tolerance, double* scratch,
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
    double* f_initial = newton_atol + n;
    double* probe = f_initial + n;
    double* f_probe = probe + n;
    View<double**> jac(f_probe + n, n, n);
    // Jacobian validated at the actual end of the first half step
    View<double**> jac_mid(jac.data() + n * n, n, n);
    double* newton_scratch = jac_mid.data() + n * n;

    for (int i = 0; i < n; ++i) {
      if (!std::isfinite(absolute_tolerance[i]) || absolute_tolerance[i] < 0.) {
        return result;
      }
      initial[i] = y[i];
      // Newton errors must remain small compared with the error estimator.
      newton_atol[i] = 1.e-4 * absolute_tolerance[i];
    }
    // The start-state f and Jacobian are reused by every solve below.
    calc_f_and_jacobian(initial, f_initial, jac);
    if (!finite_derivatives_(f_initial, jac, n)) { return result; }

    double h = proposed_dt;
    const double initial_thermal_j =
      (energy_index >= 0) ? jac(energy_index, energy_index) : 0.;
    if (initial_thermal_j > 0.) {
      h = std::min(h, 0.1 / initial_thermal_j);
    }
    const auto slaved = [&](int i, double step) {
      return jac(i, i) < 0. && -jac(i, i) * step > 10.;
    };

    // A diagonal linearized backward-Euler predictor avoids repeatedly
    // halving a cosmological step just to resolve an initial transient.
    // Acceptance still uses the full coupled solutions below; this predictor
    // makes no claim that off-diagonal couplings can be neglected.
    for (int i = 0; i < n; ++i) {
      if (slaved(i, h)) { continue; }
      const double scale = std::max(std::fabs(initial[i]),
                                   absolute_tolerance[i] / relative_tolerance_);
      const double limit = max_fractional_change_ * scale;
      const double excess = std::fabs(f_initial[i])
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
      // The full step and the first half start at `initial`, where `jac` was
      // evaluated. The second half starts at the midpoint with a Jacobian
      // evaluated at the returned midpoint (after its final correction).
      bool mid_ok = false;
      auto run_newton = [&](double* state, double step_dt,
                            const View<double**>* jacobian,
                            const double* f_start, bool jacobian_at_start,
                            View<double**>* jacobian_out, bool* exported) {
        IntegrateResult nr = newton_.step_modified(
          state, step_dt, calc_f, calc_f_and_jacobian, n, newton_scratch,
          newton_atol, jacobian, f_start, jacobian_at_start, jacobian_out,
          exported);
        result.newton_iterations += nr.iterations;
        return nr.converged() && valid_state_(state, n, energy_index);
      };

      if (!run_newton(full, h, &jac, f_initial, true, nullptr, nullptr) ||
          !run_newton(fine, 0.5 * h, &jac, f_initial, true, &jac_mid,
                      &mid_ok)) {
        h *= 0.5;
        continue;
      }
      std::copy(fine, fine + n, middle);
      // No subsequent solve reuses the final endpoint's full Jacobian.
      // Evaluate its thermal diagonal explicitly below instead.
      if (!run_newton(fine, 0.5 * h, mid_ok ? &jac_mid : &jac, nullptr,
                      mid_ok, nullptr, nullptr)) {
        h *= 0.5;
        continue;
      }

      // Check the entire candidate path, not only the initial derivative.
      // Only J_ee is needed: exported matrices are evaluated at the returned
      // states, and Newton has checked their RHS values for finiteness. If no
      // matrix was exported, evaluate the candidate and probe its slope here.
      double max_positive_j = std::max(0., initial_thermal_j);
      bool finite_candidate = true;
      if (energy_index >= 0) {
        for (int c = 0; c < 2; ++c) {
          const double* candidate = (c == 0) ? middle : fine;
          if (c == 0 && mid_ok) {
            const double jee = jac_mid(energy_index, energy_index);
            finite_candidate = finite_candidate && std::isfinite(jee);
            if (finite_candidate) {
              max_positive_j = std::max(max_positive_j, jee);
            }
            continue;
          }
          calc_f(candidate, f);
          std::copy(candidate, candidate + n, probe);
          const double de = jacobian_step_ *
            std::max(std::fabs(candidate[energy_index]),
                     std::numeric_limits<double>::min());
          probe[energy_index] += de;
          calc_f(probe, f_probe);
          const double jee = (f_probe[energy_index] - f[energy_index]) / de;
          bool ok = std::isfinite(jee);
          for (int i = 0; i < n; ++i) {
            ok = ok && std::isfinite(f[i]) && std::isfinite(f_probe[i]);
          }
          finite_candidate = finite_candidate && ok;
          if (finite_candidate) {
            max_positive_j = std::max(max_positive_j, jee);
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
        const double change_i = slaved(i, h) ? 0. :
          (change_scale > 0.) ? change_delta / change_scale
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

  /// Convenience overload for callers without a separate f-only function.
  template <typename Fn>
  AdaptiveIntegrateResult step(
      double* y, double proposed_dt, double minimum_dt,
      const Fn& calc_f_and_jacobian, int n, int energy_index,
      const double* absolute_tolerance, double* scratch,
      double local_density = 1.0) const {
    std::vector<double> jac_buf(static_cast<std::size_t>(n > 0 ? n * n : 1));
    const auto calc_f = [&](const double* state, double* f_out) {
      View<double**> unused(jac_buf.data(), n, n);
      calc_f_and_jacobian(state, f_out, unused);
    };
    return step(y, proposed_dt, minimum_dt, calc_f_and_jacobian, calc_f, n,
                energy_index, absolute_tolerance, scratch, local_density);
  }
};

}  // namespace integrate
}  // namespace GRIMPL_NAMESPACE_DECL

#endif  // MATH_ADAPTIVE_INTEGRATE_HPP
