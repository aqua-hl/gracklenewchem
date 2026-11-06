//===----------------------------------------------------------------------===//
//
// See the LICENSE file for license and copyright information
// SPDX-License-Identifier: NCSA AND BSD-3-Clause
//
//===----------------------------------------------------------------------===//
///
/// @file
/// Defines/declares generic functionality for integrating ordinary
/// differential equations. The basic premise is that this machinery should
/// generally written in a generic enough form so that we can unit test it.
///
//===----------------------------------------------------------------------===//

#ifndef MATH_INTEGRATE_HPP
#define MATH_INTEGRATE_HPP

#include <cmath>

#include "../grackle_macros.h"
#include "gaussj.hpp"
#include "../support/config.hpp"
#include "../support/View.hpp"

namespace GRIMPL_NAMESPACE_DECL {
namespace integrate {

/// @brief describes the reason that integration terminates
enum struct TerminateFlag {
  CONVERGED,     ///< integrator successfully converged
  CONVERGE_ERR,  ///< integrator simply wasn't able to converge
  OTHER_ERR      ///< some other error was detected
};

/// @brief encodes the result of a single integration step
///
/// Providing both the outcome and number of complete iterations is useful for
/// at least 2 contexts:
/// 1. Testing of the integrator is made easier
/// 2. If an integrator fails for a few different proposed timesteps without
///    completing a full iteration, that may be a good indication of a deeper
///    problem.
struct IntegrateResult {
  TerminateFlag flag;  ///< the reason for termination
  int iterations;      ///< number of complete iterations at termination

  bool converged() const noexcept { return flag == TerminateFlag::CONVERGED; }
};

/// @brief A class for integrating stiff systems of differential equations
///
/// This is intended to be a general-purpose tool that can be tested against
/// various systems of differential equations.
class StiffNewtonRaphson {
  /// the maximum number of variables that can be evolved in the current
  /// Grackle configuration
  int max_evolved_variables_;

  /// integration is considered converged when the max relative difference
  /// magnitude of any vector component does not exceed this value
  double max_rtol_;

  /// maximum number of iterations
  int maxiter_;

public:  // public API
  StiffNewtonRaphson() = delete;

  /// @brief primary constructor
  StiffNewtonRaphson(int max_evolved_variables, double max_rtol, int maxiter)
      : max_evolved_variables_{max_evolved_variables},
        max_rtol_{max_rtol},
        maxiter_{maxiter} {}

  // upper bound on the scratch space needed in step
  int num_scratch_buf_elements() const noexcept {
    int n_vectors = 4;   // f, vec, ycur_minus_ystart and (modified) scale
    int n_matrices = 2;  // jacobian_f and mtrx
    int entries_per_matrix = max_evolved_variables_ * max_evolved_variables_;
    return n_vectors * max_evolved_variables_ + n_matrices * entries_per_matrix;
  }

  /// @brief performs a single numerical integration step for a system of ODEs
  ///
  /// For the sake of discussion:
  /// - let `y(t)` specify a vector representing the state of the system of
  ///   ODEs at a time `t`
  /// - let `dy/dt = f(y)`, where `f(y)` is a vector valued function that
  ///   effectively specifies the differential equations
  ///
  /// This method uses Newton-Raphson to solve for the value of `y` at the end
  /// of the timestep, given the value at the start of the timestep. We are
  /// specifically solving the equation for updating a system of ODEs according
  /// to the backward Euler method (aka the 1-stage backwards differentiation
  /// formula).
  ///
  /// @param[in,out] y buffer of @p n elements. It specifies the initial state
  ///     for the system of ODEs and will be updated (in-place) to hold the
  ///     state at the end of the timestep.
  /// @param[in]     h the timestep
  /// @param[in]     calc_f_and_jacobian function object with the signature
  ///     `void f(const double* y_buf, double* f_buf, double* jacobian_f_buf)`.
  ///     It should treat `y_buf` argument as an input buffer specifyng a state
  ///     for the system of ODEs. When called, it should fill `f_buf` with the
  ///     the value of `f(y)` for the state in `y_buf` and `jacobian_f_buf`
  ///     with jacobian of `f(y)` (evaluated for the state in `y_buf`). The
  ///     function should expect `y_buf` and `f_buf` to have @p n elements
  ///     `jacobian_f_buf` to have space for @p n * @p n elements.
  /// @param[in]     n number of equations in the system of ODE. This must
  ///     **NOT** exceed the `max_evolved_variables` argument passed to the
  ///     constructor of this type.
  /// @param[in]    scratch_ptr buffer of scratch space for in the calculation.
  ///     The expected length is provided by the
  ///     @ref StiffNewtonRaphson::num_scratch_buf_elements method.
  /// @param[in]    enforce_positive_non_NaN when `true`, the calculation will
  ///     perform an explicit check that each guessed state of the differential
  ///     equations only contains positive, non-NaN components.
  /// @param[in]     local_density is used for rescaling. Frankly, this
  ///     parameter is a historical artifact that should be removed (when we're
  ///     prepared to update the gold-standard). All rescaling should occur
  ///     outside of this function.
  ///
  /// High Level Algorithm Overview
  /// -----------------------------
  /// At a high level, we're looking to compute the value of yₑ given by:
  /// >     yₑ = yₛ + h * f(yₑ)
  /// where
  /// - y denotes the (vector-valued) state of a system of a ODEs
  /// - f(y) is a vector-valued function that gives the time derivative of y
  /// - h is the size of a timestep
  /// - yₛ denotes the state at the system at the start of the timestep
  /// - yₑ denotes the state at the end of the timestep
  ///
  /// To solve for yₑ, let's define:
  /// >    g(yₑ) = yₑ - yₛ - h * f(yₑ)
  /// We'll use the Newton-Raphson Method to find the root of g(yₑ), which
  /// coincides with the desired value of yₑ.
  ///
  /// If J_g(y) specifies the Jacobian matrix (evaluated for some value y) for
  /// the function g(y), then the Newton-Raphson methods tells us that
  /// >    yₑ,ₖ₊₁ = yₑ,ₖ - (J_g(yₑ,ₖ))⁻¹ ⋅ g(yₑ,ₖ)
  ///
  /// More Detail
  /// -----------
  /// Let's further define J_f(y) as the Jacobian (evaluated for some value y)
  /// for the function f(y). We can then write
  /// >    J_g(y) = I - h * J_f(y)
  /// where I represents the identity matrix.
  ///
  /// We can use that formula (and the definition for g(y)) to rewrite the
  /// formula for yₑ,ₖ₊₁ as:
  /// >   yₑ,ₖ₊₁ = yₑ,ₖ - (I - h * J_f(yₑ,ₖ))⁻¹ ⋅ (yₑ,ₖ - yₛ - h * f(yₑ,ₖ))
  ///
  /// We can rewrite this as:
  /// >   δₖ = (I - h * J_f(yₑ,ₖ))⁻¹ ⋅ (h * f(yₑ,ₖ) - (yₑ,ₖ - yₛ))
  /// >   yₑ,ₖ₊₁ = yₑ,ₖ + δₖ
  /// This is the equation implemented in this function
  template <typename Fn>
  GRIMPL_FORCE_INLINE IntegrateResult step(double* y, double h,
                                           const Fn& calc_f_and_jacobian, int n,
                                           double* scratch_ptr,
                                           bool enforce_positive_non_NaN,
                                           double local_density = 1.0,
                                           const double* absolute_tolerance = nullptr,
                                           bool allow_zero = false) const {
    // I suspect that this practice of creating local buffers from a scratch
    // buffer is a useful idiom that we'll want to repeat
    int scratch_offset = 0;
    double* ycur_minus_ystart = scratch_ptr + scratch_offset;
    scratch_offset += n;
    double* vec = scratch_ptr + scratch_offset;
    scratch_offset += n;
    double* f = scratch_ptr + scratch_offset;
    scratch_offset += n;
    View<double**> jacobian_f(scratch_ptr + scratch_offset, n, n);
    scratch_offset += n * n;
    View<double**> mtrx(scratch_ptr + scratch_offset, n, n);

    // initialize ycur_minus_ystart (i.e. difference between current estimate
    // for y at the end of the timestep and y at the start of the timestep)
    for (int i = 0; i < n; i++) {
      ycur_minus_ystart[i] = 0.0;
    }

    // as we enter the loop, note that `y` holds `yₑ,₀ = yₛ`
    for (int k = 0; k < maxiter_; k++) {
      // evaluate f(yₑ,ₖ) and the jacobian matrix for f at yₑ,ₖ.
      calc_f_and_jacobian(y, f, jacobian_f);

      // A NaN correction otherwise looks converged because fmax ignores NaN.
      // Never pass non-finite coefficients to the linear solve.
      for (int i = 0; i < n; i++) {
        if (!std::isfinite(f[i])) {
          return IntegrateResult{TerminateFlag::OTHER_ERR, k};
        }
        for (int j = 0; j < n; j++) {
          if (!std::isfinite(jacobian_f(i, j))) {
            return IntegrateResult{TerminateFlag::OTHER_ERR, k};
          }
        }
      }

      // store the value of (I - h * J_f(yₑ,ₖ)) inside mtrx
      // -> I is the identity matrix and J_f is the jacobian matrix for f
      //    evaluated at the current guess for yₑ.
      // -> h is the size of the timestep
      // -> as the derivation of the method in the docstring shows, `mtrx`, is
      //    technically the jacobian of the function for which we're performing
      //    root finding
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          if (i == j) {
            mtrx(i, j) = 1.0 - h * jacobian_f(i, j);
          } else {
            mtrx(i, j) = -h * jacobian_f(i, j);
          }
        }
      }

      // fill vec with the value of (h * f(yₑ,ₖ) - (yₑ,ₖ - yₛ))
      for (int i = 0; i < n; i++) {
        vec[i] = f[i] * h - ycur_minus_ystart[i];
      }

      // Balance the equations and unknowns when component tolerances supply
      // their natural scales. Energy and species densities can differ by
      // many decades in code units; a common density factor does not fix
      // that conditioning. The RHS buffer is now free to hold the scales.
      if (absolute_tolerance != nullptr) {
        for (int i = 0; i < n; i++) {
          f[i] = std::fmax(std::fabs(y[i]),
                          absolute_tolerance[i] / max_rtol_);
          if (f[i] == 0.) { f[i] = 1.; }
          if (!std::isfinite(f[i]) || f[i] < 0.) {
            return IntegrateResult{TerminateFlag::OTHER_ERR, k};
          }
        }
        for (int i = 0; i < n; i++) {
          vec[i] /= f[i];
          for (int j = 0; j < n; j++) {
            mtrx(i, j) *= f[j] / f[i];
            if (!std::isfinite(mtrx(i, j))) {
              return IntegrateResult{TerminateFlag::OTHER_ERR, k};
            }
          }
        }
      } else {
        // Preserve the historical scaling for existing callers.
        for (int i = 0; i < n; i++) {
          vec[i] = vec[i] / local_density;
        }
      }

      // compute  δₖ = (I - h * J_f(yₑ,ₖ))⁻¹ ⋅ (h * f(yₑ,ₖ) - (yₑ,ₖ - yₛ))
      // and overwrite elements in the `vec` buffer with the result
      // - to that end, we solve following eqn for `x` (and write it to `vec`):
      //      A⋅x = b
      //      | |   |
      //      | |   ---- = `h * f(yₑ,ₖ) - (yₑ,ₖ - yₛ)`  (currently in vec)
      //      | -------- = `δₖ`                        (will be written to vec)
      //      ---------- = `(I - h * J_f(yₑ,ₖ))`       (currently in matrx)
      // - the current algorithm inverts `mtrx` in place, but that's not
      //   important to us
      //
      // todo: consider adjusting gaussj_g's return value so that its more
      //       consistent with GR_SUCCESS
      int ierror = gaussj(n, mtrx.data(), vec);
      if (ierror == 1) {
        // we intentionally say there have been k rather than k+1 iterations
        // since the current iteration isn't complete
        return IntegrateResult{TerminateFlag::OTHER_ERR, k};
      }

      // Convert the scaled correction back to each component's own units.
      for (int i = 0; i < n; i++) {
        vec[i] *= (absolute_tolerance != nullptr) ? f[i] : local_density;
      }

      // update values in ycur_minus_ystart and y for next iteration using:
      //  (yₑ,ₖ₊₁ - yₛ) = (yₑ,ₖ - yₛ) + δₖ
      //   yₑ,ₖ₊₁ = yₑ,ₖ + δₖ
      for (int i = 0; i < n; i++) {
        ycur_minus_ystart[i] = ycur_minus_ystart[i] + vec[i];
        y[i] = y[i] + vec[i];
      }

      {
        for (int i = 0; i < n; i++) {
          const bool invalid_sign = enforce_positive_non_NaN &&
            ((allow_zero && y[i] < 0.) || (!allow_zero && y[i] <= 0.));
          if (!std::isfinite(y[i]) || !std::isfinite(vec[i]) || invalid_sign) {
            // a compelling case could be made for saying there are k+1 complete
            // iterations, but saying there are k iterations is probably more
            // useful for identifying underlying issues
            return IntegrateResult{TerminateFlag::OTHER_ERR, k};
          }
        }
      }

      double max_rel_diff_mag = 0.0;
      for (int i = 0; i < n; i++) {
        double cur = y[i];
        // reminder: at this point, vec holds δₖ = yₑ,ₖ₊₁ - yₑ,ₖ
        double err;
        if (absolute_tolerance != nullptr) {
          // An abundance-aware absolute tolerance prevents absent species
          // from making the relative convergence test singular.
          double scale = std::fabs(cur) + absolute_tolerance[i] / max_rtol_;
          err = (scale > 0.) ? std::fabs(vec[i]) / scale
                            : (vec[i] == 0. ? 0. : HUGE_VAL);
        } else {
          // Bound the scale near zero without ignoring a nonzero correction.
          err = std::fabs(vec[i]) / std::fmax(std::fabs(cur), tiny8);
        }
        max_rel_diff_mag = std::fmax(err, max_rel_diff_mag);
      }

      if (max_rel_diff_mag <= max_rtol_) {
        return IntegrateResult{TerminateFlag::CONVERGED, k + 1};
      }
    }

    // we have exceeded maxiter
    return IntegrateResult{TerminateFlag::CONVERGE_ERR, maxiter_ + 1};
  }

  /// Modified Newton variant of @ref step for the adaptive integrator.
  ///
  /// `shared_jacobian` (may be null) is reused for every iteration, so an
  /// iteration costs one evaluation of `calc_f` instead of n + 1. When the
  /// correction shrinks by less than half per iteration the Jacobian is
  /// rebuilt at the current iterate (at most 3 times). `f_at_start` (may be
  /// null) is f(y) at entry. Components and equations are scaled as in
  /// @ref step; `absolute_tolerance` is required. Iterates must stay
  /// non-negative and finite. A Jacobian whose columns conserve the RHS's
  /// linear invariants keeps those invariants for every iterate.
  template <typename FnF, typename FnFJ>
  IntegrateResult step_modified(double* y, double h, const FnF& calc_f,
                                const FnFJ& calc_f_and_jacobian, int n,
                                double* scratch_ptr,
                                const double* absolute_tolerance,
                                const View<double**>* shared_jacobian,
                                const double* f_at_start) const {
    double* ycur_minus_ystart = scratch_ptr;
    double* vec = ycur_minus_ystart + n;
    double* f = vec + n;
    double* s = f + n;
    View<double**> jac_local(s + n, n, n);
    View<double**> mtrx(s + n + n * n, n, n);
    const View<double**>* J = shared_jacobian;
    bool f_ready = false;
    if (f_at_start != nullptr) {
      for (int i = 0; i < n; i++) { f[i] = f_at_start[i]; }
      f_ready = true;
    }
    if (J == nullptr) {
      calc_f_and_jacobian(y, f, jac_local);
      J = &jac_local;
      f_ready = true;
    }
    for (int i = 0; i < n; i++) { ycur_minus_ystart[i] = 0.0; }
    double previous = -1.0;
    int refreshes = 0;
    for (int k = 0; k < maxiter_; k++) {
      if (!f_ready) { calc_f(y, f); }
      f_ready = false;
      for (int i = 0; i < n; i++) {
        if (!std::isfinite(f[i])) {
          return IntegrateResult{TerminateFlag::OTHER_ERR, k};
        }
        s[i] = std::fmax(std::fabs(y[i]), absolute_tolerance[i] / max_rtol_);
        if (s[i] == 0.) { s[i] = 1.; }
      }
      // gaussj overwrites the matrix, so (I - h J) is rebuilt each iteration;
      // that costs O(n^2) and no RHS evaluations.
      for (int i = 0; i < n; i++) {
        vec[i] = (f[i] * h - ycur_minus_ystart[i]) / s[i];
        for (int j = 0; j < n; j++) {
          const double a = ((i == j) ? 1.0 : 0.0) - h * (*J)(i, j);
          mtrx(i, j) = a * s[j] / s[i];
          if (!std::isfinite(mtrx(i, j))) {
            return IntegrateResult{TerminateFlag::OTHER_ERR, k};
          }
        }
      }
      if (gaussj(n, mtrx.data(), vec) == 1) {
        return IntegrateResult{TerminateFlag::OTHER_ERR, k};
      }
      double max_rel = 0.0;
      for (int i = 0; i < n; i++) {
        vec[i] *= s[i];
        ycur_minus_ystart[i] += vec[i];
        y[i] += vec[i];
        if (!std::isfinite(y[i]) || !std::isfinite(vec[i]) || y[i] < 0.) {
          return IntegrateResult{TerminateFlag::OTHER_ERR, k};
        }
        const double scale = std::fabs(y[i]) + absolute_tolerance[i] / max_rtol_;
        const double err = (scale > 0.) ? std::fabs(vec[i]) / scale
                                        : (vec[i] == 0. ? 0. : HUGE_VAL);
        max_rel = std::fmax(max_rel, err);
      }
      if (max_rel <= max_rtol_) {
        return IntegrateResult{TerminateFlag::CONVERGED, k + 1};
      }
      if (previous > 0. && max_rel > 0.5 * previous && refreshes < 3) {
        calc_f_and_jacobian(y, f, jac_local);
        J = &jac_local;
        f_ready = true;
        ++refreshes;
        previous = -1.0;
        continue;
      }
      previous = max_rel;
    }
    return IntegrateResult{TerminateFlag::CONVERGE_ERR, maxiter_ + 1};
  }
};

}  // namespace integrate
}  // namespace GRIMPL_NAMESPACE_DECL

#endif  // MATH_INTEGRATE_HPP
