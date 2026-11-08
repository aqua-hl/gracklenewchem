//===----------------------------------------------------------------------===//
//
// See the LICENSE file for license and copyright information
// SPDX-License-Identifier: NCSA AND BSD-3-Clause
//
//===----------------------------------------------------------------------===//
///
/// @file
/// add some basic tests of integration
///
//===----------------------------------------------------------------------===//

#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "gtest/gtest.h"
#include "math/integrate.hpp"
#include "math/adaptive_integrate.hpp"
#include "support/View.hpp"

static double L2_error_norm(const std::vector<double>& a,
                            const std::vector<double>& b) {
  std::size_t len = a.size();
  if (len != b.size()) {  // sanity check!
    return std::numeric_limits<double>::infinity();
  }
  double sum = 0.0;
  for (std::size_t i = 0; i < len; i++) {
    sum += std::pow(a[i] - b[i], 2);
  }
  return std::sqrt(sum);
}

std::string prep_L2_matcher_descr(const std::vector<double>& reference,
                                  double thresh, bool negation) {
  std::string op = (negation) ? ">=" : "<";
  return ("L2 Error " + op + " " + testing::PrintToString(thresh) +
          " (ref-val: " + testing::PrintToString(reference) + ")");
}

// This is used by writing:
//   EXPECT_THAT(actual, L2ErrorNormLT(expected, thresh));
// where:
//  actual and expected are each coercible to const std::vector<double>&
//  thresh is a double precision value
MATCHER_P2(L2ErrorNormLT, expected, thresh,
           prep_L2_matcher_descr(expected, thresh, negation)) {
  double L2_error = L2_error_norm(arg, expected);
  if (L2_error < thresh) {
    return true;
  }
  // describe what went wrong (to make error message more descriptive)
  if (arg.size() != expected.size()) {
    *result_listener << "(length mismatch)";
  } else {
    *result_listener << "(L2 Error: " << L2_error << ')';
  }
  return false;
}

template <typename Fn>
static GRIMPL_NS::integrate::IntegrateResult integrate(std::vector<double>& y,
                                                       double h,
                                                       Fn& calc_f_and_jacobian,
                                                       int maxiter = 20) {
  // initialize the ode_solver
  // -> max number of equations handled by solver
  int n = y.size();
  // -> a solution is considered converged when the max relative difference
  //    magnitude of any vector component does not exceed this value
  double max_rtol = 1.e-8;
  GRIMPL_NS::integrate::StiffNewtonRaphson ode_solver(n, max_rtol, maxiter);

  // initialize scratch space
  std::vector<double> scratch(ode_solver.num_scratch_buf_elements());

  return ode_solver.step(y.data(), h, calc_f_and_jacobian, n, scratch.data(),
                         false);
}

// Move onto defining the actual tests
// - for batches of tests, we often find it convenient to organize the systems
//   of ODEs in structs (but this is by no means a requirement)

/// This struct is used to model the system of differential equations for a
/// simple harmonic oscillator
///
/// Let's set the scene:
/// - state vector, y, is given by:
///   > y = {v, x}
/// - the time derivative is given by:
///   > f(y) = dy/dt = {dv/dt, dx/dt} = {d²x/dt², dx/dt} = {-ω²*x, dx/dt}
//    >      = {-ω²*y[1], y[0]}
class HarmonicOscillator {
  static constexpr double omega_ = 0.159154943092;  // ~ 1 / (2 * pi)
  static constexpr double phi_ = 6.28318530718;     // ~ 1 (2 * pi)

  static double omega2_() { return omega_ * omega_; }
  static double pos_(double t) { return std::cos(omega_ * t + phi_); }
  static double vel_(double t) { return -omega_ * std::sin(omega_ * t + phi_); }

public:
  static double half_period() { return 0.5; }

  // initial conditions reproduced at t = 0
  static std::vector<double> analytic_solution(double t) {
    return {vel_(t), pos_(t)};
  }

  static std::vector<double> initial_condition() {
    return analytic_solution(0);
  }

  /// calculates f (time derivatives of y) a its Jacobian
  ///
  /// If we didn't want this type to be a function object, we could rename this
  /// method and make it into a static method
  ///
  /// @note
  /// For less experienced C++ devs, this is equivalent to Python's __call__
  /// magic methods
  void operator()(const double* y, double* f,
                  GRIMPL_NS::View<double**> jacobian_f) const {
    // set f(y)[0] = (dy0/dt) = dv/dt = d²x/dt² = -ω²*x = -ω²*y[1]
    f[0] = -omega2_() * y[1];
    // set f(y)[1] = (dy1/dt) = dx/dt = y[0]
    f[1] = y[0];

    // now handle the Jacobian:
    // (∂f₀/∂y₀) = (∂/∂v)(d²x/dt²) = 0  (partial deriv holds x constant)
    jacobian_f(0, 0) = 0.0;
    // (∂f₀/∂y₁) = (∂/∂x)(d²x/dt²) = -ω²
    jacobian_f(0, 1) = omega2_();
    // (∂f₁/∂y₀) = (∂/∂v)(dx/dt) = 1
    jacobian_f(1, 0) = 1.0;
    // (∂f₁/∂y₁) = (∂/∂x)(dx/dt) = 0
    jacobian_f(1, 1) = 0.0;
  }
};

TEST(IntegrateHarmonicOscillator, dt1Div1000) {
  using ODESystem = HarmonicOscillator;
  ODESystem callback_fn;  // <- use default constructor

  double dt = 0.001;
  std::vector<double> y = ODESystem::initial_condition();
  EXPECT_TRUE(integrate(y, dt, callback_fn).converged())
      << "integration failed for dt = " << dt;
  EXPECT_THAT(y, L2ErrorNormLT(ODESystem::analytic_solution(dt), 1.3e-8))
      << "for dt = " << dt;
}

TEST(IntegrateHarmonicOscillator, dt1Div100) {
  using ODESystem = HarmonicOscillator;
  ODESystem callback_fn;  // <- use default constructor

  double dt = 0.01;
  std::vector<double> y = ODESystem::initial_condition();
  EXPECT_TRUE(integrate(y, dt, callback_fn).converged())
      << "integration failed for dt = " << dt;
  EXPECT_THAT(y, L2ErrorNormLT(ODESystem::analytic_solution(dt), 1.3e-6))
      << "for dt = " << dt;
}

TEST(IntegrateHarmonicOscillator, dt1Div10) {
  using ODESystem = HarmonicOscillator;
  ODESystem callback_fn;  // <- use default constructor

  double dt = 0.1;
  std::vector<double> y = ODESystem::initial_condition();
  EXPECT_TRUE(integrate(y, dt, callback_fn).converged())
      << "integration failed for dt = " << dt;
  EXPECT_THAT(y, L2ErrorNormLT(ODESystem::analytic_solution(dt), 2.7e-3))
      << "for dt = " << dt;
}

TEST(IntegrateHarmonicOscillator, HalfPeriodStep) {
  using ODESystem = HarmonicOscillator;
  ODESystem callback_fn;  // <- use default constructor

  double dt = ODESystem::half_period();
  std::vector<double> y = ODESystem::initial_condition();
  GRIMPL_NS::integrate::IntegrateResult rslt1 = integrate(y, dt, callback_fn);
  EXPECT_TRUE(rslt1.converged()) << "integration failed for dt = " << dt;
  // honestly, it's not very surprising that the error in this case isn't good
  // (the signs totally flip for both components of y)
  EXPECT_THAT(y, L2ErrorNormLT(ODESystem::analytic_solution(dt), 0.020))
      << "for dt = " << dt;

  // the main purpose of this test-case is to check that behavior is valid if
  // we don't converge within maxiter
  ASSERT_GT(rslt1.iterations, 1);
  int maxiter = rslt1.iterations - 1;
  y = ODESystem::initial_condition();
  GRIMPL_NS::integrate::IntegrateResult rslt2 =
      integrate(y, dt, callback_fn, maxiter);

  EXPECT_FALSE(rslt2.converged())
      << "integration should not succeed within maxiter";
}

/// This struct is used to model a stiff system of ODEs known as the
/// "Kaps Problem"
///
/// Full Disclosure
/// ---------------
/// This test-case was suggested by an LLM. I had asked for examples of systems
/// of stiff ODEs with analytic solutions. This is apparently a system that
/// is known to be challenging for numerical integrators
struct KapsProblem {
  // initial conditions reproduced at t = 0
  static std::vector<double> analytic_solution(double t) {
    return {std::exp(-2.0 * t), std::exp(-t)};
  }

  static std::vector<double> initial_condition() {
    return analytic_solution(0);
  }

  /// calculates f (time derivatives of y) a its Jacobian
  ///
  /// If we didn't want this type to be a function object, we could rename this
  /// method and make it into a static method
  ///
  /// @note
  /// For less experienced C++ devs, this is equivalent to Python's __call__
  /// magic methods
  void operator()(const double* y, double* f,
                  GRIMPL_NS::View<double**> jacobian_f) const {
    // set f(y)[0] = (dy0/dt)
    f[0] = -1002.0 * y[0] + 1000.0 * std::pow(y[1], 2);
    // set f(y)[1] = (dy1/dt)
    f[1] = y[0] - y[1] - std::pow(y[1], 2);

    jacobian_f(0, 0) = -1002.0;          // <- (∂f₀/∂y₀)
    jacobian_f(0, 1) = 2000.0 * y[1];    // <- (∂f₀/∂y₁)
    jacobian_f(1, 0) = 1.0;              // <- (∂f₁/∂y₀)
    jacobian_f(1, 1) = -1.0 - 2 * y[1];  // <- (∂f₁/∂y₁)
  }
};

TEST(IntegrateKapsProblem, dt1Div1000) {
  using ODESystem = KapsProblem;
  ODESystem callback_fn;  // <- use default constructor

  double dt = 0.001;
  std::vector<double> y = ODESystem::initial_condition();
  EXPECT_TRUE(integrate(y, dt, callback_fn).converged())
      << "integration failed for dt = " << dt;
  EXPECT_THAT(y, L2ErrorNormLT(ODESystem::analytic_solution(dt), 1.6e-6))
      << "for dt = " << dt;
}

TEST(IntegrateKapsProblem, dt1Div100) {
  using ODESystem = KapsProblem;
  ODESystem callback_fn;  // <- use default constructor

  double dt = 0.01;
  std::vector<double> y = ODESystem::initial_condition();
  EXPECT_TRUE(integrate(y, dt, callback_fn).converged())
      << "integration failed for dt = " << dt;
  EXPECT_THAT(y, L2ErrorNormLT(ODESystem::analytic_solution(dt), 1.2e-4))
      << "for dt = " << dt;
}

TEST(IntegrateKapsProblem, dt1Div10) {
  using ODESystem = KapsProblem;
  ODESystem callback_fn;  // <- use default constructor

  double dt = 0.1;
  std::vector<double> y = ODESystem::initial_condition();
  EXPECT_TRUE(integrate(y, dt, callback_fn).converged())
      << "integration failed for dt = " << dt;
  EXPECT_THAT(y, L2ErrorNormLT(ODESystem::analytic_solution(dt), 8.9e-3))
      << "for dt = " << dt;
}

namespace {

template <typename Fn>
GRIMPL_NS::integrate::AdaptiveIntegrateResult adaptive_step(
    std::vector<double>& y, double dt, Fn& rhs, int energy_index = 0,
    double relative_tolerance = 1.e-3, double max_change = 0.1,
    double minimum_dt = 1.e-14, int max_retries = 40) {
  GRIMPL_NS::integrate::AdaptiveBackwardEuler solver(
    static_cast<int>(y.size()), relative_tolerance, max_change, max_retries);
  std::vector<double> scratch(solver.num_scratch_buf_elements());
  std::vector<double> absolute_tolerance(y.size(), 1.e-13);
  return solver.step(y.data(), dt, minimum_dt, rhs,
                     static_cast<int>(y.size()), energy_index,
                     absolute_tolerance.data(), scratch.data());
}

struct ThermalMode {
  double slope;
  void operator()(const double* y, double* f,
                  GRIMPL_NS::View<double**> jac) const {
    f[0] = slope * (y[0] - 1.);
    jac(0, 0) = slope;
  }
};

}  // namespace

TEST(StiffNewtonRaphson, ComponentScalingPreservesEquivalentUnits) {
  // Coupled exchange conserves y[0] + y[1]/energy_unit. Code-unit choices
  // can put the energy equation dozens of decades above the species equation.
  for (double energy_unit : {1.e-35, 1., 1.e35}) {
    auto rhs = [energy_unit](const double* y, double* f,
                            GRIMPL_NS::View<double**> jac) {
      f[0] = 1000. * (y[1] / energy_unit - y[0]);
      f[1] = -energy_unit * f[0];
      jac(0, 0) = -1000.;
      jac(0, 1) = 1000. / energy_unit;
      jac(1, 0) = 1000. * energy_unit;
      jac(1, 1) = -1000.;
    };
    GRIMPL_NS::integrate::StiffNewtonRaphson solver(2, 1.e-10, 20);
    std::vector<double> scratch(solver.num_scratch_buf_elements());
    double y[] = {0.8, 0.2 * energy_unit};
    double absolute_tolerance[] = {1.e-13, 1.e-13 * energy_unit};
    const auto result = solver.step(y, 0.1, rhs, 2, scratch.data(), true,
                                    1., absolute_tolerance, true);
    ASSERT_TRUE(result.converged()) << "energy_unit=" << energy_unit;
    EXPECT_NEAR(y[0], 0.5 + 0.3 / 201., 2.e-14);
    EXPECT_NEAR(y[1] / energy_unit, 0.5 - 0.3 / 201., 2.e-14);
    EXPECT_NEAR(y[0] + y[1] / energy_unit, 1., 2.e-14);
  }
}

TEST(AdaptiveBackwardEuler, StableEquilibriumAllowsLargeStep) {
  ThermalMode rhs{-1.};
  std::vector<double> y{1.};
  auto result = adaptive_step(y, 1.e20, rhs);
  ASSERT_TRUE(result.converged());
  EXPECT_EQ(result.accepted_dt, 1.e20);
  EXPECT_EQ(y[0], 1.);
  EXPECT_EQ(result.error_norm, 0.);
}

TEST(AdaptiveBackwardEuler, CachedProposalGrowsAfterTransient) {
  ThermalMode rhs{-1.};
  std::vector<double> y{1.3};
  double elapsed = 0.;
  double proposal = 1.e5;
  bool grew = false;
  int substeps = 0;
  while (elapsed < 1.e5 && substeps < 200) {
    const double requested = std::min(proposal, 1.e5 - elapsed);
    auto result = adaptive_step(y, requested, rhs);
    ASSERT_TRUE(result.converged());
    grew = grew || result.suggested_dt > requested;
    elapsed += result.accepted_dt;
    proposal = result.suggested_dt;
    ++substeps;
  }
  EXPECT_TRUE(grew);
  EXPECT_LT(substeps, 200);
  EXPECT_DOUBLE_EQ(elapsed, 1.e5);
  EXPECT_NEAR(y[0], 1., 1.e-8);
}

TEST(AdaptiveBackwardEuler, StableRelaxationUsesTwoHalfSteps) {
  ThermalMode rhs{-1.};
  std::vector<double> y{1.01};
  auto result = adaptive_step(y, 1.e8, rhs);
  ASSERT_TRUE(result.converged());
  EXPECT_EQ(result.accepted_dt, 1.e8);
  const double expected = 1. + .01 / std::pow(1. + .5 * result.accepted_dt, 2);
  EXPECT_DOUBLE_EQ(y[0], expected);
  EXPECT_LT(std::fabs(y[0] - 1.), 1.e-12);
}

TEST(AdaptiveBackwardEuler, ResolvesHotAndColdTransientsThenRelaxes) {
  for (double initial : {0.7, 1.3}) {
    ThermalMode rhs{-1.};
    std::vector<double> y{initial};
    double elapsed = 0.;
    int substeps = 0;
    while (elapsed < 1.e5 && substeps < 200) {
      const double old = y[0];
      auto result = adaptive_step(y, 1.e5 - elapsed, rhs);
      ASSERT_TRUE(result.converged()) << "initial energy = " << initial;
      ASSERT_GT(result.accepted_dt, 0.);
      // |J| = 1: the 10% change limit applies until the mode is slaved by
      // backward Euler (h|J| > 10); then the step may land on equilibrium.
      if (result.accepted_dt <= 10.) {
        EXPECT_LE(std::fabs(y[0] - old), .1 * old * (1. + 1.e-12));
      }
      EXPECT_LE(result.error_norm, 1.);
      elapsed += result.accepted_dt;
      ++substeps;
    }
    EXPECT_LT(substeps, 200);
    EXPECT_DOUBLE_EQ(elapsed, 1.e5);
    EXPECT_NEAR(y[0], 1., 1.e-8);
  }
}

TEST(AdaptiveBackwardEuler, UnstableModeGrowsInsteadOfRelaxingToWrongRoot) {
  ThermalMode rhs{1.};
  for (double initial : {.7, 1.3}) {
    std::vector<double> y{initial};
    auto result = adaptive_step(y, 1.e5, rhs);
    ASSERT_TRUE(result.converged());
    EXPECT_LE(result.accepted_dt, .1);
    EXPECT_GT(std::fabs(y[0] - 1.), std::fabs(initial - 1.));
    const double exact = 1. + (initial - 1.) * std::exp(result.accepted_dt);
    EXPECT_NEAR(y[0], exact, 1.e-3 * initial);
  }
}

TEST(AdaptiveBackwardEuler, ChecksStabilityAtCandidateNotOnlyAtStart) {
  // The thermal derivative starts negative and becomes positive as y passes
  // one. A large BE step can end on that unstable side despite its stable
  // initial derivative. Loose accuracy bounds isolate the stability guard.
  auto rhs = [](const double* y, double* f, GRIMPL_NS::View<double**> jac) {
    f[0] = (y[0] - 1.) * (y[0] - 1.) + .01;
    jac(0, 0) = 2. * (y[0] - 1.);
  };
  std::vector<double> y{.8};
  auto result = adaptive_step(y, 16., rhs, 0, 1., 1.);
  ASSERT_TRUE(result.converged());
  EXPECT_LT(result.accepted_dt, 16.);
  EXPECT_GT(y[0], .8);
  const double final_j = 2. * (y[0] - 1.);
  if (final_j > 0.) { EXPECT_LE(result.accepted_dt * final_j, .1); }
}

TEST(AdaptiveBackwardEuler, PreservesReactionInvariantAndAbsentSpecies) {
  auto rhs = [](const double* y, double* f, GRIMPL_NS::View<double**> jac) {
    f[0] = 1000. * (y[1] - y[0]);
    f[1] = -f[0];
    f[2] = 0.;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) { jac(i, j) = 0.; }
    }
    jac(0, 0) = jac(1, 1) = -1000.;
    jac(0, 1) = jac(1, 0) = 1000.;
  };
  std::vector<double> y{.499, .501, 0.};
  auto result = adaptive_step(y, 10., rhs, -1);
  ASSERT_TRUE(result.converged());
  EXPECT_EQ(result.accepted_dt, 10.);
  EXPECT_NEAR(y[0] + y[1], 1., 1.e-11);
  EXPECT_NEAR(y[0], .5, 1.e-10);
  EXPECT_EQ(y[2], 0.);
}

TEST(AdaptiveBackwardEuler, ZeroSpeciesCanBeProducedWithAbsoluteTolerance) {
  auto rhs = [](const double*, double* f, GRIMPL_NS::View<double**> jac) {
    f[0] = 1.;
    jac(0, 0) = 0.;
  };
  std::vector<double> y{0.};
  auto result = adaptive_step(y, 1., rhs, -1);
  ASSERT_TRUE(result.converged());
  EXPECT_GT(y[0], 0.);
  EXPECT_NEAR(y[0], result.accepted_dt, 1.e-25);
}

TEST(AdaptiveBackwardEuler, RetriesAreBoundedAndFailureRollsBack) {
  int calls = 0;
  auto rhs = [&calls](const double* y, double* f,
                      GRIMPL_NS::View<double**> jac) {
    ++calls;
    // Deliberately reject every attempted update to model an RHS domain
    // failure. The initial evaluation is valid, so retries really execute.
    f[0] = (y[0] > 1.) ? std::numeric_limits<double>::quiet_NaN() : 1.;
    jac(0, 0) = 0.;
  };
  std::vector<double> y{1.};
  auto result = adaptive_step(y, 1., rhs, 0, 1.e-3, .1, 1.e-14, 3);
  EXPECT_FALSE(result.converged());
  EXPECT_EQ(result.retries, 3);
  EXPECT_GT(calls, 1);
  EXPECT_LE(calls, 12);
  EXPECT_EQ(result.accepted_dt, 0.);
  EXPECT_EQ(y[0], 1.);
}

TEST(AdaptiveBackwardEuler, CannotAdvanceBelowCallerProgressFloor) {
  ThermalMode rhs{1.};
  std::vector<double> y{1.3};
  auto result = adaptive_step(y, 1., rhs, 0, 1.e-3, .1, .2);
  EXPECT_FALSE(result.converged());
  EXPECT_EQ(result.accepted_dt, 0.);
  EXPECT_EQ(y[0], 1.3);
}

TEST(IntegrateFiniteChecks, RejectsNaNAndInfiniteRightHandSide) {
  for (double invalid : {std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::infinity()}) {
    auto rhs = [invalid](const double*, double* f,
                         GRIMPL_NS::View<double**> jac) {
      f[0] = invalid;
      jac(0, 0) = 0.;
    };
    std::vector<double> y{1.};
    EXPECT_FALSE(integrate(y, 1., rhs).converged());
    EXPECT_EQ(y[0], 1.);
  }
}

// Cooling that is steep at the start and nearly flat at the solution, as when
// cooling onto a floor or past a table cutoff: y' = -x (1 + B x^4), x = y - 1.
// A Jacobian reused from a steep point shrinks later corrections, so a small
// correction alone does not show that the backward-Euler equation is solved.
namespace {
struct SteepThenFlat {
  double B;
  void operator()(const double* y, double* f,
                  GRIMPL_NS::View<double**> jac) const {
    const double x = y[0] - 1.;
    f[0] = -x * (1. + B * x * x * x * x);
    jac(0, 0) = -(1. + 5. * B * x * x * x * x);
  }
};

// Root of y - y0 - h f(y) = 0, which is increasing in y; bisect on [1, y0].
double steep_then_flat_be_root(double y0, double h, double B) {
  double lo = 1., hi = y0;
  for (int it = 0; it < 200; ++it) {
    const double mid = 0.5 * (lo + hi), x = mid - 1.;
    const double g = mid - y0 + h * x * (1. + B * x * x * x * x);
    (g > 0. ? hi : lo) = mid;
  }
  return 0.5 * (lo + hi);
}

// One modified-Newton backward-Euler solve from y_start, reusing the Jacobian
// evaluated at y_jacobian.
GRIMPL_NS::integrate::IntegrateResult steep_then_flat_solve(
    double B, double h, double y_start, double y_jacobian, double& y_out) {
  SteepThenFlat rhs{B};
  const auto calc_f = [&](const double* y, double* f) {
    double unused[1];
    rhs(y, f, GRIMPL_NS::View<double**>(unused, 1, 1));
  };
  double point[] = {y_jacobian}, f_unused[1], jac_buf[1];
  GRIMPL_NS::View<double**> jac(jac_buf, 1, 1);
  rhs(point, f_unused, jac);
  GRIMPL_NS::integrate::StiffNewtonRaphson solver(1, 1.e-6, 20);
  std::vector<double> scratch(solver.num_scratch_buf_elements());
  double y[] = {y_start};
  const double atol[] = {1.e-13};
  const auto result = solver.step_modified(
    y, h, calc_f, rhs, 1, scratch.data(), atol, &jac, nullptr,
    y_jacobian == y_start);
  y_out = y[0];
  return result;
}
}  // namespace

TEST(StiffNewtonRaphson, StaleJacobianFirstCorrectionIsNotConvergence) {
  // The second half step starts at the midpoint but reuses the start-state
  // Jacobian. Here it is ~1e6 times steeper than the local one: the first
  // correction is ~1e-8 although the iterate is 5% from the root.
  double y;
  const auto result = steep_then_flat_solve(1.e6, 1.e3, 1.05, 2., y);
  ASSERT_TRUE(result.converged());
  EXPECT_GT(result.iterations, 1);
  const double exact = steep_then_flat_be_root(1.05, 1.e3, 1.e6);
  EXPECT_NEAR(y, exact, 1.e-5 * exact);
}

TEST(StiffNewtonRaphson, ReusedJacobianConvergesToTheRootOrFails) {
  // Steep at the start, flat at the solution, over a range of stiffness.
  // Failing to converge is safe (the caller shortens the step); reporting
  // convergence on a wrong answer is not.
  for (double B : {1.e2, 1.e4, 1.e6, 1.e9, 1.e12}) {
    for (double h : {1.e-3, 1.e-1, 1.e1, 1.e3}) {
      double y;
      const auto result = steep_then_flat_solve(B, h, 2., 2., y);
      const double exact = steep_then_flat_be_root(2., h, B);
      if (result.converged()) {
        EXPECT_NEAR(y, exact, 1.e-5 * exact) << "B=" << B << " h=" << h;
      }
    }
  }
}

// A smooth switch from steep to flat cooling: y' = -x (1 + B s(x)), x = y - 1,
// s a sigmoid at x = 0.5. The first full-Newton step from the steep side lands
// in the flat region; the next correction, made with the steep Jacobian, is
// ~1e7 times too small. That single sharp drop looks like fast contraction,
// but the equation is unsolved: acceptance must be validated with a Jacobian
// evaluated at the current iterate.
namespace {
struct SmoothSteepToFlat {
  double B = 1.e6, x_c = 0.5, w = 0.01;
  void operator()(const double* y, double* f,
                  GRIMPL_NS::View<double**> jac) const {
    const double x = y[0] - 1.;
    const double s = 1. / (1. + std::exp(-(x - x_c) / w));
    const double ds = s * (1. - s) / w;
    f[0] = -x * (1. + B * s);
    jac(0, 0) = -(1. + B * s) - x * B * ds;
  }
};
}  // namespace

TEST(StiffNewtonRaphson, SharpDropInCorrectionIsNotConvergence) {
  SmoothSteepToFlat rhs;
  const auto calc_f = [&](const double* y, double* f) {
    double unused[1];
    rhs(y, f, GRIMPL_NS::View<double**>(unused, 1, 1));
  };
  for (double h : {1., 10., 100.}) {
    double y[] = {2.}, f0[1], jac_buf[1];
    GRIMPL_NS::View<double**> jac(jac_buf, 1, 1);
    rhs(y, f0, jac);
    GRIMPL_NS::integrate::StiffNewtonRaphson solver(1, 1.e-6, 20);
    std::vector<double> scratch(solver.num_scratch_buf_elements());
    const double atol[] = {1.e-13};
    const auto result = solver.step_modified(y, h, calc_f, rhs, 1, scratch.data(),
                                             atol, &jac, f0, true);
    // y - 2 - h f(y) is increasing in y: bisect for the backward-Euler root.
    double lo = 1., hi = 2.;
    for (int it = 0; it < 200; ++it) {
      const double mid = 0.5 * (lo + hi), x = mid - 1.;
      const double s = 1. / (1. + std::exp(-(x - rhs.x_c) / rhs.w));
      ((mid - 2. + h * x * (1. + rhs.B * s)) > 0. ? hi : lo) = mid;
    }
    const double exact = 0.5 * (lo + hi);
    // Failing to converge is safe (the caller shortens the step); reporting
    // convergence on a wrong answer is not. For h >= 10 the root is on the
    // flat side, the case a reused steep Jacobian used to accept falsely.
    if (result.converged()) {
      EXPECT_NEAR(y[0], exact, 1.e-5 * exact) << "h=" << h;
    }
    if (h >= 10.) { EXPECT_TRUE(result.converged()) << "h=" << h; }
  }
}

namespace {
// A steep cooling branch followed by constant cooling. The smooth version
// replaces max(x, 0) by a softplus of width 1e-10, so this regression does not
// rely on a discontinuity of the Jacobian.
struct ConstantCoolingTail {
  bool smooth;
  static constexpr double rate = 1.e8;
  void operator()(const double* y, double* f,
                  GRIMPL_NS::View<double**> jac) const {
    const double x = y[0] - 1.;
    if (!smooth) {
      f[0] = -1. - rate * std::max(x, 0.);
      jac(0, 0) = x > 0. ? -rate : 0.;
      return;
    }
    constexpr double width = 1.e-10;
    const double z = x / width;
    const double softplus = z > 30. ? z : std::log1p(std::exp(z));
    const double logistic = z >= 0. ? 1. / (1. + std::exp(-z))
                                     : std::exp(z) / (1. + std::exp(z));
    f[0] = -1. - rate * width * softplus;
    jac(0, 0) = -rate * logistic;
  }
};
}  // namespace

TEST(StiffNewtonRaphson, ReturnedStateSatisfiesEquationAfterSlopeTransition) {
  // The second case is the midpoint from the formerly failing adaptive
  // step: even a current Jacobian gives a sub-tolerance correction that
  // crosses the transition. Check the equation at the returned state.
  for (bool smooth : {false, true}) {
    ConstantCoolingTail rhs{smooth};
    const auto calc_f = [&](const double* y, double* f) {
      double unused;
      rhs(y, f, GRIMPL_NS::View<double**>(&unused, 1, 1));
    };
    for (const auto initial_and_step :
         {std::pair<double, double>{1.3, .5},
          std::pair<double, double>{1.000000109999956, .025}}) {
      const double initial = initial_and_step.first;
      const double h = initial_and_step.second;
      double y = initial, f0, jac_buf;
      GRIMPL_NS::View<double**> jac(&jac_buf, 1, 1);
      rhs(&y, &f0, jac);
      GRIMPL_NS::integrate::StiffNewtonRaphson solver(1, 1.e-6, 20);
      std::vector<double> scratch(solver.num_scratch_buf_elements());
      const double atol = 1.e-17;
      const auto result = solver.step_modified(
        &y, h, calc_f, rhs, 1, scratch.data(), &atol, &jac, &f0, true);
      ASSERT_TRUE(result.converged()) << "smooth=" << smooth << " h=" << h;
      double f_end;
      calc_f(&y, &f_end);
      const double residual = y - initial - h * f_end;
      EXPECT_LE(std::fabs(residual), 2.e-6 * (std::fabs(y) + initial))
        << "smooth=" << smooth << " h=" << h;
      // Both roots are below the transition, where f = -1 (to roundoff
      // also in the smooth case).
      EXPECT_NEAR(y, initial - h, 5.e-6);
    }
  }
}

TEST(AdaptiveBackwardEuler, SteepToConstantCoolingTrajectoryRemainsAccurate) {
  for (bool smooth : {false, true}) {
    for (double initial : {1.3, 1.0000005}) {
      ConstantCoolingTail rhs{smooth};
      // Keep the production nonlinear tolerance, with a tighter local
      // integration tolerance for comparison to the full analytic trajectory.
      GRIMPL_NS::integrate::AdaptiveBackwardEuler solver(
        1, 1.e-4, .1, 40, 1.e-6);
      std::vector<double> scratch(solver.num_scratch_buf_elements());
      const double atol = 1.e-13;
      double y = initial, elapsed = 0., proposal = .5;
      int steps = 0;
      while (elapsed < .5 && steps < 1000) {
        const double floor = std::max(1.e-18, std::nextafter(
          elapsed, std::numeric_limits<double>::infinity()) - elapsed);
        const auto result = solver.step(
          &y, std::min(proposal, .5 - elapsed), floor, rhs, 1, 0,
          &atol, scratch.data());
        ASSERT_TRUE(result.converged())
          << "smooth=" << smooth << " initial=" << initial
          << " elapsed=" << elapsed;
        ASSERT_GT(result.accepted_dt, 0.);
        EXPECT_LE(result.error_norm, 1.);
        elapsed += result.accepted_dt;
        proposal = result.suggested_dt;
        ++steps;
      }
      ASSERT_LT(steps, 1000);
      EXPECT_DOUBLE_EQ(elapsed, .5);
      // Above y=1, x'=-(1+rate*x); below it, y'=-1. The time to reach
      // y=1 is log(1+rate*(initial-1))/rate. Softening a width of 1e-10
      // changes this reference by far less than the integration tolerance.
      const double crossing_time =
        std::log1p(ConstantCoolingTail::rate * (initial - 1.)) /
        ConstantCoolingTail::rate;
      const double expected = .5 + crossing_time;
      EXPECT_NEAR(y, expected, 1.e-3)
        << "smooth=" << smooth << " initial=" << initial;
    }
  }
}

TEST(AdaptiveBackwardEuler, ThermalGuardUsesDerivativeAtReturnedEndpoint) {
  const auto rhs = [](const double* y, double* f,
                      GRIMPL_NS::View<double**> jac) {
    const double threshold = 1. + 4.e-7;
    f[0] = 1.e-6 + std::max(y[0] - threshold, 0.);
    jac(0, 0) = y[0] > threshold ? 1. : 0.;
  };
  GRIMPL_NS::integrate::AdaptiveBackwardEuler solver(
    1, 1.e-3, .1, 40, 1.e-6);
  std::vector<double> scratch(solver.num_scratch_buf_elements());
  const double atol = 1.e-13;
  double y = 1.;
  const auto result = solver.step(
    &y, 1., 1.e-14, rhs, 1, 0, &atol, scratch.data());
  ASSERT_TRUE(result.converged());
  double f_end, actual_j;
  rhs(&y, &f_end, GRIMPL_NS::View<double**>(&actual_j, 1, 1));
  // A tiny last correction can cross the threshold. Using its pre-correction
  // J=0 used to accept h=1 even though the returned state's J=1.
  EXPECT_LT(result.accepted_dt, 1.);
  EXPECT_LE(result.accepted_dt * std::max(0., actual_j), .1 * (1. + 1.e-12));
}

TEST(AdaptiveBackwardEuler, NonfiniteReturnedEndpointFailsWithoutCommit) {
  for (double invalid : {std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::infinity()}) {
    const auto rhs = [invalid](const double* y, double* f,
                              GRIMPL_NS::View<double**> jac) {
      f[0] = y[0] > 1. + 8.e-7 ? invalid : 1.e-6;
      jac(0, 0) = 0.;
    };
    GRIMPL_NS::integrate::AdaptiveBackwardEuler solver(
      1, 1.e-3, .1, 40, 1.e-6);
    std::vector<double> scratch(solver.num_scratch_buf_elements());
    const double atol = 1.e-13;
    double y = 1.;
    // Every allowed substep reaches the invalid domain; a shorter retry is
    // below the caller's progress floor. The original input must survive.
    const auto result = solver.step(
      &y, 1., .9, rhs, 1, 0, &atol, scratch.data());
    EXPECT_FALSE(result.converged());
    EXPECT_EQ(result.accepted_dt, 0.);
    EXPECT_EQ(y, 1.);
  }
}

TEST(AdaptiveBackwardEuler, FloatQuantizedStiffRelaxationAcceptsLargeStep) {
  // The continuous BE root is less than one float ULP above equilibrium.
  // No solution of y-y0-h*f(float(y)) exists in that rounding bin: requiring
  // a zero unscaled residual would turn this stable relaxation into tiny steps.
  const auto rhs = [](const double* y, double* f,
                      GRIMPL_NS::View<double**> jac) {
    f[0] = 1. - static_cast<double>(static_cast<float>(y[0]));
    jac(0, 0) = -1.;
  };
  constexpr double precision = std::numeric_limits<float>::epsilon();
  constexpr double denormal = std::numeric_limits<float>::denorm_min();
  GRIMPL_NS::integrate::AdaptiveBackwardEuler solver(
    1, 1.e-3, .1, 40, 32. * precision, 1.e-2, precision, denormal);
  std::vector<double> scratch(solver.num_scratch_buf_elements());
  const double atol = 1.e-13;
  const double initial = static_cast<double>(1.01f);
  double y = initial;
  const auto result = solver.step(
    &y, 1.e8, 1.e-14, rhs, 1, 0, &atol, scratch.data());
  ASSERT_TRUE(result.converged());
  EXPECT_EQ(result.accepted_dt, 1.e8);
  EXPECT_LE(result.retries, 1);
  EXPECT_LE(result.newton_iterations, 60);
  EXPECT_NEAR(y, 1., 4. * precision);
  EXPECT_LE(result.error_norm, 1.);
}

TEST(StiffNewtonRaphson, FloatRoundoffCannotHideWrongSideOfCoolingTransition) {
  constexpr double precision = std::numeric_limits<float>::epsilon();
  constexpr double denormal = std::numeric_limits<float>::denorm_min();
  for (bool finite_difference : {false, true}) {
    const auto calc_f = [](const double* y, double* f) {
      const double stored = static_cast<double>(static_cast<float>(y[0]));
      f[0] = -1. - 1.e8 * std::max(stored - 1., 0.);
    };
    const auto rhs = [&](const double* y, double* f,
                         GRIMPL_NS::View<double**> jac) {
      calc_f(y, f);
      const double stored = static_cast<double>(static_cast<float>(y[0]));
      if (finite_difference) {
        // Match the production float Jacobian perturbation and denominator.
        // Its wide probe can straddle the transition even when y is on the
        // flat side, so |J|*epsilon alone is not a safe residual allowance.
        const double perturbed = static_cast<double>(static_cast<float>(
          stored + 1.e-2 * std::max(std::fabs(stored), 1.e-40)));
        double offset_f;
        calc_f(&perturbed, &offset_f);
        jac(0, 0) = (offset_f - f[0]) / (perturbed - stored);
      } else {
        jac(0, 0) = stored > 1. ? -1.e8 : 0.;
      }
    };
    const double initial = static_cast<double>(
      static_cast<float>(1.000000109999956));
    constexpr double h = .025;
    double y = initial, f0, jac_buf;
    GRIMPL_NS::View<double**> jac(&jac_buf, 1, 1);
    rhs(&y, &f0, jac);
    GRIMPL_NS::integrate::StiffNewtonRaphson solver(
      1, 32. * precision, 20, precision, denormal);
    std::vector<double> scratch(solver.num_scratch_buf_elements());
    const double atol = 1.e-17;
    const auto result = solver.step_modified(
      &y, h, calc_f, rhs, 1, scratch.data(), &atol, &jac, &f0, true);
    // A forward-difference Jacobian across this sharp artificial feature may
    // need a smaller substep. Failure is safe; accepting y~1 with residual
    // ~0.025 is not. The analytic Jacobian should solve this simple case.
    if (!finite_difference) { ASSERT_TRUE(result.converged()); }
    if (result.converged()) {
      double f_end;
      calc_f(&y, &f_end);
      const double residual = y - initial - h * f_end;
      EXPECT_NEAR(y, initial - h, 1.e-5)
        << "finite_difference=" << finite_difference;
      EXPECT_LE(std::fabs(residual), 64. * precision * initial)
        << "finite_difference=" << finite_difference;
    }
  }
}

TEST(AdaptiveBackwardEuler, FloatQuantizedCoupledRelaxationPreservesInvariant) {
  // Quantization of both reacting abundances leaves a gap around the exact
  // BE root. A common rounding-neighborhood solution must retain their sum.
  const auto rhs = [](const double* y, double* f,
                      GRIMPL_NS::View<double**> jac) {
    const double left = static_cast<double>(static_cast<float>(y[0]));
    const double right = static_cast<double>(static_cast<float>(y[1]));
    f[0] = right - left;
    f[1] = -f[0];
    jac(0, 0) = jac(1, 1) = -1.;
    jac(0, 1) = jac(1, 0) = 1.;
  };
  constexpr double precision = std::numeric_limits<float>::epsilon();
  constexpr double denormal = std::numeric_limits<float>::denorm_min();
  GRIMPL_NS::integrate::AdaptiveBackwardEuler solver(
    2, 1.e-3, .1, 40, 32. * precision, 1.e-2, precision, denormal);
  std::vector<double> scratch(solver.num_scratch_buf_elements());
  const double atol[] = {1.e-13, 1.e-13};
  double y[] = {static_cast<double>(1.01f), static_cast<double>(.99f)};
  const double initial_sum = y[0] + y[1];
  const auto result = solver.step(
    y, 1.e8, 1.e-14, rhs, 2, -1, atol, scratch.data());
  ASSERT_TRUE(result.converged());
  EXPECT_EQ(result.accepted_dt, 1.e8);
  EXPECT_LE(result.retries, 1);
  EXPECT_LE(result.newton_iterations, 60);
  EXPECT_NEAR(y[0], .5 * initial_sum, 4. * precision);
  EXPECT_NEAR(y[1], .5 * initial_sum, 4. * precision);
  EXPECT_NEAR(y[0] + y[1], initial_sum, 1.e-9);
}

TEST(StiffNewtonRaphson, FloatRoundingRequiresACommonCoupledSolution) {
  // Around (1,1), both equations share K*(q0-q1), but their residuals
  // have opposite offsets. Each row separately brackets zero in the ULP
  // box, at incompatible values of q0-q1. A wide production-style finite
  // difference also sees the stiff cutoff at 1.005, making the fresh Newton
  // correction misleadingly tiny. A common solution of the local residual
  // equations is necessary before accepting the rounding allowance.
  const auto calc_f = [](const double* y, double* f) {
    const double left = static_cast<double>(static_cast<float>(y[0]));
    const double right = static_cast<double>(static_cast<float>(y[1]));
    f[0] = 1.e7 * (left - right) + 1.
           - 1.e12 * std::max(left - 1.005, 0.);
    f[1] = 1.e7 * (left - right) - 1.
           - 1.e12 * std::max(right - 1.005, 0.);
  };
  const auto rhs = [&](const double* y, double* f,
                       GRIMPL_NS::View<double**> jac) {
    calc_f(y, f);
    for (int j = 0; j < 2; ++j) {
      const double base = static_cast<double>(static_cast<float>(y[j]));
      double perturbed[] = {y[0], y[1]}, offset_f[2];
      perturbed[j] = static_cast<double>(static_cast<float>(
        base + 1.e-2 * std::max(std::fabs(base), 1.e-10)));
      calc_f(perturbed, offset_f);
      for (int i = 0; i < 2; ++i) {
        jac(i, j) = (offset_f[i] - f[i]) / (perturbed[j] - base);
      }
    }
  };
  constexpr double precision = std::numeric_limits<float>::epsilon();
  constexpr double denormal = std::numeric_limits<float>::denorm_min();
  GRIMPL_NS::integrate::StiffNewtonRaphson solver(
    2, 32. * precision, 20, precision, denormal);
  std::vector<double> scratch(solver.num_scratch_buf_elements());
  double y[] = {1., 1.}, f0[2], jac_buf[4];
  const double atol[] = {1.e-17, 1.e-17};
  GRIMPL_NS::View<double**> jac(jac_buf, 2, 2);
  rhs(y, f0, jac);
  const auto result = solver.step_modified(
    y, 1., calc_f, rhs, 2, scratch.data(), atol, &jac, f0, true);
  // Failure permits the adaptive caller to shorten the step. If a future
  // solver does converge at h=1, the actual BE root is near the cutoff,
  // (1.005, 1.0049999), not the false rounding-limited state near (1,1).
  if (result.converged()) {
    EXPECT_NEAR(y[0], 1.005, 1.e-5);
    EXPECT_NEAR(y[1], 1.005, 1.e-5);
  }
}

TEST(StiffNewtonRaphson, FailedResidualKeepsAContractingJacobian) {
  // Stiff, mildly nonlinear relaxation: the start Jacobian stays a good
  // approximation, so after the first failed residual check (the unscaled
  // residual is ~h*D times the state error) further corrections with it
  // contract quickly. They must reach the actual root without a rebuild.
  constexpr double D = 1.e6, h = 1., initial = 1.01;
  int jacobian_calls = 0;
  const auto calc_f = [&](const double* y, double* f) {
    const double x = y[0] - 1.;
    f[0] = -D * x * (1. + x);
  };
  const auto rhs = [&](const double* y, double* f,
                       GRIMPL_NS::View<double**> jac) {
    ++jacobian_calls;
    calc_f(y, f);
    jac(0, 0) = -D * (1. + 2. * (y[0] - 1.));
  };
  double y = initial, f0, jac_buf, f_end = -1.;
  GRIMPL_NS::View<double**> jac(&jac_buf, 1, 1);
  rhs(&y, &f0, jac);
  jacobian_calls = 0;
  GRIMPL_NS::integrate::StiffNewtonRaphson solver(1, 1.e-6, 20);
  std::vector<double> scratch(solver.num_scratch_buf_elements());
  const double atol = 1.e-17;
  const auto result = solver.step_modified(
    &y, h, calc_f, rhs, 1, scratch.data(), &atol, &jac, &f0, true, &f_end);
  ASSERT_TRUE(result.converged());
  EXPECT_EQ(jacobian_calls, 0);
  // x = y - 1 solves D x^2 + (D + 1) x - (initial - 1) = 0.
  const double c = initial - 1.;
  const double root =
    1. + 2. * c / ((D + 1.) + std::sqrt((D + 1.) * (D + 1.) + 4. * D * c));
  double f_actual;
  calc_f(&y, &f_actual);
  EXPECT_LE(std::fabs(y - initial - h * f_actual), 1.e-6 * initial);
  EXPECT_NEAR(y, root, 1.e-12);
  // f_out is f at the returned state
  EXPECT_EQ(f_end, f_actual);
}

namespace {
// Start state p = (1, 1), h = 1, R(y) = y - p - f(y), u = y - p, s = u0 + u1.
// Below the diagonal kink s = d the residual is stiff and affine,
// R = sigma*(u - ustar), and its root ustar lies past the kink. Past the kink
// a steep term folds R away from zero, so the backward-Euler equations have
// no root at all. Probes of each coordinate separately over a state-tolerance
// box (|u_j| <= 1e-6 < d) stay below the kink and see only the affine part.
struct DiagonalFold {
  bool smooth;
  static constexpr double sigma = 1.e3, K = 1.e6, d = 1.2e-6, ustar = .8e-6;
  static constexpr double width = 1.e-8;
  void fold(double s, double& v, double& dv) const {
    if (!smooth) {
      v = std::max(s - d, 0.);
      dv = s > d ? 1. : 0.;
      return;
    }
    const double z = (s - d) / width;
    v = width * (z > 30. ? z : std::log1p(std::exp(z)));
    dv = z >= 0. ? 1. / (1. + std::exp(-z)) : std::exp(z) / (1. + std::exp(z));
  }
  void residual(const double* y, double* r) const {
    const double u0 = y[0] - 1., u1 = y[1] - 1.;
    double v, dv;
    fold(u0 + u1, v, dv);
    r[0] = sigma * (u0 - ustar) - K * v;
    r[1] = sigma * (u1 - ustar) - K * v;
  }
  void operator()(const double* y, double* f,
                  GRIMPL_NS::View<double**> jac) const {
    double r[2], v, dv;
    residual(y, r);
    fold(y[0] - 1. + y[1] - 1., v, dv);
    for (int i = 0; i < 2; ++i) {
      f[i] = (y[i] - 1.) - r[i];
      for (int j = 0; j < 2; ++j) {
        jac(i, j) = ((i == j) ? 1. - sigma : 0.) + K * dv;
      }
    }
  }
};
}  // namespace

TEST(StiffNewtonRaphson, RoundingFallbackDoesNotAcceptAModelOnlyRoot) {
  // A reused very stiff Jacobian makes the first corrections tiny, so the
  // solver reaches its rounding fallback at p with a fresh affine Jacobian.
  // A coupled secant model from coordinate probes then predicts a root within
  // the state tolerance, but the actual equations have none: accepting p
  // (residual ~8e-4) would be false convergence. Failure is safe.
  for (bool smooth : {false, true}) {
    DiagonalFold rhs{smooth};
    const auto calc_f = [&](const double* y, double* f) {
      double unused[4];
      rhs(y, f, GRIMPL_NS::View<double**>(unused, 2, 2));
    };
    double y[] = {1., 1.}, f0[2], stale_buf[4] = {1. - 1.e9, 0., 0., 1. - 1.e9};
    calc_f(y, f0);
    GRIMPL_NS::View<double**> stale(stale_buf, 2, 2);
    GRIMPL_NS::integrate::StiffNewtonRaphson solver(
      2, 1.e-6, 20, std::numeric_limits<double>::epsilon(),
      std::numeric_limits<double>::denorm_min());
    std::vector<double> scratch(solver.num_scratch_buf_elements());
    const double atol[] = {1.e-17, 1.e-17};
    const auto result = solver.step_modified(
      y, 1., calc_f, rhs, 2, scratch.data(), atol, &stale, f0, false);
    if (result.converged()) {
      double r[2];
      rhs.residual(y, r);
      EXPECT_LE(std::fabs(r[0]), 2.e-6) << "smooth=" << smooth;
      EXPECT_LE(std::fabs(r[1]), 2.e-6) << "smooth=" << smooth;
    }
  }
}
