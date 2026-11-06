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
