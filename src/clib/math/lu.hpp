//===----------------------------------------------------------------------===//
// See the LICENSE file for license and copyright information
// SPDX-License-Identifier: NCSA AND BSD-3-Clause
//===----------------------------------------------------------------------===//
#ifndef MATH_LU_HPP
#define MATH_LU_HPP

#include <cmath>
#include <utility>

#include "gaussj.hpp"  // eps_zero
#include "../support/config.hpp"

#if defined(__GNUC__) || defined(__clang__)
#define GRIMPL_LU_RESTRICT __restrict__
#else
#define GRIMPL_LU_RESTRICT
#endif

namespace GRIMPL_NAMESPACE_DECL {

/// Factors of the scaled (I - h J) for a Jacobian that several solves share.
/// A solve with the same J and h reuses them instead of factorising again.
struct SharedLU {
  static constexpr int max_n = 64;
  bool valid = false;
  int n = 0;
  double h = 0.;
  int piv[max_n];
  double scale[max_n];
  double lu[max_n * max_n];
};


/// Row-pivoted LU of an n x n row-major matrix, in place. Unlike gaussj the
/// factors are kept, so further right-hand sides cost only two triangular
/// solves. Returns 1 for a pivot below eps_zero (the gaussj test), else 0.
inline int lu_factor(int n, double* a, int* piv) {
  for (int col = 0; col < n; col++) {
    int p = col;
    double big = std::fabs(a[col * n + col]);
    for (int row = col + 1; row < n; row++) {
      const double v = std::fabs(a[row * n + col]);
      if (v > big) { big = v; p = row; }
    }
    if (big < eps_zero) { return 1; }
    piv[col] = p;
    double* GRIMPL_LU_RESTRICT pivot_row = a + col * n;
    if (p != col) {
      double* other = a + p * n;
      for (int c = 0; c < n; c++) { std::swap(pivot_row[c], other[c]); }
    }
    const double inv = 1.0 / pivot_row[col];
    for (int row = col + 1; row < n; row++) {
      double* GRIMPL_LU_RESTRICT r = a + row * n;
      const double f = r[col] * inv;
      r[col] = f;
      if (f == 0.0) { continue; }  // structural zero: nothing to eliminate
      for (int c = col + 1; c < n; c++) { r[c] -= f * pivot_row[c]; }
    }
  }
  return 0;
}

/// Solve with the factors from lu_factor; b is overwritten by the solution.
inline void lu_solve(int n, const double* a, const int* piv, double* b) {
  for (int i = 0; i < n; i++) {
    std::swap(b[i], b[piv[i]]);
    double s = b[i];
    for (int c = 0; c < i; c++) { s -= a[i * n + c] * b[c]; }
    b[i] = s;
  }
  for (int i = n - 1; i >= 0; i--) {
    double s = b[i];
    for (int c = i + 1; c < n; c++) { s -= a[i * n + c] * b[c]; }
    b[i] = s / a[i * n + i];
  }
}

}  // namespace GRIMPL_NAMESPACE_DECL

#endif  // MATH_LU_HPP
