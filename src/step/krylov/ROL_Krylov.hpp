// @HEADER
// *****************************************************************************
//               Rapid Optimization Library (ROL) Package
//
// Copyright 2014 NTESS and the ROL contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef ROL_KRYLOV_H
#define ROL_KRYLOV_H

/** \class ROL::Krylov
    \brief Provides definitions for Krylov solvers.
*/

#include <iostream>

#include "ROL_Vector.hpp"
#include "ROL_LinearOperator.hpp"
#include "ROL_ParameterList.hpp"

namespace ROL {

template<class Real>
class Krylov {

  Real absTol_;      // Absolute residual tolerance
  Real relTol_;      // Relative residual tolerance
  unsigned  maxit_;  // Maximum number of iterations
  int verbosity_;    // Per-iterate diagnostic output level

public:
  virtual ~Krylov(void) {}

  Krylov( Real absTol = 1.e-4, Real relTol = 1.e-2, unsigned maxit = 100 )
    : absTol_(absTol), relTol_(relTol), maxit_(maxit), verbosity_(0) {}

  Krylov( ROL::ParameterList &parlist ) {
    ROL::ParameterList &krylovList = parlist.sublist("General").sublist("Krylov");
    absTol_ = krylovList.get("Absolute Tolerance", 1.e-4);
    relTol_ = krylovList.get("Relative Tolerance", 1.e-2);
    maxit_  = krylovList.get("Iteration Limit", 100);
    verbosity_ = krylovList.get("Verbosity", 0);
  }

  // Run Krylov Method
  virtual Real run( Vector<Real> &x, LinearOperator<Real> &A,
              const Vector<Real> &b, LinearOperator<Real> &M,
                    int &iter, int &flag, std::ostream &outStream = std::cout ) = 0;

  void resetAbsoluteTolerance(const Real absTol) {
    absTol_ = absTol;
  }
  void resetRelativeTolerance(const Real relTol) {
    relTol_ = relTol;
  }
  void resetMaximumIteration(const unsigned maxit) {
    maxit_ = maxit;
  }
  void resetVerbosity(const int verbosity) {
    verbosity_ = verbosity;
  }
  Real getAbsoluteTolerance(void) const {
    return absTol_;
  }
  Real getRelativeTolerance(void) const {
    return relTol_;
  }
  unsigned getMaximumIteration(void) const {
    return maxit_;
  }
  int getVerbosity(void) const {
    return verbosity_;
  }
};

}

#endif
