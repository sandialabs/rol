// @HEADER
// *****************************************************************************
//               Rapid Optimization Library (ROL) Package
//
// Copyright 2014 NTESS and the ROL contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef ROL_TRUSTREGION_P_HPP
#define ROL_TRUSTREGION_P_HPP

/** \class ROL::TrustRegion_P
    \brief Provides interface for and implements trust-region subproblem solvers.
*/

#include "ROL_Vector.hpp"
#include "ROL_TrustRegionModel_U.hpp"

namespace ROL {

template<typename Real>
class TrustRegion_P {
public:
  virtual ~TrustRegion_P() {}
  
  virtual void pgstep(Vector<Real>       &pgiter,
                      Vector<Real>       &pgstep,
                      Objective<Real>    &nobj,
                      const Vector<Real> &x,
                      const Vector<Real> &dg,
                      Real                t,
                      Real               &tol) const {
    pgstep.set(x);
    pgstep.axpy(-t,dg);
    nobj.prox(pgiter,pgstep,t,tol);
    //state_->nprox++;
    pgstep.set(pgiter);
    pgstep.axpy(static_cast<Real>(-1),x);
  }
  virtual void initialize(const Vector<Real> &x, const Vector<Real> &g) {}

  virtual void solve(Vector<Real>             &y,
                     Real                     &sval,
                     Real                     &nval,
                     Vector<Real>             &gmod,
                     const Vector<Real>       &x,
                     Real                      del,
                     TrustRegionModel_U<Real> &model,
                     Objective<Real>          &nobj,
                     int                      &SPflag_, 
                     int                      &SPiter_,                      
                     int                      &nhess_,                      
                     int                      &nprox_,                      
                     int                      &nnval_,                      
                     std::ostream             &outStream) = 0; 

};// Trust Region
} // namespace ROL

#endif
