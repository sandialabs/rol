// @HEADER
// ************************************************************************
//
//               Rapid Optimization Library (ROL) Package
//                 Copyright (2014) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact lead developers:
//              Drew Kouri   (dpkouri@sandia.gov) and
//              Denis Ridzal (dridzal@sandia.gov)
//
// ************************************************************************
// @HEADER

#ifndef ROL_CAUCHYPOINT_P_H
#define ROL_CAUCHYPOINT_P_H

/** \class ROL::CauchyPoint_P
    \brief Provides interface for the Cauchy point trust-region subproblem solver.
*/

#include "ROL_TrustRegion_P.hpp"
#include "ROL_Types.hpp"

  // Compute Cauchy point, i.e., the minimizer of q(P(x - alpha*g)-x)
  // subject to the trust region constraint ||P(x - alpha*g)-x|| <= del
  //   s     -- The Cauchy step upon return: Primal optimization space vector
  //   alpha -- The step length for the Cauchy point upon return
  //   x     -- The anchor vector x (unchanged): Primal optimization space vector
  //   g     -- The (dual) gradient vector g (unchanged): Primal optimization space vector
  //   del   -- The trust region radius (unchanged)
  //   model -- Trust region model
  //   dwa   -- Dual working array, stores Hessian applied to step
  //   dwa1  -- Dual working array
  //Real dcauchy(Vector<Real>             &s, 
  //             Real                     &alpha, 
  //             Real                     &sval,
  //             Real                     &nval,
  //             const Vector<Real>       &x, 
  //             const Vector<Real>       &g,
  //             Real                      del, 
  //             TrustRegionModel_U<Real> &model,
  //             Objective<Real>          &nobj,
  //             int                      flag,
  //             int                      spiter,
  //             std::ostream             &outStream = std::cout);
namespace ROL {

template<typename Real>
class CauchyPoint_P : public TrustRegion_P<Real> {
private:

  Ptr<Vector<Real>> s;
  Ptr<Vector<Real>> px;
  Ptr<Vector<Real>> dwa;
  Ptr<Vector<Real>> dwa1;
  int  redlim_;    ///< Maximum number of Cauchy point reduction steps (default: 10)
  int  explim_;    ///< Maximum number of Cauchy point expansion steps (default: 10)
  Real alpha_;     ///< Initial Cauchy point step length (default: 1.0)
  bool normAlpha_; ///< Normalize initial Cauchy point step length (default: false)
  Real interpf_;   ///< Backtracking rate for Cauchy point computation (default: 1e-1)
  Real extrapf_;   ///< Extrapolation rate for Cauchy point computation (default: 1e1)
  Real qtol_;      ///< Relative tolerance for computed decrease in Cauchy point computation (default: 1-8)
  Real mu0_;       ///< Sufficient Decrease parameter
  Real t0_;        ///< TR stepsize parameter
  int verbosity_; 

public:
  using TrustRegion_P<Real>::pgstep; 
  
  CauchyPoint_P(ParameterList &parlist) {
    ParameterList &list = parlist.sublist("Step").sublist("TRN").sublist("Cauchy Point"); 
    
  redlim_    = list.sublist("Cauchy Point").get("Maximum Number of Reduction Steps", 10);
  explim_    = list.sublist("Cauchy Point").get("Maximum Number of Expansion Steps", 10);
  alpha_     = list.sublist("Cauchy Point").get("Initial Step Size",                 1.0);
  normAlpha_ = list.sublist("Cauchy Point").get("Normalize Initial Step Size",       false);
  interpf_   = list.sublist("Cauchy Point").get("Reduction Rate",                    0.1);
  extrapf_   = list.sublist("Cauchy Point").get("Expansion Rate",                    10.0);
  qtol_      = list.sublist("Cauchy Point").get("Decrease Tolerance",                1e-8);
  mu0_       = list.get("Sufficient Decrease Parameter",                             1e-2);
  t0_        = parlist.sublist("Status Test").get("Proximal Gradient Parameter",     1.0); 
  verbosity_ = parlist.sublist("Step").sublist("Trust Region").sublist("General").get("Output Level",      0);
  
  }//constructor

  void initialize(const Vector<Real> &x, const Vector<Real> &g) {
    s    = x.clone();
    px   = x.clone();
    dwa  = g.clone();
    dwa1 = g.clone();
    if (normAlpha_) alpha_ /= g.norm(); //from trust-region::init 
  }

  void solve(Vector<Real>             &y,
             Real                     &sval,
             Real                     &nval,
             Vector<Real>             &g,
             const Vector<Real>       &x,
             Real                      del,
             TrustRegionModel_U<Real> &model,
             Objective<Real>          &nobj,
             int                      &SPflag_, 
             int                      &SPiter_,                      
             int                      &nhess_,                      
             int                      &nprox_,                      
             int                      &nnval_,                      
             std::ostream             &outStream) { 
    // setup
    const Real half(0.5), sold(sval), nold(nval);
    Real tol = std::sqrt(ROL_EPSILON<Real>());
    bool interp = false;
    Real gs(0), snorm(0), Qk(0), pRed(0);
    // Compute s = P(x[0] - alpha g[0]) - x[0]
    pgstep(*px, *s, nobj, x, g, alpha_, tol); nprox_++; 
    snorm = s->norm();
    if (snorm > del) {
      interp = true;
    }
    else {
      model.hessVec(*dwa,*s,x,tol); nhess_++;
      nobj.update(*px, UpdateType::Trial);
      nval   = nobj.value(*px, tol); nnval_++;
      gs     = s->dot(g);
      sval   = sold + gs + half * s->apply(*dwa);
      pRed   = (sold + nold) - (sval + nval);
      Qk     = gs + nval - nold;
      interp = (pRed < -mu0_*Qk);
    }
    // Either increase or decrease alpha to find approximate Cauchy point
    int cnt = 0;
    if (interp) {//decrease loop
      while ((pRed < -mu0_*Qk) && (cnt < redlim_) && (snorm <= del)) {
        alpha_ *= interpf_; 
        pgstep(*px, *s, nobj, x, g, alpha_, tol); nprox_++; 
        snorm = s->norm();
        model.hessVec(*dwa,*s,x,tol); nhess_++;
        nobj.update(*px, UpdateType::Trial);
        nval   = nobj.value(*px, tol); nnval_++;
        gs     = s->dot(g);
        sval   = sold + gs + half * s->apply(*dwa);
        pRed   = (sold + nold) - (sval + nval);
        Qk     = gs + nval - nold;
        cnt++;
      }
    }
    else {
      bool search = true;
      Real alphas = alpha_;
      Real mvals  = pRed;
      Real svals  = sval;
      dwa1->set(*dwa);
      while (search) {
        alpha_ *= extrapf_;
        pgstep(*px, *s, nobj, x, g, alpha_, tol); nprox_++; 
        snorm = s->norm();
        if (snorm <= del && cnt < explim_){// && mnew < mold + mu0_*Qk) {
          model.hessVec(*dwa,*s,x,tol); nhess_++;
          nobj.update(*px, UpdateType::Trial);
          nval = nobj.value(*px, tol); nnval_++;
          gs   = s->dot(g);
          sval = sold + gs + half * s->apply(*dwa);
          pRed = (sold + nold) - (sval + nval);
          Qk   = gs + nval - nold;
          if (pRed >= -mu0_*Qk && std::abs(pRed-mvals) > qtol_*std::abs(mvals)) {
            dwa1->set(*dwa);
            alphas = alpha_;
            mvals  = pRed;
            svals  = sval;
            search = true;
          }
          else {
            dwa->set(*dwa1);
            pRed   = mvals;
            sval   = svals;
            search = false;
          }
        }
        else {
          search = false;
        }
        cnt++;
      }
      alpha_ = alphas;
      pgstep(*px, *s, nobj, x, g, alpha_, tol); nprox_++; 
      snorm = s->norm();
    }
    if (verbosity_ > 1) {
      outStream << "    Cauchy point"                         << std::endl;
      outStream << "    Step length (alpha):              " << alpha_ << std::endl;
      outStream << "    Step length (alpha*g):            " << snorm << std::endl;
      outStream << "    Model decrease (pRed):            " << pRed  << std::endl;
      if (!interp)
        outStream << "    Number of extrapolation steps:    " << cnt << std::endl;
    } 
    //clean-up
    y.plus(*s);
    g.plus(*dwa);
    //Real nvalt    = nobj.value(y, tol); nnval_++;
    //Real nvalpt    = nobj.value(*px, tol); nnval_++;
    SPiter_ = cnt; 
    SPflag_ = 2; 
    //return nnval;
  } // solve
}; // CP

} // namespace ROL

#endif
