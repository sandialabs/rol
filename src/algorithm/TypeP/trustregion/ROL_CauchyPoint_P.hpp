// @HEADER
// *****************************************************************************
//               Rapid Optimization Library (ROL) Package
//
// Copyright 2014 NTESS and the ROL contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
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
  Ptr<Vector<Real>> ss;
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
  int verbosity_; 

public:
  using TrustRegion_P<Real>::pgstep; 
  Real t0_;        ///< TR stepsize parameter
  
  CauchyPoint_P(ParameterList &parlist) {
    ParameterList &list = parlist.sublist("Step").sublist("TRN").sublist("Cauchy Point"); 
    
  redlim_    = list.sublist("Cauchy Point").get("Maximum Number of Reduction Steps", 10);
  explim_    = list.sublist("Cauchy Point").get("Maximum Number of Expansion Steps", 10);
  alpha_     = list.sublist("Cauchy Point").get("Initial Step Size",                 1.0);
  normAlpha_ = list.sublist("Cauchy Point").get("Normalize Initial Step Size",       false);
  interpf_   = list.sublist("Cauchy Point").get("Reduction Rate",                    0.1);
  extrapf_   = list.sublist("Cauchy Point").get("Expansion Rate",                    10.0);
  qtol_      = list.sublist("Cauchy Point").get("Decrease Tolerance",                1e-8);
  mu0_       = list.get("Sufficient Decrease Parameter",                             1e-4);
  t0_        = parlist.sublist("Status Test").get("Proximal Gradient Parameter",     1.0); 
  verbosity_ = parlist.sublist("Step").sublist("Trust Region").sublist("General").get("Output Level",      0);
  
  }//constructor

  void initialize(const Vector<Real> &x, const Vector<Real> &g) {
    s    = x.clone();
    ss   = x.clone();
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
    bool extrap = false;
    Real gs(0), snorm(0), Qk(0), pRed(0), sHs(0);
    // Compute s = P(x[0] - alpha g[0]) - x[0]
    pgstep(*px, *s, nobj, x, g, alpha_, tol); nprox_++; 
    snorm = s->norm();
    model.hessVec(*dwa,*s,x,tol); nhess_++;
    gs     = s->dot(g);
    sval   = sold + gs + half * s->apply(*dwa);
    nobj.update(*px, UpdateType::Trial);
    nval   = nobj.value(*px, tol); nnval_++;
    pRed   = (sold + nold) - (sval + nval);
    Qk     = gs + nval - nold;
    extrap = (-pRed <= mu0_*Qk) && (snorm <= (1-tol)*del);
    // Either increase or decrease alpha to find approximate Cauchy point
    int cnt = 0;
    //printf("top %f %f\n", nval + sval, snorm); 
    if (extrap) { // increase loop
      //printf("inc loop\n"); 
      Real alpha1 = alpha_;
      Real pRed1  = pRed;
      Real sval1  = sval; // 
      Real nval1  = nval; // phinew1
      Real snorm1 = snorm; //snorm1
      dwa1->set(*dwa);    // Hs1
      while (cnt < explim_) {
        alpha1 *= extrapf_;
        pgstep(*px, *ss, nobj, x, g, alpha1, tol); nprox_++; 
        snorm1 = ss->norm();
        model.hessVec(*dwa1,*ss,x,tol); nhess_++;
        nobj.update(*px, UpdateType::Trial);
        nval1  = nobj.value(*px, tol); nnval_++;
        gs     = ss->dot(g);
        sHs    = ss->apply(*dwa1); 
        sval1  = sold + gs + half * sHs;
        pRed1  = (sold + nold) - (sval1 + nval1);
        Qk     = gs + nval1 - nold;
        //printf("cp inc %f  %f %f %f\n", -pRed, mu0_*Qk, snorm, nval+sval); 
        if ((-pRed1 > mu0_*Qk && std::abs(pRed-pRed1) > qtol_*std::abs(pRed1)) || (snorm1 > (1-tol)*del)) {
          break; 
        }
        else {
          alpha_ = alpha1; 
          s->set(*ss); 
          snorm  = snorm1;
          nval   = nval1;
          dwa->set(*dwa1);
          pRed   = pRed1;
          sval   = sval1;
        }
        cnt++;
      }
    } // end inc
    else {//decrease loop
      //printf("dec loop\n"); 
      while (((-pRed > mu0_*Qk) || (snorm > (1-tol)*del)) && (cnt < redlim_)) {
        alpha_ *= interpf_; 
        pgstep(*px, *s, nobj, x, g, alpha_, tol); nprox_++; 
        snorm = s->norm();
        model.hessVec(*dwa,*s,x,tol); nhess_++;
        gs     = s->dot(g);
        sHs    = s->apply(*dwa); 
        sval   = sold + gs + half * sHs;
        nobj.update(*px, UpdateType::Trial);
        nval   = nobj.value(*px, tol); nnval_++;
        pRed   = (sold + nold) - (sval + nval);
        Qk     = gs + nval - nold;
        //printf("cp dec %f  %f  %f  %f  %f\n", pRed, -mu0_*Qk, snorm, sval, nval); 
        cnt++;
      }
    }// end dec
    if (verbosity_ > 1) {
      outStream << "    Cauchy point"                       << std::endl;
      outStream << "    Step length (alpha):              " << alpha_ << std::endl;
      outStream << "    Step length (alpha*g):            " << snorm << std::endl;
      outStream << "    Model decrease (pRed):            " << pRed  << std::endl;
      if (extrap)
        outStream << "    Number of extrapolation steps:    " << cnt << std::endl;
    } 
    //clean-up
    y.plus(*s);
    g.plus(*dwa);
    //Real nvalt    = nobj.value(y, tol); nnval_++;
    //Real nvalpt    = nobj.value(*px, tol); nnval_++;
    //printf("%f  %f %f\n", nvalt, nval, nvalpt); 
    //nval = nobj.value(y, tol); nnval_++;  
    //printf("%f  %f \n", sval, sHs*alpha_); 
    SPiter_ = cnt; 
    SPflag_ = 2; 
  } // solve
}; // CP

} // namespace ROL

#endif
