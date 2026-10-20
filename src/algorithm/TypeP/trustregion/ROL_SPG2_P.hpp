// @HEADER
// *****************************************************************************
//               Rapid Optimization Library (ROL) Package
//
// Copyright 2014 NTESS and the ROL contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef ROL_SPG2_P_H
#define ROL_SPG2_P_H

/** \class ROL::SPG2_P
    \brief Provides interface for truncated CG trust-region subproblem solver.
*/

// Use SPG to approximately solve TR subproblem:
//   min 1/2 <H(y-x), (y-x)> + <g, (y-x)>  subject to y\in C, ||y|| \le del

#include "ROL_TrustRegion_P.hpp"
#include "ROL_Types.hpp"

#include <deque>

namespace ROL { 

template<typename Real>
class SPG2_P: public TrustRegion_P<Real> {
private:
  Ptr<Vector<Real>> dwa, s, pwa, pwa0, pwa1, pwa2;
  Real t0_; 
  Real lambdaMin_;
  Real lambdaMax_;
  int maxSize_;
  int maxit_;
  Real tol1_;
  Real tol2_;
  bool useMin_;
  bool useNMSP_;
  int  verbosity_; 

public:
  using TrustRegion_P<Real>::pgstep; 
  // Constructor
  SPG2_P( ParameterList &parlist ) {
    ParameterList &list = parlist.sublist("Step").sublist("Trust Region").sublist("TRN").sublist("Solver").sublist("SPG2");
    verbosity_         = parlist.sublist("Step").sublist("Trust Region").sublist("General").get("Output Level",      0);
    // Spectral projected gradient parameters
    lambdaMin_ = list.sublist("Solver").get("Minimum Spectral Step Size",          1e-12);
    lambdaMax_ = list.sublist("Solver").get("Maximum Spectral Step Size",          1e12);
    maxSize_   = list.sublist("Solver").get("Maximum Storage Size",                10);
    maxit_     = list.sublist("Solver").get("Iteration Limit",                     15);
    tol1_      = list.sublist("Solver").get("Absolute Tolerance",                  1e-4);
    tol2_      = list.sublist("Solver").get("Relative Tolerance",                  1e-2);
    useMin_    = list.sublist("Solver").get("Use Smallest Model Iterate",          true);
    useNMSP_   = list.sublist("Solver").get("Use Nonmonotone Search",              false);
   t0_         = list.sublist("Status Test").get("Proximal Gradient Parameter", 1.0);
  }

  void initialize(const Vector<Real> &x, const Vector<Real> &g) {
    s    = x.clone();
    pwa  = x.clone();
    pwa0 = x.clone();
    pwa1 = x.clone();
    pwa2 = x.clone();
    dwa  = g.clone();
  }

  void solve(Vector<Real>             &y,
             Real                     &sval,
             Real                     &nval,
             Vector<Real>             &gmod,
             const Vector<Real>       &x,
             Real                     del,
             TrustRegionModel_U<Real> &model,
             Objective<Real>          &nobj,
             int                      &SPflag_, 
             int                      &SPiter_,                      
             int                      &nhess_, 
             int                      &nprox_,                      
             int                      &nnval_,                      
             std::ostream             &outStream
    ) {

    const Real half(0.5), one(1), safeguard(1e2*ROL_EPSILON<Real>());
    Real pRed(0), snorm(0), snorm0(0); 
    Real tol(std::sqrt(ROL_EPSILON<Real>()));
    Real alpha(1), alphaMax(1), lambda(1), lambdaTmp(1);
    Real gs(0), ds(0), dd(0),  gnorm(0), sHs(0), sold(sval), nold(nval);
    pwa1->zero();

    // Set y = x
    y.set(x); // y = x0 
    // take gcp
    gcp2(*pwa1, sval, nval, gmod, x, del, model, nobj, SPflag_, SPiter_, nhess_, nprox_, nnval_, outStream); 
    // y should be x0 (check), pwa1 is x1, s is already stored
    s->set(*pwa1); 
    s->axpy(-one, x); 
    // Compute initial step
    gnorm  = s->norm();                            // norm(step) / lambda

    // Compute initial projected gradient norm
    const Real gtol = std::min(tol1_,tol2_*std::pow((gnorm/t0_),2));

    if (verbosity_ > 1)
      outStream << "  Spectral Projected Gradient 2"          << std::endl;

    SPiter_ = 0;
    while (SPiter_ < maxit_) {
      SPiter_++;
      // Perform line search
      alphaMax = one;
      snorm0   = snorm;
      pwa2->set(*pwa1); pwa2->axpy(-one, x); // just for snorm 
      snorm    =  pwa2->norm();
      if (snorm >= (one - tol)*del){
        pwa0->set(y); pwa0->axpy(-one, x); // just for x0 norm 
        ds = s->dot(*pwa0);
        dd = std::pow(gnorm,2); 
        alphaMax = std::min(one, (-ds + std::sqrt(std::pow(ds,2) + dd * (std::pow(del,2) - std::pow(snorm0, 2))))/dd);
      } 
      // Evaluate model Hessian
      model.hessVec(*dwa,*s,x,tol); nhess_++; // dwa = H step
      sHs   = s->apply(dwa->dual());                   // sHs = <step, H step>
      gs    = s->apply(gmod.dual());
      // Evaluate nonsmooth term
      nobj.update(*pwa1,UpdateType::Trial);
      nval = nobj.value(*pwa1,tol); nnval_++;
      if (sHs <= safeguard*std::pow(gnorm,2)){
        alpha = alphaMax; 
      }
      else {
        alpha = std::min(alphaMax, -(gs + nval - nold)/sHs); 
      }
      // Update model quantities
      if (alpha == one) {
        y.set(*pwa1);
        sval = sold + gs + half * sHs;
        gmod.plus(*dwa);
      }
      else {
        y.axpy(alpha,*s); // New iterate
        nobj.update(y,UpdateType::Trial);
        nval  = nobj.value(y, tol); nnval_++;
        sval  = sold + alpha * (gs + half * alpha * sHs);
        gmod.axpy(alpha,*dwa);
        pwa->set(y); pwa->axpy(-one, x); // just for norm 
        snorm = pwa->norm(); 
      }
      // update model information
      sold = sval; 
      nold = nval; 
      if (snorm >= (one - safeguard)*del) { // Trust-region constraint is violated
        SPflag_ = 2;  
        break; 
      }
        
      if (sHs <= safeguard*std::pow(gnorm,2)) {
        lambdaTmp = t0_/gmod.norm(); 
      }
      else {
        lambdaTmp = std::pow(gnorm,2)/sHs; 
      } 
      // Compute new spectral step
      lambda = std::max(lambdaMin_, std::min(lambdaMax_, lambdaTmp)); 
      pwa->set(y); pwa->axpy(-lambda, gmod.dual()); 
      nobj.prox(*pwa1, *pwa, lambda, tol); nprox_++; // sets x1 
      //pgstep(*pwa1, *pwa, nobj, y, gmod.dual(), alpha, tol); nprox_++; //pwa1 = x1, pwa is evaluation 
      s->set(*pwa1); s->axpy(-one, y); 
      gnorm = s->norm(); 
      if (gnorm/t0_ <= gtol) { SPflag_ = 0; break; }


      if (verbosity_ > 1) {
        outStream << std::endl;
        outStream << "    Iterate:                          " << SPiter_ << std::endl;
        outStream << "    Spectral step length (lambda):    " << lambda  << std::endl;
        outStream << "    Step length (alpha):              " << alpha   << std::endl;
        outStream << "    Model decrease (pRed):            " << pRed    << std::endl;
        outStream << "    Optimality criterion:             " << gnorm   << std::endl;
        outStream << "    Step norm:                        " << snorm   << std::endl;
        outStream << std::endl;
      }

    } // end loop
    SPflag_ = (SPiter_==maxit_) ? 1 : SPflag_;
}; // solve
void gcp2( Vector<Real>             &y,        
           Real                     &sval,
           Real                     &nval,   
           Vector<Real>             &gmod,
           const Vector<Real>       &x,
           Real                     del,
           TrustRegionModel_U<Real> &model,
           Objective<Real>          &nobj,
           int                      &SPflag_, 
           int                      &SPiter_,  
           int                      &nhess_, 
           int                      &nprox_,   
           int                      &nnval_,   
           std::ostream             &outStream) {
   // take a gcp2 step:: TODO maybe abstract this out?     
   Real gg(0), gHg(0), t0Tmp(0), lam(0), alpha(1.0), snorm(0);
   Real sHs(0), gs(0), nold(nval); 
   Real one(1.0), half(0.5);  
   Real tol(std::sqrt(ROL_EPSILON<Real>()));
   // Compute cp as single spg step
   pwa2->set(gmod.dual()); 
   model.hessVec(*dwa, *pwa2, x, tol);
   gHg = dwa->apply(gmod);
   gg  = pwa2->apply(*pwa2);  
   
   if (gHg > tol*gg) {
     t0Tmp = gg / gHg; 
   }
   else{
     t0Tmp = t0_ / std::sqrt(gg); 
   }
   lam = std::min(lambdaMax_, std::max(lambdaMin_, t0Tmp));
   //xc = pwa1, s = pwa
   pwa->set(x); pwa->axpy(-lam, *pwa2); 
   nobj.prox(*pwa1, *pwa, lam, tol); nprox_++; 
   s->set(*pwa1); s->axpy(-one, x); 
   snorm = s->norm(); 

   // Hs
   model.hessVec(*dwa, *s, x, tol); 
   sHs = s->apply(*dwa);
   gs  = pwa2->dot(*s); //pwa2 = gmod.dual()
   
   nobj.update(*pwa1, UpdateType::Trial); 
   nval = nobj.value(*pwa1, tol); nnval_++;
   
   // Spectral Step
   if (snorm >= (one - tol)*del) {
     alpha = std::min(one, del/snorm); 
   } 
   if (sHs > tol*std::pow(snorm,2)) {
     alpha = std::min(alpha, -(gs + nval - nold)/sHs); 
   }
   if (alpha != one){
     s->scale(alpha); 
     snorm *= alpha;
     gs    *= alpha; 
     dwa->scale(alpha); 
     sHs   *= std::pow(alpha,2); 
     pwa1->set(x); pwa1->axpy(one, *s); 
     
     nobj.update(*pwa1, UpdateType::Trial); 
     nval = nobj.value(*pwa1, tol); nnval_++;
   }
   sval += gs + half*sHs; 
   y.set(*pwa1);  
}  // gcp2
}; // SPG  
}  // namespace ROL

#endif
