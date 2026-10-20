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

#ifndef ROL_SPG_P_H
#define ROL_SPG_P_H

/** \class ROL::SPG_P
    \brief Provides interface for the nonsmooth SPG trust-region subproblem solver.
*/

#include "ROL_TrustRegion_P.hpp"
#include "ROL_CauchyPoint_P.hpp"
#include "ROL_Types.hpp"

#include <deque>

namespace ROL { 

template<typename Real>
class SPG_P : public TrustRegion_P<Real> {
private:
  Ptr<CauchyPoint_P<Real>> CP_; 

  Ptr<Vector<Real>> ymin, dwa, pwa, pwa1, pwa2, pwa3, pwa4, pwa5; 
  Real alpha_; 
  Real lambdaMin_;
  Real lambdaMax_;
  Real gamma_;
  int maxSize_;
  int maxit_;
  Real tol1_;
  Real tol2_;
  bool useMin_;
  bool useNMSP_;
  int  verbosity_;
  Real t0_; 
  Real mu0_;  
  Real interpf_;  
public:
  using TrustRegion_P<Real>::pgstep; 
  SPG_P(ParameterList &parlist) {
   ParameterList &list = parlist.sublist("Step").sublist("Trust Region").sublist("TRN").sublist("Solver").sublist("SPG");
   
   // Cauchy Point
   CP_ = makePtr<CauchyPoint_P<Real>>(parlist); 
   verbosity_         = parlist.sublist("Step").sublist("Trust Region").sublist("General").get("Output Level",      0);
   // Subsolver (general) parameters
   lambdaMin_ = list.sublist("Solver").get("Minimum Spectral Step Size",    1e-8);
   lambdaMax_ = list.sublist("Solver").get("Maximum Spectral Step Size",    1e8);
   gamma_     = list.sublist("Solver").get("Sufficient Decrease Tolerance", 1e-4);
   maxSize_   = list.sublist("Solver").get("Maximum Storage Size",          10);
   maxit_     = list.sublist("Solver").get("Iteration Limit",               25);
   tol1_      = list.sublist("Solver").get("Absolute Tolerance",            1e-4);
   tol2_      = list.sublist("Solver").get("Relative Tolerance",            1e-2);
   mu0_       = list.get("Sufficient Decrease Parameter",                             1e-2);
   interpf_   = list.sublist("Cauchy Point").get("Reduction Rate",                    0.1);
   // Subsolver (spectral projected gradient) parameters
   useMin_    = list.sublist("Solver").get("Use Smallest Model Iterate",    true);
   useNMSP_   = list.sublist("Solver").get("Use Nonmonotone Search",        false);
   t0_        = list.sublist("Status Test").get("Proximal Gradient Parameter", 1.0);
   alpha_ = 10; 
  } // constructor

  void initialize(const Vector<Real> &x, const Vector<Real> &g) {
    CP_->initialize(x,g);  
    ymin = x.clone(); 
    pwa  = x.clone(); 
    pwa1 = x.clone(); 
    pwa2 = x.clone(); 
    pwa3 = x.clone(); 
    pwa4 = x.clone(); 
    pwa5 = x.clone(); 
    dwa  = g.clone(); 
  } //init
  void solve(Vector<Real>             &y,
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
             std::ostream             &outStream) { 
  // Use SPG to approximately solve TR subproblem:
  //   min 1/2 <H(y-x), (y-x)> + <g, (y-x)> + phi(y)  subject to  ||y|| \le del
  
  const Real half(0.5), one(1), safeguard(1e2*ROL_EPSILON<Real>());
  const Real mval(sval+nval);
  pwa1->set(y);
  CP_->solve(y, sval, nval, gmod, x, del, model, nobj, SPflag_, SPiter_, nhess_, nprox_, nnval_, outStream); 
  // needs s, snorm, pRed (not really), phinew, gmod 
  pwa1->axpy(-one, y); 
  if (pwa1->norm() >= (1-safeguard)*del) {
    SPflag_=0;
    return; 
  }
  Real tol(std::sqrt(ROL_EPSILON<Real>()));
  Real mcomp(0), mval_min(0), sval_min(0), nval_min(0);
  Real alpha(1), coeff(1), lambda(1), lambdaTmp(1);
  Real snew(sval), nnew(nval), mnew(mval);
  Real sold(sval), nold(nval), mold(mval);
  Real sHs(0), ss(0), gs(0), Qk(0), gnorm(0);
  std::deque<Real> mqueue; mqueue.push_back(mold);

  if (useNMSP_ && useMin_) {
    mval_min = mval; sval_min = sval; nval_min = nval; ymin->set(y);
  }

  // Compute initial proximal gradient norm
  pwa1->set(gmod.dual());
  pwa->set(y); pwa->axpy(-t0_,*pwa1);
  dprox(x,t0_,del,nobj,nprox_,outStream);
  pwa->axpy(-one,y);
  gnorm = pwa->norm() / t0_;
  const Real gtol = std::min(tol1_,tol2_*gnorm);

  // Compute initial spectral step size
  coeff  = one / gmod.norm();
  lambda = std::max(lambdaMin_,std::min(coeff,lambdaMax_));

  if (verbosity_ > 1)
    outStream << "  Spectral Projected Gradient"          << std::endl;

  SPiter_ = 0;
  while (SPiter_ < maxit_) {
    SPiter_++;

    // Compuate SPG step
    alpha = one;
    pwa->set(y); pwa->axpy(-lambda,*pwa1);               // pwa = y - lambda gmod.dual()
    dprox(x,lambda,del,nobj,nprox_,outStream); // pwa = P(y - lambda gmod.dual())
    pwa2->set(*pwa);                        // pwa2 = P(y - lambda gmod.dual())
    pwa->axpy(-one,y);                       // pwa = P(y - lambda gmod.dual()) - y = step
    ss = pwa->dot(*pwa);                    // Norm squared of step

    // Evaluate model Hessian
    model.hessVec(*dwa,*pwa,x,tol); nhess_++; // dwa = H step
    nobj.update(*pwa2, UpdateType::Trial);
    nnew  = nobj.value(*pwa2, tol); nnval_++;
    sHs   = dwa->apply(*pwa);                 // sHs = <step, H step>
    gs    = gmod.apply(*pwa);                // gs  = <step, model gradient>
    snew  = half * sHs + gs + sold;
    mnew  = snew + nnew;
    Qk    = gs + nnew - nold;

    if (useNMSP_) { // Nonmonotone
      mcomp = *std::max_element(mqueue.begin(),mqueue.end());
      while( mnew > mcomp + mu0_*Qk ) {
        alpha *= interpf_;
        pwa2->set(y); pwa2->axpy(alpha, *pwa);
        nobj.update(*pwa2, UpdateType::Trial);
        nnew  = nobj.value(*pwa2, tol); nnval_++;
        snew  = half * alpha * alpha * sHs + alpha * gs + sold;
        mnew  = nnew + snew;
        Qk    = alpha * gs + nnew - nold;
      }
    }
    else {
      alpha = (sHs <= safeguard) ? one : std::min(one,-(gs + nnew - nold)/sHs);
    }

    // Update model quantities
    y.set(*pwa2);
    sold = snew;
    nold = nnew;
    mold = mnew;
    gmod.axpy(alpha,*dwa);                      // Update model gradient
    nobj.update(y, UpdateType::Accept);

    // Update nonmonotone line search information
    if (useNMSP_) {
      if (static_cast<int>(mqueue.size())==maxSize_) mqueue.pop_front();
      mqueue.push_back(sval+nval);
      if (useMin_ && mval <= mval_min) {
        mval_min = mval; sval_min = sval; nval_min = nval; ymin->set(y);
      }
    }

    // Compute projected gradient norm
    pwa1->set(gmod.dual());
    pwa->set(y); pwa->axpy(-t0_, *pwa1);
    dprox(x,t0_,del,nobj,nprox_,outStream);
    pwa->axpy(-one,y);
    gnorm = pwa->norm() / t0_;

    if (verbosity_ > 1) {
      outStream << std::endl;
      outStream << "    Iterate:                          " << SPiter_   << std::endl;
      outStream << "    Spectral step length (lambda):    " << lambda    << std::endl;
      outStream << "    Step length (alpha):              " << alpha     << std::endl;
      outStream << "    Model decrease (pRed):            " << mval-mold << std::endl;
      outStream << "    Optimality criterion:             " << gnorm     << std::endl;
      outStream << std::endl;
    }
    if (gnorm < gtol) break;

    // Compute new spectral step
    lambdaTmp = (sHs == 0 ? coeff : ss / sHs);
    lambda    = std::max(lambdaMin_, std::min(lambdaTmp, lambdaMax_));
  }
  if (useNMSP_ && useMin_) {
    sval = sval_min; nval = nval_min; y.set(*ymin);
  }
  else {
    sval = sold; nval = nold;
  }
  SPflag_ = (SPiter_==maxit_) ? 1 : 0;
} //solve
void dprox(//Vector<Real>       &x,
           const Vector<Real> &x0,
           Real               t,
           Real               del,
           Objective<Real>    &nobj,
           int                &nprox_,
           std::ostream       &outStream) const {
  // Solve ||P(t*x0 + (1-t)*(x-x0))-x0|| = del using Brent's method
  Ptr<Vector<Real>> y0, y1, yc, px;
  y0 = pwa2->clone(); 
  y1 = pwa3->clone(); 
  yc = pwa4->clone(); 
  px = pwa5->clone(); 
  const Real zero(0), half(0.5), one(1), two(2), three(3);
  const Real eps(ROL_EPSILON<Real>()), tol0(1e1*eps), fudge(1.0-1e-2*sqrt(eps));
  Real f0(0), f1(0), fc(0), t0(0), t1(1), tc(0), d1(1), d2(1), tol(1);
  Real p(0), q(0), r(0), s(0), m(0);
  int cnt(nprox_++);
  //nobj.prox(*y1, x, t, tol); nprox_++;
  nobj.prox(*y1, *pwa, t, tol); nprox_++;
  px->set(*y1); px->axpy(-one,x0);
  f1 = px->norm();
  if (f1 <= del) {
    //x.set(*y1);
    pwa->set(*y1);
    return;
  }
  y0->set(x0);
  tc = t0; fc = f0; yc->set(*y0);
  d1 = t1-t0; d2 = d1;
  int code = 0;
  while (true) {
    if (std::abs(fc-del) < std::abs(f1-del)) {
      t0 = t1; t1 = tc; tc = t0;
      f0 = f1; f1 = fc; fc = f0;
      y0->set(*y1); y1->set(*yc); yc->set(*y0);
    }
    tol = two*eps*std::abs(t1) + half*tol0;
    m   = half*(tc - t1);
    if (std::abs(m) <= tol) { code = 1; break; }
    if ((f1 >= fudge*del && f1 <= del)) break;
    if (std::abs(d1) < tol || std::abs(f0-del) <= std::abs(f1-del)) {
      d1 = m; d2 = d1;
    }
    else {
      s = (f1-del)/(f0-del);
      if (t0 == tc) {
        p = two*m*s;
        q = one-s;
      }
      else {
        q = (f0-del)/(fc-del);
        r = (f1-del)/(fc-del);
        p = s*(two*m*q*(q-r)-(t1-t0)*(r-one));
        q = (q-one)*(r-one)*(s-one);
      }
      if (p > zero) q = -q;
      else          p = -p;
      s  = d1;
      d1 = d2;
      if (two*p < three*m*q-std::abs(tol*q) && p < std::abs(half*s*q)) {
        d2 = p/q;
      }
      else {
        d1 = m; d2 = d1;
      }
    }
    t0 = t1; f0 = f1; y0->set(*y1);
    if (std::abs(d2) > tol) t1 += d2;
    else if (m > zero)      t1 += tol;
    else                    t1 -= tol;
    //px->set(x); px->scale(t1); px->axpy(one-t1,x0);
    px->set(*pwa); px->scale(t1); px->axpy(one-t1,x0);
    nobj.prox(*y1, *px, t1*t, tol); nprox_++;
    px->set(*y1); px->axpy(-one,x0);
    f1 = px->norm();
    if ((f1 > del && fc > del) || (f1 <= del && fc <= del)) {
      tc = t0; fc = f0; yc->set(*y0);
      d1 = t1-t0; d2 = d1;
    }
  }
  if (code==1 && f1>del) pwa->set(*yc); //x.set(*yc);
  else                   pwa->set(*y1); //x.set(*y1); 
  if (verbosity_ > 1) {
    outStream << std::endl;
    outStream << "  Trust-Region Subproblem Proximity Operator" << std::endl;
    outStream << "    Number of proxes:                 " << nprox_ - cnt << std::endl; //FIX: state_->nprox-cnt << std::endl;
    if (code == 1 && f1 > del) {
      outStream << "    Transformed Multiplier:           " << tc << std::endl;
      outStream << "    Dual Residual:                    " << fc-del << std::endl;
    }
    else {
      outStream << "    Transformed Multiplier:           " << t1 << std::endl;
      outStream << "    Dual Residual:                    " << f1-del << std::endl;
    }
    outStream << "    Exit Code:                        " << code << std::endl;
    outStream << std::endl;
  }
} // dprox
}; // SPG

} // namespace ROL

#endif
