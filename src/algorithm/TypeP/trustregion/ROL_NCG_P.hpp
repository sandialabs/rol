// @HEADER
// *****************************************************************************
//               Rapid Optimization Library (ROL) Package
//
// Copyright 2014 NTESS and the ROL contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef ROL_NCG_P_H
#define ROL_NCG_P_H

/** \class ROL::TruncatedCG_U
    \brief Provides interface for truncated CG trust-region subproblem solver.
        dncg(x,smodel,ntrial,*gmod,*state_->iterateVec,state_->searchSize,
             *model_,nobj,*pwa1,*pwa2,*pwa3,*pwa4,*pwa5,*pwa6,*dwa1,
             outStream);
        pRed = state_->value - (smodel+ntrial);
*/

#include "ROL_TrustRegion_P.hpp"
#include "ROL_Types.hpp"

namespace ROL { 

template<class Real>
class NCG_P : public TrustRegion_P<Real> {
private:
  Ptr<Vector<Real>> s, pwa, pwa1, pwa2, pwa3, pwa4, pwa5, dwa;

  Real lambdaMin_;
  Real lambdaMax_;
  Real gamma_;
  int  maxSize_;
  int  maxit_;
  Real tol1_;
  Real tol2_;
  bool useMin_;
  bool useNMSP_;
  int  ncgType_; 
  Real etaNCG_; 
  Real desPar_;
  int  verbosity_; 
  Real t0_; 

public:

  using TrustRegion_P<Real>::pgstep; 
  
  // Constructor
  NCG_P( ParameterList &parlist ) {
    // Unravel Parameter List
    ParameterList &list = parlist.sublist("Step").sublist("Trust Region").sublist("TRN").sublist("Solver").sublist("NCG");
    verbosity_         = parlist.sublist("Step").sublist("Trust Region").sublist("General").get("Output Level",      0);
    lambdaMin_ = list.get("Minimum Spectral Step Size",          1e-8);
    lambdaMax_ = list.get("Maximum Spectral Step Size",          1e8);
    gamma_     = list.get("Sufficient Decrease Tolerance",       1e-4);
    maxSize_   = list.get("Maximum Storage Size",                10);
    maxit_     = list.get("Iteration Limit",                     15);
    tol1_      = list.get("Absolute Tolerance",                  1e-4);
    tol2_      = list.get("Relative Tolerance",                  1e-2);
    useMin_    = list.get("Use Smallest Model Iterate",          true);
    useNMSP_   = list.get("Use Nonmonotone Search",              false);
    etaNCG_    = list.get("Truncation Parameter for HZ CG",      1e-2); 
    ncgType_   = list.get("Nonlinear CG Type",                   4); 
    desPar_    = list.get("Descent Parameter",                   0.2); 
    t0_        = list.sublist("Status Test").get("Proximal Gradient Parameter", 1.0);
  }

  void initialize(const Vector<Real> &x, const Vector<Real> &g) {
    s    = x.clone();
    pwa  = x.clone();
    pwa1 = x.clone();
    pwa2 = x.clone();
    pwa3 = x.clone();
    pwa4 = x.clone();
    pwa5 = x.clone();
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
             std::ostream             &outStream) {
  // Use NCG to approximately solve TR subproblem:
  //   min 1/2 <H(y-x), (y-x)> + <g, (y-x)> + phi(y)  subject to  ||y-x|| \le del
  //
  //   Input:
  //       y     = computed iterate
  //       sval  = smooth model value
  //       nval  = nonsmooth value
  //       gmod  = current gradient
  //       x     = current iterate
  //       del   = trust region radius
  //       model = trust region model
  //       nobj  = nonsmooth objective function
  //       s     = the current step
  //       pwa1  = the SPG iterate
  //       pwa2  = the "negative gradient"
  //       pwa3  = y - x
  //       pwa4  = the previous "negative gradient"
  //       pwa5  = temporary storage
  //       dwa   = the Hessian applied to the step
  const Real zero(0), half(0.5), one(1), two(2);
  const Real del2(del*del);
  Real tol(std::sqrt(ROL_EPSILON<Real>())), safeguard(tol);
  Real mold(sval+nval), nold(nval), sold(sval); ;
  Real snorm(0), snorm0(0), gnorm(0), gnorm0(0), gnorm2(0);
  Real alpha(1), beta(1), lambdaTmp(1), lambda(1), eta(etaNCG_);
  Real alphaMax(1), pred(0), lam1(1);
  Real sy(0), gg(0), sHs(0), gs(0), ds(0), ss(0), ss0(0);
  bool reset(true);

  // Set y = x
  y.set(x);
  pwa3->zero(); // Initially y - x = 0
  pwa2->set(gmod.dual()); 
  model.hessVec(*dwa, *pwa2, x, tol); nhess_++; // Hg
  Real gHg = dwa->apply(gmod); 
  gg       = pwa2->dot(*pwa2); 
  // Compute initial spectral step length
  if (gHg > safeguard * gg){
    lambdaTmp = gg / gHg; 
  }
  else {
    lambdaTmp = t0_ / std::sqrt(gg);
  } 
  lambda    = std::max(lambdaMin_,std::min(lambdaTmp,lambdaMax_));
  lam1      = lambda;

  // Compute Cauchy point via SPG
  pgstep(*pwa1, *pwa2, nobj, y, gmod.dual(), lambda, tol); nprox_++;  // pwa1  = prox(x-lambda g), pwa2  = pwa1 - x
  pwa2->scale(one/lambda);                                            // pwa2  = (pwa1 - x) / lambda (for smooth: pwa2 = negative gradient)
  s->set(*pwa2);                                                      // s     = (pwa1 - x) / lambda
  gs     = s->dot(gmod.dual());                                       // gs    = <g, prox(x-lambda g)-x> / lambda
  snorm  = s->norm();                                                 // snorm = norm(prox(x-lambda g)-x) / lambda
  gnorm  = snorm;
  const Real gtol = std::min(tol1_,tol2_*std::pow(gnorm,2));

  if (verbosity_>1) outStream << "  Nonlinear Conjugate Gradient" << std::endl;

  SPiter_ = 0;
  SPflag_ = 1;
  while (SPiter_ < maxit_) {
    SPiter_++;

    // Compute the model curvature
    model.hessVec(*dwa,*s,x,tol); nhess_++; // dwa = H step
    sHs   = dwa->apply(*s);                 // sHs = <step, H step>

    // Compute alpha as the 1D minimize in the s direction
    pwa3->set(y); pwa3->axpy(-one, x); 
    ds = s->dot(*pwa3);
    ss = snorm*snorm;
    ss0 = snorm0*snorm0; 
    alphaMax = (-ds + std::sqrt(ds*ds + ss*(del2 - ss0)))/ss;
    //should output nold and alpha - pass s, pwa5
    dbls(alpha,nold,pred,y,lam1,alphaMax,sHs,gs,nobj, nnval_, outStream);
    
    // Update quantities to evaluate quadratic model value and gradient
    y.axpy(alpha, *s);
    gmod.axpy(alpha, *dwa); // need dual here?
    sold = sold + alpha*(gs + half*alpha*sHs);

    // Check step size
    ss0   += alpha*(alpha*ss + two*ds);
    snorm0 = std::sqrt(ss0); // pwa3.norm();
    
    if (snorm0 >= (one-safeguard)*del) { SPflag_ = 2; break; }

    // Update spectral step length
    lambdaTmp = (sHs <= safeguard) ? t0_/gmod.norm() : ss/sHs;
    lambda    = std::max(lambdaMin_, std::min(lambdaMax_, lambdaTmp));

    // Compute SPG direction
    pwa4->set(*pwa2);                                                   // store previous "negative gradient"
    pgstep(*pwa1, *pwa2, nobj, y, gmod.dual(), lambda, tol); nprox_++;  // pwa1 = prox(x-lambda g), pwa2 = pwa1 - x
    pwa2->scale(one/lambda);                                            // pwa2 = (pwa1 - x) / lambda (for smooth: pwa2 = negative gradient)
    gnorm0 = gnorm;
    gnorm  = pwa2->norm();
    if (gnorm <= gtol) { SPflag_ = 0; break; }

    gnorm2 = gnorm * gnorm;
    switch (ncgType_) {
      case 0: // Fletcher-Reeves
        beta = gnorm2/(gnorm0 * gnorm0);
        break;
      default:
      case 1: // Polyak-Ribiere+
        pwa5->set(*pwa4); 
        pwa5->axpy(-one,*pwa2);
        beta = std::max(zero, -pwa5->dot(*pwa2)/(gnorm0*gnorm0));
        break;
      case 2: // Hager-Zhang
        pwa5->set(*pwa4); 
        pwa5->axpy(-one,*pwa2);
        sy   = s->dot(*pwa5);
        gg   = pwa2->dot(*pwa4);
        eta  = -one/(s->norm()*std::min(etaNCG_, gnorm0));
        beta = std::max(eta, (gnorm2-gg-two*pwa2->dot(*s)*(gnorm2-two*gg+(gnorm0*gnorm0))/sy)/sy);
        break;
      case 3: // Hestenes-Stiefel+
        pwa5->set(*pwa4); 
        pwa5->axpy(-one,*pwa2);
        beta = std::max(zero, -pwa2->dot(*pwa5)/s->dot(*pwa5));
        break;
      case 4: // Dai-Yuan+
        pwa5->set(*pwa4); 
        pwa5->axpy(-one,*pwa2);
        beta = std::max(zero, gnorm2/s->dot(*pwa5));
        break;
      case 5: // Fletcher-Reeves-Polyak-Ribiere
        pwa5->set(*pwa4); 
        pwa5->axpy(-one,*pwa2);
        beta = std::max(-gnorm2, std::min(gnorm2, -pwa5->dot(*pwa2)))/(gnorm0*gnorm0);
        break;
      case 6: //Dai-Yuan-Hestenes-Stiefles
        pwa5->set(*pwa4); 
        pwa5->axpy(-one,*pwa2);
        beta = std::max(zero, std::min(-pwa2->dot(*pwa5), gnorm2)/s->dot(*pwa5));
        break;
    }

    reset = true;
    if (beta != zero && beta < ROL_INF<Real>()){
      pwa5->set(*pwa2);         // pwa5 = pwa2 (SPG step)
      pwa5->axpy(beta, *s);     // pwa5 = pwa2 + beta*s
      pwa4->set(y);             // pwa4 = y
      gs = gmod.apply(*pwa5);
      pwa4->plus(*pwa5);        // pwa4 = y + pwa5
      nobj.update(*pwa4,UpdateType::Trial);
      nval = nobj.value(*pwa4,tol); nnval_++;
      if (gs + nval - nold <= -(one - desPar_)*gnorm2) {
        s->set(*pwa5);
        lam1  = one;
        reset = false;
      }
    }
    if (reset){ // Reset because either beta=0 or step does not produce descent
      beta = zero;
      lam1 = lambda;
      s->set(*pwa2);
      gs   = gmod.apply(*s);
    }
    snorm = s->norm();

    if (verbosity_ > 1) {
      outStream << std::endl;
      outStream << "    Iterate:                          " << SPiter_          << std::endl;
      outStream << "    Spectral step length (lambda):    " << lambda           << std::endl;
      outStream << "    Step length (alpha):              " << alpha            << std::endl;
      outStream << "    NCG parameter (beta):             " << beta             << std::endl;
      outStream << "    Model decrease (pRed):            " << mold-(sval+nold) << std::endl;
      outStream << "    Step size:                        " << snorm0           << std::endl;
      outStream << "    Optimality criterion:             " << gnorm            << std::endl;
      outStream << "    Optimality tolerance:             " << gtol             << std::endl;
      outStream << std::endl;
    }
  } // loop

  //pRed = mold - (sval+nold);
  sval = sold; 
  nval = nold;
  };// solve

  void dbls(     Real  &alpha, 
                 Real  &nval, 
                 Real  &pred,
    const Vector<Real> &y,
                 Real  lambda, 
                 Real  tmax,
                 Real  kappa, 
                 Real  gs,
       Objective<Real> &nobj, 
                  int  &nnval, 
          std::ostream &outStream) {
  
    Real tol(std::sqrt(ROL_EPSILON<Real>()));
    const Real eps1(1e-2*tol), tol0(1e4*tol);
    //const Real eps0(1e2*ROL_EPSILON<Real>());
    const unsigned maxit(std::max(5.0, std::ceil(-std::log(1e-2*tol/tmax)/std::log(2))));
    const Real zero(0), half(0.5), two(2); //one(1), two(2);
    const Real lam(0.5*(3.0-std::sqrt(5.0)));
    const Real nold(nval);
    const Real mu(1e-4);

    // Evaluate model at initial left end point (t = 0)
    Real tL(0), pL(0), nL(nold), nR(ROL_INF<Real>());
    // Evaluate model at right end point (t = tmax)
    Real tR = tmax;
    pwa5->set(y); pwa5->axpy(tR, *s);
    nobj.update(*pwa5,UpdateType::Trial);
    nR = nobj.value(*pwa5,tol); nnval++;
    Real pR(0), n0(0), p0(0), t0(0);
    Real t2(0), tolBS(0), t1(0), tp(0), n1(0),  pp(p0), np(n0); //n2(0)
    if (nR >= ROL_INF<Real>()) {
      t0 = lambda;
      pwa5->set(y);
      pwa5->axpy(t0, *s);
      nobj.update(*pwa5,UpdateType::Trial);
      n0 = nobj.value(*pwa5,tol); nnval++;
      p0 = t0 * (half * t0 * kappa + gs) + n0 - nold;
      if (p0 >= zero){ // this t0 increases the pred from t0, so use as end point
        tR = t0; 
        nR = n0; 
        pR = p0; 
      }// p0 if
      else { // pred is decreasing at t0, so find t0 that gives finite n0
        t2    = tR;
        //n2    = nR; 
        tolBS = tol * (tR - lambda); 
        while (t2 > t0 + tolBS) { 
          t1 = half * (t0 + t2); 
          pwa5->set(y); 
          pwa5->axpy(t1, *s); 
          n1 = nobj.value(*pwa5, tol); nnval++; 
          if ( n1 >= ROL_INF<Real>() ) { // test if inf
            t2 = t1; 
            //n2 = n1; //n2 not used? 
          } // inf if
          else {
            tp = t0; 
            np = n0; //not used with nL
            pp = p0; 
            t0 = t1; 
            n0 = n1; 
            p0 = t0 * (half * t0 * kappa + gs) + n0 - nold; 
            if (p0 >= pp) { // pred is increasing between tp and t0
              tL = tp; 
              nL = np; //not used? 
              pL = pp; 
            } // pp if
          } // inf else
        } // p0 while
        tR = t0; 
        nR = n0; 
        pR = p0; 
      } // p0 else
    }//nR
    else {
     pR = tR * (half * tR * kappa + gs) + nR - nold;
    }
    // Compute minimizer of quadratic upper bound
    //Real t = (kappa > 0) ? std::min(tR,(((nold - nR)/(tR - tL)) + gs) / kappa) : tR;
    Real t = tR; 
    if (kappa > zero) {
      t = std::min(tR, -(((nR - nL)/(tR - tL)) + gs)/kappa); 
    }
    bool useOptT = true;
    if (t <= tL) {
      //t = (t0 < tR ? t0 : half*tR);
      t = tL + lambda*(tR - tL);
      useOptT = false;
    }

    // Evaluate model at t (t = minimizer of quadratic upper bound)
    pwa5->set(y); 
    pwa5->axpy(t, *s);
    nobj.update(*pwa5,UpdateType::Trial);
    Real nt = nobj.value(*pwa5,tol); nnval++;
    Real Qt = t * gs + nt - nold, Qu = Qt;
    Real pt = half * t * t * kappa + Qt;
    // If phi(x) = phi(x+tmax s) = phi(x+ts), then
    // phi(x+ts) is constant for all t, so the TR
    // model is quadratic---use the minimizer
    if (useOptT && pt < zero && nt == nold && nR == nold) {
      alpha = t;
      pred  = pt ; //t * (half * t * kappa) + gs;
      nval  = nold;
      return;
    }

    // If pt >= max(pL, pR), then the model is concave
    // and the minimum is obtained at the end points
    if (pt >= std::max(pL, pR) && (pR < zero)) {
      alpha = tR;
      pred  = pR;
      nval  = nR;
      return;
    }

    // Run Brent's method (quadratic interpolation + golden section)
    // to minimize m_k(x+ts) with respect to t
    Real v = tL, pv = pL, w = tR, pw = pR; 
    Real d(0), pu(0), nu(0);
    Real u(0), p(0), q(0), r(0), etmp(0), e(0), dL(0), dR(0);
    Real tm   = half * (tL + tR);
    Real tol1 = tol0 * std::abs(t)+eps1;
    Real tol2 = two * tol1;
    //Real tol3 = eps1;
    e         = std::max(t - tL, tR - t); 
    int iter(0); 
    for (unsigned it = 0u; it < maxit; ++it) {
      iter +=1; 
      dL = tL-t; 
      dR = tR-t;
      if (std::abs(e) > tol1) {
        r = (t - w)*(pt - pv);
        q = (t - v)*(pt - pw);
        p = (t - v)*q - (t - w)*r;
        q = two*(q - r);
        if (q > zero) p = -p;
        q    = std::abs(q);
        etmp = e;
        e    = d;
        if ( std::abs(p) >= std::abs(half*q*etmp) || p <= q*dL || p >= q*dR ) {
          e = (t > tm ? dL : dR);
          d = lam * e;
        }//p if
        else {
          d = p/q;
          u = t+d;
          if (u - tL < tol2 || tR - u < tol2) {
            d = (tm >= t ? tol1 : -tol1);
          }// tol2 if
        }//p else
      }//tol1 if
      else { // e loop
        e = (t > tm ? dL : dR);
        d = lam * e;
      }
      //u = t + (std::abs(d) >= tol1 ? d : (d >= zero ? tol1 : -tol1));
      u = t + d; 
      if (std::abs(d) < tol1){
        u = t + tol1; 
        if (d<0) u = t - tol1; 
      }//d if
      pwa5->set(y);
      pwa5->axpy(u, *s);
      nobj.update(*pwa5, UpdateType::Trial);
      nu = nobj.value(*pwa5, tol); nnval++;
      Qu = u * gs + nu - nold;
      pu = half * u * u * kappa + Qu;

      if (pu <= pt) {
        if (u >= t) tL = t;
        else        tR = t;
        v  = w;  w  = t;  t  = u;
        pv = pw; pw = pt; pt = pu;
        nt = nu; Qt = Qu;
      }//pu/pt if
      else {
        if (u < t) tL = u;
        else       tR = u;
        if (pu <= pw || w == t) {
          v  = w;  w  = u;
          pv = pw; pw = pu;
        }
        else if (pu <= pv || v == t || v == w) {
          v = u; pv = pu;
        }
      }
      tm   = half * (tL+tR);
      tol1 = tol0*std::abs(t)+eps1;
      tol2 = two*tol1;
      //tol3 = eps0 * std::max(std::abs(Qt),one);
      //if (pt <= (mu*std::min(zero,Qt)+tol3) && std::abs(t-tm) <= (tol2-half*(tR-tL))) break;
      if (pt <= (mu*std::min(zero,Qt)) && std::abs(t-tm) <= (tol2-half*(tR-tL))){ 
        //printf("pt = %2.2e  rhs = %2.2e   |t - tm| = %2.2e  rhs = %2.2e\n",   pt, (mu*std::min(zero,Qt)), std::abs(t-tm), tol2-half*(tR-tL)); 
        break;
      }
    }
    alpha = t;
    pred  = pt;
    nval  = nt;
    //if (pt >= 0 && verbosity_>1) outStream << "  dbls: No acceptable step-size found." << "pred = " << pred << std::endl;
    //printf("out dbls\n"); 
    if (pt >= 0) outStream << "  dbls: No acceptable step-size found. pred = " << pred << std::endl;
}
}; 
} // namespace ROL

#endif
