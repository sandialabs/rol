// @HEADER
// *****************************************************************************
//               Rapid Optimization Library (ROL) Package
//
// Copyright 2014 NTESS and the ROL contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef ROL_TYPEP_NCG2_HPP
#define ROL_TYPEP_NCG2_HPP

/** \class ROL::TypeP::NCG2
    \brief Archived legacy nonlinear-CG solver retained for reference.

    This implementation is not registered in the TypeP trust-region factory.
*/

#include "ROL_TypeP_TruncatedCG.hpp"
#include <cmath>

namespace ROL {
namespace TypeP {

template<class Real>
class NCG2 : public TruncatedCG<Real> {
private:
  using Base = TruncatedCG<Real>;

public:
  using Base::Base;

  void solve(Vector<Real>             &y,
             Real                     &sval,
             Real                     &nval,
             Vector<Real>             &gmod,
             const Vector<Real>       &x,
             Real                      del,
             Objective<Real>          &model,
             Objective<Real>          &nobj,
             int                      &SPflag_,
             int                      &SPiter_,
             int                      &nhess_,
             int                      &nprox_,
             int                      &nnval_,
             std::ostream             &outStream) override {
    using std::max;
    using std::min;
    using std::sqrt;

    const Real zero(0), half(0.5), one(1), two(2);
    Real tol(std::sqrt(ROL_EPSILON<Real>()));
    const Real safeguard(tol);
    const Real del2(del*del);
    const Real smallStep(tol*max(one,x.norm()));
    const Real modelInitial(sval+nval);
    Real nold(nval), snorm(0), snorm0(0), gnorm(0), gnorm0(0), gnorm2(0);
    Real alpha(1), beta(1), lambdaTmp(1), lambda(1), eta(this->etaNCG_);
    Real alphaMax(1), pred(0), lam1(1);
    Real sy(0), gg(0), sHs(0), gs(0), ds(0), ss(0), ss0(0);
    bool reset(true);

    y.set(x);
    this->pwa3->zero();

    this->pwa2->set(gmod.dual());
    model.hessVec(*this->dwa,*this->pwa2,x,tol);
    nhess_++;
    const Real gHg = this->dwa->apply(*this->pwa2);
    gg = gmod.apply(*this->pwa2);
    lambdaTmp = (gHg > safeguard*gg) ? gg/gHg : this->t0_/sqrt(gg);
    lambda = max(this->lambdaMin_,min(lambdaTmp,this->lambdaMax_));
    lam1 = lambda;

    this->pgstep(*this->pwa1,*this->pwa2,nobj,y,gmod.dual(),lambda,
                 tol);
    nprox_++;
    this->pwa2->scale(one/lambda);
    this->s->set(*this->pwa2);
    gs = gmod.apply(*this->s);
    snorm = this->s->norm();
    gnorm = snorm;
    const Real gtol = min(this->tol1_,this->tol2_*gnorm);

    if (gnorm <= gtol) {
      SPiter_ = 0;
      SPflag_ = 0;
      nval = nold;
      return;
    }

    if (this->verbosity_ > 2) {
      outStream << "  NCG2 (legacy nonlinear CG)" << std::endl;
    }

    SPiter_ = 0;
    SPflag_ = 1;
    while (SPiter_ < this->maxit_) {
      ++SPiter_;

      ss = snorm*snorm;
      ds = this->s->dot(*this->pwa3);
      alphaMax = static_cast<Real>(1e6);
      if (snorm > smallStep) {
        const Real radicand = max(zero,ds*ds+ss*(del2-snorm0*snorm0));
        alphaMax = max(zero,(-ds+sqrt(radicand))/ss);
      }

      // The legacy routine carried curvature from the initial gradient here.
      // Recompute it along the actual search direction before the line search.
      model.hessVec(*this->dwa,*this->s,x,tol);
      ++nhess_;
      sHs = this->dwa->apply(*this->s);
      this->dbls(alpha,nold,pred,y,lam1,alphaMax,sHs,gs,nobj,nnval_,outStream);

      y.axpy(alpha,*this->s);
      gmod.axpy(alpha,*this->dwa);
      sval += alpha*(gs+half*alpha*sHs);

      this->pwa3->set(y);
      this->pwa3->axpy(-one,x);
      ss0 += alpha*(alpha*ss+two*ds);
      snorm0 = sqrt(max(zero,ss0));
      if (snorm0 >= (one-safeguard)*del) {
        SPflag_ = 2;
        break;
      }

      lambdaTmp = (sHs <= safeguard*ss) ? this->t0_/gmod.norm() : ss/sHs;
      lambda = max(this->lambdaMin_,min(lambdaTmp,this->lambdaMax_));

      this->pwa4->set(*this->pwa2);
      this->pgstep(*this->pwa1,*this->pwa2,nobj,y,gmod.dual(),lambda,tol);
      ++nprox_;
      this->pwa2->scale(one/lambda);
      gnorm0 = gnorm;
      gnorm = this->pwa2->norm();
      if (gnorm <= gtol) {
        SPflag_ = 0;
        break;
      }

      gnorm2 = gnorm*gnorm;
      switch (this->ncgType_) {
        case 0:
          beta = gnorm2/(gnorm0*gnorm0);
          break;
        case 1:
          this->pwa5->set(*this->pwa4);
          this->pwa5->axpy(-one,*this->pwa2);
          beta = max(zero,-this->pwa5->dot(*this->pwa2)/(gnorm0*gnorm0));
          break;
        case 2:
          this->pwa5->set(*this->pwa4);
          this->pwa5->axpy(-one,*this->pwa2);
          sy = this->s->dot(*this->pwa5);
          gg = this->pwa2->dot(*this->pwa4);
          eta = -one/(this->s->norm()*min(this->etaNCG_,gnorm0));
          beta = max(eta,(gnorm2-gg-two*this->pwa2->dot(*this->s)*
                    (gnorm2-two*gg+gnorm0*gnorm0)/sy)/sy);
          break;
        case 3:
          this->pwa5->set(*this->pwa4);
          this->pwa5->axpy(-one,*this->pwa2);
          beta = max(zero,-this->pwa2->dot(*this->pwa5)/
                    this->s->dot(*this->pwa5));
          break;
        case 4:
          this->pwa5->set(*this->pwa4);
          this->pwa5->axpy(-one,*this->pwa2);
          beta = max(zero,gnorm2/this->s->dot(*this->pwa5));
          break;
        case 5:
          this->pwa5->set(*this->pwa4);
          this->pwa5->axpy(-one,*this->pwa2);
          beta = max(-gnorm2,min(gnorm2,-this->pwa5->dot(*this->pwa2)))/
                 (gnorm0*gnorm0);
          break;
        case 6:
          this->pwa5->set(*this->pwa4);
          this->pwa5->axpy(-one,*this->pwa2);
          beta = max(zero,min(-this->pwa2->dot(*this->pwa5),gnorm2)/
                    this->s->dot(*this->pwa5));
          break;
        default:
          this->pwa5->set(*this->pwa4);
          this->pwa5->axpy(-one,*this->pwa2);
          beta = max(zero,-this->pwa5->dot(*this->pwa2)/(gnorm0*gnorm0));
          break;
      }
      if (!(beta < ROL_INF<Real>() && beta > -ROL_INF<Real>())) beta = zero;

      reset = true;
      if (beta != zero && beta < ROL_INF<Real>() && beta > -ROL_INF<Real>()) {
        this->pwa5->set(*this->pwa2);
        this->pwa5->axpy(beta,*this->s);
        this->pwa4->set(y);
        this->pwa4->plus(*this->pwa5);
        nobj.update(*this->pwa4,UpdateType::Trial);
        nval = nobj.value(*this->pwa4,tol);
        ++nnval_;
        gs = gmod.apply(*this->pwa5);
        if (gs+nval-nold <= -(one-this->desPar_)*gnorm2) {
          this->s->set(*this->pwa5);
          lam1 = one;
          reset = false;
        }
      }
      if (reset) {
        this->s->set(*this->pwa2);
        gs = gmod.apply(*this->s);
        lam1 = lambda;
        beta = zero;
      }
      snorm = this->s->norm();

      if (this->verbosity_ > 2) {
        outStream << "\n    NCG2 iterate: " << SPiter_
                  << ", alpha: " << alpha
                  << ", beta: " << beta
                  << ", model decrease: " << modelInitial-(sval+nold)
                  << ", step norm: " << snorm0
                  << ", optimality: " << gnorm
                  << ", tolerance: " << gtol << std::endl;
      }
    }

    nval = nold;
  }
};

} // namespace TypeP
} // namespace ROL

#endif
