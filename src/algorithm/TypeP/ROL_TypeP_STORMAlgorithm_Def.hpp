// @HEADER
// *****************************************************************************
//               Rapid Optimization Library (ROL) Package
//
// Copyright 2014 NTESS and the ROL contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef ROL_STORMALGORITHM_DEF_HPP
#define ROL_STORMALGORITHM_DEF_HPP

#include "ROL_TypeP_STORMAlgorithm.hpp"
#include "ROL_TrustRegion_P_Factory.hpp"
#include "ROL_TypeP_TrustRegionAlgorithm.hpp"
#include "ROL_TypeP_TrustRegionAlgorithm_Def.hpp"
#include "ROL_TrustRegionModel_U.hpp"
#include "ROL_TrustRegionUtilities.hpp"
#include "ROL_Types.hpp"
#include <deque>

namespace ROL {
namespace TypeP {

template<typename Real>
STORMAlgorithm<Real>::STORMAlgorithm(ParameterList &parlist, 
                                     const Ptr<Secant<Real>> &secant )
    : TrustRegionAlgorithm<Real>(parlist, secant) {
  // Set status test
  status_->reset();
  status_->add(makePtr<StatusTest<Real>>(parlist));

  // Trust-Region Parameters
  ParameterList &slist = parlist.sublist("Step");
  ParameterList &trlist  = slist.sublist("Trust Region");
  state_->searchSize = trlist.get("Initial Radius",            static_cast<Real>(-1));
  delMax_ = trlist.get("Maximum Radius",                       ROL_INF<Real>());
  eta0_   = trlist.get("Step Acceptance Threshold",            static_cast<Real>(0.05));
  eta1_   = trlist.get("Radius Shrinking Threshold",           static_cast<Real>(0.05));
  eta2_   = trlist.get("Radius Growing Threshold",             static_cast<Real>(0.9));
  gamma0_ = trlist.get("Radius Shrinking Rate (Negative rho)", static_cast<Real>(0.2));
  gamma1_ = trlist.get("Radius Shrinking Rate (Positive rho)", static_cast<Real>(0.25));
  gamma2_ = trlist.get("Radius Growing Rate",                  static_cast<Real>(5.0));
  TRsafe_ = trlist.get("Safeguard Size",                       static_cast<Real>(100.0));
  eps_    = TRsafe_*ROL_EPSILON<Real>();
  // Nonmonotone Information
  storageNM_ = trlist.get("Nonmonotone Storage Limit", 0);
  useNM_     = (storageNM_ <= 0 ? false : true);
  // Initialize nonmonotone data
  rhoNM_  = 0;
  sigmac_ = 0;
  sigmar_ = 0;
  L_      = 0;
  // Algorithm-Specific Parameters
  ROL::ParameterList &lmlist = trlist.sublist("TRN");
  mu0_               = lmlist.get("Sufficient Decrease Parameter",                             1e-2);

  interpRad_         = trlist.get("Use Radius Interpolation",             false);
  verbosity_         = trlist.sublist("General").get("Output Level",      0);
  initProx_          = trlist.get("Apply Prox to Initial Guess",          false);
  t0_                = parlist.sublist("Status Test").get("Proximal Gradient Parameter", 1.0);
  // Inexactness Information
  ParameterList &glist = parlist.sublist("General");
  useInexact_.clear();
  useInexact_.push_back(glist.get("Inexact Objective Function",     false));
  useInexact_.push_back(glist.get("Inexact Gradient",               false));
  useInexact_.push_back(glist.get("Inexact Hessian-Times-A-Vector", false));
  // Trust-Region Inexactness Parameters
  ParameterList &ilist = trlist.sublist("Inexact").sublist("Gradient");
  scale0_ = ilist.get("Tolerance Scaling",  static_cast<Real>(0.1));
  scale1_ = ilist.get("Relative Tolerance", static_cast<Real>(2));
  // Inexact Function Evaluation Information
  ParameterList &vlist = trlist.sublist("Inexact").sublist("Value");
  scale_       = vlist.get("Tolerance Scaling",                 static_cast<Real>(1.e-1));
  omega_       = vlist.get("Exponent",                          static_cast<Real>(0.9));
  force_       = vlist.get("Forcing Sequence Initial Value",    static_cast<Real>(1.0));
  updateIter_  = vlist.get("Forcing Sequence Update Frequency", static_cast<int>(10));
  forceFactor_ = vlist.get("Forcing Sequence Reduction Factor", static_cast<Real>(0.1));
  // Initialize Trust Region Subproblem Solver Object
  std::string solverName = trlist.get("Subproblem Solver", "Cauchy Point");
  algSelect_ = StringToETrustRegionP(solverName);
  solver_    = TrustRegionPFactory<Real>(parlist);
  verbosity_ = glist.get("Output Level", 0);
  // Secant Information
  useSecantPrecond_ = glist.sublist("Secant").get("Use as Preconditioner", false);
  useSecantHessVec_ = glist.sublist("Secant").get("Use as Hessian",        false);
  if (secant == nullPtr) {
    std::string secantType = glist.sublist("Secant").get("Type","Limited-Memory BFGS");
    esec_ = StringToESecant(secantType);
  }
  // Initialize trust region model
  model_ = makePtr<TrustRegionModel_U<Real>>(parlist,secant);
  writeHeader_ = verbosity_ > 2;
}

template<typename Real>
void STORMAlgorithm<Real>::writeName( std::ostream& os ) const {
  std::ios_base::fmtflags osFlags(os.flags());
  os << std::endl << ETrustRegionPToString(algSelect_) << " ProxSTORM Trust-Region Solver";
  if ( useSecantPrecond_ || useSecantHessVec_ ) {
    if ( useSecantPrecond_ && !useSecantHessVec_ ) {
      os << " with " << ESecantToString(esec_) << " Preconditioning" << std::endl;
    }
    else if ( !useSecantPrecond_ && useSecantHessVec_ ) {
      os << " with " << ESecantToString(esec_) << " Hessian Approximation" << std::endl;
    }
    else {
      os << " with " << ESecantToString(esec_) << " Preconditioning and Hessian Approximation" << std::endl;
    }
  }
  else {
    os << std::endl;
  }
  os << "Trust-Region Method (Type P)" << std::endl;
  os.flags(osFlags);
}

template<typename Real>
Real STORMAlgorithm<Real>::computeValue(Real inTol,
                                        Real &outTol,
                                        Real pRed,
                                        Real &fold,
                                        Real del,
                                        int  iter,
                                        const Vector<Real> &x,
                                        const Vector<Real> &xold,
                                        Objective<Real>    &sobj) {
  outTol = std::sqrt(ROL_EPSILON<Real>());
  if ( useInexact_[0] ) {
    int two(2); 
    outTol   = scale_*static_cast<Real>(0.999)*std::pow(del, two);
    fold     = sobj.value(xold,outTol); state_->nsval++;
  }
  // Evaluate objective function at new iterate
  sobj.update(x,UpdateType::Trial);
  Real fval = sobj.value(x,outTol); state_->nsval++;
  return fval;
}

template<typename Real>
void STORMAlgorithm<Real>::computeGradient(const Vector<Real> &x,
                                                 Vector<Real> &g,
                                                 Vector<Real> &px,
                                                 Vector<Real> &dg,
                                                 Vector<Real> &step,
                                                 Real del,
                                                 Objective<Real> &sobj,
                                                 Objective<Real> &nobj,
                                                 bool accept,
                                                 Real &gtol,
                                                 Real &gnorm,
                                                 std::ostream &outStream) const {
  if ( useInexact_[1] ) {
    Real gtol0 = scale0_*static_cast<Real>(0.999)*del;
    sobj.gradient(g,x,gtol0); state_->ngrad++;
  }
  else {
    if (accept) {
      gtol = std::sqrt(ROL_EPSILON<Real>());
      sobj.gradient(g,x,gtol); state_->ngrad++;
    }
  }
  dg.set(g.dual());
  solver_->pgstep(px, step, nobj, x, dg, t0_, gtol); state_->nprox++; // gtol for prox might be different 
  gnorm = step.norm() / t0_;
}

template<typename Real>
void STORMAlgorithm<Real>::stepUpdate(Vector<Real>               &x, 
                                      Vector<Real>               &g,
                                      Vector<Real>               &px,             
                                      Vector<Real>               &dg, 
                                      Objective<Real>            &sobj,
                                      Objective<Real>            &nobj,
                                      TrustRegionModel_U<Real>   &model,
                                      Real                       &rho,
                                      Real                       &pRed,
                                      Real                       &Ftrial,
                                      Real                       &strial,
                                      Real                       &ntrial,
                                      bool                       accept,
                                      Real                       &gtol,
                                      Real                       &inTol,
                                      Real                       &outTol,
                                      Vector<Real>               &pwa1,
                                      Vector<Real>               &dwa1,
                                      std::ostream               &outStream 
      )  {
    if (rho < eta0_  || state_->gnorm < eta2_*state_->searchSize) { // || (TRflag_ >= 2 || g.norm() < eta2_*state_->searchSize)) { // Step Rejected
      x.set(*state_->iterateVec);
      sobj.update(x,UpdateType::Revert,state_->iter);
      nobj.update(x,UpdateType::Revert,state_->iter);
      state_->searchSize = gamma0_*std::min(state_->snorm,state_->searchSize);
      //}
      while (true) {
      		computeGradient(x,*state_->gradientVec,px,dg,pwa1,state_->searchSize,sobj,nobj,false,gtol,state_->gnorm,outStream);
          if (state_->gnorm >= eta2_*state_->searchSize) break;
          state_->searchSize = std::min(gamma0_*state_->searchSize, delMax_);
      }
    } // Step Rejected
    else {// if (rho >= eta0_ && g.norm() >= eta2_*state_->searchSize){// Step Accepted
    //else if ((rho >= eta0_ && TRflag_ != TRUtils::NPOSPREDNEG && (g.norm() >= eta2_*state_->searchSize))
    //         || (TRflag_ == TRUtils::POSPREDNEG)) { // Step Accepted
      state_->value  = Ftrial;
      state_->svalue = strial;
      state_->nvalue = ntrial;
      sobj.update(x,UpdateType::Accept,state_->iter);
      nobj.update(x,UpdateType::Accept,state_->iter);
      inTol = outTol;
      // Increase trust-region radius
      state_->searchSize = std::min(gamma2_*state_->searchSize, delMax_);
      // Compute gradient at new iterate
      while (true) {
      		computeGradient(x,*state_->gradientVec,px,dg,pwa1,state_->searchSize,sobj,nobj,true,gtol,state_->gnorm,outStream);
          if (state_->gnorm >= eta2_*state_->searchSize ) break;
          state_->searchSize = std::min(gamma0_*state_->searchSize, delMax_);
      }
      dwa1.set(*state_->gradientVec);
      state_->iterateVec->set(x);
      // Update secant information in trust-region model
      model_->update(x,*state_->stepVec,dwa1,*state_->gradientVec,
                     state_->snorm,state_->iter);
    } // Step Accepted
} // stepUpdate function

} // namespace TypeP
} // namespace ROL

#endif
