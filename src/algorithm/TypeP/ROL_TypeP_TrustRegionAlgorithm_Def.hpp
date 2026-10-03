// @HEADER
// *****************************************************************************
//               Rapid Optimization Library (ROL) Package
//
// Copyright 2014 NTESS and the ROL contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef ROL_TYPEP_TRUSTREGIONALGORITHM_DEF_HPP
#define ROL_TYPEP_TRUSTREGIONALGORITHM_DEF_HPP

#include "ROL_TrustRegion_P_Factory.hpp"

namespace ROL {
namespace TypeP {

template<typename Real>
TrustRegionAlgorithm<Real>::TrustRegionAlgorithm( ParameterList &parlist,
                                                  const Ptr<Secant<Real>> &secant )
    : Algorithm<Real>(), esec_(SECANT_USERDEFINED) {
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
  gamma0_ = trlist.get("Radius Shrinking Rate (Negative rho)", static_cast<Real>(0.25)); // 0.0625
  gamma1_ = trlist.get("Radius Shrinking Rate (Positive rho)", static_cast<Real>(0.25));
  gamma2_ = trlist.get("Radius Growing Rate",                  static_cast<Real>(2.5));
  TRsafe_ = trlist.get("Safeguard Size",                       static_cast<Real>(1.0e10));
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
  mu0_               = lmlist.get("Sufficient Decrease Parameter",                             1e-4);

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
void TrustRegionAlgorithm<Real>::initialize(Vector<Real>          &x,
                                            const Vector<Real>    &g,
                                            Real                   ftol,
                                            Objective<Real>       &sobj,
                                            Objective<Real>       &nobj,
                                            Vector<Real>          &px,
                                            Vector<Real>          &dg,
                                            std::ostream          &outStream) {
  // Initialize data
  Algorithm<Real>::initialize(x,g);
  model_->initialize(x,g);
  nhess_ = 0;
  // Update approximate gradient and approximate objective function.
  if (initProx_){
    nobj.prox(*state_->iterateVec,x,t0_, ftol); state_->nprox++;
    x.set(*state_->iterateVec);
  }
  sobj.update(x,UpdateType::Initial,state_->iter);
  state_->svalue = sobj.value(x,ftol); state_->nsval++;
  nobj.update(x, UpdateType::Initial,state_->iter);
  state_->nvalue = nobj.value(x,ftol); state_->nnval++;
  state_->value = state_->svalue + state_->nvalue;
  computeGradient(x,*state_->gradientVec,px,dg,*state_->stepVec,state_->searchSize,sobj,nobj,true,gtol_,state_->gnorm,outStream);

  solver_->initialize(x,*state_->gradientVec);
  state_->snorm = ROL_INF<Real>();
  // Compute initial trust region radius if desired.
  if ( state_->searchSize <= static_cast<Real>(0) )
    state_->searchSize = state_->gradientVec->norm();
  SPiter_ = 0;
  SPflag_ = 0;
  fr_     = state_->value; 
  fc_     = state_->value;
  fmin_   = state_->value;
}

template<typename Real>
Real TrustRegionAlgorithm<Real>::computeValue(Real inTol,
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
    if (!(iter%updateIter_) && (iter!=0)) force_ *= forceFactor_;
    const Real one(1);
    Real eta = static_cast<Real>(0.999)*std::min(eta1_,one-eta2_);
    outTol   = scale_*std::pow(eta*std::min(pRed,force_),one/omega_);
    if (inTol > outTol) {
      fold = sobj.value(xold,outTol); state_->nsval++;
    }
  }
  // Evaluate objective function at new iterate
  sobj.update(x,UpdateType::Trial);
  Real fval = sobj.value(x,outTol); state_->nsval++;
  return fval;
}

template<typename Real>
void TrustRegionAlgorithm<Real>::computeGradient(const Vector<Real> &x,
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
    Real gtol0 = scale0_*del;
    if (accept) gtol  = gtol0 + static_cast<Real>(1);
    else        gtol0 = scale0_*std::min(gnorm,del);
    while ( gtol > gtol0 ) {
      gtol = gtol0;
      sobj.gradient(g,x,gtol); state_->ngrad++;
      dg.set(g.dual());
      solver_->pgstep(px, step, nobj, x, dg, t0_, gtol0); state_->nprox++; // change gtol? one or ocScale?
      gnorm = step.norm() / t0_;
      gtol0 = scale0_*std::min(gnorm,del);
    }
  }
  else {
    if (accept) {
      gtol = std::sqrt(ROL_EPSILON<Real>());
      sobj.gradient(g,x,gtol); state_->ngrad++;
      dg.set(g.dual());
      solver_->pgstep(px, step, nobj, x, dg, t0_, gtol); state_->nprox++;
      gnorm = step.norm() / t0_;
    }
  }
}

template<typename Real>
void TrustRegionAlgorithm<Real>::run(Vector<Real>       &x,
                                     const Vector<Real> &g,
                                     Objective<Real>    &sobj,
                                     Objective<Real>    &nobj,
                                     std::ostream       &outStream ) {
  const Real one(1);
  // Initialize trust-region data
  Real inTol = static_cast<Real>(0.1)*ROL_OVERFLOW<Real>(), outTol(inTol);
  Real strial(0), ntrial(0), smodel(0), Ftrial(0), pRed(0), rho(1);
  // Initialize trust-region data
  std::vector<std::string> output;
  Ptr<Vector<Real>> gmod = g.clone();
  Ptr<Vector<Real>> px   = x.clone();
  Ptr<Vector<Real>> dg   = x.clone();
  // Initialize Algorithm
  initialize(x,g,inTol,sobj,nobj, *px, *dg, outStream);
  // Initialize storage vectors
  Ptr<Vector<Real>> pwa1 = x.clone();
  Ptr<Vector<Real>> dwa1 = g.clone();
  // Output
  if (verbosity_ > 0) writeOutput(outStream,true);

  TRUtils::ETRFlag TRflagNM_;
  while (status_->check(*state_)) {
    // Build trust-region model
    model_->setData(sobj,*state_->iterateVec,*state_->gradientVec,gtol_);

    /**** SOLVE TRUST-REGION SUBPROBLEM ****/
    gmod->set(*state_->gradientVec);
    smodel = state_->svalue;
    ntrial = state_->nvalue;
    SPflag_ = 0; 
    SPiter_ = 0;
    solver_->solve(x, smodel, ntrial, *gmod,
                   *state_->iterateVec, state_->searchSize, 
                   *model_, nobj, SPflag_, SPiter_, nhess_, state_->nprox, state_->nnval, outStream);
    pRed = state_->value - (smodel+ntrial);
    state_->stepVec->set(x); state_->stepVec->axpy(-one,*state_->iterateVec);
    state_->snorm = state_->stepVec->norm();

    // Compute trial objective value
    strial = computeValue(inTol,outTol,pRed,state_->svalue,state_->searchSize, state_->iter,x,*state_->iterateVec,sobj);
    Ftrial = strial + ntrial;

    // Compute ratio of actual and predicted reduction
    TRflag_ = TRUtils::SUCCESS;
    TRUtils::analyzeRatio<Real>(rho,TRflag_,state_->value,Ftrial,pRed,eps_,outStream,verbosity_>1);
    if (useNM_) {
      TRUtils::analyzeRatio<Real>(rhoNM_,TRflagNM_,fr_,Ftrial,pRed+sigmar_,eps_,outStream,verbosity_>1);
      TRflag_ = (rho < rhoNM_ ? TRflagNM_ : TRflag_);
      rho     = (rho < rhoNM_ ?    rhoNM_ :    rho );
    }
    // Update algorithm state
    state_->iter++;
    // Accept/reject step and update trust region radius
    stepUpdate(x, *gmod, *px, *dg, sobj, nobj, *model_, rho, pRed,
               Ftrial, strial, ntrial, 
               false, gtol_,  inTol, outTol, 
               *pwa1, *dwa1, outStream); // put by computeGradient/value

    // Update Output
    if (verbosity_ > 0) writeOutput(outStream,writeHeader_);
  }
  if (verbosity_ > 0) Algorithm<Real>::writeExitStatus(outStream);
} // run
template<typename Real>
void TrustRegionAlgorithm<Real>::stepUpdate(Vector<Real>               &x, 
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
    Real zero(0); 
    if ((rho < eta0_ && TRflag_ == TRUtils::SUCCESS) || (TRflag_ >= 2)) { // Step Rejected
      x.set(*state_->iterateVec);
      sobj.update(x,UpdateType::Revert,state_->iter);
      nobj.update(x,UpdateType::Revert,state_->iter);
      if (interpRad_ && (rho < zero && TRflag_ != TRUtils::TRNAN)) {
        // Negative reduction, interpolate to find new trust-region radius
        state_->searchSize = TRUtils::interpolateRadius<Real>(*state_->gradientVec,*state_->stepVec,
          state_->snorm,pRed,state_->value,Ftrial,state_->searchSize,gamma0_,gamma1_,eta2_,
          outStream,verbosity_>1);
      }
      else { // Shrink trust-region radius
        state_->searchSize = gamma1_*std::min(state_->snorm,state_->searchSize);
      }
      computeGradient(x,*state_->gradientVec,px,dg,pwa1,state_->searchSize,sobj,nobj,false,gtol,state_->gnorm,outStream);
    } // Step Rejected
    else if ((rho >= eta0_ && TRflag_ != TRUtils::NPOSPREDNEG)
             || (TRflag_ == TRUtils::POSPREDNEG)) { // Step Accepted
      state_->value  = Ftrial;
      state_->svalue = strial;
      state_->nvalue = ntrial;
      sobj.update(x,UpdateType::Accept,state_->iter);
      nobj.update(x,UpdateType::Accept,state_->iter);
      inTol = outTol;
      if (useNM_) {
        sigmac_ += pRed; sigmar_ += pRed;
        if (Ftrial < fmin_) { fmin_ = Ftrial; fc_ = fmin_; sigmac_ = zero; L_ = 0; }
        else {
          L_++;
          if (Ftrial > fc_)     { fc_ = Ftrial; sigmac_ = zero;   }
          if (L_ == storageNM_) { fr_ = fc_;     sigmar_ = sigmac_; }
        }
      } // useNM_
      // Increase trust-region radius
      if (rho >= eta2_) state_->searchSize = std::min(gamma2_*state_->searchSize, delMax_);
      // Compute gradient at new iterate
      dwa1.set(*state_->gradientVec);
      computeGradient(x,*state_->gradientVec,px,dg,pwa1,state_->searchSize,sobj,nobj,true,gtol,state_->gnorm,outStream);
      state_->iterateVec->set(x);
      // Update secant information in trust-region model
      model_->update(x,*state_->stepVec,dwa1,*state_->gradientVec,
                     state_->snorm,state_->iter);
    } // Step Accepted

} // stepUpdate function


template<typename Real>
void TrustRegionAlgorithm<Real>::writeHeader( std::ostream& os ) const {
  std::ios_base::fmtflags osFlags(os.flags());
  if(verbosity_ > 1) {
    os << std::string(114,'-') << std::endl;
    os << "Trust-Region status output definitions" << std::endl << std::endl;
    os << "  iter    - Number of iterates (steps taken)" << std::endl;
    os << "  value   - Objective function value" << std::endl;
    os << "  gnorm   - Norm of the gradient" << std::endl;
    os << "  snorm   - Norm of the step (update to optimization vector)" << std::endl;
    os << "  delta   - Trust-Region radius" << std::endl;
    os << "  #sval   - Number of times the smooth objective function was evaluated" << std::endl;
    os << "  #nval   - Number of times the nonsmooth objective function was evaluated" << std::endl;
    os << "  #grad   - Number of times the gradient was computed" << std::endl;
    os << "  #hess   - Number of times the Hessian was applied" << std::endl;
    os << "  #prox   - Number of times the proximal operator was computed" << std::endl;
    os << std::endl;
    os << "  tr_flag - Trust-Region flag" << std::endl;
    for( int flag = TRUtils::SUCCESS; flag != TRUtils::UNDEFINED; ++flag ) {
      os << "    " << NumberToString(flag) << " - "
           << TRUtils::ETRFlagToString(static_cast<TRUtils::ETRFlag>(flag)) << std::endl;
    }
    if( algSelect_  == TRUSTREGION_P_NCG ) {
      os << std::endl;
      os << "  iterCG - Number of Truncated CG iterations" << std::endl << std::endl;
      os << "  flagGC - Trust-Region Truncated CG flag" << std::endl;
      for( int flag = CG_FLAG_SUCCESS; flag != CG_FLAG_UNDEFINED; ++flag ) {
        os << "    " << NumberToString(flag) << " - "
             << ECGFlagToString(static_cast<ECGFlag>(flag)) << std::endl;
      }            
    }
    else if( algSelect_ == TRUSTREGION_P_SPG || algSelect_ == TRUSTREGION_P_SPG2 ) {
      os << std::endl;
      os << "  iterCG - Number of spectral projected gradient iterations" << std::endl << std::endl;
      os << "  flagGC - Trust-Region spectral projected gradient flag" << std::endl;
    }
    os << std::string(114,'-') << std::endl;
  }
  os << "  ";
  os << std::setw(6)  << std::left << "iter";
  os << std::setw(15) << std::left << "value";
  os << std::setw(15) << std::left << "gnorm";
  os << std::setw(15) << std::left << "snorm";
  os << std::setw(15) << std::left << "delta";
  os << std::setw(10) << std::left << "#sval";
  os << std::setw(10) << std::left << "#nval";
  os << std::setw(10) << std::left << "#grad";
  os << std::setw(10) << std::left << "#hess";
  os << std::setw(10) << std::left << "#prox";
  os << std::setw(10) << std::left << "tr_flag";
  if ( algSelect_ == TRUSTREGION_P_NCG ) {
    os << std::setw(10) << std::left << "iterCG";
    os << std::setw(10) << std::left << "flagCG";
  }
  else if (algSelect_ == TRUSTREGION_P_SPG || algSelect_ == TRUSTREGION_P_SPG2) {
    os << std::setw(10) << std::left << "iterSPG";
    os << std::setw(10) << std::left << "flagSPG";
  }
  os << std::endl;
  os.flags(osFlags);
}

template<typename Real>
void TrustRegionAlgorithm<Real>::writeName( std::ostream& os ) const {
  std::ios_base::fmtflags osFlags(os.flags());
  os << std::endl << ETrustRegionPToString(algSelect_) << " Trust-Region Solver";
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
void TrustRegionAlgorithm<Real>::writeOutput(std::ostream& os, bool write_header ) const {
  std::ios_base::fmtflags osFlags(os.flags());
  os << std::scientific << std::setprecision(6);
  if ( state_->iter == 0 ) {
    writeName(os);
  }
  if ( write_header ) {
    writeHeader(os);
  }
  if ( state_->iter == 0 ) {
    os << "  ";
    os << std::setw(6)  << std::left << state_->iter;
    os << std::setw(15) << std::left << state_->value;
    os << std::setw(15) << std::left << state_->gnorm;
    os << std::setw(15) << std::left << "---";
    os << std::setw(15) << std::left << state_->searchSize;
    os << std::setw(10) << std::left << state_->nsval;
    os << std::setw(10) << std::left << state_->nnval;
    os << std::setw(10) << std::left << state_->ngrad;
    os << std::setw(10) << std::left << nhess_;
    os << std::setw(10) << std::left << state_->nprox;
    os << std::setw(10) << std::left << "---";
    os << std::setw(10) << std::left << "---";
    os << std::setw(10) << std::left << "---";
    os << std::endl;
  }
  else {
    os << "  ";
    os << std::setw(6)  << std::left << state_->iter;
    os << std::setw(15) << std::left << state_->value;
    os << std::setw(15) << std::left << state_->gnorm;
    os << std::setw(15) << std::left << state_->snorm;
    os << std::setw(15) << std::left << state_->searchSize;
    os << std::setw(10) << std::left << state_->nsval;
    os << std::setw(10) << std::left << state_->nnval;
    os << std::setw(10) << std::left << state_->ngrad;
    os << std::setw(10) << std::left << nhess_;
    os << std::setw(10) << std::left << state_->nprox;
    os << std::setw(10) << std::left << TRflag_;
    os << std::setw(10) << std::left << SPiter_;
    os << std::setw(10) << std::left << SPflag_;
    os << std::endl;
  }
  os.flags(osFlags);
}
} // namespace TypeP
} // namespace ROL

#endif
