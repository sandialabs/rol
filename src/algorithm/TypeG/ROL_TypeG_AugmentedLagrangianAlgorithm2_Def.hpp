// @HEADER
// *****************************************************************************
//               Rapid Optimization Library (ROL) Package
//
// Copyright 2014 NTESS and the ROL contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef ROL_TYPEG_AUGMENTEDLAGRANGIANALGORITHM2_DEF_H
#define ROL_TYPEG_AUGMENTEDLAGRANGIANALGORITHM2_DEF_H

#include <cmath>

#include "ROL_AugmentedLagrangianObjective2.hpp" 
#include "ROL_Solver.hpp" 

namespace ROL {
namespace TypeG {

template<typename Real>
AugmentedLagrangianAlgorithm2<Real>::AugmentedLagrangianAlgorithm2( ParameterList &list, const Ptr<Secant<Real>> &secant )
  : TypeG::Algorithm<Real>::Algorithm(), secant_(secant), list_(list), subproblemIter_(0) {
  // Set status test
  status_->reset();
  status_->add(makePtr<ConstraintStatusTest<Real>>(list));

  Real one(1), p1(0.1), p9(0.9), ten(1.e1), oe8(1.e8), oem8(1.e-8);
  ParameterList& sublist = list.sublist("Step").sublist("Augmented Lagrangian");
  useDefaultInitPen_     = sublist.get("Use Default Initial Penalty Parameter", true);
  state_->searchSize     = sublist.get("Initial Penalty Parameter",             ten);
  // Multiplier update parameters
  scaleLagrangian_      = sublist.get("Use Scaled Augmented Lagrangian",          false);
  minPenaltyLowerBound_ = sublist.get("Penalty Parameter Reciprocal Lower Bound", p1);  // \theta, no superscript
  penaltyUpdate_        = sublist.get("Penalty Parameter Growth Factor",          ten); // \eta (or \overline{\eta} which overrides after sufficiently many iterations)
  maxPenaltyParam_      = sublist.get("Maximum Penalty Parameter",                oe8);
  minPenaltyReciprocal_ = p1;                                                           // \theta with superscript
  // Optimality tolerance update
  optIncreaseExponent_ = sublist.get("Optimality Tolerance Update Exponent",   one);
  optDecreaseExponent_ = sublist.get("Optimality Tolerance Decrease Exponent", one);
  optToleranceInitial_ = sublist.get("Initial Optimality Tolerance",           one);
  // Feasibility tolerance update
  feasIncreaseExponent_ = sublist.get("Feasibility Tolerance Update Exponent",   p1);  // \alpha
  feasDecreaseExponent_ = sublist.get("Feasibility Tolerance Decrease Exponent", p9);  // \beta
  feasToleranceInitial_ = sublist.get("Initial Feasibility Tolerance",           one);
  // Subproblem information
  print_         = sublist.get("Print Intermediate Optimization History", false);
  maxit_         = sublist.get("Subproblem Iteration Limit",              1000);
  subStep_       = sublist.get("Subproblem Step Type",                    "Trust Region");
  hessianApprox_ = sublist.get("Level of Hessian Approximation",          0);
  list_.sublist("Step").set("Type",subStep_);
  list_.sublist("Status Test").set("Iteration Limit",maxit_);
  list_.sublist("Status Test").set("Use Relative Tolerances",false);
  // Verbosity setting
  verbosity_          = list.sublist("General").get("Output Level", 0);
  printHeader_        = verbosity_ > 0;
  print_              = (verbosity_ > 2 ? true : print_);
  list_.sublist("General").set("Output Level",(print_ ? verbosity_ : 0));
  // Outer iteration tolerances
  outerFeasTolerance_ = list.sublist("Status Test").get("Constraint Tolerance",    oem8);
  outerOptTolerance_  = list.sublist("Status Test").get("Gradient Tolerance",      oem8);
  outerStepTolerance_ = list.sublist("Status Test").get("Step Tolerance",          oem8);
  useRelTol_          = list.sublist("Status Test").get("Use Relative Tolerances", false);
  // Scaling
  useDefaultScaling_  = sublist.get("Use Default Problem Scaling", true);
  fscale_             = sublist.get("Objective Scaling",           one);
  cscale_             = sublist.get("Constraint Scaling",          one);

  // > nu and gamma
}

template<typename Real>
void AugmentedLagrangianAlgorithm2<Real>::initialize( Vector<Real>                        &x,
                                                     const Vector<Real>                   &g,
                                                     AugmentedLagrangianObjective2<Real>  &alobj,
                                                     std::ostream                         &outStream ) {
  const Real one(1), TOL(1.e-2);
  Real tol = std::sqrt(ROL_EPSILON<Real>());
  // > TypeG::Algorithm<Real>::initialize(x,g,l,c);
  if (state_->iterateVec == nullPtr) {
    state_->iterateVec = x.clone();
  }
  state_->iterateVec->set(x);
  // > if (state_->lagmultVec == nullPtr) {
  // >   state_->lagmultVec = mul.clone();
  // > }
  // > state_->lagmultVec->set(mul);
  if (state_->stepVec == nullPtr) {
    state_->stepVec = x.clone();
  }
  state_->stepVec->zero();
  if (state_->gradientVec == nullPtr) {
    state_->gradientVec = g.clone();
  }
  state_->gradientVec->set(g);
  // > if (state_->constraintVec == nullPtr) {
  // >   state_->constraintVec = c.clone();
  // > }
  // > state_->constraintVec->zero();
  if (state_->minIterVec == nullPtr) {
    state_->minIterVec = x.clone();
  }
  state_->minIterVec->set(x);
  state_->minIter = state_->iter;
  state_->minValue = state_->value;

  // Initialize the algorithm state
  state_->nfval = 0;
  state_->ncval = 0;
  state_->ngrad = 0;

  // Compute objective value
  alobj.update(x,UpdateType::Initial,state_->iter);
  state_->value = alobj.getObjectiveValue(x,tol);
  // > alobj.gradient(*state_->gradientVec,x,tol);

  // ========================================================================
  // lagmultVec, constraintVec, and gradientVec are unused state_ elements.
  // ========================================================================

  if (useDefaultScaling_) {
    fscale_ = one/std::max(one,alobj.getObjectiveGradient(x,tol)->norm());
    alobj.setScaling(fscale_);
  }

  unsigned numberPenalties = alobj.getNumberConstraints();

  // Compute constraint violation
  std::vector<Real> feasibilities(numberPenalties);
  for (unsigned i = 0; i < numberPenalties; ++i)
    feasibilities[i] = alobj.feasibility(x,tol,i);
  state_->cnorm = 0;
  if (!feasibilities.empty())
    state_->cnorm = *std::max_element(feasibilities.begin(),feasibilities.end());

  // Update evaluation counters
  state_->ncval += alobj.getNumberConstraintEvaluations();
  state_->nfval += alobj.getNumberFunctionEvaluations();
  state_->ngrad += alobj.getNumberGradientEvaluations();

  // Compute problem scaling
  std::vector<Real> constraintScalings(numberPenalties);
  Ptr<Vector<Real>> constraintVec;
  // > TODO
  // >if (useDefaultScaling_) {
  // >  Ptr<Vector<Real>> ji = x.clone();
  // >  // Helper function
  // >  auto getConstraintScale = [&] (const Vector<Real> &c) -> Real {
  // >    Real cscale(1);
  // >    try {
  // >      Real maxji(0), normji(0);
  // >      for (int i = 0; i < c.dimension(); ++i) {
  // >        c.applyAdjointJacobian(*ji,*c.basis(i),x,tol);
  // >        normji = ji->norm();
  // >        maxji  = std::max(normji,maxji);
  // >      }
  // >      cscale = one/std::max(one,maxji);
  // >    }
  // >    catch (std::exception &e) {}
  // >    return cscale;
  // >  };
  // >  for (unsigned k = 0; k < numberPenalties; ++k) {
  // >    constraintVec = alobj.getConstraintVec(x,tol,k);
  // >    constraintScalings[k] = getConstraintScale(*constraintVec);
  // >    alobj.setScaling(constraintScalings[k],k);
  // >  }
  // >  if (!constraintScalings.empty())
  // >    cscale_ = *std::max_element(constraintScalings.begin(),constraintScalings.end());
  // >}
  // >else {
    for (unsigned i = 0; i < numberPenalties; ++i) {
      constraintScalings[i] = cscale_;
      alobj.setScaling(constraintScalings[i],i);
  // >}
  }

  // > // Compute gradient of the lagrangian
  // > x.axpy(-one,state_->gradientVec->dual());
  // > proj_->project(x,outStream);
  // > x.axpy(-one/std::min(fscale_,cscale_),*state_->iterateVec);
  // > state_->gnorm = x.norm();
  // > x.set(*state_->iterateVec);
  // > TODO: problem.getSubproblemStationarityMeasure(*state_->iterateVec,x); which is used below to set tolerances

  // Compute initial penalty parameter
  if (useDefaultInitPen_) {
    const Real oem8(1e-8), oem2(1e-2), two(2), ten(10);
    Real penaltyParameter;
    for (unsigned i = 0; i < numberPenalties; ++i) {
      penaltyParameter = std::max(oem8, std::min(ten*std::max(one,std::abs(fscale_*state_->value)) / std::max(one,std::pow(constraintScalings[i]*feasibilities[i],two)),
                                                 oem2*maxPenaltyParam_)); // ROL convention
      alobj.setPenaltyParameter(penaltyParameter,i);
      state_->searchSize = std::max(state_->searchSize,penaltyParameter);
    }
  }
  else {
    for (unsigned i = 0; i < numberPenalties; ++i)
      alobj.setPenaltyParameter(state_->searchSize,i);
  }

  // Initialize intermediate stopping tolerances
  // > if (useRelTol_) outerOptTolerance_ *= state_->gnorm;
  if (verbosity_ > 0)
    outStream << "Warning: \"Use Relative Tolerances\" parameter is unsupported!" << std::endl;
  minPenaltyReciprocal_ = std::min(one/state_->searchSize,minPenaltyLowerBound_);
  optTolerance_  = std::max(TOL*outerOptTolerance_,
                            optToleranceInitial_*std::pow(minPenaltyReciprocal_,optDecreaseExponent_));
  // > optTolerance_  = std::min<Real>(optTolerance_,TOL*state_->gnorm);
  feasTolerance_ = std::max(TOL*outerFeasTolerance_,
                            feasToleranceInitial_*std::pow(minPenaltyReciprocal_,feasDecreaseExponent_));

  alobj.reset();

  if (verbosity_ > 1) {
    outStream << std::endl;
    outStream << "Augmented Lagrangian Initialize"                    << std::endl;
    outStream << "Objective Scaling:  "         << fscale_            << std::endl;
    outStream << "Maximum Constraint Scaling: " << cscale_            << std::endl;
    outStream << "Maximum Penalty Parameter:  " << state_->searchSize << std::endl;
    outStream << std::endl;
  }
}

template<typename Real>
void AugmentedLagrangianAlgorithm2<Real>::run( Vector<Real>  &x,
                                               Vector<Real>  &g,
                                               Problem<Real> &problem,
                                               std::ostream  &outStream ) {

  // ========================================================================
  // STEP 1: Get subproblem and objective
  // ========================================================================
  Ptr<Problem<Real>> subproblem = problem.getAugmentedLagrangianSubproblem();
  Ptr<AugmentedLagrangianObjective2<Real>> alobj = staticPtrCast<AugmentedLagrangianObjective2<Real>>(subproblem->getObjective());

  // ========================================================================
  // STEP 2: Set up
  // ========================================================================
  initialize(x,g,*alobj,outStream);

  // Constants
  const Real one(1), two(2), oem2(1e-2);
  Real tol(std::sqrt(ROL_EPSILON<Real>()));

  // Additional parameters
  unsigned maxSubproblemFails = 2;
  unsigned k0 = 100;
  Real nu = 1;
  Real gamma = 1;

  state_->gnorm = ROL_INF<Real>();

  // Output
  if (verbosity_ > 0) writeOutput(outStream,true);

  // Data

  unsigned numberPenalties = alobj->getNumberConstraints();

  std::vector<Real> feasibilities(numberPenalties);

  std::vector<Real> dualResiduals(numberPenalties);

  std::vector<Real> dualTolerances(numberPenalties); // \tau
  for (unsigned i = 0; i < numberPenalties; ++i)
    dualTolerances[i] = feasTolerance_;

  std::vector<Real> theta(numberPenalties);
  for (unsigned i = 0; i < numberPenalties; ++i)
    theta[i] = minPenaltyReciprocal_;

  EExitStatus statusFlag;
  bool isSubproblemConverged = false;
  Real penaltyParameter;
  bool isForcedUpdate = false;
  bool isUpdated      = false;

  // ========================================================================
  // STEP 3: Run algorithm
  // ========================================================================
  while (status_->check(*state_)) {
    // Solve augmented Lagrangian subproblem
    list_.sublist("Status Test").set("Gradient Tolerance",optTolerance_);
    list_.sublist("Status Test").set("Step Tolerance",1.e-6*optTolerance_);
    list_.sublist("Status Test").set("Gradient Tolerance",optTolerance_);
    Solver<Real> solver(subproblem,list_);
    isSubproblemConverged = false;
    for (unsigned i = 0; i < maxSubproblemFails; ++i) {
      solver.reset();
      alobj->update(x,UpdateType::Initial,state_->iter);
      solver.solve(outStream);
      statusFlag = solver.getAlgorithmState()->statusFlag;
      isSubproblemConverged = (statusFlag == EXITSTATUS_CONVERGED) || (statusFlag == EXITSTATUS_USERDEFINED);
      if (isSubproblemConverged)
        break;
    }

    if (!isSubproblemConverged) {
      if (verbosity_ > 0)
        outStream << "Warning: Augmented Lagrangian subproblem failed to converge!" << std::endl;
    }

    subproblemIter_ = solver.getAlgorithmState()->iter;

    // Compute step
    state_->stepVec->set(x);
    state_->stepVec->axpy(-one,*state_->iterateVec);
    state_->snorm = state_->stepVec->norm();

    // Update iteration information
    state_->iter++;
    state_->iterateVec->set(x);
    state_->value = alobj->getObjectiveValue(x,tol);
    for (unsigned i = 0; i < numberPenalties; ++i)
      feasibilities[i] = alobj->feasibility(x,tol,i);
    if (!feasibilities.empty())
      state_->cnorm = *std::max_element(feasibilities.begin(),feasibilities.end());
    state_->gnorm = solver.getAlgorithmState()->gnorm;
    //alobj.update(x,UpdateType::Accept,state_->iter);

    // Update evaluation counters
    state_->nfval += alobj->getNumberFunctionEvaluations();
    state_->ngrad += alobj->getNumberGradientEvaluations();
    state_->ncval += alobj->getNumberConstraintEvaluations();

    // ========================================================================
    // ALESQP Updates (Algorithm 4.1)
    // ========================================================================

    for (unsigned i = 0; i < numberPenalties; ++i)
      dualResiduals[i] = alobj->dualResidual(x,tol,i);

    // Forced updates for large iterations (line 10)
    isForcedUpdate = false;
    if (state_->iter > k0) {
      for (unsigned i = 0; i < numberPenalties; ++i) {
        penaltyParameter = alobj->getPenaltyParameter(i);
        if (alobj->getScaling(i)*dualResiduals[i] > penaltyParameter*dualTolerances[i])
          isForcedUpdate = true;
      }
    }

    // Update (line 13)
    isUpdated = false;
    for (unsigned i = 0; i < numberPenalties; ++i) {
      penaltyParameter = alobj->getPenaltyParameter(i);
      if (isForcedUpdate || alobj->getScaling(i)*dualResiduals[i] > penaltyParameter*dualTolerances[i]) {
        penaltyParameter  *= penaltyUpdate_;
        penaltyParameter   = std::min(penaltyParameter,maxPenaltyParam_);
        state_->searchSize = std::max(state_->searchSize,penaltyParameter);
        theta[i]           = std::min(one/penaltyParameter,theta[i]);
        dualTolerances[i]  = feasToleranceInitial_*std::pow(theta[i],feasDecreaseExponent_);
        dualTolerances[i]  = std::max(dualTolerances[i],oem2*outerFeasTolerance_);   // ROL convention
        alobj->setPenaltyParameter(penaltyParameter,i);
        isUpdated = true;
      }
      else {
        theta[i]           = std::min(one/penaltyParameter,theta[i]);
        dualTolerances[i] *= std::pow(theta[i],feasIncreaseExponent_);
        dualTolerances[i]  = std::max(dualTolerances[i],oem2*outerFeasTolerance_); // ROL convention
      }
      if (alobj->getScaling(i)*dualResiduals[i] <= nu*std::pow(penaltyParameter,gamma))
        alobj->updateMultiplier(x,tol,i);
    }
    if (!dualTolerances.empty())
      feasTolerance_ = *std::max_element(dualTolerances.begin(),dualTolerances.end());
    // ROL update to \epsilon. Updates to the constraint tolerance, \delta, are possible as well.
    outStream << std::endl << "isUpdated: " << isUpdated << std::endl;
    if (isUpdated) {
      optTolerance_ = std::max(oem2*outerOptTolerance_,
                      optToleranceInitial_/std::pow(std::max(two,state_->searchSize),optDecreaseExponent_));
    }
    else {
      optTolerance_ = std::max(oem2*outerOptTolerance_,
                      optTolerance_/std::pow(std::max(two,state_->searchSize),optIncreaseExponent_));
    }
    alobj->reset();

    // Update Output
    if (verbosity_ > 0) writeOutput(outStream,printHeader_);
  }
  if (verbosity_ > 0) TypeG::Algorithm<Real>::writeExitStatus(outStream);
}


template<typename Real>
void AugmentedLagrangianAlgorithm2<Real>::run( Vector<Real>          &x,
                                              const Vector<Real>    &g,
                                              Objective<Real>       &obj,
                                              BoundConstraint<Real> &bnd,
                                              Constraint<Real>      &econ,
                                              Vector<Real>          &emul,
                                              const Vector<Real>    &eres,
                                              std::ostream          &outStream ) {
  throw Exception::NotImplemented(">>> ROL::TypeG""AugmentedLagrangianAlgorithm2::run: Overload Not Implemented!");
}

template<typename Real>
void AugmentedLagrangianAlgorithm2<Real>::writeHeader( std::ostream& os ) const {
  std::ios_base::fmtflags osFlags(os.flags());
  if(verbosity_>1) {
    os << std::string(114,'-') << std::endl;
    os << "Augmented Lagrangian status output definitions" << std::endl << std::endl;
    os << "  iter    - Number of iterates (steps taken)"            << std::endl;
    os << "  fval    - Objective function value"                    << std::endl;
    os << "  cnorm   - Norm of the constraint violation"            << std::endl;
    os << "  gLnorm  - Norm of the gradient of the Lagrangian"      << std::endl;
    os << "  snorm   - Norm of the step"                            << std::endl;
    os << "  penalty - Penalty parameter"                           << std::endl;
    os << "  feasTol - Feasibility tolerance"                       << std::endl;
    os << "  optTol  - Optimality tolerance"                        << std::endl;
    os << "  #fval   - Number of times the objective was computed"  << std::endl;
    os << "  #grad   - Number of times the gradient was computed"   << std::endl;
    os << "  #cval   - Number of times the constraint was computed" << std::endl;
    os << "  subIter - Number of iterations to solve subproblem"    << std::endl;
    os << std::string(114,'-') << std::endl;
  }
  os << "  ";
  os << std::setw(6)  << std::left << "iter";
  os << std::setw(15) << std::left << "fval";
  os << std::setw(15) << std::left << "cnorm";
  os << std::setw(15) << std::left << "gLnorm";
  os << std::setw(15) << std::left << "snorm";
  os << std::setw(10) << std::left << "penalty";
  os << std::setw(10) << std::left << "feasTol";
  os << std::setw(10) << std::left << "optTol";
  os << std::setw(8)  << std::left << "#fval";
  os << std::setw(8)  << std::left << "#grad";
  os << std::setw(8)  << std::left << "#cval";
  os << std::setw(8)  << std::left << "subIter";
  os << std::endl;
  os.flags(osFlags);
}

template<typename Real>
void AugmentedLagrangianAlgorithm2<Real>::writeName( std::ostream& os ) const {
  std::ios_base::fmtflags osFlags(os.flags());
  os << std::endl << "Augmented Lagrangian Solver (Type G, General Constraints)";
  os << std::endl;
  os << "Subproblem Solver: " << subStep_ << std::endl;
  os.flags(osFlags);
}

template<typename Real>
void AugmentedLagrangianAlgorithm2<Real>::writeOutput( std::ostream& os, const bool print_header ) const {
  std::ios_base::fmtflags osFlags(os.flags());
  os << std::scientific << std::setprecision(6);
  if ( state_->iter == 0 ) writeName(os);
  if ( print_header )      writeHeader(os);
  if ( state_->iter == 0 ) {
    os << "  ";
    os << std::setw(6)  << std::left << state_->iter;
    os << std::setw(15) << std::left << state_->value;
    os << std::setw(15) << std::left << state_->cnorm;
    os << std::setw(15) << std::left << state_->gnorm;
    os << std::setw(15) << std::left << "---";
    os << std::scientific << std::setprecision(2);
    os << std::setw(10) << std::left << state_->searchSize;
    os << std::setw(10) << std::left << std::max(feasTolerance_,outerFeasTolerance_);
    os << std::setw(10) << std::left << std::max(optTolerance_,outerOptTolerance_);
    os << std::scientific << std::setprecision(6);
    os << std::setw(8) << std::left << state_->nfval;
    os << std::setw(8) << std::left << state_->ngrad;
    os << std::setw(8) << std::left << state_->ncval;
    os << std::setw(8) << std::left << "---";
    os << std::endl;
  }
  else {
    os << "  ";
    os << std::setw(6)  << std::left << state_->iter;
    os << std::setw(15) << std::left << state_->value;
    os << std::setw(15) << std::left << state_->cnorm;
    os << std::setw(15) << std::left << state_->gnorm;
    os << std::setw(15) << std::left << state_->snorm;
    os << std::scientific << std::setprecision(2);
    os << std::setw(10) << std::left << state_->searchSize;
    os << std::setw(10) << std::left << feasTolerance_;
    os << std::setw(10) << std::left << optTolerance_;
    os << std::scientific << std::setprecision(6);
    os << std::setw(8) << std::left << state_->nfval;
    os << std::setw(8) << std::left << state_->ngrad;
    os << std::setw(8) << std::left << state_->ncval;
    os << std::setw(8) << std::left << subproblemIter_;
    os << std::endl;
  }
  os.flags(osFlags);
}

} // namespace TypeG
} // namespace ROL

#endif
