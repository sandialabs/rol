// @HEADER
// *****************************************************************************
//               Rapid Optimization Library (ROL) Package
//
// Copyright 2014 NTESS and the ROL contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef ROL_PROBLEM_DEF_HPP
#define ROL_PROBLEM_DEF_HPP

#include <iostream>

#include "ROL_AugmentedLagrangianObjective2.hpp"
#include "ROL_AugmentedLagrangianPenalty.hpp"
#include "ROL_IdentityOperator.hpp"
#include "ROL_PartitionedVector.hpp"
#include "ROL_PolyhedralProjection.hpp"
#include "ROL_Projection_Partitioned.hpp"
#include "ROL_ZeroProjection.hpp"

namespace ROL {

template<typename Real>
Problem<Real>::Problem( const Ptr<Objective<Real>> &obj,
                        const Ptr<Vector<Real>>    &x,
                        const Ptr<Vector<Real>>    &g)
  : isFinalized_(false), hasBounds_(false),
    hasEquality_(false), hasInequality_(false),
    hasLinearEquality_(false), hasLinearInequality_(false),
    hasProximableObjective_(false),
    cnt_econ_(0), cnt_icon_(0), cnt_linear_econ_(0), cnt_linear_icon_(0),
    obj_(nullPtr), nobj_(nullPtr), xprim_(nullPtr), xdual_(nullPtr), bnd_(nullPtr),
    con_(nullPtr), mul_(nullPtr), res_(nullPtr), proj_(nullPtr),
    problemType_(TYPE_U) {
  INPUT_obj_   = obj;
  INPUT_nobj_  = nullPtr;
  INPUT_xprim_ = x;
  INPUT_bnd_   = nullPtr;
  INPUT_con_.clear();
  INPUT_linear_con_.clear();
  if (g==nullPtr) INPUT_xdual_ = x->dual().clone();
  else            INPUT_xdual_ = g;
}

template<typename Real>
void Problem<Real>::addBoundConstraint(const Ptr<BoundConstraint<Real>> &bnd) {
  ROL_TEST_FOR_EXCEPTION(isFinalized_,std::invalid_argument,
    ">>> ROL::Problem: Cannot add bounds after problem is finalized!");

  INPUT_bnd_ = bnd;
  hasBounds_ = true;
}

template<typename Real>
void Problem<Real>::removeBoundConstraint() {
  ROL_TEST_FOR_EXCEPTION(isFinalized_,std::invalid_argument,
    ">>> ROL::Problem: Cannot remove bounds after problem is finalized!");

  INPUT_bnd_ = nullPtr;
  hasBounds_ = false;
}

template<typename Real>
void Problem<Real>::addConstraint( std::string                  name,
                                   const Ptr<Constraint<Real>> &econ,
                                   const Ptr<Vector<Real>>     &emul,
                                   const Ptr<Vector<Real>>     &eres,
                                   bool                         reset) {
  ROL_TEST_FOR_EXCEPTION(isFinalized_,std::invalid_argument,
    ">>> ROL::Problem: Cannot add constraint after problem is finalized!");

  if (reset) INPUT_con_.clear();

  auto it = INPUT_con_.find(name);
  ROL_TEST_FOR_EXCEPTION(it != INPUT_con_.end(),std::invalid_argument,
    ">>> ROL::Problem: Constraint names must be distinct!");
  it = INPUT_linear_con_.find(name);
  ROL_TEST_FOR_EXCEPTION(it != INPUT_linear_con_.end(),std::invalid_argument,
    ">>> ROL::Problem: Constraint names must be distinct!");
  auto itp = INPUT_proj_.find(name);
  ROL_TEST_FOR_EXCEPTION(itp != INPUT_proj_.end(),std::invalid_argument,
    ">>> ROL::Problem: Constraint names must be distinct!");

  INPUT_con_.insert({name,ConstraintData<Real>(econ,emul,eres)});
  hasEquality_ = true;
  cnt_econ_++;
}

template<typename Real>
void Problem<Real>::addConstraint( std::string                       name,
                                   const Ptr<Constraint<Real>>      &icon,
                                   const Ptr<Vector<Real>>          &imul,
                                   const Ptr<BoundConstraint<Real>> &ibnd,
                                   const Ptr<Vector<Real>>          &ires,
                                   bool                              reset) {
  ROL_TEST_FOR_EXCEPTION(isFinalized_,std::invalid_argument,
    ">>> ROL::Problem: Cannot add constraint after problem is finalized!");

  if (reset) INPUT_con_.clear();

  auto it = INPUT_con_.find(name);
  ROL_TEST_FOR_EXCEPTION(it != INPUT_con_.end(),std::invalid_argument,
    ">>> ROL::Problem: Constraint names must be distinct!");
  it = INPUT_linear_con_.find(name);
  ROL_TEST_FOR_EXCEPTION(it != INPUT_linear_con_.end(),std::invalid_argument,
    ">>> ROL::Problem: Constraint names must be distinct!");
  auto itp = INPUT_proj_.find(name);
  ROL_TEST_FOR_EXCEPTION(itp != INPUT_proj_.end(),std::invalid_argument,
    ">>> ROL::Problem: Constraint names must be distinct!");

  INPUT_con_.insert({name,ConstraintData<Real>(icon,imul,ires,ibnd)});
  hasInequality_ = true;
  cnt_icon_++;
}

template<typename Real>
void Problem<Real>::addConstraint( std::string                  name,
                                   const Ptr<Constraint<Real>> &pcon,
                                   const Ptr<Vector<Real>>     &pmul,
                                   const Ptr<Projection<Real>> &proj,
                                   const Ptr<Vector<Real>>     &pres,
                                   bool                         reset) {
  ROL_TEST_FOR_EXCEPTION(isFinalized_,std::invalid_argument,
    ">>> ROL::Problem: Cannot add constraint after problem is finalized!");

  if (reset) INPUT_proj_.clear();

  auto it = INPUT_con_.find(name);
  ROL_TEST_FOR_EXCEPTION(it != INPUT_con_.end(),std::invalid_argument,
    ">>> ROL::Problem: Constraint names must be distinct!");
  it = INPUT_linear_con_.find(name);
  ROL_TEST_FOR_EXCEPTION(it != INPUT_linear_con_.end(),std::invalid_argument,
    ">>> ROL::Problem: Constraint names must be distinct!");
  auto itp = INPUT_proj_.find(name);
  ROL_TEST_FOR_EXCEPTION(itp != INPUT_proj_.end(),std::invalid_argument,
    ">>> ROL::Problem: Constraint names must be distinct!");

  if (proj == nullPtr) {
    INPUT_con_.insert({name,ConstraintData<Real>(pcon,pmul,pres)});
    hasEquality_ = true;
    cnt_econ_++;
  }
  else {
    INPUT_proj_.insert({name,{ConstraintData<Real>(pcon,pmul,pres),proj}});
  }
}

template<typename Real>
void Problem<Real>::removeConstraint(std::string name) {
  ROL_TEST_FOR_EXCEPTION(isFinalized_,std::invalid_argument,
    ">>> ROL::Problem: Cannot remove constraint after problem is finalized!");

  auto it = INPUT_con_.find(name);
  if (it!=INPUT_con_.end()) {
    if (it->second.bounds==nullPtr) cnt_econ_--;
    else                            cnt_icon_--;
    INPUT_con_.erase(it);
  }
  if (cnt_econ_==0) hasEquality_   = false;
  if (cnt_icon_==0) hasInequality_ = false;
  auto itp = INPUT_proj_.find(name);
  if (itp!=INPUT_proj_.end()) {
    INPUT_proj_.erase(itp);
  }
}

template<typename Real>
void Problem<Real>::addLinearConstraint( std::string                  name,
                                         const Ptr<Constraint<Real>> &linear_econ,
                                         const Ptr<Vector<Real>>     &linear_emul,
                                         const Ptr<Vector<Real>>     &linear_eres,
                                         bool                         reset) {
  ROL_TEST_FOR_EXCEPTION(isFinalized_,std::invalid_argument,
    ">>> ROL::Problem: Cannot add linear constraint after problem is finalized!");

  if (reset) INPUT_linear_con_.clear();

  auto it = INPUT_con_.find(name);
  ROL_TEST_FOR_EXCEPTION(it != INPUT_con_.end(),std::invalid_argument,
    ">>> ROL::Problem: Constraint names must be distinct!");
  it = INPUT_linear_con_.find(name);
  ROL_TEST_FOR_EXCEPTION(it != INPUT_linear_con_.end(),std::invalid_argument,
    ">>> ROL::Problem: Constraint names must be distinct!");
  auto itp = INPUT_proj_.find(name);
  ROL_TEST_FOR_EXCEPTION(itp != INPUT_proj_.end(),std::invalid_argument,
    ">>> ROL::Problem: Constraint names must be distinct!");

  INPUT_linear_con_.insert({name,ConstraintData<Real>(linear_econ,linear_emul,linear_eres)});
  hasLinearEquality_ = true;
  cnt_linear_econ_++;
}

template<typename Real>
void Problem<Real>::addLinearConstraint( std::string                       name,
                                         const Ptr<Constraint<Real>>      &linear_icon,
                                         const Ptr<Vector<Real>>          &linear_imul,
                                         const Ptr<BoundConstraint<Real>> &linear_ibnd,
                                         const Ptr<Vector<Real>>          &linear_ires,
                                         bool                              reset) {
  ROL_TEST_FOR_EXCEPTION(isFinalized_,std::invalid_argument,
    ">>> ROL::Problem: Cannot add linear constraint after problem is finalized!");

  if (reset) INPUT_linear_con_.clear();

  auto it = INPUT_con_.find(name);
  ROL_TEST_FOR_EXCEPTION(it != INPUT_con_.end(),std::invalid_argument,
    ">>> ROL::Problem: Constraint names must be distinct!");
  it = INPUT_linear_con_.find(name);
  ROL_TEST_FOR_EXCEPTION(it != INPUT_linear_con_.end(),std::invalid_argument,
    ">>> ROL::Problem: Constraint names must be distinct!");
  auto itp = INPUT_proj_.find(name);
  ROL_TEST_FOR_EXCEPTION(itp != INPUT_proj_.end(),std::invalid_argument,
    ">>> ROL::Problem: Constraint names must be distinct!");

  INPUT_linear_con_.insert({name,ConstraintData<Real>(linear_icon,linear_imul,linear_ires,linear_ibnd)});
  hasLinearInequality_ = true;
  cnt_linear_icon_++;
}

template<typename Real>
void Problem<Real>::removeLinearConstraint(std::string name) {
  ROL_TEST_FOR_EXCEPTION(isFinalized_,std::invalid_argument,
    ">>> ROL::Problem: Cannot remove linear inequality after problem is finalized!");

  auto it = INPUT_linear_con_.find(name);
  if (it!=INPUT_linear_con_.end()) {
    if (it->second.bounds==nullPtr) cnt_linear_econ_--;
    else                            cnt_linear_icon_--;
    INPUT_linear_con_.erase(it);
  }
  if (cnt_linear_econ_==0) hasLinearEquality_ = false;
  if (cnt_linear_icon_==0) hasLinearInequality_ = false;
}

template<typename Real>
void Problem<Real>::setProjectionAlgorithm(ParameterList &list) {
  ROL_TEST_FOR_EXCEPTION(isFinalized_,std::invalid_argument,
    ">>> ROL::Problem: Cannot set polyhedral projection algorithm after problem is finalized!");

  ppa_list_ = list;
}

template<typename Real>
void Problem<Real>::addProximableObjective(const Ptr<Objective<Real>> &nobj) {
  ROL_TEST_FOR_EXCEPTION(isFinalized_,std::invalid_argument,
    ">>> ROL::Problem: Cannot add regularizer after problem is finalized!");

  INPUT_nobj_ = nobj;
  hasProximableObjective_ = true;
}

template<typename Real>
void Problem<Real>::removeProximableObjective() {
  ROL_TEST_FOR_EXCEPTION(isFinalized_,std::invalid_argument,
    ">>> ROL::Problem: Cannot remove regularizer after problem is finalized!");

  INPUT_nobj_ = nullPtr;
  hasProximableObjective_ = false;
}

template<typename Real>
void Problem<Real>::addAugmentedLagrangianGroup(std::string                     name,
                                                const std::vector<std::string> &con_names) {
  ROL_TEST_FOR_EXCEPTION(isFinalized_,std::invalid_argument,
    ">>> ROL::Problem: Cannot add augmented Lagrangian group after problem is finalized!");
  bool isFound;
  for (const auto& str : con_names) {
    ROL_TEST_FOR_EXCEPTION(al_constraints_.count(str),std::invalid_argument,
      ">>> ROL::Problem: Cannot include the same constraint in two augmented Lagrangian groups!");
    isFound = INPUT_con_.count(str) + INPUT_linear_con_.count(str) + INPUT_proj_.count(str);
    ROL_TEST_FOR_EXCEPTION(!isFound,std::invalid_argument,
      ">>> ROL::Problem: Group defined with nonexistent constraint name!");
  }
  for (const auto& str : con_names) {
    al_constraints_.insert(str);
    al_groups_.insert({name,str});
  }
}

template<typename Real>
void Problem<Real>::removeAugmentedLagrangianGroup(std::string name) {
  ROL_TEST_FOR_EXCEPTION(isFinalized_,std::invalid_argument,
    ">>> ROL::Problem: Cannot remove augmented Lagrangian group after problem is finalized!");
  auto it = al_groups_.find(name);
  if (it!=al_groups_.end()) {
    const std::vector<std::string>& constraint_names = it->second;
    for (const auto& str : constraint_names) {
      al_constraints_.erase(str);
    }
    al_groups_.erase(it);
  }
}

template<typename Real>
Ptr<Problem<Real>> Problem<Real>::getAugmentedLagrangianSubproblem() {

  ROL_TEST_FOR_EXCEPTION(isFinalized_,std::invalid_argument,
    ">>> ROL::Problem: Cannot build augmented Lagrangian subproblem after problem is finalized!");

  std::unordered_set<std::string> unprocessed_constraints;
  for (const auto& kv : INPUT_proj_)       unprocessed_constraints.insert(kv.first);
  for (const auto& kv : INPUT_con_)        unprocessed_constraints.insert(kv.first);
  for (const auto& kv : INPUT_linear_con_) unprocessed_constraints.insert(kv.first);

  Ptr<Projection<Real>> projection;

  auto makePenalty = [&] (const ConstraintData<Real>  &constraint_data,
                          const Ptr<Projection<Real>> &projection)
                         -> Ptr<AugmentedLagrangianPenalty<Real>> {
    Ptr<AugmentedLagrangianPenalty<Real>> penalty = makePtr<AugmentedLagrangianPenalty<Real>>(
                                                      constraint_data.constraint,
                                                      projection,
                                                      Real(1.0),
                                                      *INPUT_xdual_,
                                                      *constraint_data.residual,
                                                      constraint_data.residual->dual(),
                                                      0);
    penalty->setMultiplier(*constraint_data.multiplier);
    return penalty;
  };

  Ptr<AugmentedLagrangianPenalty<Real>> penalty;
  std::vector<Ptr<AugmentedLagrangianPenalty<Real>>> penalties;

  // ========================================================================
  // STEP 1: Process user-defined groups.
  // ========================================================================

  std::vector<Ptr<Constraint<Real>>> constraints;
  std::vector<Ptr<Projection<Real>>> projections;
  std::vector<Ptr<Vector<Real>>>     con_vectors;
  std::vector<Ptr<Vector<Real>>>     mul_vectors;

  Ptr<Constraint_Partitioned<Real>> partitioned_constraint;
  Ptr<Projection_Partitioned<Real>> partitioned_projection;
  Ptr<PartitionedVector<Real>>      partitioned_con_vector;
  Ptr<PartitionedVector<Real>>      partitioned_mul_vector;

  for (const auto& pair : al_groups_) {
    for (const auto& constraint_name : pair.second) {
      if (INPUT_proj_.count(constraint_name)) {
        std::pair<ConstraintData<Real>,Ptr<Projection<Real>>>& value = INPUT_proj_.at(constraint_name);
        auto& constraint_data = value.first;
        constraints.push_back(constraint_data.constraint);
        con_vectors.push_back(constraint_data.residual);
        mul_vectors.push_back(constraint_data.multiplier);
        projections.push_back(value.second);
        unprocessed_constraints.erase(constraint_name);
      }
      else if (INPUT_con_.count(constraint_name)) {
        auto& constraint_data = INPUT_con_.at(constraint_name);
        constraints.push_back(constraint_data.constraint);
        con_vectors.push_back(constraint_data.residual);
        mul_vectors.push_back(constraint_data.multiplier);
        if (constraint_data.bounds != nullPtr)
          projection = staticPtrCast<Projection<Real>>(makePtr<PolyhedralProjection<Real>>(constraint_data.bounds));
        else 
          projection = staticPtrCast<Projection<Real>>(makePtr<ZeroProjection<Real>>());
        projections.push_back(projection);
        unprocessed_constraints.erase(constraint_name);
      }
      else if (INPUT_linear_con_.count(constraint_name)) {
        auto& constraint_data = INPUT_linear_con_.at(constraint_name);
        constraints.push_back(constraint_data.constraint);
        con_vectors.push_back(constraint_data.residual);
        mul_vectors.push_back(constraint_data.multiplier);
        if (constraint_data.bounds != nullPtr)
          projection = makePtr<PolyhedralProjection<Real>>(constraint_data.bounds);
        else
          projection = makePtr<ZeroProjection<Real>>();
        projections.push_back(projection);
        unprocessed_constraints.erase(constraint_name);
      }
    }

    partitioned_constraint = makePtr<Constraint_Partitioned<Real>>(constraints);
    partitioned_projection = makePtr<Projection_Partitioned<Real>>(projections);
    partitioned_con_vector = makePtr<PartitionedVector<Real>>(con_vectors);
    partitioned_mul_vector = makePtr<PartitionedVector<Real>>(mul_vectors);
    ConstraintData<Real> constraint_data(partitioned_constraint,
                                         partitioned_mul_vector,
                                         partitioned_con_vector);
    penalty = makePenalty(constraint_data,partitioned_projection);
    penalties.push_back(penalty);
  }

  // ========================================================================
  // STEP 2: Process remaining constraints.
  // ========================================================================

  Ptr<Problem<Real>> subproblem = makePtr<Problem<Real>>(INPUT_obj_,INPUT_xprim_,INPUT_xdual_);
  bool isSubproblemTypeE = false;

  for (const auto& constraint_name : unprocessed_constraints) {
    if (INPUT_proj_.count(constraint_name)) {
      std::pair<ConstraintData<Real>,Ptr<Projection<Real>>>& value = INPUT_proj_.at(constraint_name);
      penalty = makePenalty(value.first,value.second);
      penalties.push_back(penalty);
    }
    else if (INPUT_con_.count(constraint_name)) {
      auto& constraint_data = INPUT_con_.at(constraint_name);
      if (constraint_data.bounds != nullPtr) {
        projection = staticPtrCast<Projection<Real>>(makePtr<PolyhedralProjection<Real>>(constraint_data.bounds));
        penalty = makePenalty(constraint_data,projection);
        penalties.push_back(penalty);
      }
      else {
        subproblem->addConstraint(constraint_name,
                                  constraint_data.constraint,
                                  constraint_data.multiplier,
                                  constraint_data.residual);
        isSubproblemTypeE = true;
      }
    }
    else if (INPUT_linear_con_.count(constraint_name)) {
      auto& constraint_data = INPUT_linear_con_.at(constraint_name);
      if (constraint_data.bounds != nullPtr) {
        projection = makePtr<PolyhedralProjection<Real>>(constraint_data.bounds);
        penalty = makePenalty(constraint_data,projection);
        penalties.push_back(penalty);
      }
      else {
        subproblem->addConstraint(constraint_name,
                                  constraint_data.constraint,
                                  constraint_data.multiplier,
                                  constraint_data.residual);
      }
    }
  }

  // ========================================================================
  // STEP 3: Process the bound constraint.
  // ========================================================================

  if (hasBounds_) {
    if (isSubproblemTypeE) {
      projection = staticPtrCast<Projection<Real>>(makePtr<PolyhedralProjection<Real>>(INPUT_bnd_));
      Ptr<Vector<Real>> b = INPUT_xprim_->clone();
      b->zero();
      Ptr<LinearOperator<Real>> A = makePtr<IdentityOperator<Real>>();
      Ptr<Constraint<Real>> constraint = makePtr<LinearConstraint<Real>>(A,b);
      ConstraintData<Real> constraint_data(constraint,INPUT_xdual_,INPUT_xprim_);
      penalty = makePenalty(constraint_data,projection);
      penalties.push_back(penalty);
    }
    else subproblem->addBoundConstraint(INPUT_bnd_);
  }

  subproblem->INPUT_obj_ = makePtr<AugmentedLagrangianObjective2<Real>>(subproblem->INPUT_obj_,penalties,*INPUT_xdual_,false);

  return subproblem;
}

template<typename Real>
void Problem<Real>::finalize(bool lumpConstraints, bool printToStream, std::ostream &outStream) {
  if (!isFinalized_) {
    std::unordered_map<std::string,ConstraintData<Real>> con, lcon, icon;
    bool hasEquality            = hasEquality_;
    bool hasLinearEquality      = hasLinearEquality_;
    bool hasInequality          = hasInequality_;
    bool hasLinearInequality    = hasLinearInequality_;
    bool hasProximableObjective = hasProximableObjective_;
    con.insert(INPUT_con_.begin(),INPUT_con_.end());
    if (lumpConstraints) {
      con.insert(INPUT_linear_con_.begin(),INPUT_linear_con_.end());
      hasEquality = (hasEquality || hasLinearEquality);
      hasInequality = (hasInequality || hasLinearInequality);
      hasLinearEquality = false;
      hasLinearInequality = false;
    }
    else {
      lcon.insert(INPUT_linear_con_.begin(),INPUT_linear_con_.end());
    }
    // Transform optimization problem
    //std::cout << hasBounds_ << "  " << hasEquality << "  " << hasInequality << "  " << hasLinearEquality << "  " << hasLinearInequality << std::endl;
    nobj_            = nullPtr;
    if (hasProximableObjective){
      if (!hasEquality && !hasInequality && !hasBounds_ && !hasLinearEquality && !hasLinearInequality){
        problemType_ = TYPE_P;
        obj_         = INPUT_obj_;
        nobj_        = INPUT_nobj_;
        xprim_       = INPUT_xprim_;
        xdual_       = INPUT_xdual_;
        bnd_         = nullPtr;
        con_         = nullPtr;
        mul_         = nullPtr;
        res_         = nullPtr;
     }
     else {
       throw Exception::NotImplemented(">>> ROL::TypeP - with constraints is not supported");
     }
    }
    else {
      if (!hasLinearEquality && !hasLinearInequality) {
        proj_ = nullPtr;
        if (!hasEquality && !hasInequality && !hasBounds_ ) {
          problemType_ = TYPE_U;
          obj_         = INPUT_obj_;
          xprim_       = INPUT_xprim_;
          xdual_       = INPUT_xdual_;
          bnd_         = nullPtr;
          con_         = nullPtr;
          mul_         = nullPtr;
          res_         = nullPtr;
        }
        else if (!hasEquality && !hasInequality && hasBounds_) {
          problemType_ = TYPE_B;
          obj_         = INPUT_obj_;
          xprim_       = INPUT_xprim_;
          xdual_       = INPUT_xdual_;
          bnd_         = INPUT_bnd_;
          con_         = nullPtr;
          mul_         = nullPtr;
          res_         = nullPtr;
        }
        else if (hasEquality && !hasInequality && !hasBounds_) {
          ConstraintAssembler<Real> cm(con,INPUT_xprim_,INPUT_xdual_);
          problemType_ = TYPE_E;
          obj_         = INPUT_obj_;
          xprim_       = INPUT_xprim_;
          xdual_       = INPUT_xdual_;
          bnd_         = nullPtr;
          con_         = cm.getConstraint();
          mul_         = cm.getMultiplier();
          res_         = cm.getResidual();
        }
        else {
          ConstraintAssembler<Real> cm(con,INPUT_xprim_,INPUT_xdual_,INPUT_bnd_);
          problemType_ = TYPE_EB;
          obj_         = INPUT_obj_;
          if (cm.hasInequality()) {
            obj_      = makePtr<SlacklessObjective<Real>>(INPUT_obj_);
          }
          xprim_       = cm.getOptVector();
          xdual_       = cm.getDualOptVector();
          bnd_         = cm.getBoundConstraint();
          con_         = cm.getConstraint();
          mul_         = cm.getMultiplier();
          res_         = cm.getResidual();
        }
      }
      else {
        if (!hasBounds_ && !hasLinearInequality) {
          ConstraintAssembler<Real> cm(lcon,INPUT_xprim_,INPUT_xdual_);
          xfeas_ = cm.getOptVector()->clone(); xfeas_->set(*cm.getOptVector());
          rlc_   = makePtr<ReduceLinearConstraint<Real>>(cm.getConstraint(),xfeas_,cm.getResidual());
          proj_  = nullPtr;
          if (!hasEquality && !hasInequality) {
            problemType_ = TYPE_U;
            obj_         = rlc_->transform(INPUT_obj_);
            xprim_       = xfeas_->clone(); xprim_->zero();
            xdual_       = cm.getDualOptVector();
            bnd_         = nullPtr;
            con_         = nullPtr;
            mul_         = nullPtr;
            res_         = nullPtr;
          }
          else {
            for (auto it = con.begin(); it != con.end(); ++it) {
              icon.insert(std::pair<std::string,ConstraintData<Real>>(it->first,
                ConstraintData<Real>(rlc_->transform(it->second.constraint),
                  it->second.multiplier,it->second.residual,it->second.bounds)));
            }
            Ptr<Vector<Real>> xtmp = xfeas_->clone(); xtmp->zero();
            ConstraintAssembler<Real> cm1(icon,xtmp,cm.getDualOptVector());
            xprim_         = cm1.getOptVector();
            xdual_         = cm1.getDualOptVector();
            con_           = cm1.getConstraint();
            mul_           = cm1.getMultiplier();
            res_           = cm1.getResidual();
            if (!hasInequality) {
              problemType_ = TYPE_E;
              obj_         = rlc_->transform(INPUT_obj_);
              bnd_         = nullPtr;
            }
            else {
              problemType_ = TYPE_EB;
              obj_         = makePtr<SlacklessObjective<Real>>(rlc_->transform(INPUT_obj_));
              bnd_         = cm1.getBoundConstraint();
            }
          }
        }
        else if ((hasBounds_ || hasLinearInequality) && !hasEquality && !hasInequality) {
          ConstraintAssembler<Real> cm(lcon,INPUT_xprim_,INPUT_xdual_,INPUT_bnd_);
          problemType_ = TYPE_B;
          obj_         = INPUT_obj_;
          if (cm.hasInequality()) {
            obj_       = makePtr<SlacklessObjective<Real>>(INPUT_obj_);
          }
          xprim_       = cm.getOptVector();
          xdual_       = cm.getDualOptVector();
          bnd_         = cm.getBoundConstraint();
          con_         = nullPtr;
          mul_         = nullPtr;
          res_         = nullPtr;
          proj_        = PolyhedralProjectionFactory<Real>(*xprim_,*xdual_,bnd_,
                           cm.getConstraint(),*cm.getMultiplier(),*cm.getResidual(),ppa_list_);
        }
        else {
          ConstraintAssembler<Real> cm(con,lcon,INPUT_xprim_,INPUT_xdual_,INPUT_bnd_);
          problemType_ = TYPE_EB;
          obj_         = INPUT_obj_;
          if (cm.hasInequality()) {
            obj_       = makePtr<SlacklessObjective<Real>>(INPUT_obj_);
          }
          xprim_       = cm.getOptVector();
          xdual_       = cm.getDualOptVector();
          con_         = cm.getConstraint();
          mul_         = cm.getMultiplier();
          res_         = cm.getResidual();
          bnd_         = cm.getBoundConstraint();
          proj_        = PolyhedralProjectionFactory<Real>(*xprim_,*xdual_,bnd_,
                          cm.getLinearConstraint(),*cm.getLinearMultiplier(),
                          *cm.getLinearResidual(),ppa_list_);
        }
      }
    }
    isFinalized_ = true;
    if (printToStream) {
      outStream << std::endl;
      outStream << "  ROL::Problem::finalize" << std::endl;
      outStream << "    Problem Summary:" << std::endl;
      outStream << "      Has Proximable Objective? .......... " << (hasProximableObjective ? "yes" : "no") << std::endl;
      outStream << "      Has Bound Constraint? .............. " << (hasBounds_ ? "yes" : "no") << std::endl;
      outStream << "      Has Equality Constraint? ........... " << (hasEquality ? "yes" : "no") << std::endl;
      if (hasEquality) {
        int cnt = 0;
	for (auto it = con.begin(); it != con.end(); ++it) {
          if (it->second.bounds==nullPtr) {
            if (cnt==0) {
              outStream << "        Names: ........................... ";
	      cnt++;
	    }
	    else {
              outStream << "                                           ";
	    }
            outStream << it->first << std::endl;
	  }
	}
        outStream << "        Total: ........................... " << cnt_econ_+(lumpConstraints ? cnt_linear_econ_ : 0) << std::endl;
      }
      outStream << "      Has Inequality Constraint? ......... " << (hasInequality ? "yes" : "no") << std::endl;
      if (hasInequality) {
        int cnt = 0;
	for (auto it = con.begin(); it != con.end(); ++it) {
          if (it->second.bounds!=nullPtr) {
            if (cnt==0) {
              outStream << "        Names: ........................... ";
	      cnt++;
	    }
	    else {
              outStream << "                                           ";
	    }
            outStream << it->first << std::endl;
	  }
	}
        outStream << "        Total: ........................... " << cnt_icon_+(lumpConstraints ? cnt_linear_icon_ : 0) << std::endl;
      }
      if (!lumpConstraints) {
        outStream << "      Has Linear Equality Constraint? .... " << (hasLinearEquality ? "yes" : "no") << std::endl;
        if (hasLinearEquality) {
          int cnt = 0;
	  for (auto it = lcon.begin(); it != lcon.end(); ++it) {
            if (it->second.bounds==nullPtr) {
              if (cnt==0) {
                outStream << "        Names: ........................... ";
		cnt++;
	      }
	      else {
                outStream << "                                           ";
	      }
              outStream << it->first << std::endl;
	    }
	  }
          outStream << "        Total: ........................... " << cnt_linear_econ_ << std::endl;
        }
        outStream << "      Has Linear Inequality Constraint? .. " << (hasLinearInequality ? "yes" : "no") << std::endl;
        if (hasLinearInequality) {
          int cnt = 0;
	  for (auto it = lcon.begin(); it != lcon.end(); ++it) {
            if (it->second.bounds!=nullPtr) {
              if (cnt==0) {
                outStream << "        Names: ........................... ";
		cnt++;
	      }
	      else {
                outStream << "                                           ";
	      }
              outStream << it->first << std::endl;
	    }
	  }
          outStream << "        Total: ........................... " << cnt_linear_icon_ << std::endl;
        }
      }
      outStream << std::endl;
    }
  }
  else {
    if (printToStream) {
      outStream << std::endl;
      outStream << "  ROL::Problem::finalize" << std::endl;
      outStream << "    Problem already finalized!" << std::endl;
      outStream << std::endl;
    }
  }
}

template<typename Real>
const Ptr<Objective<Real>>& Problem<Real>::getObjective() {
  finalize();
  return obj_;
}

template<typename Real>
const Ptr<Objective<Real>>& Problem<Real>::getProximableObjective(){
  finalize();
  return nobj_;
}

template<typename Real>
const Ptr<Vector<Real>>& Problem<Real>::getPrimalOptimizationVector() {
  finalize();
  return xprim_;
}

template<typename Real>
const Ptr<Vector<Real>>& Problem<Real>::getDualOptimizationVector() {
  finalize();
  return xdual_;
}

template<typename Real>
const Ptr<BoundConstraint<Real>>& Problem<Real>::getBoundConstraint() {
  finalize();
  return bnd_;
}

template<typename Real>
const Ptr<Constraint<Real>>& Problem<Real>::getConstraint() {
  finalize();
  return con_;
}

template<typename Real>
const Ptr<Vector<Real>>& Problem<Real>::getMultiplierVector() {
  finalize();
  return mul_;
}

template<typename Real>
const Ptr<Vector<Real>>& Problem<Real>::getResidualVector() {
  finalize();
  return res_;
}

template<typename Real>
const Ptr<PolyhedralProjection<Real>>& Problem<Real>::getPolyhedralProjection() {
  finalize();
  return proj_;
}

template<typename Real>
EProblem Problem<Real>::getProblemType() {
  finalize();
  return problemType_;
}

template<typename Real>
Real Problem<Real>::checkLinearity(bool printToStream, std::ostream &outStream) const {
  std::ios_base::fmtflags state(outStream.flags());
  if (printToStream) {
    outStream << std::setprecision(8) << std::scientific;
    outStream << std::endl;
    outStream << "  ROL::Problem::checkLinearity" << std::endl;
  }
  const Real one(1), two(2), eps(1e-2*std::sqrt(ROL_EPSILON<Real>()));
  Real tol(std::sqrt(ROL_EPSILON<Real>())), cnorm(0), err(0), maxerr(0);
  Ptr<Vector<Real>> x  = INPUT_xprim_->clone(); x->randomize(-one,one);
  Ptr<Vector<Real>> y  = INPUT_xprim_->clone(); y->randomize(-one,one);
  Ptr<Vector<Real>> z  = INPUT_xprim_->clone(); z->zero();
  Ptr<Vector<Real>> xy = INPUT_xprim_->clone();
  Real alpha = two*static_cast<Real>(rand())/static_cast<Real>(RAND_MAX)-one;
  xy->set(*x); xy->axpy(alpha,*y);
  Ptr<Vector<Real>> c1, c2;
  for (auto it = INPUT_linear_con_.begin(); it != INPUT_linear_con_.end(); ++it) {
    c1 = it->second.residual->clone();
    c2 = it->second.residual->clone();
    it->second.constraint->update(*xy,UpdateType::Temp);
    it->second.constraint->value(*c1,*xy,tol);
    cnorm = c1->norm();
    it->second.constraint->update(*x,UpdateType::Temp);
    it->second.constraint->value(*c2,*x,tol);
    c1->axpy(-one,*c2);
    it->second.constraint->update(*y,UpdateType::Temp);
    it->second.constraint->value(*c2,*y,tol);
    c1->axpy(-alpha,*c2);
    it->second.constraint->update(*z,UpdateType::Temp);
    it->second.constraint->value(*c2,*z,tol);
    c1->axpy(alpha,*c2);
    err = c1->norm();
    maxerr = std::max(err,maxerr);
    if (printToStream) {
      outStream << "    Constraint " << it->first;
      outStream << ":  ||c(x+alpha*y) - (c(x)+alpha*(c(y)-c(0)))|| = " << err << std::endl;
      if (err > eps*cnorm) {
        outStream << "      Constraint " << it->first << " may not be linear!" << std::endl;
      }
    }
  }
  if (printToStream) {
    outStream << std::endl;
  }
  outStream.flags(state);
  return maxerr;
}

template<typename Real>
void Problem<Real>::checkVectors(bool printToStream, std::ostream &outStream) const {
  const Real one(1);
  Ptr<Vector<Real>> x, y;
  // Primal optimization space vector
  x = INPUT_xprim_->clone(); x->randomize(-one,one);
  y = INPUT_xprim_->clone(); y->randomize(-one,one);
  if (printToStream) {
    outStream << std::endl << "  Check primal optimization space vector" << std::endl;
  }
  INPUT_xprim_->checkVector(*x,*y,printToStream,outStream);

  // Dual optimization space vector
  x = INPUT_xdual_->clone(); x->randomize(-one,one);
  y = INPUT_xdual_->clone(); y->randomize(-one,one);
  if (printToStream) {
    outStream << std::endl << "  Check dual optimization space vector" << std::endl;
  }
  INPUT_xdual_->checkVector(*x,*y,printToStream,outStream);

  // Check constraint space vectors
  for (auto it = INPUT_con_.begin(); it != INPUT_con_.end(); ++it) {
    // Primal constraint space vector
    x = it->second.residual->clone(); x->randomize(-one,one);
    y = it->second.residual->clone(); y->randomize(-one,one);
    if (printToStream) {
      outStream << std::endl << "  " << it->first << ": Check primal constraint space vector" << std::endl;
    }
    it->second.residual->checkVector(*x,*y,printToStream,outStream);

    // Dual optimization space vector
    x = it->second.multiplier->clone(); x->randomize(-one,one);
    y = it->second.multiplier->clone(); y->randomize(-one,one);
    if (printToStream) {
      outStream << std::endl << "  " << it->first << ": Check dual constraint space vector" << std::endl;
    }
    it->second.multiplier->checkVector(*x,*y,printToStream,outStream);
  }

  // Check constraint space vectors
  for (auto it = INPUT_linear_con_.begin(); it != INPUT_linear_con_.end(); ++it) {
    // Primal constraint space vector
    x = it->second.residual->clone(); x->randomize(-one,one);
    y = it->second.residual->clone(); y->randomize(-one,one);
    if (printToStream) {
      outStream << std::endl << "  " << it->first << ": Check primal linear constraint space vector" << std::endl;
    }
    it->second.residual->checkVector(*x,*y,printToStream,outStream);

    // Dual optimization space vector
    x = it->second.multiplier->clone(); x->randomize(-one,one);
    y = it->second.multiplier->clone(); y->randomize(-one,one);
    if (printToStream) {
      outStream << std::endl << "  " << it->first << ": Check dual linear constraint space vector" << std::endl;
    }
    it->second.multiplier->checkVector(*x,*y,printToStream,outStream);
  }
}

template<typename Real>
void Problem<Real>::checkDerivatives(bool printToStream, std::ostream &outStream, const Ptr<Vector<Real>> &x0, Real scale) const {
  const Real one(1);
  Ptr<Vector<Real>> x, d, v, g, c, w;
  // Objective check
  x = x0;
  if (x == nullPtr) { x = INPUT_xprim_->clone(); x->randomize(-one,one); }
  d = INPUT_xprim_->clone(); d->randomize(-scale,scale);
  v = INPUT_xprim_->clone(); v->randomize(-scale,scale);
  g = INPUT_xdual_->clone(); g->randomize(-scale,scale);
  if (printToStream)
    outStream << std::endl << "  Check objective function" << std::endl << std::endl;
  INPUT_obj_->checkGradient(*x,*g,*d,printToStream,outStream);
  INPUT_obj_->checkHessVec(*x,*g,*d,printToStream,outStream);
  INPUT_obj_->checkHessSym(*x,*g,*d,*v,printToStream,outStream);
  //TODO: Proximable Objective Check
  // Constraint check
  for (auto it = INPUT_con_.begin(); it != INPUT_con_.end(); ++it) {
    c = it->second.residual->clone();   c->randomize(-scale,scale);
    w = it->second.multiplier->clone(); w->randomize(-scale,scale);
    if (printToStream)
      outStream << std::endl << "  " << it->first << ": Check constraint function" << std::endl << std::endl;
    it->second.constraint->checkApplyJacobian(*x,*v,*c,printToStream,outStream);
    it->second.constraint->checkAdjointConsistencyJacobian(*w,*v,*x,printToStream,outStream);
    it->second.constraint->checkApplyAdjointHessian(*x,*w,*v,*g,printToStream,outStream);
  }

  // Linear constraint check
  for (auto it = INPUT_linear_con_.begin(); it != INPUT_linear_con_.end(); ++it) {
    c = it->second.residual->clone();   c->randomize(-scale,scale);
    w = it->second.multiplier->clone(); w->randomize(-scale,scale);
    if (printToStream)
      outStream << std::endl << "  " << it->first << ": Check constraint function" << std::endl << std::endl;
    it->second.constraint->checkApplyJacobian(*x,*v,*c,printToStream,outStream);
    it->second.constraint->checkAdjointConsistencyJacobian(*w,*v,*x,printToStream,outStream);
    it->second.constraint->checkApplyAdjointHessian(*x,*w,*v,*g,printToStream,outStream);
  }
}

template<typename Real>
void Problem<Real>::check(bool printToStream, std::ostream &outStream, const Ptr<Vector<Real>> &x0, Real scale) const {
  checkVectors(printToStream,outStream);
  if (hasLinearEquality_ || hasLinearInequality_)
    checkLinearity(printToStream,outStream);
  checkDerivatives(printToStream,outStream,x0,scale);
// if (hasProximableObjective)
// checkProximableObjective(printToStream, outStream);
}

template<typename Real>
bool Problem<Real>::isFinalized() const {
  return isFinalized_;
}

template<typename Real>
void Problem<Real>::edit() {
  isFinalized_ = false;
  rlc_  = nullPtr;
  proj_ = nullPtr;
}

template<typename Real>
void Problem<Real>::finalizeIteration() {
  if (rlc_ != nullPtr) {
    if (!hasInequality_) {
      rlc_->project(*INPUT_xprim_,*xprim_);
      INPUT_xprim_->plus(*rlc_->getFeasibleVector());
    }
    else {
      Ptr<Vector<Real>> xprim = dynamic_cast<PartitionedVector<Real>&>(*xprim_).get(0)->clone();
      xprim->set(*dynamic_cast<PartitionedVector<Real>&>(*xprim_).get(0));
      rlc_->project(*INPUT_xprim_,*xprim);
      INPUT_xprim_->plus(*rlc_->getFeasibleVector());
    }
  }
}

}  // namespace ROL

#endif // ROL_NEWOPTIMIZATIONPROBLEM_DEF_HPP
