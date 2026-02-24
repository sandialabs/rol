// @HEADER
// *****************************************************************************
//               Rapid Optimization Library (ROL) Package
//
// Copyright 2014 NTESS and the ROL contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef ROL_AUGMENTEDLAGRANGIANPENALTY_H
#define ROL_AUGMENTEDLAGRANGIANPENALTY_H

namespace ROL {

template <class Real>
class AugmentedLagrangianPenalty : public Objective<Real> {
private:
  // Required for Augmented Lagrangian definition
  const Ptr<Constraint<Real>> con_;
  const Ptr<Projection<Real>> proj_;
  nullstream bhs_;

  Real penaltyParameter_;
  Ptr<Vector<Real>> multiplier_;

  // Auxiliary storage
  Ptr<Vector<Real>> dualOptVector_;
  Ptr<Vector<Real>> dualConVector_;
  Ptr<Vector<Real>> primConVector_;

  // Constraint evaluations
  Ptr<VectorController<Real,int>> conValue_;
  Ptr<VectorController<Real,int>> dualValue_;

  // Constraint scaling
  Real cscale_;

  // Evaluation counter
  int ncval_;

  // User defined options
  int hessianApprox_;

public:

  AugmentedLagrangianPenalty(const Ptr<Constraint<Real>> &con,
                             const Ptr<Projection<Real>> &proj,
                             const Real penaltyParameter,
                             const Vector<Real> &dualOptVec,
                             const Vector<Real> &primConVec,
                             const Vector<Real> &dualConVec,
                             ParameterList &parlist)
    : con_(con), proj_(proj), penaltyParameter_(penaltyParameter),
      cscale_(1), ncval_(0) {

    conValue_ = makePtr<VectorController<Real,int>>();
    dualValue_ = makePtr<VectorController<Real,int>>();

    multiplier_    = dualConVec.clone();
    multiplier_->zero();
    dualOptVector_ = dualOptVec.clone();
    dualConVector_ = dualConVec.clone();
    primConVector_ = primConVec.clone();

    ParameterList& sublist = parlist.sublist("Step").sublist("Augmented Lagrangian");
    hessianApprox_ = sublist.get("Level of Hessian Approximation",  0);
  }

  AugmentedLagrangianPenalty(const Ptr<Constraint<Real>> &con,
                             const Ptr<Projection<Real>> &proj,
                             const Real penaltyParameter,
                             const Vector<Real> &dualOptVec,
                             const Vector<Real> &primConVec,
                             const Vector<Real> &dualConVec,
                             const int hessianApprox)
    : con_(con), proj_(proj), penaltyParameter_(penaltyParameter),
      cscale_(1), ncval_(0), hessianApprox_(hessianApprox) {

    conValue_ = makePtr<VectorController<Real,int>>();
    dualValue_ = makePtr<VectorController<Real,int>>();

    multiplier_    = dualConVec.clone();
    multiplier_->zero();
    dualOptVector_ = dualOptVec.clone();
    dualConVector_ = dualConVec.clone();
    primConVector_ = primConVec.clone();
  }

  virtual void update( const Vector<Real> &x, UpdateType type, int iter = -1 ) {
    con_->update(x,type,iter);
    conValue_->objectiveUpdate(type);
    dualValue_->objectiveUpdate(type);
  }

  void setScaling(const Real cscale = 1.0) {
    cscale_ = cscale;
  }

  Real getScaling() {
    return cscale_;
  }

  virtual Real value( const Vector<Real> &x, Real &tol ) {
    // Compute penalty function value
    Real val = getDualVec(x,tol)->norm();
    Real temp = multiplier_->norm();
    val = (val*val - temp*temp)/(Real(2)*penaltyParameter_);
    val *= cscale_;
    return val;
  }

  virtual void gradient( Vector<Real> &g, const Vector<Real> &x, Real &tol ) {
    // Compute penalty function gradient
    con_->applyAdjointJacobian(g,*getDualVec(x,tol),x,tol);
    g.scale(cscale_);
  }

  virtual void hessVec( Vector<Real> &hv, const Vector<Real> &v, const Vector<Real> &x, Real &tol ) {
    // Apply penalty Hessian to a vector
    if (hessianApprox_ >= 3) {
      hv.zero();
      return;
    }
    con_->applyAdjointHessian(hv,*getDualVec(x,tol),v,x,tol);
    con_->applyJacobian(*primConVector_,v,x,tol);
    dualConVector_->set(getConstraintVec(x,tol)->dual());
    dualConVector_->axpy(Real(1)/penaltyParameter_,*multiplier_);
    proj_->applyJacobian(*primConVector_,dualConVector_->dual());
    con_->applyAdjointJacobian(*dualOptVector_,primConVector_->dual(),x,tol);
    hv.plus(*dualOptVector_);
    hv.scale(cscale_);
  }

  Real dualNorm( const Vector<Real> &x, Real &tol ) {
    return getDualVec(x,tol)->norm();
  }

  Real dualResidual( const Vector<Real> &x, Real &tol ) {
    dualConVector_->set(*getDualVec(x,tol));
    dualConVector_->axpy(Real(-1),*multiplier_);
    return dualConVector_->norm();
  }

  Real feasibility( const Vector<Real> &x, Real &tol ) {
    primConVector_->set(*getConstraintVec(x,tol));
    proj_->project(*primConVector_,bhs_);
    primConVector_->axpy(Real(-1),*getConstraintVec(x,tol));
    return primConVector_->norm();
  }

  void setPenaltyParameter( const Real penaltyParameter ) {
    penaltyParameter_ = penaltyParameter;
  }

  Real getPenaltyParameter() {
    return penaltyParameter_;
  }

  void setMultiplier( const Vector<Real> &multiplier ) {
    multiplier_->set(multiplier);
  }

  void updateMultiplier( const Vector<Real> &x, Real &tol ) {
    multiplier_->set(*getDualVec(x,tol));
  }

  // Return constraint value
  const Ptr<const Vector<Real>> getConstraintVec( const Vector<Real> &x, Real &tol ) {
    int key(0);
    if (!conValue_->isComputed(key)) {
      if (conValue_->isNull(key)) conValue_->allocate(*primConVector_,key);
      con_->value(*conValue_->set(key),x,tol); ncval_++;
    }
    return conValue_->get(key);
  }

  // Return dual value (proximal point step)
  const Ptr<const Vector<Real>> getDualVec( const Vector<Real> &x, Real &tol ) {
    int key(0);
    if (!dualValue_->isComputed(key)) {
      if (dualValue_->isNull(key)) dualValue_->allocate(*dualConVector_,key);
      primConVector_->set(*getConstraintVec(x,tol));
      primConVector_->axpy(Real(1)/penaltyParameter_,multiplier_->dual());
      dualConVector_->set(primConVector_->dual());
      proj_->project(*primConVector_,bhs_);
      dualConVector_->axpy(Real(-1),primConVector_->dual());
      dualConVector_->scale(penaltyParameter_);
      dualValue_->set(key)->set(*dualConVector_);
    }
    return dualValue_->get(key);
  }

  // Return total number of constraint evaluations
  int getNumberConstraintEvaluations(void) const {
    return ncval_;
  }

  // Reset counter
  void reset(void) {
    ncval_ = 0;
  }

}; // class AugmentedLagrangianPenalty

}  // namespace ROL

#endif
