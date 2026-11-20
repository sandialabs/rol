// @HEADER
// *****************************************************************************
//               Rapid Optimization Library (ROL) Package
//
// Copyright 2014 NTESS and the ROL contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

/*! \file  test_20.cpp
    \brief Test the Hessian approximations of Reduced_Objective_SimOpt.

    With \f$S(z)\f$ the solution of \f$c(S(z),z)=0\f$, the Gauss-Newton
    Hessian keeps the second derivatives of the objective function and drops
    those of the equality constraint,
    \f[
       H_{GN}(z) = S'(z)^* f_{uu} S'(z) + S'(z)^* f_{uz} + f_{zu} S'(z) + f_{zz}.
    \f]
    The test problem is
    \f[
       \min_{u,z}\; \frac{1}{2}\sum_{i<m} (u_i-d_i)^2 + \frac{\alpha}{2}\sum_i z_i^2
                     + \gamma\sum_i u_i z_i
       \quad\text{subject to}\quad
       u_i + \beta(z_i^2 u_i + u_i^3) - z_{i+1} = 0,
    \f]
    where the index of \f$z\f$ is periodic.  All four blocks of the constraint
    Hessian are nonzero for \f$\beta\neq 0\f$, and the objective function has
    mixed second derivatives for \f$\gamma\neq 0\f$.
*/

#include "ROL_StdVector.hpp"
#include "ROL_Objective_SimOpt.hpp"
#include "ROL_Constraint_SimOpt.hpp"
#include "ROL_Reduced_Objective_SimOpt.hpp"
#include "ROL_TypeU_LineSearchAlgorithm.hpp"
#include "ROL_ParameterList.hpp"
#include "ROL_LAPACK.hpp"
#include "ROL_Stream.hpp"
#include "ROL_GlobalMPISession.hpp"

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <cmath>

typedef double RealT;

template<class Real>
class Constraint_Cubic : public ROL::Constraint_SimOpt<Real> {
private:
  const Real beta_;
  int nhess_;

  ROL::Ptr<const std::vector<Real>> getVector(const ROL::Vector<Real> &x) const {
    return dynamic_cast<const ROL::StdVector<Real>&>(x).getVector();
  }

  ROL::Ptr<std::vector<Real>> getVector(ROL::Vector<Real> &x) const {
    return dynamic_cast<ROL::StdVector<Real>&>(x).getVector();
  }

public:
  Constraint_Cubic(const Real beta) : beta_(beta), nhess_(0) {}

  // Number of calls to applyAdjointHessian.
  int getNumberHessianCalls(void) const { return nhess_; }

  void value(ROL::Vector<Real> &c, const ROL::Vector<Real> &u,
             const ROL::Vector<Real> &z, Real &tol) override {
    ROL::Ptr<std::vector<Real>> cp = getVector(c);
    ROL::Ptr<const std::vector<Real>> up = getVector(u), zp = getVector(z);
    const int n = static_cast<int>(up->size());
    for (int i = 0; i < n; ++i) {
      const Real ui = (*up)[i], zi = (*zp)[i];
      (*cp)[i] = ui + beta_*(zi*zi*ui + ui*ui*ui) - (*zp)[(i+1)%n];
    }
  }

  void applyJacobian_1(ROL::Vector<Real> &jv, const ROL::Vector<Real> &v, const ROL::Vector<Real> &u,
                       const ROL::Vector<Real> &z, Real &tol) override {
    const Real one(1), three(3);
    ROL::Ptr<std::vector<Real>> jvp = getVector(jv);
    ROL::Ptr<const std::vector<Real>> vp = getVector(v), up = getVector(u), zp = getVector(z);
    const int n = static_cast<int>(up->size());
    for (int i = 0; i < n; ++i) {
      const Real ui = (*up)[i], zi = (*zp)[i];
      (*jvp)[i] = (one + beta_*(zi*zi + three*ui*ui))*(*vp)[i];
    }
  }

  void applyJacobian_2(ROL::Vector<Real> &jv, const ROL::Vector<Real> &v, const ROL::Vector<Real> &u,
                       const ROL::Vector<Real> &z, Real &tol) override {
    const Real two(2);
    ROL::Ptr<std::vector<Real>> jvp = getVector(jv);
    ROL::Ptr<const std::vector<Real>> vp = getVector(v), up = getVector(u), zp = getVector(z);
    const int n = static_cast<int>(up->size());
    for (int i = 0; i < n; ++i) {
      (*jvp)[i] = two*beta_*(*zp)[i]*(*up)[i]*(*vp)[i] - (*vp)[(i+1)%n];
    }
  }

  void applyInverseJacobian_1(ROL::Vector<Real> &ijv, const ROL::Vector<Real> &v, const ROL::Vector<Real> &u,
                              const ROL::Vector<Real> &z, Real &tol) override {
    const Real one(1), three(3);
    ROL::Ptr<std::vector<Real>> ijvp = getVector(ijv);
    ROL::Ptr<const std::vector<Real>> vp = getVector(v), up = getVector(u), zp = getVector(z);
    const int n = static_cast<int>(up->size());
    for (int i = 0; i < n; ++i) {
      const Real ui = (*up)[i], zi = (*zp)[i];
      (*ijvp)[i] = (*vp)[i]/(one + beta_*(zi*zi + three*ui*ui));
    }
  }

  void applyAdjointJacobian_1(ROL::Vector<Real> &ajv, const ROL::Vector<Real> &v, const ROL::Vector<Real> &u,
                              const ROL::Vector<Real> &z, Real &tol) override {
    applyJacobian_1(ajv,v,u,z,tol);
  }

  void applyAdjointJacobian_2(ROL::Vector<Real> &ajv, const ROL::Vector<Real> &v, const ROL::Vector<Real> &u,
                              const ROL::Vector<Real> &z, Real &tol) override {
    const Real two(2);
    ROL::Ptr<std::vector<Real>> ajvp = getVector(ajv);
    ROL::Ptr<const std::vector<Real>> vp = getVector(v), up = getVector(u), zp = getVector(z);
    const int n = static_cast<int>(up->size());
    for (int i = 0; i < n; ++i) {
      (*ajvp)[i] = two*beta_*(*zp)[i]*(*up)[i]*(*vp)[i] - (*vp)[(i+n-1)%n];
    }
  }

  void applyInverseAdjointJacobian_1(ROL::Vector<Real> &iajv, const ROL::Vector<Real> &v, const ROL::Vector<Real> &u,
                                     const ROL::Vector<Real> &z, Real &tol) override {
    applyInverseJacobian_1(iajv,v,u,z,tol);
  }

  void applyAdjointHessian_11(ROL::Vector<Real> &ahwv, const ROL::Vector<Real> &w, const ROL::Vector<Real> &v,
                              const ROL::Vector<Real> &u, const ROL::Vector<Real> &z, Real &tol) override {
    const Real six(6);
    ROL::Ptr<std::vector<Real>> ahwvp = getVector(ahwv);
    ROL::Ptr<const std::vector<Real>> wp = getVector(w), vp = getVector(v), up = getVector(u);
    const int n = static_cast<int>(up->size());
    for (int i = 0; i < n; ++i) {
      (*ahwvp)[i] = six*beta_*(*up)[i]*(*wp)[i]*(*vp)[i];
    }
    nhess_++;
  }

  void applyAdjointHessian_12(ROL::Vector<Real> &ahwv, const ROL::Vector<Real> &w, const ROL::Vector<Real> &v,
                              const ROL::Vector<Real> &u, const ROL::Vector<Real> &z, Real &tol) override {
    const Real two(2);
    ROL::Ptr<std::vector<Real>> ahwvp = getVector(ahwv);
    ROL::Ptr<const std::vector<Real>> wp = getVector(w), vp = getVector(v), zp = getVector(z);
    const int n = static_cast<int>(zp->size());
    for (int i = 0; i < n; ++i) {
      (*ahwvp)[i] = two*beta_*(*zp)[i]*(*wp)[i]*(*vp)[i];
    }
    nhess_++;
  }

  void applyAdjointHessian_21(ROL::Vector<Real> &ahwv, const ROL::Vector<Real> &w, const ROL::Vector<Real> &v,
                              const ROL::Vector<Real> &u, const ROL::Vector<Real> &z, Real &tol) override {
    applyAdjointHessian_12(ahwv,w,v,u,z,tol);
  }

  void applyAdjointHessian_22(ROL::Vector<Real> &ahwv, const ROL::Vector<Real> &w, const ROL::Vector<Real> &v,
                              const ROL::Vector<Real> &u, const ROL::Vector<Real> &z, Real &tol) override {
    const Real two(2);
    ROL::Ptr<std::vector<Real>> ahwvp = getVector(ahwv);
    ROL::Ptr<const std::vector<Real>> wp = getVector(w), vp = getVector(v), up = getVector(u);
    const int n = static_cast<int>(up->size());
    for (int i = 0; i < n; ++i) {
      (*ahwvp)[i] = two*beta_*(*up)[i]*(*wp)[i]*(*vp)[i];
    }
    nhess_++;
  }
};

template<class Real>
class Objective_Tracking : public ROL::Objective_SimOpt<Real> {
private:
  std::vector<Real> d_;
  const Real alpha_, gamma_;

  ROL::Ptr<const std::vector<Real>> getVector(const ROL::Vector<Real> &x) const {
    return dynamic_cast<const ROL::StdVector<Real>&>(x).getVector();
  }

  ROL::Ptr<std::vector<Real>> getVector(ROL::Vector<Real> &x) const {
    return dynamic_cast<ROL::StdVector<Real>&>(x).getVector();
  }

public:
  // The first d.size() entries of the state are observed.
  Objective_Tracking(const std::vector<Real> &d, const Real alpha, const Real gamma = 0)
    : d_(d), alpha_(alpha), gamma_(gamma) {}

  void setData(const std::vector<Real> &d) { d_ = d; }

  Real value(const ROL::Vector<Real> &u, const ROL::Vector<Real> &z, Real &tol) override {
    const Real half(0.5);
    ROL::Ptr<const std::vector<Real>> up = getVector(u), zp = getVector(z);
    Real val(0);
    for (std::size_t i = 0; i < d_.size(); ++i) {
      val += half*((*up)[i]-d_[i])*((*up)[i]-d_[i]);
    }
    for (std::size_t i = 0; i < zp->size(); ++i) {
      val += half*alpha_*(*zp)[i]*(*zp)[i] + gamma_*(*up)[i]*(*zp)[i];
    }
    return val;
  }

  void gradient_1(ROL::Vector<Real> &g, const ROL::Vector<Real> &u, const ROL::Vector<Real> &z, Real &tol) override {
    ROL::Ptr<std::vector<Real>> gp = getVector(g);
    ROL::Ptr<const std::vector<Real>> up = getVector(u);
    g.set(z);
    g.scale(gamma_);
    for (std::size_t i = 0; i < d_.size(); ++i) {
      (*gp)[i] += (*up)[i]-d_[i];
    }
  }

  void gradient_2(ROL::Vector<Real> &g, const ROL::Vector<Real> &u, const ROL::Vector<Real> &z, Real &tol) override {
    g.set(z);
    g.scale(alpha_);
    g.axpy(gamma_,u);
  }

  void hessVec_11(ROL::Vector<Real> &hv, const ROL::Vector<Real> &v, const ROL::Vector<Real> &u,
                  const ROL::Vector<Real> &z, Real &tol) override {
    ROL::Ptr<std::vector<Real>> hvp = getVector(hv);
    ROL::Ptr<const std::vector<Real>> vp = getVector(v);
    hv.zero();
    for (std::size_t i = 0; i < d_.size(); ++i) {
      (*hvp)[i] = (*vp)[i];
    }
  }

  void hessVec_12(ROL::Vector<Real> &hv, const ROL::Vector<Real> &v, const ROL::Vector<Real> &u,
                  const ROL::Vector<Real> &z, Real &tol) override {
    hv.set(v);
    hv.scale(gamma_);
  }

  void hessVec_21(ROL::Vector<Real> &hv, const ROL::Vector<Real> &v, const ROL::Vector<Real> &u,
                  const ROL::Vector<Real> &z, Real &tol) override {
    hv.set(v);
    hv.scale(gamma_);
  }

  void hessVec_22(ROL::Vector<Real> &hv, const ROL::Vector<Real> &v, const ROL::Vector<Real> &u,
                  const ROL::Vector<Real> &z, Real &tol) override {
    hv.set(v);
    hv.scale(alpha_);
  }
};

// Central difference of the control-to-state map, s = (S(z+hv) - S(z-hv))/(2h).
void differenceState(ROL::Vector<RealT> &s, const ROL::Vector<RealT> &v,
                     const ROL::Vector<RealT> &z, const RealT h,
                     ROL::Constraint_SimOpt<RealT> &con) {
  RealT tol = std::sqrt(ROL::ROL_EPSILON<RealT>()), half(0.5), one(1);
  ROL::Ptr<ROL::Vector<RealT>> znew = z.clone(), unew = s.clone(), c = s.clone();
  znew->set(z); znew->axpy(h,v);
  con.solve(*c,s,*znew,tol);
  znew->set(z); znew->axpy(-h,v);
  con.solve(*c,*unew,*znew,tol);
  s.axpy(-one,*unew);
  s.scale(half/h);
}

// Reduced objective that uses the Gauss-Newton Hessian while the iteration
// number passed to update is below numGN and the Hessian afterwards.
template<class Real>
class Reduced_Objective_Switch : public ROL::Reduced_Objective_SimOpt<Real> {
private:
  const int numGN_;

public:
  Reduced_Objective_Switch(const ROL::Ptr<ROL::Objective_SimOpt<Real>> &obj,
                           const ROL::Ptr<ROL::Constraint_SimOpt<Real>> &con,
                           const ROL::Ptr<ROL::Vector<Real>> &state,
                           const ROL::Ptr<ROL::Vector<Real>> &control,
                           const ROL::Ptr<ROL::Vector<Real>> &adjoint,
                           const int numGN)
    : ROL::Reduced_Objective_SimOpt<Real>(obj,con,state,control,adjoint,true,2), numGN_(numGN) {}

  void update(const ROL::Vector<Real> &z, ROL::UpdateType type, int iter = -1) override {
    ROL::Reduced_Objective_SimOpt<Real>::update(z,type,iter);
    if (iter >= 0) this->setHessApprox((iter < numGN_) ? 2 : 0);
  }
};

int main(int argc, char *argv[]) {

  ROL::GlobalMPISession mpiSession(&argc, &argv);

  // This little trick lets us print to std::cout only if a (dummy) command-line argument is provided.
  int iprint = argc - 1;
  ROL::Ptr<std::ostream> outStream;
  ROL::nullstream bhs; // outputs nothing
  if (iprint > 0)
    outStream = ROL::makePtrFromRef(std::cout);
  else
    outStream = ROL::makePtrFromRef(bhs);

  int errorFlag = 0;

  // *** Test body.

  try {
    RealT tol = std::sqrt(ROL::ROL_EPSILON<RealT>());
    RealT eps = static_cast<RealT>(100)*ROL::ROL_EPSILON<RealT>();
    RealT zero(0), half(0.5), one(1);
    *outStream << std::scientific << std::setprecision(8);

    /*************************************************************************/
    /************* INITIALIZE CONSTRAINT AND OBJECTIVE ***********************/
    /*************************************************************************/
    int dim = 10, nobs = 4;
    RealT beta(1), alpha(1e-2);
    ROL::Ptr<Constraint_Cubic<RealT>> con = ROL::makePtr<Constraint_Cubic<RealT>>(beta);
    ROL::Ptr<Objective_Tracking<RealT>> obj
      = ROL::makePtr<Objective_Tracking<RealT>>(std::vector<RealT>(nobs,one),alpha);
    ROL::ParameterList list;
    list.sublist("SimOpt").sublist("Solve").set("Absolute Residual Tolerance",eps);
    list.sublist("SimOpt").sublist("Solve").set("Zero Initial Guess",true);
    con->setSolveParameters(list);

    /*************************************************************************/
    /************* INITIALIZE VECTOR STORAGE *********************************/
    /*************************************************************************/
    ROL::Ptr<std::vector<RealT>> u_ptr = ROL::makePtr<std::vector<RealT>>(dim,0.0);
    ROL::Ptr<std::vector<RealT>> l_ptr = ROL::makePtr<std::vector<RealT>>(dim,0.0);
    ROL::Ptr<std::vector<RealT>> z_ptr = ROL::makePtr<std::vector<RealT>>(dim,0.0);
    ROL::Ptr<std::vector<RealT>> v_ptr = ROL::makePtr<std::vector<RealT>>(dim,0.0);
    ROL::Ptr<std::vector<RealT>> w_ptr = ROL::makePtr<std::vector<RealT>>(dim,0.0);
    RealT left = -1e0, right = 1e0;
    for (int i = 0; i < dim; i++) {
      (*z_ptr)[i] = ( (RealT)rand() / (RealT)RAND_MAX ) * (right - left) + left;
      (*v_ptr)[i] = ( (RealT)rand() / (RealT)RAND_MAX ) * (right - left) + left;
      (*w_ptr)[i] = ( (RealT)rand() / (RealT)RAND_MAX ) * (right - left) + left;
    }
    ROL::Ptr<ROL::Vector<RealT>> up = ROL::makePtr<ROL::StdVector<RealT>>(u_ptr);
    ROL::Ptr<ROL::Vector<RealT>> lp = ROL::makePtr<ROL::StdVector<RealT>>(l_ptr);
    ROL::Ptr<ROL::Vector<RealT>> zp = ROL::makePtr<ROL::StdVector<RealT>>(z_ptr);
    ROL::Ptr<ROL::Vector<RealT>> vp = ROL::makePtr<ROL::StdVector<RealT>>(v_ptr);
    ROL::Ptr<ROL::Vector<RealT>> wp = ROL::makePtr<ROL::StdVector<RealT>>(w_ptr);
    // State space vectors.
    ROL::Ptr<ROL::Vector<RealT>> cp = up->clone(), sp = up->clone(), sw = up->clone();
    ROL::Ptr<ROL::Vector<RealT>> pp = up->clone(), qp = up->clone(), jv = up->clone();
    ROL::Ptr<ROL::Vector<RealT>> gu = up->dual().clone(), hs = up->dual().clone(), ahwv = up->dual().clone();
    // Control space vectors.
    ROL::Ptr<ROL::Vector<RealT>> g  = zp->dual().clone(), gGN  = zp->dual().clone();
    ROL::Ptr<ROL::Vector<RealT>> hv = zp->dual().clone(), hvGN = zp->dual().clone();
    ROL::Ptr<ROL::Vector<RealT>> hz = zp->dual().clone(), dhv  = zp->dual().clone();

    /*************************************************************************/
    /************* INITIALIZE REDUCED OBJECTIVES *****************************/
    /*************************************************************************/
    // The last argument is the Hessian approximation (0: none, 1: finite differences,
    // 2: Gauss-Newton, 3: Gauss-Newton without the mixed derivatives of the objective).
    bool storage = true;
    ROL::Ptr<ROL::Objective<RealT>> robj
      = ROL::makePtr<ROL::Reduced_Objective_SimOpt<RealT>>(obj,con,up,zp,lp,storage,0);
    ROL::Ptr<ROL::Objective<RealT>> robjGN
      = ROL::makePtr<ROL::Reduced_Objective_SimOpt<RealT>>(obj,con,up,zp,lp,storage,2);

    /*************************************************************************/
    /************* CHECK DERIVATIVES OF THE TEST PROBLEM *********************/
    /*************************************************************************/
    *outStream << "\nCheck derivatives with the full Hessian\n";
    robj->update(*zp,ROL::UpdateType::Initial);
    std::vector<std::vector<RealT>> gcheck = robj->checkGradient(*zp,*vp,true,*outStream,7,2);
    std::vector<std::vector<RealT>> hcheck = robj->checkHessVec(*zp,*vp,true,*outStream,7,2);
    RealT gerr(ROL::ROL_INF<RealT>()), herr(ROL::ROL_INF<RealT>());
    for (std::size_t i = 0; i < gcheck.size(); ++i)
      gerr = std::min(gerr,gcheck[i][3]/std::abs(gcheck[i][1]));
    for (std::size_t i = 0; i < hcheck.size(); ++i)
      herr = std::min(herr,hcheck[i][3]/hcheck[i][1]);
    errorFlag += ((gerr > tol) ? 1 : 0) + ((herr > tol) ? 1 : 0);

    /*************************************************************************/
    /************* VALUE AND GRADIENT ****************************************/
    /*************************************************************************/
    // The Hessian approximation has to leave the value and the gradient alone.
    robj->update(*zp,ROL::UpdateType::Initial);
    robjGN->update(*zp,ROL::UpdateType::Initial);
    RealT f = robj->value(*zp,tol), fGN = robjGN->value(*zp,tol);
    robj->gradient(*g,*zp,tol);
    robjGN->gradient(*gGN,*zp,tol);
    gGN->axpy(-one,*g);
    *outStream << "\nCheck value and gradient with the Gauss-Newton Hessian\n";
    *outStream << "  Change in value    = " << std::abs(fGN-f) << "\n";
    *outStream << "  Change in gradient = " << gGN->norm() << "\n";
    errorFlag += ((fGN != f) ? 1 : 0) + ((gGN->norm() != zero) ? 1 : 0);

    /*************************************************************************/
    /************* GAUSS-NEWTON HESSIAN AGAINST ITS DEFINITION ***************/
    /*************************************************************************/
    // The constraint Hessian must not be applied.
    int nhess = con->getNumberHessianCalls();
    robjGN->hessVec(*hvGN,*vp,*zp,tol);
    errorFlag += (con->getNumberHessianCalls() != nhess) ? 1 : 0;
    robj->hessVec(*hv,*vp,*zp,tol);
    // Solve state equation.
    con->solve(*cp,*up,*zp,tol);
    // Solve state sensitivity equation.
    con->applyJacobian_2(*jv,*vp,*up,*zp,tol);
    jv->scale(-one);
    con->applyInverseJacobian_1(*sp,*jv,*up,*zp,tol);
    // Solve adjoint sensitivity equation without the constraint Hessian.
    obj->hessVec_11(*hs,*sp,*up,*zp,tol);
    obj->hessVec_12(*gu,*vp,*up,*zp,tol);
    hs->plus(*gu);
    hs->scale(-one);
    con->applyInverseAdjointJacobian_1(*pp,*hs,*up,*zp,tol);
    // Build hessVec.
    con->applyAdjointJacobian_2(*dhv,*pp,*up,*zp,tol);
    obj->hessVec_21(*hz,*sp,*up,*zp,tol);
    dhv->plus(*hz);
    obj->hessVec_22(*hz,*vp,*up,*zp,tol);
    dhv->plus(*hz);
    dhv->axpy(-one,*hvGN);
    RealT derr = dhv->norm()/hvGN->norm();
    *outStream << "\nCheck Gauss-Newton hessVec against its definition\n";
    *outStream << "  Relative error = " << derr << "\n";
    errorFlag += (derr > eps) ? 1 : 0;

    /*************************************************************************/
    /************* GAUSS-NEWTON HESSIAN AGAINST FINITE DIFFERENCES ***********/
    /*************************************************************************/
    // The objective has no mixed second derivatives, so
    // <w,H_GN v> = <f_uu s_v,s_w> + <f_zz v,w> with the state sensitivities
    // s_v and s_w, here central differences of the control-to-state map.
    RealT wHv = hvGN->apply(*wp);
    RealT wRv = hz->apply(*wp);
    *outStream << "\nCheck Gauss-Newton hessVec against finite differences of the state\n";
    *outStream << std::setw(16) << "Step size" << std::setw(20) << "<w,hessVec>"
               << std::setw(20) << "FD approx" << std::setw(18) << "abs error" << "\n";
    RealT h(1e-1), ferr(0), ferr0(0);
    for (int i = 0; i < 4; ++i) {
      differenceState(*sp,*vp,*zp,h,*con);
      differenceState(*sw,*wp,*zp,h,*con);
      obj->hessVec_11(*hs,*sp,*up,*zp,tol);
      RealT fd = hs->apply(*sw) + wRv;
      ferr0 = ferr;
      ferr  = std::abs(fd-wHv);
      *outStream << std::setw(16) << h << std::setw(20) << wHv
                 << std::setw(20) << fd << std::setw(18) << ferr << "\n";
      h *= static_cast<RealT>(0.1);
    }
    // The error of the central difference is second order in the step size.
    RealT rate = std::log10(ferr0/ferr);
    *outStream << "  Rate between the last two steps = " << rate << "\n";
    errorFlag += ((rate < static_cast<RealT>(1.9)) ? 1 : 0) + ((ferr > std::sqrt(tol)*std::abs(wHv)) ? 1 : 0);

    /*************************************************************************/
    /************* DIFFERENCE BETWEEN HESSIAN AND GAUSS-NEWTON HESSIAN *******/
    /*************************************************************************/
    // The difference is the constraint Hessian applied to the adjoint.
    // Solve adjoint equation.
    obj->gradient_1(*gu,*up,*zp,tol);
    con->applyInverseAdjointJacobian_1(*lp,*gu,*up,*zp,tol);
    lp->scale(-one);
    // Solve state sensitivity equation.
    con->applyJacobian_2(*jv,*vp,*up,*zp,tol);
    jv->scale(-one);
    con->applyInverseJacobian_1(*sp,*jv,*up,*zp,tol);
    // Apply adjoint Hessian of constraint.
    con->applyAdjointHessian_11(*ahwv,*lp,*sp,*up,*zp,tol);
    con->applyAdjointHessian_21(*hs,*lp,*vp,*up,*zp,tol);
    ahwv->plus(*hs);
    ahwv->scale(-one);
    con->applyInverseAdjointJacobian_1(*qp,*ahwv,*up,*zp,tol);
    con->applyAdjointJacobian_2(*dhv,*qp,*up,*zp,tol);
    con->applyAdjointHessian_12(*hz,*lp,*sp,*up,*zp,tol);
    dhv->plus(*hz);
    con->applyAdjointHessian_22(*hz,*lp,*vp,*up,*zp,tol);
    dhv->plus(*hz);
    RealT dnorm = dhv->norm();
    hv->axpy(-one,*hvGN);
    hv->axpy(-one,*dhv);
    RealT cerr = hv->norm()/dnorm;
    *outStream << "\nCheck difference between Hessian and Gauss-Newton Hessian\n";
    *outStream << "  ||hessVec - GN hessVec|| / ||GN hessVec|| = " << dnorm/hvGN->norm() << "\n";
    *outStream << "  Relative error                            = " << cerr << "\n";
    errorFlag += ((dnorm == zero) ? 1 : 0) + ((cerr > eps) ? 1 : 0);

    /*************************************************************************/
    /************* SYMMETRY AND SPECTRUM OF THE GAUSS-NEWTON HESSIAN *********/
    /*************************************************************************/
    // H_GN - f_zz = S'(z)^* f_uu S'(z) is positive semidefinite and its rank
    // is at most the number of observed state entries.
    std::vector<RealT> H(dim*dim), A(dim*dim);
    for (int j = 0; j < dim; ++j) {
      robjGN->hessVec(*hvGN,*zp->basis(j),*zp,tol);
      obj->hessVec_22(*hz,*zp->basis(j),*up,*zp,tol);
      for (int i = 0; i < dim; ++i) {
        H[i+j*dim] = (*dynamic_cast<ROL::StdVector<RealT>&>(*hvGN).getVector())[i];
        A[i+j*dim] = H[i+j*dim] - (*dynamic_cast<ROL::StdVector<RealT>&>(*hz).getVector())[i];
      }
    }
    RealT asym(0), hmax(0);
    for (int j = 0; j < dim; ++j) {
      for (int i = 0; i < j; ++i) {
        asym = std::max(asym,std::abs(H[i+j*dim]-H[j+i*dim]));
        hmax = std::max(hmax,std::abs(H[i+j*dim]));
        A[i+j*dim] = half*(A[i+j*dim]+A[j+i*dim]);
        A[j+i*dim] = A[i+j*dim];
      }
    }
    ROL::LAPACK<int,RealT> lapack;
    std::vector<RealT> lambda(dim), work(3*dim);
    int info(0);
    lapack.SYEV('N','U',dim,&A[0],dim,&lambda[0],&work[0],3*dim,&info);
    int rank(0);
    for (int i = 0; i < dim; ++i) rank += (lambda[i] > tol*lambda[dim-1]) ? 1 : 0;
    *outStream << "\nCheck symmetry and spectrum of the Gauss-Newton Hessian\n";
    *outStream << "  max|H_ij - H_ji| / max|H_ij|    = " << asym/hmax << "\n";
    *outStream << "  Smallest eigenvalue of H - f_zz = " << lambda[0] << "\n";
    *outStream << "  Largest eigenvalue of H - f_zz  = " << lambda[dim-1] << "\n";
    *outStream << "  Rank of H - f_zz                = " << rank << "\n";
    *outStream << "  Number of observations          = " << nobs << "\n";
    errorFlag += ((info != 0) ? 1 : 0) + ((asym > eps*hmax) ? 1 : 0);
    errorFlag += ((lambda[0] < -eps*lambda[dim-1]) ? 1 : 0) + ((rank != nobs) ? 1 : 0);

    /*************************************************************************/
    /************* ZERO RESIDUAL *********************************************/
    /*************************************************************************/
    // With the data equal to the state the adjoint vanishes, and the
    // Gauss-Newton Hessian is the Hessian.
    obj->setData(std::vector<RealT>(u_ptr->begin(),u_ptr->begin()+nobs));
    robj->update(*zp,ROL::UpdateType::Initial);
    robjGN->update(*zp,ROL::UpdateType::Initial);
    robj->hessVec(*hv,*vp,*zp,tol);
    robjGN->hessVec(*hvGN,*vp,*zp,tol);
    hv->axpy(-one,*hvGN);
    RealT zerr = hv->norm()/hvGN->norm();
    *outStream << "\nCheck Gauss-Newton Hessian with zero residual\n";
    *outStream << "  ||hessVec - GN hessVec|| / ||GN hessVec|| = " << zerr << "\n";
    errorFlag += (zerr > eps) ? 1 : 0;
    hcheck = robjGN->checkHessVec(*zp,*vp,true,*outStream,7,2);
    herr = ROL::ROL_INF<RealT>();
    for (std::size_t i = 0; i < hcheck.size(); ++i)
      herr = std::min(herr,hcheck[i][3]/hcheck[i][1]);
    errorFlag += (herr > tol) ? 1 : 0;
    std::vector<RealT> hsym = robjGN->checkHessSym(*zp,*vp,*wp,true,*outStream);
    errorFlag += (hsym[2] > eps*std::abs(hsym[0])) ? 1 : 0;

    /*************************************************************************/
    /************* LINEAR CONSTRAINT *****************************************/
    /*************************************************************************/
    // The reduced objective is quadratic, the Gauss-Newton Hessian is the
    // Hessian, and Newton's method converges in one iteration.
    ROL::Ptr<Constraint_Cubic<RealT>> conLin = ROL::makePtr<Constraint_Cubic<RealT>>(zero);
    conLin->setSolveParameters(list);
    obj->setData(std::vector<RealT>(nobs,one));
    ROL::Ptr<ROL::Objective<RealT>> robjLin
      = ROL::makePtr<ROL::Reduced_Objective_SimOpt<RealT>>(obj,conLin,up,zp,lp,storage,0);
    ROL::Ptr<ROL::Objective<RealT>> robjLinGN
      = ROL::makePtr<ROL::Reduced_Objective_SimOpt<RealT>>(obj,conLin,up,zp,lp,storage,2);
    robjLin->update(*zp,ROL::UpdateType::Initial);
    robjLinGN->update(*zp,ROL::UpdateType::Initial);
    RealT fz = robjLinGN->value(*zp,tol);
    robjLinGN->gradient(*g,*zp,tol);
    robjLin->hessVec(*hv,*vp,*zp,tol);
    robjLinGN->hessVec(*hvGN,*vp,*zp,tol);
    RealT model = fz + g->apply(*vp) + half*hvGN->apply(*vp);
    hv->axpy(-one,*hvGN);
    RealT lerr = hv->norm()/hvGN->norm();
    ROL::Ptr<ROL::Vector<RealT>> xp = zp->clone();
    xp->set(*zp); xp->plus(*vp);
    robjLinGN->update(*xp,ROL::UpdateType::Initial);
    RealT fx   = robjLinGN->value(*xp,tol);
    RealT merr = std::abs(fx-model)/std::abs(fx);
    *outStream << "\nCheck Gauss-Newton Hessian with linear constraint\n";
    *outStream << "  ||hessVec - GN hessVec|| / ||GN hessVec|| = " << lerr << "\n";
    *outStream << "  Relative error of the quadratic model     = " << merr << "\n";
    errorFlag += ((lerr > eps) ? 1 : 0) + ((merr > eps) ? 1 : 0);

    ROL::ParameterList parlist;
    parlist.sublist("General").set("Output Level",1);
    parlist.sublist("General").sublist("Krylov").set("Type","Conjugate Gradients");
    parlist.sublist("General").sublist("Krylov").set("Absolute Tolerance",1.e-12);
    parlist.sublist("General").sublist("Krylov").set("Relative Tolerance",1.e-12);
    parlist.sublist("General").sublist("Krylov").set("Iteration Limit",2*dim);
    parlist.sublist("Step").sublist("Line Search").sublist("Descent Method").set("Type","Newton-Krylov");
    parlist.sublist("Status Test").set("Gradient Tolerance",1.e-10);
    parlist.sublist("Status Test").set("Iteration Limit",10);
    ROL::TypeU::LineSearchAlgorithm<RealT> algo(parlist);
    xp->set(*zp);
    algo.run(*xp,*robjLinGN,*outStream);
    errorFlag += (algo.getState()->iter != 1) ? 1 : 0;
    errorFlag += (algo.getState()->statusFlag != ROL::EXITSTATUS_CONVERGED) ? 1 : 0;

    /*************************************************************************/
    /************* MIXED DERIVATIVES OF THE OBJECTIVE ************************/
    /*************************************************************************/
    // With gamma nonzero the objective has the mixed derivatives f_uz = f_zu = gamma I.
    // Dropping them gives the Gauss-Newton Hessian for gamma = 0, keeping them
    // adds S'(z)^* f_uz + f_zu S'(z).
    RealT gamma(0.5);
    ROL::Ptr<Objective_Tracking<RealT>> objMix
      = ROL::makePtr<Objective_Tracking<RealT>>(std::vector<RealT>(nobs,one),alpha,gamma);
    ROL::Ptr<ROL::Objective<RealT>> robjMix
      = ROL::makePtr<ROL::Reduced_Objective_SimOpt<RealT>>(objMix,con,up,zp,lp,storage,0);
    ROL::Ptr<ROL::Objective<RealT>> robjMixGN
      = ROL::makePtr<ROL::Reduced_Objective_SimOpt<RealT>>(objMix,con,up,zp,lp,storage,2);
    ROL::Ptr<ROL::Objective<RealT>> robjMixGN3
      = ROL::makePtr<ROL::Reduced_Objective_SimOpt<RealT>>(objMix,con,up,zp,lp,storage,3);
    *outStream << "\nCheck derivatives with mixed derivatives of the objective\n";
    robjMix->update(*zp,ROL::UpdateType::Initial);
    gcheck = robjMix->checkGradient(*zp,*vp,true,*outStream,7,2);
    hcheck = robjMix->checkHessVec(*zp,*vp,true,*outStream,7,2);
    gerr = ROL::ROL_INF<RealT>();
    herr = ROL::ROL_INF<RealT>();
    for (std::size_t i = 0; i < gcheck.size(); ++i)
      gerr = std::min(gerr,gcheck[i][3]/std::abs(gcheck[i][1]));
    for (std::size_t i = 0; i < hcheck.size(); ++i)
      herr = std::min(herr,hcheck[i][3]/hcheck[i][1]);
    errorFlag += ((gerr > tol) ? 1 : 0) + ((herr > tol) ? 1 : 0);
    robjGN->update(*zp,ROL::UpdateType::Initial);
    robjMixGN->update(*zp,ROL::UpdateType::Initial);
    robjMixGN3->update(*zp,ROL::UpdateType::Initial);
    robjGN->hessVec(*hvGN,*vp,*zp,tol);
    robjMixGN->hessVec(*hv,*vp,*zp,tol);
    // The constraint Hessian must not be applied.
    nhess = con->getNumberHessianCalls();
    robjMixGN3->hessVec(*dhv,*vp,*zp,tol);
    errorFlag += (con->getNumberHessianCalls() != nhess) ? 1 : 0;
    dhv->axpy(-one,*hvGN);
    RealT nomixerr = dhv->norm()/hvGN->norm();
    // Solve state sensitivity equation.
    con->solve(*cp,*up,*zp,tol);
    con->applyJacobian_2(*jv,*vp,*up,*zp,tol);
    jv->scale(-one);
    con->applyInverseJacobian_1(*sp,*jv,*up,*zp,tol);
    // Apply S'(z)^* f_uz + f_zu S'(z).
    objMix->hessVec_12(*hs,*vp,*up,*zp,tol);
    hs->scale(-one);
    con->applyInverseAdjointJacobian_1(*pp,*hs,*up,*zp,tol);
    con->applyAdjointJacobian_2(*dhv,*pp,*up,*zp,tol);
    objMix->hessVec_21(*hz,*sp,*up,*zp,tol);
    dhv->plus(*hz);
    RealT mixnorm = dhv->norm();
    hv->axpy(-one,*hvGN);
    hv->axpy(-one,*dhv);
    RealT mixerr = hv->norm()/mixnorm;
    *outStream << "\nCheck Gauss-Newton Hessian with and without the mixed derivatives\n";
    *outStream << "  Relative error without the mixed derivatives = " << nomixerr << "\n";
    *outStream << "  ||mixed derivatives|| / ||GN hessVec||       = " << mixnorm/hvGN->norm() << "\n";
    *outStream << "  Relative error of the mixed derivatives      = " << mixerr << "\n";
    errorFlag += ((nomixerr > eps) ? 1 : 0) + ((mixnorm == zero) ? 1 : 0) + ((mixerr > eps) ? 1 : 0);

    /*************************************************************************/
    /************* FINITE-DIFFERENCE HESSIAN *********************************/
    /*************************************************************************/
    // The finite-difference Hessian uses gradients only, so the constraint
    // Hessian must not be applied.
    ROL::Ptr<ROL::Objective<RealT>> robjFD
      = ROL::makePtr<ROL::Reduced_Objective_SimOpt<RealT>>(obj,con,up,zp,lp,storage,1);
    robj->update(*zp,ROL::UpdateType::Initial);
    robjFD->update(*zp,ROL::UpdateType::Initial);
    robj->hessVec(*hv,*vp,*zp,tol);
    nhess = con->getNumberHessianCalls();
    robjFD->hessVec(*dhv,*vp,*zp,tol);
    errorFlag += (con->getNumberHessianCalls() != nhess) ? 1 : 0;
    // Passing true, as for the old finite-difference flag, gives the same result.
    ROL::Ptr<ROL::Objective<RealT>> robjFDflag
      = ROL::makePtr<ROL::Reduced_Objective_SimOpt<RealT>>(obj,con,up,zp,lp,storage,true);
    robjFDflag->update(*zp,ROL::UpdateType::Initial);
    robjFDflag->hessVec(*hz,*vp,*zp,tol);
    hz->axpy(-one,*dhv);
    dhv->axpy(-one,*hv);
    RealT fderr = dhv->norm()/hv->norm();
    *outStream << "\nCheck finite-difference hessVec against the Hessian\n";
    *outStream << "  Relative error                 = " << fderr << "\n";
    *outStream << "  Change with a boolean argument = " << hz->norm() << "\n";
    errorFlag += ((fderr > std::sqrt(tol)) ? 1 : 0) + ((hz->norm() != zero) ? 1 : 0);

    /*************************************************************************/
    /************* INVALID HESSIAN APPROXIMATION *****************************/
    /*************************************************************************/
    ROL::Ptr<ROL::Objective<RealT>> robjBad
      = ROL::makePtr<ROL::Reduced_Objective_SimOpt<RealT>>(obj,con,up,zp,lp,storage,4);
    *outStream << "\nCheck that hessVec throws for an invalid Hessian approximation\n";
    int thrown = 0;
    try {
      robjBad->hessVec(*dhv,*vp,*zp,tol);
    }
    catch (std::invalid_argument& exc) {
      *outStream << exc.what() << "\n";
      thrown++;
    };
    errorFlag += (thrown != 1) ? 1 : 0;

    /*************************************************************************/
    /************* CHANGE THE HESSIAN APPROXIMATION IN UPDATE ****************/
    /*************************************************************************/
    // Gauss-Newton Hessian while the iteration number is below numGN, then the Hessian.
    int numGN = 2;
    ROL::Ptr<ROL::Objective<RealT>> robjSwitch
      = ROL::makePtr<Reduced_Objective_Switch<RealT>>(obj,con,up,zp,lp,numGN);
    robj->update(*zp,ROL::UpdateType::Initial);
    robjGN->update(*zp,ROL::UpdateType::Initial);
    robj->hessVec(*hv,*vp,*zp,tol);
    robjGN->hessVec(*hvGN,*vp,*zp,tol);
    robjSwitch->update(*zp,ROL::UpdateType::Initial,0);
    robjSwitch->hessVec(*dhv,*vp,*zp,tol);
    dhv->axpy(-one,*hvGN);
    RealT serrGN = dhv->norm()/hvGN->norm();
    robjSwitch->update(*zp,ROL::UpdateType::Accept,numGN);
    robjSwitch->hessVec(*dhv,*vp,*zp,tol);
    dhv->axpy(-one,*hv);
    RealT serr = dhv->norm()/hv->norm();
    *outStream << "\nCheck change of the Hessian approximation in update\n";
    *outStream << "  Relative error against GN hessVec at iteration 0 = " << serrGN << "\n";
    *outStream << "  Relative error against hessVec at iteration " << numGN << "    = " << serr << "\n";
    errorFlag += ((serrGN > eps) ? 1 : 0) + ((serr > eps) ? 1 : 0);

    // Newton-Krylov stopped at iteration numGN never applies the constraint Hessian.
    parlist.sublist("Status Test").set("Iteration Limit",numGN);
    ROL::TypeU::LineSearchAlgorithm<RealT> algoGN(parlist);
    xp->zero();
    nhess = con->getNumberHessianCalls();
    algoGN.run(*xp,*robjSwitch,*outStream);
    errorFlag += (con->getNumberHessianCalls() != nhess) ? 1 : 0;
    // Run to convergence, the Hessian takes over.
    parlist.sublist("Status Test").set("Iteration Limit",100);
    ROL::TypeU::LineSearchAlgorithm<RealT> algoSwitch(parlist);
    xp->zero();
    algoSwitch.run(*xp,*robjSwitch,*outStream);
    errorFlag += (con->getNumberHessianCalls() == nhess) ? 1 : 0;
    errorFlag += (algoSwitch.getState()->statusFlag != ROL::EXITSTATUS_CONVERGED) ? 1 : 0;
  }
  catch (std::logic_error& err) {
    *outStream << err.what() << "\n";
    errorFlag = -1000;
  }; // end try

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}
