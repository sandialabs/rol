// @HEADER
// *****************************************************************************
//               Rapid Optimization Library (ROL) Package
//
// Copyright 2014 NTESS and the ROL contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

/*! \file  test_09.cpp
    \brief Validate TypeP trust-region subproblem solvers when the primal
           space has a non-identity inner product (diagonal, lumped-mass-like
           weights).  The model gradient is a dual vector, so solvers that use
           it as a primal vector stall.

           Problem: min_x  sum_i m_i [ 0.5 a_i (x_i-d_i)^2 + beta |x_i| ]
           with <x,y> = sum_i m_i x_i y_i.  The mass weight cancels in the
           prox, so the solution is x_i = soft(d_i, beta/a_i).
*/

#include "ROL_TypeP_TrustRegionAlgorithm.hpp"
#include "ROL_Objective.hpp"
#include "ROL_Vector.hpp"
#include "ROL_Stream.hpp"
#include "ROL_GlobalMPISession.hpp"
#include <cmath>
#include <vector>

typedef double RealT;

// Vector with inner product sum_i m_i x_i y_i; the dual vector stores
// functional coefficients with inner product sum_i x_i y_i / m_i.
class WeightedVector : public ROL::Vector<RealT> {
private:
  ROL::Ptr<const std::vector<RealT>> m_;
  std::vector<RealT> v_;
  bool isDual_;
  mutable ROL::Ptr<WeightedVector> dual_;

public:
  WeightedVector(const ROL::Ptr<const std::vector<RealT>> &m, bool isDual = false)
    : m_(m), v_(m->size(), 0), isDual_(isDual) {}

  std::vector<RealT>       &get()       { return v_; }
  const std::vector<RealT> &get() const { return v_; }

  void plus(const ROL::Vector<RealT> &x) override {
    const auto &xv = static_cast<const WeightedVector&>(x).v_;
    for (size_t i = 0; i < v_.size(); ++i) v_[i] += xv[i];
  }
  void scale(const RealT alpha) override { for (auto &e : v_) e *= alpha; }
  RealT dot(const ROL::Vector<RealT> &x) const override {
    const auto &xv = static_cast<const WeightedVector&>(x).v_;
    RealT s(0);
    for (size_t i = 0; i < v_.size(); ++i)
      s += isDual_ ? v_[i]*xv[i]/(*m_)[i] : (*m_)[i]*v_[i]*xv[i];
    return s;
  }
  RealT norm() const override { return std::sqrt(dot(*this)); }
  ROL::Ptr<ROL::Vector<RealT>> clone() const override {
    return ROL::makePtr<WeightedVector>(m_, isDual_);
  }
  int dimension() const override { return static_cast<int>(v_.size()); }
  const ROL::Vector<RealT> &dual() const override {
    if (dual_ == ROL::nullPtr) dual_ = ROL::makePtr<WeightedVector>(m_, !isDual_);
    for (size_t i = 0; i < v_.size(); ++i)
      dual_->v_[i] = isDual_ ? v_[i]/(*m_)[i] : (*m_)[i]*v_[i];
    return *dual_;
  }
  RealT apply(const ROL::Vector<RealT> &x) const override {
    const auto &xv = static_cast<const WeightedVector&>(x).v_;
    RealT s(0);
    for (size_t i = 0; i < v_.size(); ++i) s += v_[i]*xv[i];
    return s;
  }
};

class WeightedQuadratic : public ROL::Objective<RealT> {
private:
  ROL::Ptr<const std::vector<RealT>> m_;
  std::vector<RealT> a_, d_;
public:
  WeightedQuadratic(const ROL::Ptr<const std::vector<RealT>> &m,
                    const std::vector<RealT> &a, const std::vector<RealT> &d)
    : m_(m), a_(a), d_(d) {}

  RealT value(const ROL::Vector<RealT> &x, RealT &) override {
    const auto &xv = static_cast<const WeightedVector&>(x).get();
    RealT v(0);
    for (size_t i = 0; i < xv.size(); ++i)
      v += (*m_)[i]*0.5*a_[i]*(xv[i]-d_[i])*(xv[i]-d_[i]);
    return v;
  }
  void gradient(ROL::Vector<RealT> &g, const ROL::Vector<RealT> &x, RealT &) override {
    auto &gv = static_cast<WeightedVector&>(g).get();
    const auto &xv = static_cast<const WeightedVector&>(x).get();
    for (size_t i = 0; i < xv.size(); ++i)
      gv[i] = (*m_)[i]*a_[i]*(xv[i]-d_[i]);
  }
  void hessVec(ROL::Vector<RealT> &hv, const ROL::Vector<RealT> &v,
               const ROL::Vector<RealT> &, RealT &) override {
    auto &hvv = static_cast<WeightedVector&>(hv).get();
    const auto &vv = static_cast<const WeightedVector&>(v).get();
    for (size_t i = 0; i < vv.size(); ++i)
      hvv[i] = (*m_)[i]*a_[i]*vv[i];
  }
};

// beta * sum_i m_i |x_i| with the prox taken in the weighted inner product
class WeightedL1 : public ROL::Objective<RealT> {
private:
  ROL::Ptr<const std::vector<RealT>> m_;
  RealT beta_;
public:
  WeightedL1(const ROL::Ptr<const std::vector<RealT>> &m, RealT beta) : m_(m), beta_(beta) {}

  RealT value(const ROL::Vector<RealT> &x, RealT &) override {
    const auto &xv = static_cast<const WeightedVector&>(x).get();
    RealT v(0);
    for (size_t i = 0; i < xv.size(); ++i) v += (*m_)[i]*std::abs(xv[i]);
    return beta_*v;
  }
  void prox(ROL::Vector<RealT> &Pv, const ROL::Vector<RealT> &v, RealT t, RealT &) override {
    auto &p = static_cast<WeightedVector&>(Pv).get();
    const auto &vv = static_cast<const WeightedVector&>(v).get();
    const RealT thr = t*beta_;
    for (size_t i = 0; i < vv.size(); ++i)
      p[i] = std::max(RealT(0), std::abs(vv[i])-thr) * (vv[i] < 0 ? -1 : 1);
  }
};

class InexactQuadratic : public ROL::Objective<RealT> {
  bool rejectFirst_;
  bool trial_ = false;
  int trials_ = 0;
public:
  bool invalidTolerance = false;
  int refreshed = 0;
  int accepted = 0, rejected = 0;

  explicit InexactQuadratic(bool rejectFirst) : rejectFirst_(rejectFirst) {}
  void update(const ROL::Vector<RealT> &, ROL::UpdateType type, int = -1) override {
    trial_ = (type == ROL::UpdateType::Trial);
    if (trial_) ++trials_;
    if (type == ROL::UpdateType::Accept) ++accepted;
    if (type == ROL::UpdateType::Revert) ++rejected;
  }
  RealT value(const ROL::Vector<RealT> &x, RealT &tol) override {
    if (!std::isfinite(tol) || tol < 0) invalidTolerance = true;
    const RealT z = static_cast<const WeightedVector&>(x).get()[0] - 1;
    if (!trial_ && tol < 1) ++refreshed;
    // The first trial can be made deliberately poor to exercise rejection.
    return 0.5*z*z + (tol > 1 ? -0.1 : 0)
         + (trial_ && rejectFirst_ && trials_ == 1 ? 1 : 0);
  }
  void gradient(ROL::Vector<RealT> &g, const ROL::Vector<RealT> &x, RealT &) override {
    static_cast<WeightedVector&>(g).get()[0] = static_cast<const WeightedVector&>(x).get()[0] - 1;
  }
  void hessVec(ROL::Vector<RealT> &hv, const ROL::Vector<RealT> &v,
               const ROL::Vector<RealT> &, RealT &) override {
    hv.set(v);
  }
};

class ConstantProx : public ROL::Objective<RealT> {
public:
  RealT value(const ROL::Vector<RealT> &, RealT &) override { return 0.02; }
  void prox(ROL::Vector<RealT> &pv, const ROL::Vector<RealT> &v, RealT, RealT &) override {
    pv.set(v);
  }
};

class ObservedTrustRegion : public ROL::TypeP::TrustRegionAlgorithm<RealT> {
public:
  using ROL::TypeP::TrustRegionAlgorithm<RealT>::TrustRegionAlgorithm;
  using ROL::TypeP::TrustRegionAlgorithm<RealT>::computeValue;
  bool badPrediction = false;
  int calls = 0;
  RealT computeValue(RealT inTol, RealT &outTol, RealT pRed, RealT &fold,
                     RealT del, int iter, const ROL::Vector<RealT> &x,
                     const ROL::Vector<RealT> &xold, ROL::Objective<RealT> &obj) override {
    const RealT oldz = static_cast<const WeightedVector&>(xold).get()[0] - 1;
    const RealT newz = static_cast<const WeightedVector&>(x).get()[0] - 1;
    if (std::abs(pRed - 0.5*(oldz*oldz-newz*newz)) > 1e-12) badPrediction = true;
    ++calls;
    return ROL::TypeP::TrustRegionAlgorithm<RealT>::computeValue(
        inTol, outTol, pRed, fold, del, iter, x, xold, obj);
  }
};

int testInexactValues(std::ostream &out) {
  int errors = 0;
  auto m = ROL::makePtr<std::vector<RealT>>(1, 1);
  for (bool rejectFirst : {false, true}) {
    ROL::ParameterList list;
    list.sublist("General").set("Inexact Objective Function", true);
    list.sublist("Step").sublist("Trust Region").set("Initial Radius", 0.1);
    list.sublist("Status Test").set("Iteration Limit", rejectFirst ? 2 : 1);
    list.sublist("Status Test").set("Gradient Tolerance", 1e-12);
    list.sublist("Status Test").set("Step Tolerance", 1e-14);
    WeightedVector x(m);
    InexactQuadratic smooth(rejectFirst);
    ConstantProx nonsmooth;
    ObservedTrustRegion algo(list);
    algo.run(x, smooth, nonsmooth, out);
    auto state = algo.getState();
    if (smooth.refreshed == 0 || smooth.invalidTolerance || algo.badPrediction ||
        smooth.accepted != 1 || smooth.rejected != (rejectFirst ? 1 : 0) ||
        algo.calls != (rejectFirst ? 2 : 1) || x.get()[0] <= 0 ||
        std::abs(state->value - state->svalue - state->nvalue) > 1e-12) {
      out << "Inexact value synchronization failed, rejectFirst=" << rejectFirst << "\n";
      ++errors;
    }

    for (RealT pRed : {-0.1, 0.0}) {
      RealT outTol = 0, oldValue = 0;
      algo.computeValue(1, outTol, pRed, oldValue, 0.1, 1, x, x, smooth);
      if (!std::isfinite(outTol) || outTol <= 0 || smooth.invalidTolerance) {
        out << "Invalid tolerance for nonpositive predicted reduction\n";
        ++errors;
      }
    }
  }
  return errors;
}

int main(int argc, char *argv[]) {

  ROL::GlobalMPISession mpiSession(&argc, &argv);

  int iprint = argc - 1;
  ROL::Ptr<std::ostream> outStream;
  ROL::nullstream bhs;
  if (iprint > 0) outStream = ROL::makePtrFromRef(std::cout);
  else            outStream = ROL::makePtrFromRef(bhs);

  int errorFlag = 0;

  try {
    errorFlag += testInexactValues(*outStream);
    const int dim = 40;
    const RealT beta = 1e-3, alpha = 1e-4, errTol = 1e-1;

    // Non-uniform lumped-mass-like weights, and a smooth quadratic
    // with a small Tikhonov term.
    auto mP = ROL::makePtr<std::vector<RealT>>(dim);
    std::vector<RealT> a(dim), d(dim), xstar(dim);
    for (int i = 0; i < dim; ++i) {
      (*mP)[i] = 1e-3*(1.0 + 9.0*std::pow(std::sin(0.7*i+0.3), 2));
      a[i]     = alpha + std::pow(std::cos(0.37*i), 2);
      d[i]     = std::sin(1.3*i+0.1);
      const RealT thr = beta/a[i];
      xstar[i] = std::max(RealT(0), std::abs(d[i])-thr) * (d[i] < 0 ? -1 : 1);
    }
    ROL::Ptr<const std::vector<RealT>> m = mP;

    auto sobj = ROL::makePtr<WeightedQuadratic>(m, a, d);
    auto nobj = ROL::makePtr<WeightedL1>(m, beta);
    auto sol  = ROL::makePtr<WeightedVector>(m);

    // Verify the test vector's dual/Riesz map and the objective derivatives
    auto xd = sol->clone(); auto yd = sol->clone();
    for (int i = 0; i < dim; ++i) {
      static_cast<WeightedVector&>(*xd).get()[i] = std::sin(2.0*i);
      static_cast<WeightedVector&>(*yd).get()[i] = std::cos(3.0*i);
    }
    sobj->checkGradient(*xd,*yd,true,*outStream);
    sobj->checkHessVec(*xd,*yd,true,*outStream);

    ROL::ParameterList list;
    list.sublist("General").set("Output Level", iprint);
    list.sublist("Step").set("Type","Trust Region");
    list.sublist("Status Test").set("Gradient Tolerance", 1e-8);
    list.sublist("Status Test").set("Constraint Tolerance", 1e-8);
    list.sublist("Status Test").set("Step Tolerance", 1e-12);
    list.sublist("Status Test").set("Iteration Limit", 500);
    list.sublist("Step").sublist("Trust Region").set("Initial Radius", 1e2);

    // Required final gradient norm within the iteration limit (the Cauchy
    // point is a steepest-descent-type step and needs many iterations).
    struct Case { std::string name; bool required; RealT gtol; };
    const std::vector<Case> cases = {
      {"Cauchy Point",  true,  1e-6},
      {"Truncated CG",  true,  1e-6},
      {"SPG",           true,  1e-6},
      {"SPG2",          true,  1e-6}};

    *outStream << std::endl
               << "Weighted-metric l1 problem (dim = " << dim << ")" << std::endl;
    for (const auto &c : cases) {
      list.sublist("Step").sublist("Trust Region").set("Subproblem Solver", c.name);
      sol->zero();
      auto algo = ROL::makePtr<ROL::TypeP::TrustRegionAlgorithm<RealT>>(list);
      algo->run(*sol, *sobj, *nobj, *outStream);
      const auto &s = static_cast<WeightedVector&>(*sol).get();
      RealT err(0);
      for (int i = 0; i < dim; ++i) err = std::max(err, std::abs(s[i]-xstar[i]));
      auto st = algo->getState();
      *outStream << "  " << c.name << ": outer iterations = " << st->iter << ", prox evaluations = "
                 << st->nprox << ", gnorm = " << st->gnorm << ", max err = "
                 << err << (st->gnorm <= c.gtol ? "" : "  (NOT CONVERGED)") << std::endl;
      if (c.required && (st->gnorm > c.gtol || err > errTol)) errorFlag += 1;
    }
  }
  catch (std::logic_error &err) {
    *outStream << err.what() << "\n";
    errorFlag = -1000;
  }

  if (errorFlag != 0) std::cout << "End Result: TEST FAILED\n";
  else                std::cout << "End Result: TEST PASSED\n";

  return 0;
}
