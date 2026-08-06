// @HEADER
// *****************************************************************************
//               Rapid Optimization Library (ROL) Package
//
// Copyright 2014 NTESS and the ROL contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef ROL_PRODUCT_SIMPLEX_DAI_FLETCHER_PROJECTION_DEF_HPP
#define ROL_PRODUCT_SIMPLEX_DAI_FLETCHER_PROJECTION_DEF_HPP

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace ROL {

template<typename Real>
RowWiseDaiFletcherProjection<Real>::RowWiseDaiFletcherProjection( const Vector<Real>               &xprim,
                                                                                const Vector<Real>               &xdual,
                                                                                const Ptr<BoundConstraint<Real>> &bnd,
                                                                                const Ptr<Constraint<Real>>      &con,
                                                                                const Vector<Real>               &mul,
                                                                                const Vector<Real>               &res)
  : PolyhedralProjection<Real>(xprim,xdual,bnd,con,mul,res),
    num_detections_(0),
    num_moths_(0),
    target_sum_(static_cast<Real>(1)),
    lower_(static_cast<Real>(0)),
    upper_(static_cast<Real>(1)),
    DEFAULT_atol_(std::sqrt(ROL_EPSILON<Real>()*std::sqrt(ROL_EPSILON<Real>()))),
    DEFAULT_rtol_(std::sqrt(ROL_EPSILON<Real>())),
    DEFAULT_ltol_(ROL_EPSILON<Real>()),
    DEFAULT_maxit_(5000),
    DEFAULT_verbosity_(0),
    atol_(DEFAULT_atol_),
    rtol_(DEFAULT_rtol_),
    ltol_(DEFAULT_ltol_),
    maxit_(DEFAULT_maxit_),
    verbosity_(DEFAULT_verbosity_) {
  initialize(xprim,xdual,bnd,con,mul,res);
}

template<typename Real>
RowWiseDaiFletcherProjection<Real>::RowWiseDaiFletcherProjection( const Vector<Real>               &xprim,
                                                                                const Vector<Real>               &xdual,
                                                                                const Ptr<BoundConstraint<Real>> &bnd,
                                                                                const Ptr<Constraint<Real>>      &con,
                                                                                const Vector<Real>               &mul,
                                                                                const Vector<Real>               &res,
                                                                                ParameterList                    &list)
  : PolyhedralProjection<Real>(xprim,xdual,bnd,con,mul,res),
    num_detections_(0),
    num_moths_(0),
    target_sum_(static_cast<Real>(1)),
    lower_(static_cast<Real>(0)),
    upper_(static_cast<Real>(1)),
    DEFAULT_atol_(std::sqrt(ROL_EPSILON<Real>()*std::sqrt(ROL_EPSILON<Real>()))),
    DEFAULT_rtol_(std::sqrt(ROL_EPSILON<Real>())),
    DEFAULT_ltol_(ROL_EPSILON<Real>()),
    DEFAULT_maxit_(5000),
    DEFAULT_verbosity_(0),
    atol_(DEFAULT_atol_),
    rtol_(DEFAULT_rtol_),
    ltol_(DEFAULT_ltol_),
    maxit_(DEFAULT_maxit_),
    verbosity_(DEFAULT_verbosity_) {

  num_detections_ = list.sublist("General").sublist("Polyhedral Projection").get("Number of Detections", 0);
  num_moths_      = list.sublist("General").sublist("Polyhedral Projection").get("Number of Moths", 0);
  target_sum_     = list.sublist("General").sublist("Polyhedral Projection").get("Target Sum", static_cast<Real>(1));
  lower_          = list.sublist("General").sublist("Polyhedral Projection").get("Lower Bound", static_cast<Real>(0));
  upper_          = list.sublist("General").sublist("Polyhedral Projection").get("Upper Bound", static_cast<Real>(1));

  atol_     = list.sublist("General").sublist("Polyhedral Projection").get("Absolute Tolerance", DEFAULT_atol_);
  rtol_     = list.sublist("General").sublist("Polyhedral Projection").get("Relative Tolerance", DEFAULT_rtol_);
  ltol_     = list.sublist("General").sublist("Polyhedral Projection").get("Multiplier Tolerance", DEFAULT_ltol_);
  maxit_    = list.sublist("General").sublist("Polyhedral Projection").get("Iteration Limit", DEFAULT_maxit_);
  verbosity_ = list.sublist("General").get("Output Level", DEFAULT_verbosity_);

  initialize(xprim,xdual,bnd,con,mul,res);
}

template<typename Real>
void RowWiseDaiFletcherProjection<Real>::initialize( const Vector<Real>               &xprim,
                                                            const Vector<Real>               &xdual,
                                                            const Ptr<BoundConstraint<Real>> &bnd,
                                                            const Ptr<Constraint<Real>>      &con,
                                                            const Vector<Real>               &mul,
                                                            const Vector<Real>               &res) {
}

template<typename Real>
Real RowWiseDaiFletcherProjection<Real>::residual(const std::vector<Real> &x) const {
  Real sum = static_cast<Real>(0);

  for (const auto &xi : x) {
    sum += xi;
  }

  return sum - target_sum_;
}

template<typename Real>
void RowWiseDaiFletcherProjection<Real>::update_primal(std::vector<Real> &y, const std::vector<Real> &x, const Real lam) const {
  const int dim = static_cast<int>(x.size());

  for (int k = 0; k < dim; ++k) {
    auto val = x[k] + lam;
    y[k] = std::max(lower_, std::min(upper_, val));
  }
}

template<typename Real>
void RowWiseDaiFletcherProjection<Real>::project_df(
    std::vector<Real> &x,
    Real              &lam,
    std::ostream      &stream,
    int               *proj_iter) const {
  const Real zero(0), one(1), half(0.5);

  const int dim = static_cast<int>(x.size());

  ROL_TEST_FOR_EXCEPTION(dim <= 0, std::logic_error,
    ">>> ROL::RowWiseDaiFletcherProjection : Empty detection row!");

  auto minmax = std::minmax_element(x.begin(), x.end());

  const Real xmin = *minmax.first;
  const Real xmax = *minmax.second;

  Real lamLower = lower_ - xmax;
  Real lamUpper = upper_ - xmin;

  std::vector<Real> xnew(dim, zero);

  update_primal(xnew, x, lamLower);
  Real resLower = residual(xnew);

  update_primal(xnew, x, lamUpper);
  Real resUpper = residual(xnew);


  if (std::abs(resLower) <= atol_) {
    update_primal(xnew, x, lamLower);
    x = xnew;
    lam = lamLower;
    return;
  }

  if (std::abs(resUpper) <= atol_) {
    update_primal(xnew, x, lamUpper);
    x = xnew;
    lam = lamUpper;
    return;
  }

  lam = (lamLower * resUpper - lamUpper * resLower)
        / (resUpper - resLower);

  update_primal(xnew, x, lam);
  Real res = residual(xnew);


  Real ctol = std::min(atol_, rtol_ * std::max(one, std::max(std::abs(resLower), std::abs(resUpper))));

  int cnt = 0;
  for (cnt = 0; cnt < maxit_; ++cnt) {
    const Real scale = std::max(one, std::max(std::abs(lamLower), std::abs(lamUpper)));
    const Real bracket_width = std::abs(lamUpper - lamLower);

    if (std::abs(res) <= ctol || bracket_width <= ltol_ * scale) {
      break;
    }
  
    if (res > zero) {
      lamUpper = lam;
      resUpper = res;
    } else {
      lamLower = lam;
      resLower = res;
    }
   
    lam = (lamLower * resUpper - lamUpper * resLower) / (resUpper - resLower);
    update_primal(xnew, x, lam);
    res = residual(xnew);
  }
  if (proj_iter != nullptr) {
    *proj_iter = cnt;
  }

  
  x = xnew;

  if (std::abs(res) > ctol && verbosity_ > 0) {
    stream << ">>> ROL::RowWiseDaiFletcherProjection : Projection may be inaccurate! "
           << "rnorm = " << std::abs(res)
           << "  rtol = " << ctol
           << std::endl;
  }
}

template<typename Real>
void RowWiseDaiFletcherProjection<Real>::project(Vector<Real> &x,
                                                        std::ostream &stream) {
  project(x, stream, nullptr);
}

template<typename Real>
void RowWiseDaiFletcherProjection<Real>::project(Vector<Real> &x,
                                                        std::ostream &stream,
                                                        int *proj_iter) {
  SV *x_sv = dynamic_cast<SV*>(&x);

  std::vector<Real> &x_vec = *(x_sv->getVector());

  const int expected_dim = num_detections_ * num_moths_;
  
  int total_iter = 0;
  for (int i = 0; i < num_detections_; ++i) {
    const int offset = i * num_moths_;

    std::vector<Real> detection_row(num_moths_);

    for (int k = 0; k < num_moths_; ++k) {
      detection_row[k] = x_vec[offset + k];
    }

    Real lam  = static_cast<Real>(0);

    int row_iter = 0;
    project_df(detection_row, lam, stream, &row_iter);
    total_iter += row_iter;

    for (int k = 0; k < num_moths_; ++k) {
      x_vec[offset + k] = detection_row[k];
    }
  }
  if (proj_iter != nullptr) {
    *proj_iter = total_iter;
  }
}

} 

#endif