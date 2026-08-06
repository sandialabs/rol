// @HEADER
// *****************************************************************************
//               Rapid Optimization Library (ROL) Package
//
// Copyright 2014 NTESS and the ROL contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef ROL_ROW_WISE_DAI_FLETCHER_PROJECTION_HPP
#define ROL_ROW_WISE_DAI_FLETCHER_PROJECTION_HPP

#include <vector>

#include "ROL_PolyhedralProjection.hpp"
#include "ROL_ParameterList.hpp"
#include "ROL_StdVector.hpp"

namespace ROL {

template<typename Real>
class RowWiseDaiFletcherProjection : public PolyhedralProjection<Real> {
private:
  using V  = Vector<Real>;
  using SV = StdVector<Real>;

  int num_detections_;
  int num_moths_;

  Real target_sum_;
  Real lower_;
  Real upper_;

  Real DEFAULT_atol_, DEFAULT_rtol_, DEFAULT_ltol_;
  int  DEFAULT_maxit_, DEFAULT_verbosity_;

  Real atol_, rtol_, ltol_;
  int  maxit_, verbosity_;

  using PolyhedralProjection<Real>::bnd_;
  using PolyhedralProjection<Real>::con_;
  using PolyhedralProjection<Real>::xprim_;
  using PolyhedralProjection<Real>::xdual_;
  using PolyhedralProjection<Real>::mul_;
  using PolyhedralProjection<Real>::res_;

  void initialize(const Vector<Real>               &xprim,
                  const Vector<Real>               &xdual,
                  const Ptr<BoundConstraint<Real>> &bnd,
                  const Ptr<Constraint<Real>>      &con,
                  const Vector<Real>               &mul,
                  const Vector<Real>               &res);

  Real residual(const std::vector<Real> &x) const;

  void update_primal(std::vector<Real>       &y,
                     const std::vector<Real> &x,
                     const Real lam) const;

  void project_df(std::vector<Real> &x,
                  Real              &lam,
                  std::ostream      &stream = std::cout, 
                  int *proj_iter = nullptr) const;

public:
  RowWiseDaiFletcherProjection(
      const Vector<Real>               &xprim,
      const Vector<Real>               &xdual,
      const Ptr<BoundConstraint<Real>> &bnd,
      const Ptr<Constraint<Real>>      &con,
      const Vector<Real>               &mul,
      const Vector<Real>               &res);

  RowWiseDaiFletcherProjection(
      const Vector<Real>               &xprim,
      const Vector<Real>               &xdual,
      const Ptr<BoundConstraint<Real>> &bnd,
      const Ptr<Constraint<Real>>      &con,
      const Vector<Real>               &mul,
      const Vector<Real>               &res,
      ParameterList                    &list);

  void project(Vector<Real> &x, std::ostream &stream = std::cout) override;

  void project(Vector<Real> &x, std::ostream &stream, int *proj_iter) override;
};

}

#include "ROL_RowWiseDaiFletcherProjection_Def.hpp"

#endif