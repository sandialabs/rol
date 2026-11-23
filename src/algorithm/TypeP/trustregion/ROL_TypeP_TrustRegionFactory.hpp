// @HEADER
// *****************************************************************************
//               Rapid Optimization Library (ROL) Package
//
// Copyright 2014 NTESS and the ROL contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

/** \file
    \brief  Contains definitions of enums for trust region algorithms.
    \author Created by D. Ridzal and D. Kouri.
 */

#ifndef ROL_TYPEP_TRUSTREGIONFACTORY_HPP
#define ROL_TYPEP_TRUSTREGIONFACTORY_HPP

#include "ROL_TypeP_TrustRegionTypes.hpp"
#include "ROL_TypeP_CauchyPoint.hpp"
#include "ROL_TypeP_SPG.hpp"
#include "ROL_TypeP_SPG2.hpp"
#include "ROL_TypeP_TruncatedCG.hpp"

namespace ROL {
namespace TypeP {

  template<typename Real>
  inline Ptr<TrustRegion<Real>> TrustRegionFactory(ParameterList &list) {
    ETrustRegionP etr = StringToETrustRegionP(
      list.sublist("Step").sublist("Trust Region").get("Subproblem Solver","Cauchy Point"));
    switch(etr) {
      case TRUSTREGION_P_CAUCHYPOINT:    return makePtr<CauchyPoint<Real>>(list);
      case TRUSTREGION_P_SPG:            return makePtr<SPG<Real>>(list);
      case TRUSTREGION_P_NCG:            return makePtr<TruncatedCG<Real>>(list);
      case TRUSTREGION_P_SPG2:           return makePtr<SPG2<Real>>(list);
      default:                           return nullPtr;
    }
  }
} // namespace TypeP
} // namespace ROL

#endif
