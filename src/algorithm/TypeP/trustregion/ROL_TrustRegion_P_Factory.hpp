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

#ifndef ROL_TRUSTREGION_P_FACTORY_HPP
#define ROL_TRUSTREGION_P_FACTORY_HPP

#include "ROL_TrustRegion_P_Types.hpp"
#include "ROL_CauchyPoint_P.hpp"
#include "ROL_SPG_P.hpp"
#include "ROL_SPG2_P.hpp"
#include "ROL_NCG_P.hpp"

namespace ROL {
  template<typename Real>
  inline Ptr<TrustRegion_P<Real>> TrustRegionPFactory(ParameterList &list) {
    ETrustRegionP etr = StringToETrustRegionP(
      list.sublist("Step").sublist("Trust Region").get("Subproblem Solver","Cauchy Point"));
    switch(etr) {
      case TRUSTREGION_P_CAUCHYPOINT:    return makePtr<CauchyPoint_P<Real>>(list);
      case TRUSTREGION_P_SPG:            return makePtr<SPG_P<Real>>(list);
      case TRUSTREGION_P_NCG:            return makePtr<NCG_P<Real>>(list);
      case TRUSTREGION_P_SPG2:           return makePtr<SPG2_P<Real>>(list);
      default:                           return nullPtr;
    }
  }
} // ROL

#endif
