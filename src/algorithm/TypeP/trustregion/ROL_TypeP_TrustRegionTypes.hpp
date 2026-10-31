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

#ifndef ROL_TYPEP_TRUSTREGIONTYPES_HPP
#define ROL_TYPEP_TRUSTREGIONTYPES_HPP

namespace ROL {
namespace TypeP {

  /** \enum   ROL::ETrustRegionP
      \brief  Enumeration of trust-region solver types.
      \arg    TRUSTREGION_P_CAUCHYPOINT describe
      \arg    TRUSTREGION_P_SPG         describe
      \arg    TRUSTREGION_P_NCG         describe
      \arg    TRUSTREGION_P_SPG2        describe
   */

  enum ETrustRegionP{
    TRUSTREGION_P_CAUCHYPOINT = 0,
    TRUSTREGION_P_SPG,
    TRUSTREGION_P_SPG2,
    TRUSTREGION_P_NCG,
    TRUSTREGION_P_LAST
  };

  inline std::string ETrustRegionPToString(ETrustRegionP tr) {
    std::string retString;
    switch(tr) {
      case TRUSTREGION_P_CAUCHYPOINT:  retString = "Cauchy Point";      break;
      case TRUSTREGION_P_SPG:          retString = "SPG";               break;
      case TRUSTREGION_P_SPG2:         retString = "SPG2";              break;
      case TRUSTREGION_P_NCG:          retString = "Truncated CG";      break;
      case TRUSTREGION_P_LAST:         retString = "Last Type (Dummy)"; break;
      default:                         retString = "INVALID ETrustRegionP";
    }
    return retString;
  }

  /** \brief  Verifies validity of a TrustRegionP enum.

      \param  tr  [in]  - enum of the TrustRegionP
      \return 1 if the argument is a valid TrustRegionP; 0 otherwise.
    */
  inline int isValidTrustRegionP(ETrustRegionP ls){
    return( (ls == TRUSTREGION_P_CAUCHYPOINT) ||
            (ls == TRUSTREGION_P_SPG)         ||
            (ls == TRUSTREGION_P_SPG2)        ||
            (ls == TRUSTREGION_P_NCG)         ||
            (ls == TRUSTREGION_P_LAST)
          );
  }

  inline ETrustRegionP & operator++(ETrustRegionP &type) {
    return type = static_cast<ETrustRegionP>(type+1);
  }

  inline ETrustRegionP operator++(ETrustRegionP &type, int) {
    ETrustRegionP oldval = type;
    ++type;
    return oldval;
  }

  inline ETrustRegionP & operator--(ETrustRegionP &type) {
    return type = static_cast<ETrustRegionP>(type-1);
  }

  inline ETrustRegionP operator--(ETrustRegionP &type, int) {
    ETrustRegionP oldval = type;
    --type;
    return oldval;
  }

  inline ETrustRegionP StringToETrustRegionP(std::string s) {
    s = removeStringFormat(s);
    if (s == removeStringFormat("NCG")) {
      return TRUSTREGION_P_NCG;
    }
    if (s == removeStringFormat("Simplified SPG")) {
      return TRUSTREGION_P_SPG2;
    }
    for ( ETrustRegionP tr = TRUSTREGION_P_CAUCHYPOINT; tr < TRUSTREGION_P_LAST; tr++ ) {
      if ( !s.compare(removeStringFormat(ETrustRegionPToString(tr))) ) {
        return tr;
      }
    }
    return TRUSTREGION_P_CAUCHYPOINT;
  }
} // namespace TypeP
} // namespace ROL

#endif
