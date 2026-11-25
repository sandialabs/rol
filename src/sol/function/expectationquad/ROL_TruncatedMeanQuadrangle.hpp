// @HEADER
// *****************************************************************************
//               Rapid Optimization Library (ROL) Package
//
// Copyright 2014 NTESS and the ROL contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef ROL_TRUNCATEDMEANQUAD_HPP
#define ROL_TRUNCATEDMEANQUAD_HPP

#include "ROL_ExpectationQuad.hpp"

namespace ROL {

template<class Real>
class TruncatedMeanQuadrangle : public ExpectationQuad<Real> {
private:

  Real beta_;

  void parseParameterList(ROL::ParameterList &parlist) {
    std::string type = parlist.sublist("SOL").get("Type","Risk Averse");
    ROL::ParameterList list;
    if (type == "Risk Averse") {
      list = parlist.sublist("SOL").sublist("Risk Measure").sublist("Truncated Mean");
    }
    else if (type == "Error") {
      list = parlist.sublist("SOL").sublist("Error Measure").sublist("Huber");
    }
    else if (type == "Deviation") {
      list = parlist.sublist("SOL").sublist("Deviation Measure").sublist("Truncated Mean");
    }
    else if (type == "Regret") {
      list = parlist.sublist("SOL").sublist("Regret Measure").sublist("Truncated Mean");
    }
    beta_ = list.get<Real>("Threshold");
  }

  void checkInputs(void) const {
    Real zero(0);
    ROL_TEST_FOR_EXCEPTION((beta_ <= zero), std::invalid_argument,
      ">>> ERROR (ROL::TruncatedMeanQuadrangle): Threshold must be positive!");
  }

public:

  TruncatedMeanQuadrangle(Real beta)
    : ExpectationQuad<Real>(), beta_(beta) {
    checkInputs();
  }

  TruncatedMeanQuadrangle(ROL::ParameterList &parlist)
    : ExpectationQuad<Real>() {
    parseParameterList(parlist);
    checkInputs();
  }

  Real error(Real x, int deriv = 0) {
    bool inside = ( std::abs(x) < beta_ ? true : false );
    Real err(0), zero(0), half(0.5), one(1), two(2);
    if (deriv==0) {
      err = (inside ? half*std::pow(x,two)/beta_ : std::abs(x)-half*beta_);
    }
    else if (deriv==1) {
      err = (inside ? x/beta_ : ((zero < x) - (x < zero)));
    }
    else {
      err = (inside ? one/beta_ : zero);
    }
    return err;
  }

  Real regret(Real x, int deriv = 0) {
    Real zero(0), one(1);
    Real X = ((deriv==0) ? x : ((deriv==1) ? one : zero));
    Real reg = error(x,deriv) + X;
    return reg;
  }

  void check(void) {
    ExpectationQuad<Real>::check();
    std::vector<Real> X = {beta_, -beta_};
    std::vector<std::string> name = {"beta", "-beta"};
    ExpectationQuad<Real>::check(X,name,1);
  }

};

}
#endif
