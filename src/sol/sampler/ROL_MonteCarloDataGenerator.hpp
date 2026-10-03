// @HEADER
// *****************************************************************************
//               Rapid Optimization Library (ROL) Package
//
// Copyright 2014 NTESS and the ROL contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef ROL_MONTECARLODATAGENERATOR_HPP
#define ROL_MONTECARLODATAGENERATOR_HPP

#include "ROL_SampleGenerator.hpp"
#include "ROL_Distribution.hpp"

namespace ROL {

template<typename Real>
class MonteCarloDataGenerator : public SampleGenerator<Real> {
private: 
  std::string file_;
  
  Real sum_val_;
  Real sum_val2_;
  Real sum_ng_;
  Real sum_ng2_;

protected:
  int nSamp_;
  int dim_;
  int currentLine_;
  const bool adaptive_;
  const int  nNewSamp_; 
  int nSampTotal_;

  virtual std::vector<std::vector<Real>> sample(int nSamp);

public:
  MonteCarloDataGenerator(int nSamp,
                          int dim,
                          std::string file,
                          const Ptr<BatchManager<Real>> &bman, 
                          bool adaptive = false,
                          int numNewSamps = 0);
  void update( const Vector<Real> &x ) override;
  Real computeError( std::vector<Real> &vals ) override;
  Real computeError( std::vector<Ptr<Vector<Real>>> &vals, const Vector<Real> &x ) override;
  void refine(void) override;
  int numGlobalSamples(void) const override;
};

}

#include "ROL_MonteCarloDataGeneratorDef.hpp"

#endif
