// @HEADER
// *****************************************************************************
//               Rapid Optimization Library (ROL) Package
//
// Copyright 2014 NTESS and the ROL contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef ROL_MONTECARLODATAGENERATORDEF_HPP
#define ROL_MONTECARLODATAGENERATORDEF_HPP

#include <limits>


namespace ROL {

template<typename Real>
std::vector<std::vector<Real>> MonteCarloDataGenerator<Real>::sample(int n) {
  std::fstream input_pt;

  input_pt.open(file_.c_str(),std::ios::in);
  if ( !input_pt.is_open() ) {
    if ( SampleGenerator<Real>::batchID() == 0 ) {
        std::cout << "CANNOT OPEN " << file_.c_str() << std::endl;
    }
    input_pt.close();
    return std::vector<std::vector<Real>>();
  }
  else {
    // Skip until currentLine_
    n = std::min(nSampTotal_-currentLine_, n);
    for (int i=0;i<currentLine_;++i) {
      input_pt.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    std::vector<std::vector<Real>> pt(n);
    std::vector<Real> point(dim_,0.0);
    for (int i = 0; i < n ; i++) {
      for (int j = 0; j < dim_; j++) {
        input_pt >> point[j];
      } 
      pt[i] = point;
    }
    // Get process rankd and number of processes
    int rank  = SampleGenerator<Real>::batchID();
    int nProc = SampleGenerator<Real>::numBatches();
    // Separate samples across processes
    int frac = n/nProc;
    int rem  = n%nProc;
    int N    = frac;
    if ( rank < rem ) N++;
    std::vector<std::vector<Real>> my_pt(N);
    int index = 0;
    for (int i = 0; i < N; i++) {
      index = i*nProc + rank;
      my_pt[i] = pt[index];  
    }
    input_pt.close();
    currentLine_ += n;
    nSamp_ += n;
    return my_pt;
  }
}

template<typename Real>
void MonteCarloDataGenerator<Real>::update( const Vector<Real> &x ) {
  SampleGenerator<Real>::update(x);
  sum_val_  = static_cast<Real>(0);
  sum_val2_ = static_cast<Real>(0);
  sum_ng_   = static_cast<Real>(0);
  sum_ng2_  = static_cast<Real>(0);
}

template<typename Real>
Real MonteCarloDataGenerator<Real>::computeError( std::vector<Real> &vals ) {
  Real err(0);
  if ( adaptive_ && nSamp_ < nSampTotal_ ) {
    const Real zero(0), one(1);
    // Compute unbiased sample variance
    int cnt = 0;
    for ( int i = SampleGenerator<Real>::start(); i < SampleGenerator<Real>::numMySamples(); ++i ) {
      sum_val_  += vals[cnt];
      sum_val2_ += vals[cnt]*vals[cnt];
      cnt++;
    }
    Real mymean = sum_val_ / static_cast<Real>(nSamp_);
    Real mean   = zero;
    SampleGenerator<Real>::sumAll(&mymean,&mean,1);

    Real myvar  = (sum_val2_ - static_cast<Real>(nSamp_)*mean*mean)/(static_cast<Real>(nSamp_)-one);
    Real var    = zero;
    SampleGenerator<Real>::sumAll(&myvar,&var,1);
    // Return Monte Carlo error
    vals.clear();
    err = std::sqrt(var/static_cast<Real>(nSamp_));
  }
  else {
    vals.clear();
  }
  return err;
}

template<typename Real>
Real MonteCarloDataGenerator<Real>::computeError( std::vector<Ptr<Vector<Real>>> &vals,
                                              const Vector<Real> &x ) {
  Real err(0);
  if ( adaptive_ && nSamp_ < nSampTotal_) {
    const Real zero(0), one(1);
    // Compute unbiased sample variance
    int cnt = 0;
    Real ng = zero;
    for ( int i = SampleGenerator<Real>::start(); i < SampleGenerator<Real>::numMySamples(); ++i ) {
      ng = (vals[cnt])->norm();
      sum_ng_  += ng;
      sum_ng2_ += ng*ng;
      cnt++;
    }
    Real mymean = sum_ng_ / static_cast<Real>(nSamp_);
    Real mean   = zero;
    SampleGenerator<Real>::sumAll(&mymean,&mean,1);

    Real myvar  = (sum_ng2_ - static_cast<Real>(nSamp_)*mean*mean)/(static_cast<Real>(nSamp_)-one);
    Real var    = zero;
    SampleGenerator<Real>::sumAll(&myvar,&var,1);
    // Return Monte Carlo error
    vals.clear();
    err = std::sqrt(var/static_cast<Real>(nSamp_));
  }
  else {
    vals.clear();
  }
  return err;
}

template<typename Real>
MonteCarloDataGenerator<Real>::MonteCarloDataGenerator(int nSampInitial,
                                                       int dim, 
                                                       std::string file,
                                                       const Ptr<BatchManager<Real>> &bman, 
                                                       bool adaptive,
                                                       int nNewSamp)
  : SampleGenerator<Real>(bman),
    file_(file),
    sum_val_(0),
    sum_val2_(0),
    sum_ng_(0),
    sum_ng2_(0),
    nSamp_(0),
    dim_(dim),
    currentLine_(0),
    adaptive_(adaptive),
    nNewSamp_(nNewSamp),
    nSampTotal_(0) {
  // Count number of lines to see total number of data samples.
  std::fstream input_pt;
  input_pt.open(file_.c_str(),std::ios::in);
  int line_count = 0;
  std::string line;
  while (std::getline(input_pt, line)) {
    line_count++;
  }
  nSampTotal_ += line_count;
  input_pt.close();

  std::vector<std::vector<Real>> pts = sample(nSampInitial);
  std::vector<Real> wts(pts.size(),static_cast<Real>(1)/static_cast<Real>(nSamp_));
  SampleGenerator<Real>::setPoints(pts);
  SampleGenerator<Real>::setWeights(wts);
}

template<typename Real>
void MonteCarloDataGenerator<Real>::refine(void) {
  if ( adaptive_ ) {
    std::vector<std::vector<Real>> pts;
    for ( int i = 0; i < SampleGenerator<Real>::numMySamples(); ++i )
      pts.push_back(SampleGenerator<Real>::getMyPoint(i));
    std::vector<std::vector<Real>> pts_new = sample(nNewSamp_);
    pts.insert(pts.end(),pts_new.begin(),pts_new.end());
    nSamp_ = std::min(nSamp_+nNewSamp_, nSampTotal_);
    std::vector<Real> wts(pts.size(),static_cast<Real>(1)/static_cast<Real>(nSamp_));
    SampleGenerator<Real>::refine();
    SampleGenerator<Real>::setPoints(pts);
    SampleGenerator<Real>::setWeights(wts);
  }
}

template<typename Real>
int MonteCarloDataGenerator<Real>::numGlobalSamples(void) const {
  return nSamp_;
}

}
#endif
