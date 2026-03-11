// @HEADER
// *****************************************************************************
//               Rapid Optimization Library (ROL) Package
//
// Copyright 2014 NTESS and the ROL contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef ROL_PARTITIONEDBATCHMANAGER_HPP
#define ROL_PARTITIONEDBATCHMANAGER_HPP

#include "ROL_BatchManager.hpp"
#include "ROL_PartitionedVector.hpp"

namespace ROL {


template<class Real>
class PartitionedBatchManager : public BatchManager<Real> {
private:
  ROL::Ptr<BatchManager<Real>> bman_;

public:

  PartitionedBatchManager(const ROL::Ptr<BatchManager<Real>> &bman)
    : bman_(bman) {}

  virtual int batchID() {
    return bman_->batchID();
  }

  virtual int numBatches() {
    return bman_->numBatches();
  }

  virtual void sumAll(Vector<Real> &input, Vector<Real> &output) {
    PartitionedVector<Real>& ivec = static_cast<PartitionedVector<Real>&>(input);
    PartitionedVector<Real>& ovec = static_cast<PartitionedVector<Real>&>(output);

    size_t ilength = ivec.numVectors(), olength = ovec.numVectors();
    ROL_TEST_FOR_EXCEPTION(ilength != olength, std::invalid_argument,
      ">>> (PartitionedTeuchosBatchManager::sumAll): Inconsistent local number of lengths!");

    for (int i=0; i<ilength; ++i) {
      bman_->sumAll(*(ivec.get(i)),*(ovec.get(i)));
    }
  }

};


}

#endif
