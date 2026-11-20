// @HEADER
// *****************************************************************************
//               Rapid Optimization Library (ROL) Package
//
// Copyright 2014 NTESS and the ROL contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef ROL_ITERATIONPRINTER_H
#define ROL_ITERATIONPRINTER_H

/** \class ROL::IterationPrinter
    \brief Provides per-iterate diagnostic output for Krylov methods and
           trust-region subproblem solvers.
*/

#include <cmath>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

namespace ROL {

template<class Real>
class IterationPrinter {
private:

  const std::string name_;
  std::ostream &os_;
  const bool active_;
  const int ncol_;
  const bool flagColumn_;  // Append a trailing flag column

  // Widths and precision match the Type* algorithm tables.
  static constexpr int iwidth_ =  6;
  static constexpr int cwidth_ = 15;

public:

  // Value marking a column that does not apply to a given row.
  static Real none() { return std::numeric_limits<Real>::quiet_NaN(); }

  // Writes the header.
  IterationPrinter(const char *name, int verbosity,
                   std::initializer_list<const char*> labels,
                   std::ostream &outStream = std::cout, bool flagColumn = false)
    : name_(name), os_(outStream),
      active_(verbosity > 0),
      ncol_(static_cast<int>(labels.size())), flagColumn_(flagColumn) {
    if ( !active_ ) return;
    os_ << std::string(2+iwidth_*(1+flagColumn_)+cwidth_*ncol_,'-') << "\n";
    os_ << "  " << std::setw(iwidth_) << std::left << "iter";
    for (const char *label : labels) os_ << std::setw(cwidth_) << std::left << label;
    if ( flagColumn_ ) os_ << std::setw(iwidth_) << std::left << "flag";
    os_ << "\n";
  }

  // Values equal to none() and a negative flag print as "---".
  void writeRow(int iter, std::initializer_list<Real> values, int flag = -1) const {
    if ( !active_ ) return;
    std::ios_base::fmtflags f = os_.flags();
    std::streamsize p = os_.precision();
    os_ << "  " << std::setw(iwidth_) << std::left << iter;
    os_ << std::scientific << std::setprecision(6);
    for (Real value : values) {
      if ( std::isnan(value) ) os_ << std::setw(cwidth_) << std::left << "---";
      else                     os_ << std::setw(cwidth_) << std::left << value;
    }
    os_.flags(f);
    os_.precision(p);
    for (int i = static_cast<int>(values.size()); i < ncol_; ++i)
      os_ << std::setw(cwidth_) << std::left << "---";
    if ( flagColumn_ ) {
      if ( flag < 0 ) os_ << std::setw(iwidth_) << std::left << "---";
      else            os_ << std::setw(iwidth_) << std::left << flag;
    }
    os_ << "\n";
  }

  void writeSummary(int iter, int flag) const {
    if ( !active_ ) return;
    os_ << "  " << name_ << " done: iter=" << iter << " flag=" << flag << "\n" << std::endl;
  }
};

}

#endif
