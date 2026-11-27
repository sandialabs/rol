// @HEADER
// *****************************************************************************
//               Rapid Optimization Library (ROL) Package
//
// Copyright 2014 NTESS and the ROL contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

/*! \file  test_04.cpp
    \brief Test Krylov solver per-iterate diagnostics
*/

#include "ROL_KrylovFactory.hpp"
#include "ROL_StdTridiagonalOperator.hpp"
#include "ROL_DiagonalOperator.hpp"
#include "ROL_StdVector.hpp"
#include "ROL_ParameterList.hpp"

#include "ROL_Stream.hpp"
#include "ROL_GlobalMPISession.hpp"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

typedef double RealT;

class CoutRedirect {
public:
  CoutRedirect(std::streambuf* to) : saved_(std::cout.rdbuf(to)) {}
  ~CoutRedirect() { std::cout.rdbuf(saved_); }
private:
  std::streambuf* saved_;
}; // class CoutRedirect

struct RunResult {
  std::string captured;
  RealT       xnorm;
};

// A negative verbosity omits the parameter.
static RunResult runCaptured(const std::string &type, int verbosity) {
  ROL::ParameterList parlist;
  ROL::ParameterList &klist = parlist.sublist("General").sublist("Krylov");
  klist.set("Type",               type);
  klist.set("Iteration Limit",    20);
  klist.set("Absolute Tolerance", 1.e-12);
  klist.set("Relative Tolerance", 1.e-10);
  if (verbosity >= 0) klist.set("Verbosity", verbosity);

  // 1D Laplacian, symmetric positive definite.
  const int dim = 10;
  ROL::Ptr<std::vector<RealT>> diag = ROL::makePtr<std::vector<RealT>>(dim, 2.0);
  ROL::Ptr<std::vector<RealT>> offd = ROL::makePtr<std::vector<RealT>>(dim,-1.0);
  ROL::StdTridiagonalOperator<RealT> A(diag,offd,offd);
  // Diagonal preconditioner so CR's prnorm row differs from ||b||.
  ROL::StdVector<RealT> mdiag(ROL::makePtr<std::vector<RealT>>(dim, 2.0));
  ROL::DiagonalOperator<RealT> M(mdiag);

  ROL::StdVector<RealT> x(ROL::makePtr<std::vector<RealT>>(dim,0.0));
  ROL::StdVector<RealT> b(ROL::makePtr<std::vector<RealT>>(dim,1.0));

  int iter(0), flag(0);
  std::stringstream captured;
  {
    CoutRedirect redirect(captured.rdbuf());
    ROL::KrylovFactory<RealT>(parlist)->run(x,A,b,M,iter,flag);
  }
  RunResult result;
  result.captured = captured.str();
  result.xnorm    = x.norm();
  return result;
}

int main(int argc, char *argv[]) {
  ROL::GlobalMPISession mpiSession(&argc, &argv);

  int iprint     = argc - 1;
  ROL::Ptr<std::ostream> outStream;
  ROL::nullstream bhs; // outputs nothing
  if (iprint > 0)
    outStream = ROL::makePtrFromRef(std::cout);
  else
    outStream = ROL::makePtrFromRef(bhs);

  int errorFlag = 0;

  try {

    const std::string types[] = {"Conjugate Gradients", "Conjugate Residuals",
                                 "GMRES", "MINRES", "BiCGSTAB"};
    const std::string names[] = {"CG done: iter=", "CR done: iter=",
                                 "GMRES done: iter=", "MINRES done: iter=",
                                 "BiCGSTAB done: iter="};
    const std::string cols[]  = {"pAp", "prnorm", "rnorm", "rnorm", "omega"};

    for (int k = 0; k < 5; ++k) {
      const RunResult omitted = runCaptured(types[k],-1);
      const RunResult v0      = runCaptured(types[k], 0);
      const RunResult v1      = runCaptured(types[k], 1);

      if (!omitted.captured.empty() || !v0.captured.empty()) {
        *outStream << types[k] << ": produced output without Verbosity\n";
        errorFlag += 1;
      }
      if (v0.xnorm != v1.xnorm) {
        *outStream << types[k] << ": diagnostic altered the solution ("
                   << v0.xnorm << " vs " << v1.xnorm << ")\n";
        errorFlag += 1;
      }
      const std::string labels[] = {"iter", cols[k], names[k]};
      for (const std::string &label : labels) {
        if (v1.captured.find(label) == std::string::npos) {
          *outStream << types[k] << ": missing '" << label << "'\n";
          errorFlag += 1;
        }
      }
      *outStream << types[k] << "\n" << v1.captured;
    }

  }
  catch (std::logic_error& err) {
    *outStream << err.what() << "\n";
    errorFlag = -1000;
  }; // end try

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;
}
