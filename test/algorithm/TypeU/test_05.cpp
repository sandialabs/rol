// @HEADER
// *****************************************************************************
//               Rapid Optimization Library (ROL) Package
//
// Copyright 2014 NTESS and the ROL contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

/*! \file  test_05.cpp
    \brief Test trust-region subproblem solver per-iterate diagnostics
           (Verbosity parameter).
*/

#define USE_HESSVEC 1

#include "ROL_GetTestProblems.hpp"
#include "ROL_TypeU_TrustRegionAlgorithm.hpp"
#include "ROL_Stream.hpp"
#include "ROL_GlobalMPISession.hpp"

#include <cctype>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

typedef double RealT;

class CoutRedirect {
public:
  CoutRedirect(std::streambuf* to) : saved_(std::cout.rdbuf(to)) {}
  ~CoutRedirect() { std::cout.rdbuf(saved_); }
private:
  std::streambuf* saved_;
};

struct RunResult {
  std::string captured;
  RealT       finalNorm;
};

// A negative verbosity omits the parameter.
static RunResult runCaptured(int verbosity, RealT radius = -1,
                             int outerLimit = 3,
                             const char *solver = "Truncated CG") {
  auto parlist = ROL::makePtr<ROL::ParameterList>();
  parlist->sublist("General").set("Output Level", 0);
  parlist->sublist("General").sublist("Krylov").set("Iteration Limit", 20);
  parlist->sublist("Step").set("Type","Trust Region");
  auto &trlist = parlist->sublist("Step").sublist("Trust Region");
  if (verbosity >= 0) {
    parlist->sublist("General").sublist("Krylov").set("Verbosity", verbosity);
    trlist.sublist("SPG").sublist("Solver").set("Verbosity", verbosity);
  }
  trlist.set("Subproblem Solver",solver);
  if (radius > 0) trlist.set("Initial Radius", radius);
  parlist->sublist("Status Test").set("Iteration Limit", outerLimit);

  ROL::Ptr<ROL::OptimizationProblem<RealT>> problem;
  ROL::Ptr<ROL::Vector<RealT>> x0;
  std::vector<ROL::Ptr<ROL::Vector<RealT>>> z;
  ROL::GetTestProblem<RealT>(problem, x0, z, ROL::TESTOPTPROBLEM_ROSENBROCK);
  auto x = x0->clone(); x->set(*x0);

  std::stringstream captured;
  RunResult r;
  {
    CoutRedirect redirect(captured.rdbuf());
    auto algo = ROL::makePtr<ROL::TypeU::TrustRegionAlgorithm<RealT>>(*parlist);
    algo->run(*x, *problem->getObjective(), std::cout);
  }
  r.captured  = captured.str();
  r.finalNorm = x->norm();
  return r;
}

// Run to a dedicated stream (not std::cout) so a dropped outStream is caught.
static std::string runToOwnStream(int outerLevel, int verbosity,
                                  const char *solver = "Truncated CG") {
  auto parlist = ROL::makePtr<ROL::ParameterList>();
  parlist->sublist("General").set("Output Level", outerLevel);
  parlist->sublist("General").sublist("Krylov").set("Iteration Limit", 20);
  parlist->sublist("General").sublist("Krylov").set("Verbosity", verbosity);
  parlist->sublist("Step").set("Type", "Trust Region");
  parlist->sublist("Step").sublist("Trust Region").set("Subproblem Solver", solver);
  parlist->sublist("Status Test").set("Iteration Limit", 3);

  ROL::Ptr<ROL::OptimizationProblem<RealT>> problem;
  ROL::Ptr<ROL::Vector<RealT>> x0;
  std::vector<ROL::Ptr<ROL::Vector<RealT>>> z;
  ROL::GetTestProblem<RealT>(problem, x0, z, ROL::TESTOPTPROBLEM_ROSENBROCK);
  auto x = x0->clone(); x->set(*x0);

  std::ostringstream os;
  auto algo = ROL::makePtr<ROL::TypeU::TrustRegionAlgorithm<RealT>>(*parlist);
  algo->run(*x, *problem->getObjective(), os);
  return os.str();
}

// Extract the iteration column from each CG row.
static std::vector<int> extractIterColumn(const std::string& s) {
  std::vector<int> out;
  std::size_t pos = 0;
  while (pos < s.size()) {
    const std::size_t eol = s.find('\n', pos);
    const std::string line = s.substr(pos, eol - pos);
    if (line.size() >= 3 && line[0] == ' ' && line[1] == ' '
        && std::isdigit(static_cast<unsigned char>(line[2]))) {
      out.push_back(std::atoi(line.c_str() + 2));
    }
    if (eol == std::string::npos) break;
    pos = eol + 1;
  }
  return out;
}

int main(int argc, char *argv[]) {
  ROL::GlobalMPISession mpiSession(&argc, &argv);

  // This little trick lets us print to std::cout only if a (dummy) command-line argument is provided.
  int iprint     = argc - 1;
  ROL::Ptr<std::ostream> outStream;
  ROL::nullstream bhs; // outputs nothing
  if (iprint > 0)
    outStream = ROL::makePtrFromRef(std::cout);
  else
    outStream = ROL::makePtrFromRef(bhs);

  int errorFlag  = 0;

  try {
    const RunResult r_omitted = runCaptured(-1);
    if (!r_omitted.captured.empty()) {
      *outStream << "Omitted Verbosity produced diagnostic output" << std::endl;
      errorFlag += 1;
    }

    const RunResult r_v0 = runCaptured(0);
    if (!r_v0.captured.empty()) {
      *outStream << "Verbosity=0 produced diagnostic output" << std::endl;
      errorFlag += 1;
    }

    const RunResult r_v1 = runCaptured(1);
    if (r_v0.finalNorm != r_v1.finalNorm) {
      *outStream << "Diagnostic altered optimizer state ("
                 << r_v0.finalNorm << " vs " << r_v1.finalNorm << ")" << std::endl;
      errorFlag += 1;
    }

    if (r_v1.captured.find("TCG done: iter=") == std::string::npos) {
      *outStream << "Verbosity=1 missing 'TCG done: iter=' summary" << std::endl;
      errorFlag += 1;
    }
    const char* labels[] = {"iter", "rnorm", "snorm", "alpha", "pRed", "flag"};
    for (const char* lbl : labels) {
      if (r_v1.captured.find(lbl) == std::string::npos) {
        *outStream << "Verbosity=1 missing header label '" << lbl << "'" << std::endl;
        errorFlag += 1;
      }
    }

    // Each outer iteration starts a new 0,1,...,N sequence.
    const std::vector<int> iters = extractIterColumn(r_v1.captured);
    if (iters.empty()) {
      *outStream << "Verbosity=1 produced no CG rows" << std::endl;
      errorFlag += 1;
    }
    int expect = 0;
    for (std::size_t i = 0; i < iters.size(); ++i) {
      if (iters[i] != expect) {
        if (iters[i] == 0) { expect = 1; }
        else {
          *outStream << "CG row index gap at position " << i
                     << " (got " << iters[i] << ", expected " << expect << ")" << std::endl;
          errorFlag += 1;
          break;
        }
      } else {
        ++expect;
      }
    }

    const RunResult r_trunc = runCaptured(1, 1e-8, 1);
    const std::vector<int> trunc_iters = extractIterColumn(r_trunc.captured);
    if (r_trunc.captured.find("TCG done:") == std::string::npos
        || r_trunc.captured.find("flag=3") == std::string::npos
        || trunc_iters.size() != 2
        || trunc_iters[0] != 0 || trunc_iters[1] != 1) {
      *outStream << "Trust-region truncation diagnostic is malformed" << std::endl;
      errorFlag += 1;
    }

    const RunResult r_spg0 = runCaptured(0,-1,3,"SPG");
    const RunResult r_spg1 = runCaptured(1,-1,3,"SPG");
    if (!r_spg0.captured.empty()
        || r_spg0.finalNorm != r_spg1.finalNorm
        || r_spg1.captured.find("SPG done: iter=") == std::string::npos) {
      *outStream << "SPG subproblem diagnostic is malformed" << std::endl;
      errorFlag += 1;
    }

    // Outer trust-region table and inner subproblem table must share the driver stream.
    const std::string both = runToOwnStream(1, 1, "Truncated CG");
    if (both.find("iterCG") == std::string::npos
        || both.find("TCG done: iter=") == std::string::npos
        || both.find("rnorm") == std::string::npos) {
      *outStream << "Inner subproblem table did not follow the driver stream" << std::endl;
      errorFlag += 1;
    }

    std::cout.precision(3);
    std::cout.setf(std::ios::fixed);
    const std::ios_base::fmtflags before_flags = std::cout.flags();
    const std::streamsize before_prec = std::cout.precision();
    runCaptured(1);
    if (std::cout.flags() != before_flags || std::cout.precision() != before_prec) {
      *outStream << "Diagnostic leaked stream state" << std::endl;
      errorFlag += 1;
    }
    std::cout.unsetf(std::ios::fixed);
    std::cout.precision(6);

    *outStream << "Verbosity omitted captured " << r_omitted.captured.size() << " chars" << std::endl;
    *outStream << "Verbosity=0 captured "       << r_v0.captured.size()      << " chars" << std::endl;
    *outStream << "Verbosity=1 captured "       << r_v1.captured.size()      << " chars" << std::endl;
    *outStream << "CG rows: "                   << iters.size() << std::endl;
    *outStream << "SPG captured "               << r_spg1.captured.size() << " chars" << std::endl;

    *outStream << std::endl << "Truncated CG" << std::endl << r_v1.captured;
    *outStream << "SPG" << std::endl << r_spg1.captured;
    *outStream << "Outer + inner (Output Level 1)" << std::endl << both;
  }
  catch (std::logic_error& err) {
    *outStream << err.what() << std::endl;
    errorFlag = -1000;
  }; // end try

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED" << std::endl;
  else
    std::cout << "End Result: TEST PASSED" << std::endl;

  return 0;
}
