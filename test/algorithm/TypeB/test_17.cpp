// @HEADER
// *****************************************************************************
//               Rapid Optimization Library (ROL) Package
//
// Copyright 2014 NTESS and the ROL contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

/*! \file  test_17.cpp
    \brief Test TypeB Newton-Krylov and PDAS per-iterate diagnostics on a shared stream.
*/

#define USE_HESSVEC 1

#include "ROL_GetTestProblems.hpp"
#include "ROL_TypeB_NewtonKrylovAlgorithm.hpp"
#include "ROL_TypeB_PrimalDualActiveSetAlgorithm.hpp"
#include "ROL_Stream.hpp"
#include "ROL_GlobalMPISession.hpp"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

typedef double RealT;

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
    auto parlist = ROL::makePtr<ROL::ParameterList>();
    parlist->sublist("General").set("Output Level", 1);          // outer TypeB table
    auto &klist = parlist->sublist("General").sublist("Krylov");
    klist.set("Type", "Conjugate Gradients");
    klist.set("Verbosity", 1);                                   // inner Krylov table
    klist.set("Iteration Limit", 20);
    parlist->sublist("Status Test").set("Iteration Limit", 3);

    // First bound-constrained (TYPE_B) test problem.
    ROL::Ptr<ROL::OptimizationProblem<RealT>> problem;
    ROL::Ptr<ROL::Vector<RealT>> x0;
    std::vector<ROL::Ptr<ROL::Vector<RealT>>> z;
    for (ROL::ETestOptProblem p = ROL::TESTOPTPROBLEM_ROSENBROCK;
         p < ROL::TESTOPTPROBLEM_LAST; p++) {
      ROL::GetTestProblem<RealT>(problem, x0, z, p);
      if (problem->getProblemType() == ROL::TYPE_B) break;
    }
    if (problem->getProblemType() != ROL::TYPE_B) {
      *outStream << "No bound-constrained test problem found" << std::endl;
      errorFlag += 1;
    }

    // Diagnostics go to a dedicated stream (not std::cout) so a dropped outStream is caught.

    // Newton-Krylov: outer table interleaves with the inner CG table.
    {
      std::ostringstream diag;
      auto x = x0->clone(); x->set(*x0);
      ROL::TypeB::NewtonKrylovAlgorithm<RealT> algo(*parlist);
      algo.run(*x, *problem->getObjective(), *problem->getBoundConstraint(), diag);
      const std::string s = diag.str();
      if (s.find("iterCG") == std::string::npos
          || s.find("CG done: iter=") == std::string::npos
          || s.find("rnorm") == std::string::npos) {
        *outStream << "NewtonKrylov: inner Krylov table did not follow the driver stream"
                   << std::endl;
        errorFlag += 1;
      }
      *outStream << "NewtonKrylov" << std::endl << s;
    }

    // PDAS: outer table interleaves with its inner Krylov table.
    {
      std::ostringstream diag;
      auto x = x0->clone(); x->set(*x0);
      ROL::TypeB::PrimalDualActiveSetAlgorithm<RealT> algo(*parlist);
      algo.run(*x, *problem->getObjective(), *problem->getBoundConstraint(), diag);
      const std::string s = diag.str();
      if (s.find("feasible") == std::string::npos
          || s.find("done: iter=") == std::string::npos
          || s.find("prnorm") == std::string::npos) {
        *outStream << "PDAS: inner Krylov table did not follow the driver stream"
                   << std::endl;
        errorFlag += 1;
      }
      *outStream << "PDAS" << std::endl << s;
    }
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
