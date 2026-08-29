// @HEADER
// *****************************************************************************
//               Rapid Optimization Library (ROL) Package
//
// Copyright 2014 NTESS and the ROL contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include "ROL_ParameterList.hpp"
#include "ROL_Stream.hpp"
#include "ROL_GlobalMPISession.hpp"

#include "ROL_StdVector.hpp"
#include "ROL_Types.hpp"

#include "ROL_l1Objective.hpp"
#include "ROL_STORMAlgorithm.hpp"
#include "ROL_StdObjective.hpp"
#include "ROL_BatchManager.hpp"
#include "ROL_MonteCarloGenerator.hpp"

#include "ROL_SampleGenerator.hpp"
#include "ROL_Distribution.hpp"
#include "ROL_Problem.hpp"

typedef double RealT;

template<class Real> 
class ParametrizedObjectiveEx1 : public ROL::Objective<Real> {
public:
  Real value( const ROL::Vector<Real> &x, Real &tol ) {
    ROL::Ptr<const std::vector<Real> > ex = 
      dynamic_cast<const ROL::StdVector<Real>&>(x).getVector();
    Real quad = 0.0, lin = 0.0;
    std::vector<Real> p = this->getParameter();   
    unsigned size = ex->size();
    for ( unsigned i = 0; i < size; i++ ) {
      quad += (*ex)[i]*(*ex)[i]; 
      lin  += (*ex)[i]*p[i+1];
    }
    return p[0]*quad + lin + p[size+1];
  }

  void gradient( ROL::Vector<Real> &g, const ROL::Vector<Real> &x, Real &tol ) {
    ROL::Ptr<const std::vector<Real> > ex = 
      dynamic_cast<const ROL::StdVector<Real>&>(x).getVector();
    ROL::Ptr<std::vector<Real> > eg =
      dynamic_cast<ROL::StdVector<Real>&>(g).getVector();
    std::vector<Real> p = this->getParameter();
    unsigned size = ex->size();
      for ( unsigned i = 0; i < size; i++ ) {
      (*eg)[i] = 2.0*p[0]*(*ex)[i] + p[i+1];
    }
  }

  void hessVec( ROL::Vector<Real> &hv, const ROL::Vector<Real> &v, const ROL::Vector<Real> &x, Real &tol ) {
    ROL::Ptr<const std::vector<Real> > ex = 
      dynamic_cast<const ROL::StdVector<Real>&>(x).getVector();
    ROL::Ptr<const std::vector<Real> > ev = 
      dynamic_cast<const ROL::StdVector<Real>&>(v).getVector();
    ROL::Ptr<std::vector<Real> > ehv =
      dynamic_cast<ROL::StdVector<Real>&>(hv).getVector();
    std::vector<Real> p = this->getParameter();
    unsigned size = ex->size();
    for ( unsigned i = 0; i < size; i++ ) {
      // (*ehv)[i] = 2.0*std::exp(p[0])*(*ev)[i]; 
      (*ehv)[i] = 2.0*p[0]*(*ev)[i]; 
    } 
  }
};

template<typename Real>
class MonteCarloGeneratorFiniteSamples : public ROL::MonteCarloGenerator<Real> {

private:
  int maxSamp_;

public:
  MonteCarloGeneratorFiniteSamples(int maxSamp,
                                   int nSamp,
                                   const std::vector<ROL::Ptr<ROL::Distribution<Real>>> &dist, 
                                   const ROL::Ptr<ROL::BatchManager<Real>> &bman, 
                                   bool use_SA = false,
                                   bool adaptive = false,
                                   int numNewSamps = 0,
                                   int seed = 123454321)
    : ROL::MonteCarloGenerator<Real>(nSamp, dist, bman, use_SA, adaptive, numNewSamps, seed),
      maxSamp_(maxSamp) {}      

  MonteCarloGeneratorFiniteSamples(int maxSamp,
                                   int nSamp,
                                   std::vector<std::vector<Real>> &bounds, 
                                   const ROL::Ptr<ROL::BatchManager<Real>> &bman,  
                                   bool use_SA = false,
                                   bool adaptive = false,
                                   int numNewSamps = 0,
                                   int seed = 123454321)
    : ROL::MonteCarloGenerator<Real>(nSamp, bounds, bman, use_SA, adaptive, numNewSamps, seed),
      maxSamp_(maxSamp) {}

  MonteCarloGeneratorFiniteSamples(int maxSamp,
                                   int nSamp,
                                   const std::vector<Real> &mean,
                                   const std::vector<Real> &std, 
                                   const ROL::Ptr<ROL::BatchManager<Real>> &bman,
                                   bool use_SA = false,
                                   bool adaptive = false,
                                   int numNewSamps = 0,
                                   int seed = 123454321)
    : ROL::MonteCarloGenerator<Real>(nSamp, mean, std, bman, use_SA, adaptive, numNewSamps, seed),
      maxSamp_(maxSamp) {}

  Real computeError( std::vector<Real> &vals ) {
    if (this->numMySamples() >= maxSamp_) {
      return static_cast<Real>(0.);
    } else {
      return ROL::MonteCarloGenerator<Real>::computeError(vals);
    }
  };

  Real computeError( std::vector<ROL::Ptr<ROL::Vector<Real>>> &vals, const ROL::Vector<Real> &x ) {
    if (this->numMySamples() >= maxSamp_) {
      return static_cast<Real>(0.);
    } else {
      return ROL::MonteCarloGenerator<Real>::computeError(vals, x);
    }
  };
};


void setRandomVector(std::vector<RealT> &x) {
  unsigned dim = x.size();
  for ( unsigned i = 0; i < dim; i++ ) {
    x[i] = (RealT)rand()/(RealT)RAND_MAX;
  }
}

void printSolution(const std::vector<RealT> &x,
                   std::ostream & outStream) {
// void printSolution(const ROL::StdVector<RealT> &x,
//                    std::ostream & outStream) {
  unsigned dim = x.size();
  outStream << "x = (";
  for ( unsigned i = 0; i < dim-1; i++ ) {
    outStream << x[i] << ", ";
  }
  outStream << x[dim-1] << ")\n";
}


void run_and_check_proxstorm(ROL::Ptr<ROL::Objective<RealT>> sObj,
                             ROL::Ptr<ROL::StdVector<RealT>> l1Wts,
                             unsigned dim, 
                             unsigned sdim,
                             std::string subproblemSolver, 
                             ROL::Ptr<ROL::ParameterList> & parlist, 
                             ROL::Ptr<std::ostream> & outStream,
                             int & errorFlag) {
  if (sdim - dim != 2) {
    throw std::invalid_argument("sdim should be equal to dim+2");
  }

  // Build vectors
  // Optimization variable
  ROL::Ptr<std::vector<RealT>> x_ptr  = ROL::makePtr<std::vector<RealT>>(dim,0.0);
  ROL::Ptr<ROL::Vector<RealT>> x      = ROL::makePtr<ROL::StdVector<RealT>>(x_ptr);
  // Variables for holding true solution
  ROL::Ptr<std::vector<RealT>> xr_ptr = ROL::makePtr<std::vector<RealT>>(dim,0.0);
  ROL::Ptr<ROL::Vector<RealT>> xr     = ROL::makePtr<ROL::StdVector<RealT>>(xr_ptr);
  ROL::Ptr<std::vector<RealT>> xg_ptr = ROL::makePtr<std::vector<RealT>>(dim,0.0);
  ROL::Ptr<ROL::Vector<RealT>> xg     = ROL::makePtr<ROL::StdVector<RealT>>(xg_ptr);
  // Vector to store difference between optimization variable and true solution
  ROL::Ptr<std::vector<RealT>> d_ptr  = ROL::makePtr<std::vector<RealT>>(dim,0.0);
  ROL::Ptr<ROL::Vector<RealT>> d      = ROL::makePtr<ROL::StdVector<RealT>>(d_ptr);

  // Build samplers
  int nSamp = 1000;
  int nSampPerRefinement = 1000;
  int maxSamp = 10000;

  ROL::Ptr<ROL::BatchManager<RealT> > bman = ROL::makePtr<ROL::BatchManager<RealT>>();

  std::vector<RealT> tmp(2,0.); tmp[0] = 0.5; tmp[1] = 1.5;
  std::vector<std::vector<RealT> > bounds(sdim,tmp);
  ROL::Ptr<ROL::SampleGenerator<RealT>> sampler = 
      ROL::makePtr<MonteCarloGeneratorFiniteSamples<RealT>>(
          maxSamp, nSamp, bounds, bman, false, true, nSampPerRefinement
      );

  // Set up problem and solve using STORM
  ROL::Ptr<ROL::Problem<RealT>> problem
    = ROL::makePtr<ROL::Problem<RealT>>(sObj,x);
  ROL::Ptr<ROL::l1Objective<RealT>> nObj = ROL::makePtr<ROL::l1Objective<RealT>>(
      l1Wts
  );
  problem->addProximableObjective(nObj);
  problem->finalize(false, true, *outStream);

  parlist->sublist("Step").sublist("Trust Region").set("Subproblem Solver", subproblemSolver);
  ROL::STORMAlgorithm<RealT> storm_solver(problem, sampler, *parlist);

  // Add check?
  // storm_solver.check(*outStream);
  storm_solver.run(*outStream);

  // Print solution found by Proxstorm
  *outStream << std::endl << "STORM Solution" << std::endl;
  printSolution(*x_ptr,*outStream);

  // Calculate true solution. For the weights in this example, the solution
  // will be negative and so |x_true| = -x_true
  RealT denom(0), denom_g(0);
  for (int i = 0; i < sampler->numMySamples(); ++i) {
    denom += sampler->getMyWeight(i)*(sampler->getMyPoint(i)[0]);
    for (unsigned j = 0; j < dim; ++j) {
      (*xr_ptr)[j] += sampler->getMyWeight(i)*sampler->getMyPoint(i)[j+1];
    }
  }
  sampler->sumAll(&denom,&denom_g,1);
  sampler->sumAll(*xr,*xg);
  for (unsigned j = 0; j < dim; ++j) {
    (*xg_ptr)[j] = ((*xg_ptr)[j] - (*l1Wts)[j]) / (-2.0*denom_g);
  }
  *outStream << std::endl << "True Solution" << std::endl;
  printSolution(*xg_ptr,*outStream);

  // Print difference
  d->set(*x); d->axpy(-1.0,*xg);
  RealT err = d->norm();
  *outStream << std::endl << "Error: " << err << std::endl << std::endl;
  errorFlag += (err > 1e-2 ? 1 : 0);

}

int main(int argc, char* argv[]) {

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
    /**********************************************************************************************/
    /************************* CONSTRUCT ROL ALGORITHM ********************************************/
    /**********************************************************************************************/
    // Get ROL parameterlist
    std::string filename = "input_18.xml";
//    ROL::RCP<ROL::ParameterList> parlist = ROL::rcp( new ROL::ParameterList() );
//    ROL::updateParametersFromXmlFile( filename, parlist.ptr() );
//    ROL::ParameterList list = *parlist;
      ROL::Ptr<ROL::ParameterList> parlist = ROL::getParametersFromXmlFile( filename );
    /**********************************************************************************************/
    /************************* CONSTRUCT SOL COMPONENTS *******************************************/
    /**********************************************************************************************/
    // Build vectors
    unsigned dim = 4;
    unsigned sdim = dim + 2;

    // Set up objective.
    ROL::Ptr<ROL::StdVector<RealT>> sol = ROL::makePtr<ROL::StdVector<RealT>>(dim,1.0);
    // Build stochastic, smooth objective function
    ROL::Ptr<ROL::Objective<RealT> > sObj =
      ROL::makePtr<ParametrizedObjectiveEx1<RealT>>();

    // Build weight vectors for nonsmooth but exact objective function    
    ROL::Ptr<std::vector<RealT>> wtsZero = ROL::makePtr<std::vector<RealT>>(dim, 0.);    
    ROL::Ptr<ROL::StdVector<RealT>> l1WtsZero = ROL::makePtr<ROL::StdVector<RealT>>(wtsZero);
    ROL::Ptr<std::vector<RealT>> wts = ROL::makePtr<std::vector<RealT>>(dim, 0.);    
    ROL::Ptr<ROL::StdVector<RealT>> l1Wts = ROL::makePtr<ROL::StdVector<RealT>>(wts);
    l1Wts->randomize(static_cast<RealT>(0),static_cast<RealT>(0.2));

    // Check derivatives of smooth problem
    std::vector<RealT> param = std::vector<RealT>(dim,0.0);    
    setRandomVector(param);
    *outStream << "Check Derivatives of Parametrized Smooth Objective Function\n";
    sObj->setParameter(param);

    ROL::Ptr<ROL::Vector<RealT>> xd = sol->clone();
    xd->randomize(-1.0,1.0);
    ROL::Ptr<ROL::Vector<RealT>> yd = sol->clone();
    yd->randomize(-1.0,1.0);
    ROL::Ptr<ROL::Vector<RealT>> zd = sol->clone();
    zd->randomize(-1.0,1.0);
    sObj->checkGradient(*xd,*yd,true,*outStream);
    sObj->checkHessVec(*xd,*yd,true,*outStream);
    sObj->checkHessSym(*xd,*yd,*zd,true,*outStream);

    // First check with a ``dummy'' L1 regularizer.
    std::cout << "Proxstorm with 'dummy' L1 regularizer (0 weight)";
    run_and_check_proxstorm(sObj, l1WtsZero, dim, sdim, "SPG", parlist, outStream, errorFlag);
    run_and_check_proxstorm(sObj, l1WtsZero, dim, sdim, "SPG2", parlist, outStream, errorFlag);
    run_and_check_proxstorm(sObj, l1WtsZero, dim, sdim, "NCG", parlist, outStream, errorFlag);

    // Then check with L1 regularizer with nonzero weights.
    std::cout << "Proxstorm with L1 regularizer";
    run_and_check_proxstorm(sObj, l1Wts, dim, sdim, "SPG", parlist, outStream, errorFlag);
    run_and_check_proxstorm(sObj, l1Wts, dim, sdim, "SPG2", parlist, outStream, errorFlag);
    run_and_check_proxstorm(sObj, l1Wts, dim, sdim, "NCG", parlist, outStream, errorFlag);
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
