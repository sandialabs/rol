// @HEADER
// @HEADER

/*! \file  example_02.cpp
    \brief Logistic regression example with L1 penalty.
    Uses proxstorm algorithm to increase the amount of data as needed.
*/

#include "ROL_Bounds.hpp"
#include "ROL_StdObjective.hpp"
#include "ROL_StdConstraint.hpp"
#include "ROL_Stream.hpp"

#include "ROL_GlobalMPISession.hpp"

#include "ROL_TypeB_Algorithm.hpp"
#include "ROL_STORMAlgorithm.hpp"
#include "ROL_STORMAlgorithm_Def.hpp"
#include "ROL_TypeP_TrustRegionAlgorithm.hpp"
#include "ROL_TypeP_TrustRegionAlgorithm_Def.hpp"
#include "ROL_ScalarLinearConstraint.hpp"
#include "ROL_l1Objective.hpp"
#include "ROL_MonteCarloDataGenerator.hpp"


#include <iostream>
#include <random>
#include <fstream>
#include <vector>


typedef double RealT;

template<class Real>
class ParametrizedLogisticObjective : public ROL::StdObjective<Real> {
private:
  std::vector<Real> a_;
  Real b_;

public:
  ParametrizedLogisticObjective() {}

  virtual Real value(const std::vector<Real> &x, Real &tol) {
    // log (1 + exp(-(a.dot(xw) + xv).*b))
    Real val(x[0]);
    Real one(1);

    // The "parameter" is a data point, where all but the last entry is a
    // vector of features a and the last entry is the label b (a scalar).
    std::vector<Real> a_b = this->getParameter();
    Real b = a_b.back();
    int nfeatures = a_b.size() - 1;
    
    for (int i = 0; i < nfeatures; ++i) {
      val += a_b[i]*x[1+i];
    }
    val *= -b;
    val = std::log(one + std::exp(val));
    return val;
  } 

  // Todo: implement gradient
  void gradient(std::vector<Real> &g, const std::vector<Real> &x, Real &tol) {
    Real tmp(0);  
    Real one(1); 

    std::vector<Real> a_b = this->getParameter();
    Real b = a_b.back();
    int nfeatures = a_b.size() - 1;

    g.assign(x.size(),static_cast<Real>(0));
    // x[0] = -t * b /(1 + t)
    // x[1:end] = -a.*(t*b/(1 + t)) for t = exp(-b(a.dot(x) + xv))

    tmp = x[0];
    for (int i = 0; i<nfeatures; ++i) {
      tmp += a_b[i] * x[1+i];
    }
    tmp = std::exp(-b * tmp); // exp(-b(a.dot(x) + xv))
    tmp = -tmp * b / (one+tmp); // -t * b /(1 + t)

    g[0] = tmp;
    for (int i=1; i<nfeatures+1; ++i) {
      g[i] = tmp*a_b[i-1];
    }
  }

  void hessVec(std::vector<Real> &hv, const std::vector<Real> &v, const std::vector<Real> &x, Real &tol) {
    hv.assign(x.size(),static_cast<Real>(0));
    Real one(1);
    Real sv(0);
    Real tmp(x[0]);
    Real tmpv(v[0]);

    std::vector<Real> a_b = this->getParameter();
    Real b = a_b.back();
    int nfeatures = a_b.size() - 1;

    for (int i=0; i<nfeatures; ++i) {
      tmp += a_b[i]*x[i+1];
      tmpv += a_b[i]*v[1+i];
    }
    tmp *= -b;
    tmp = std::exp(tmp);
    tmp = (tmpv*b*b*tmp) / ((one + tmp) * (one + tmp));    
    for (int i = 0; i < x.size(); ++i) {
      if (i == 0) hv[i] = tmp;
      else {
        hv[i] += a_b[i-1]*tmp;
      }
    }
  }
};


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

  // *** Example body.
 
  try {
    int dim = 68;
    /*** Read in XML input ***/
    std::string filename = "input.xml";
    auto parlist = ROL::getParametersFromXmlFile(filename);

    //ROL::Ptr<ROL::ParameterList> parlist = ROL::getParametersFromXmlFile("input.xml");
    //ROL::ParameterList list;
    parlist->sublist("General").set("Output Level",iprint);
    parlist->sublist("Step").set("Type","STORM");
    parlist->sublist("Status Test").set("Gradient Tolerance",1e-7);
    parlist->sublist("Status Test").set("Constraint Tolerance",1e-8);
    parlist->sublist("Status Test").set("Step Tolerance",1e-12);
    parlist->sublist("Status Test").set("Iteration Limit", 10000);
    parlist->sublist("Step").sublist("Trust Region").set("Subproblem Solver", "SPG");  
    
    // set up objective
    ROL::Ptr<ParametrizedLogisticObjective<RealT>>  obj = ROL::makePtr<ParametrizedLogisticObjective<RealT>>();
    ROL::Ptr<ROL::l1Objective<RealT>> nobj; 
    ROL::Ptr<ROL::StdVector<RealT>> wts = ROL::makePtr<ROL::StdVector<RealT>>(dim, 0);
    ROL::Ptr<ROL::StdVector<RealT>> x = ROL::makePtr<ROL::StdVector<RealT>>(dim, 0);
    ROL::Ptr<ROL::StdVector<RealT>> xy = ROL::makePtr<ROL::StdVector<RealT>>(dim, 0);
    wts->setScalar(0.05); // Note: A relatively small weight is needed to get a solution that isn't just zero.
    x->randomize();
    nobj = ROL::makePtr<ROL::l1Objective<RealT>>(wts); 

    // Set up samplers
    int nSamp = 10000;
    int nNewSamp = 1000;
    std::string file = "data_with_labels.txt";
    ROL::Ptr<ROL::BatchManager<RealT> > bman =
      ROL::makePtr<ROL::BatchManager<RealT>>();
    ROL::Ptr<ROL::SampleGenerator<RealT> > sampler =
      ROL::makePtr<ROL::MonteCarloDataGenerator<RealT>>(
        nSamp,
        dim+1, // Dim is the number of features. We need dim+1 because 
        // data_with_labels.txt contains the features and the label.
        file,
        bman,
        true,
        nNewSamp
      );

    // Check derivatives of parametrized objective function
    bool checkDeriv = parlist->sublist("Problem").get("Check Derivatives", true);
    if ( checkDeriv) {
      *outStream << "Check Derivatives of Parametrized Objective Function\n";
      ROL::Ptr<ROL::StdVector<RealT>> dx = ROL::makePtr<ROL::StdVector<RealT>>(dim,0);
      ROL::Ptr<ROL::StdVector<RealT>> hx = ROL::makePtr<ROL::StdVector<RealT>>(dim,0);
      dx->randomize(); hx->randomize();  
 
      obj->setParameter(sampler->getMyPoint(0));
      obj->checkGradient(*x,*dx,true,*outStream);
      obj->checkHessVec(*x,*dx,true,*outStream);
      obj->checkHessSym(*x,*dx,*hx,true,*outStream);
    }

    // Set up and run proxstorm
    ROL::Ptr<ROL::Problem<RealT>> problem = ROL::makePtr<ROL::Problem<RealT>>(
        obj, x
    );
    problem->addProximableObjective(nobj);
    problem->finalize(false, true, *outStream);

    ROL::Ptr<ROL::STORMAlgorithm<RealT>> algo_storm = ROL::makePtr<ROL::STORMAlgorithm<RealT>>(
      problem,
      sampler,
      *parlist
    );
    algo_storm->run(*outStream); // storm

    *outStream << "x = ";
    for (int i=0; i<dim; ++i) {
      *outStream << (*x)[i] << " ";
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
