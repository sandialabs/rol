// @HEADER
// @HEADER

/*! \file  example_01.cpp
    \brief Logistic regression example with L1 penalty.
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


#include <iostream>
#include <random>
#include <fstream>
#include <vector>


typedef double RealT;

template<class Real>
class LogisticObjective : public ROL::StdObjective<Real> {
private:
  const std::vector<std::vector<Real>> A_;
  const std::vector<Real> b_; 
  std::vector<Real> p_; 
  const int nrows_;
  const int ncols_; 

public:
  LogisticObjective(const std::vector<std::vector<Real>> &A, const std::vector<Real> &b) : A_(A), b_(b), p_(b), nrows_(A.size()), ncols_(A[0].size()) {} 

  virtual Real value(const std::vector<Real> &x, Real &tol) {
    // sum(log (1 + exp(-(A * xw + xv).*b)))/N
    Real val(0);
    Real N(nrows_);  
    Real one(1);
    Real tmp(0);
    for (int i = 0; i < nrows_; ++i) {
      tmp = x[0]; 
      for (int j = 0; j < ncols_; ++j) {
        tmp += A_[i][j]*x[1+j];
      }
      tmp *= -b_[i];
      val += std::log(one + std::exp(tmp));
    }
    val /= N; 
    return val;
  } 

  void gradient(std::vector<Real> &g, const std::vector<Real> &x, Real &tol) {
    Real one(1);
    Real sv(0); 
    Real N(nrows_);  
    Real tmp(0);  
    Real tmp2(0);  
    g.assign(x.size(),static_cast<Real>(0));
    // x[0] = -sum(t .* b ./(1 + t))
    // x[1:end] = -A'*(t.*b./(1 + t)) for t = exp(-b .* (A *x + xv))
    for (int i = 0; i < nrows_; ++i) {
      tmp = x[0]; 
      for (int j = 0; j < ncols_; ++j) {
        tmp += A_[i][j]*x[1+j];
      }
      tmp   *= -b_[i];
      tmp2   = std::exp(tmp);
      p_[i]  = (tmp2*b_[i])/(one + tmp2);
      sv    += p_[i]; 
   }// row for loop
   g[0] = -sv/N; 
   for (unsigned int i = 1; i<x.size(); ++i){
     for (int j=0; j<nrows_; ++j) {
       g[i] += p_[j]*(-A_[j][i-1]); 
     } //for
     g[i] /= N;  
   } // g for loop
  }

  void hessVec(std::vector<Real> &hv, const std::vector<Real> &v, const std::vector<Real> &x, Real &tol) {
    hv.assign(x.size(),static_cast<Real>(0));
    Real one(1); 
    Real sv(0); 
    Real N(nrows_);  
    Real tmp(0);  
    Real tmp2(0); 
    Real tmpv(0); 
    for (int i = 0; i < nrows_; ++i) {
      tmp  = x[0]; 
      tmpv = v[0]; 
      for (int j = 0; j < ncols_; ++j) { 
        tmp  += A_[i][j]*x[1+j]; 
        tmpv += A_[i][j]*v[1+j]; 
      } // end col for
      tmp  *= -b_[i]; 
      tmp2  = std::exp(tmp); 
      p_[i] = (tmpv*b_[i]*b_[i]*tmp2) / ((one + tmp2)*(one + tmp2));
      sv += p_[i];  
    } // end row for
    
   for (unsigned int i = 0; i<x.size(); ++i){
     if (i == 0) hv[i] = sv/N; 
     else{
       for (int j=0; j<nrows_; ++j) {
         hv[i] += A_[j][i-1]*p_[j]; 
       } //for
       hv[i] /= N; 
     }// else 
   } // g for loop
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

    /*** Read in XML input ***/
    std::string filename = "input.xml";
    auto parlist = ROL::getParametersFromXmlFile(filename);

    // initialize data
    std::vector<RealT> b;
    std::vector<std::vector<RealT>> A; 
    int nVars, nInstances; 
    //input data
    std::ifstream fileD, fileX, fileY;
    fileD.open("info.txt");
    fileX.open("data_matrix.txt");
    fileY.open("label_vector.txt");
    fileD >> nInstances; 
    fileD >> nVars; 
    A.resize(nInstances);
    b.resize(nInstances,0.0);
    for (int i = 0; i < nInstances; ++i) {
      fileY >> b[i];
      A[i].resize(nVars);
      for (int j = 0; j < nVars; ++j) fileX >> A[i][j];
    }
    fileD.close();
    fileX.close();
    fileY.close();
    
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
    int dim = nVars + 1 ; 
    ROL::Ptr<LogisticObjective<RealT>>  obj = ROL::makePtr<LogisticObjective<RealT>>(A,b);
    ROL::Ptr<ROL::l1Objective<RealT>> nobj; 
    ROL::Ptr<ROL::StdVector<RealT>> wts = ROL::makePtr<ROL::StdVector<RealT>>(dim, 0); 
    ROL::Ptr<ROL::StdVector<RealT>> x = ROL::makePtr<ROL::StdVector<RealT>>(dim, 0); 
    ROL::Ptr<ROL::StdVector<RealT>> xy = ROL::makePtr<ROL::StdVector<RealT>>(dim, 0); 
    wts->setScalar(0.05); // Note: A relatively small weight is needed to get a solution that isn't just zero.
    x->randomize();
    xy->randomize(); 
    nobj = ROL::makePtr<ROL::l1Objective<RealT>>(wts); 

    ROL::Ptr<ROL::TypeP::TrustRegionAlgorithm<RealT>> algo_nonsmooth_tr = ROL::makePtr<ROL::TypeP::TrustRegionAlgorithm<RealT>>(*parlist);
    bool checkDeriv = parlist->sublist("Problem").get("Check Derivatives",true);
    if ( checkDeriv ) {
      ROL::Ptr<ROL::StdVector<RealT>> dx = ROL::makePtr<ROL::StdVector<RealT>>(dim,0);
      ROL::Ptr<ROL::StdVector<RealT>> hx = ROL::makePtr<ROL::StdVector<RealT>>(dim,0);
      x->randomize(); dx->randomize(); hx->randomize();  
      //ROL::ValidateFunction<RealT> validate(1,13,20,11,true,*outStream);
      //ROL::ObjectiveCheck<RealT>::check(*obj,validate,*x);
      obj->checkGradient(*x,*dx,true,*outStream);
      obj->checkHessVec(*x,*dx,true,*outStream);
      obj->checkHessSym(*x,*dx,*hx,true,*outStream);
    }
    
    //std::clock_t timer = std::clock(); 
    algo_nonsmooth_tr->run(*xy, *obj, *nobj, *outStream); // nonsmooth tr 
    *outStream << "xy = ";
    for (int i=0; i<dim; ++i) {
      *outStream << (*xy)[i] << " ";
    }

    // Set up and run proxstorm
    ROL::Ptr<ROL::Problem<RealT>> problem = ROL::makePtr<ROL::Problem<RealT>>(
        obj, x
    );
    problem->addProximableObjective(nobj);
    problem->finalize(false, true, *outStream);

    ROL::Ptr<ROL::STORMAlgorithm<RealT>> algo_storm = ROL::makePtr<ROL::STORMAlgorithm<RealT>>(
      problem,
      nullptr,
      *parlist
    );
    algo_storm->run(*outStream); // storm

    *outStream << "x = ";
    for (int i=0; i<dim; ++i) {
      *outStream << (*x)[i] << " ";
    }

    ROL::Ptr<std::vector<RealT>> d_ptr  = ROL::makePtr<std::vector<RealT>>(dim,0.0);
    ROL::Ptr<ROL::Vector<RealT>> d      = ROL::makePtr<ROL::StdVector<RealT>>(d_ptr);
    d->set(*x); d->axpy(-1.0,*xy);
    RealT err = d->norm();
    *outStream << std::endl << "Norm of difference: " << err<< std::endl <<std::endl;
    errorFlag += (err > 1e-2 ? 1 : 0);
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


