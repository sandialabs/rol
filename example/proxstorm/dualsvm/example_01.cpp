// @HEADER
// @HEADER

/*! \file  example_01.cpp
    \brief Shows how to minimize a function with binary (0/1) constraints.
*/

#include "ROL_Bounds.hpp"
#include "ROL_StdObjective.hpp"
#include "ROL_StdConstraint.hpp"
#include "ROL_Stream.hpp"

#include "ROL_GlobalMPISession.hpp"

#include "ROL_TypeB_Algorithm.hpp"
#include "ROL_TypeP_STORMAlgorithm.hpp"
#include "ROL_TypeP_STORMAlgorithm_Def.hpp"
#include "ROL_TypeP_TrustRegionAlgorithm.hpp"
#include "ROL_TypeP_TrustRegionAlgorithm_Def.hpp"
#include "ROL_ScalarLinearConstraint.hpp"

#include <iostream>
#include <random>

typedef double RealT;

template<class Real>
class DualSVMObjective : public ROL::StdObjective<Real> {
private:
  const int nrows_;
  std::vector<std::vector<Real>> K_;

  Real kernel(const std::vector<Real> &x, const std::vector<Real> &y) const {
    typename std::vector<Real>::size_type size = x.size();
    assert(size==y.size());
    Real val(0);
    for (typename std::vector<Real>::size_type i = 0; i < size; ++i) val += x[i]*y[i];
    return val;
  }

public:
  DualSVMObjective(const std::vector<std::vector<Real>> &X, const ROL::StdVector<Real> &y)
    : nrows_(X.size()) {
    K_.resize(nrows_);
    for (int i = 0; i < nrows_; ++i) {
      K_[i].resize(nrows_);
      for (int j = 0; j < nrows_; ++j) {
        K_[i][j] = y[i]*y[j]*kernel(X[i],X[j]);
      }
    }
  }

  virtual Real value(const std::vector<Real> &x, Real &tol) {
    Real val(0);
    for (int i = 0; i < nrows_; ++i) {
      val -= x[i];
      for (int j = 0; j < nrows_; ++j) {
        val += static_cast<Real>(0.5)*x[i]*K_[i][j]*x[j];
      }
    }
    return val;
  } 

  void gradient(std::vector<Real> &g, const std::vector<Real> &x, Real &tol) {
    g.assign(x.size(),static_cast<Real>(-1));
    for (int i = 0; i < nrows_; ++i) {
      for (int j = 0; j < nrows_; ++j) {
        g[i] += K_[i][j]*x[j];
      }
    }
  }

  void hessVec(std::vector<Real> &hv, const std::vector<Real> &v, const std::vector<Real> &x, Real &tol) {
    hv.assign(x.size(),static_cast<Real>(0));
    for (int i = 0; i < nrows_; ++i) {
      for (int j = 0; j < nrows_; ++j) {
        hv[i] += K_[i][j]*v[j];
      }
    }
  }

};

template<typename Real>
class DualSVMConstraint : public ROL::StdObjective<Real> {
private:
  const std::vector<Real> y_;
  const int size_;
  const ROL::Ptr<ROL::PolyhedralProjection<Real>> proj_; 
   

public:
  DualSVMConstraint(const ROL::Ptr<ROL::StdVector<Real>> y,
                    const ROL::Ptr<ROL::PolyhedralProjection<Real>> &proj) :  y_(*y->getVector()), size_(y->dimension()), proj_(proj) {
  }

  Real value(const std::vector<Real> &x, Real &tol) {
    Real val_eq  = 0.0;
    Real zero(0.0); 
    for (int i = 0; i < size_; ++i) {
      val_eq += x[i]*y_[i];
      if (x[i] < -std::sqrt(ROL::ROL_EPSILON<Real>())){
         return ROL::ROL_INF<Real>(); }
    }
    if (val_eq > std::sqrt(ROL::ROL_EPSILON<Real>()))
      return ROL::ROL_INF<Real>(); 
    else
      return zero;   
  } // end value
  
  void prox(ROL::Vector<Real>      &Pz, 
           const ROL::Vector<Real> &z,  
           Real                    t, 
           Real                    &tol) {
       Pz.set(z);
       proj_->project(Pz);
  } // end prox
}; // end DualSVMConstraint

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

    parlist->sublist("General").set("Output Level",iprint);
    parlist->sublist("Step").set("Type","Trust Region");
    parlist->sublist("Status Test").set("Gradient Tolerance",1e-7);
    parlist->sublist("Status Test").set("Constraint Tolerance",1e-8);
    parlist->sublist("Status Test").set("Step Tolerance",1e-12);
    parlist->sublist("Status Test").set("Iteration Limit", 100);
    parlist->sublist("Step").sublist("Trust Region").set("Subproblem Solver", "SPG2");  
    parlist->sublist("General").sublist("Polyhedral Projection").set("Type", "Dai-Fletcher");
    // Set up problem data
    int nInstances = 50;
    int nVars      = 100;
    unsigned seed  = 111;
    std::default_random_engine generator(seed);
    std::normal_distribution<RealT> dist(0.0,1.0);
    std::vector<std::vector<RealT>> X(nInstances);
    for (int i = 0; i < nInstances; ++i) {
      for (int j = 0; j < nVars; ++j) {
        X[i].push_back(dist(generator));
      }
    }
    std::vector<RealT> w(nVars);
    for (int j = 0; j < nVars; ++j) {
      w[j] = dist(generator);
    }
    RealT yval(0);
    ROL::Ptr<ROL::StdVector<RealT>> y = ROL::makePtr<ROL::StdVector<RealT>>(nInstances, 0);
    //std::vector<RealT> y(nInstances,0);
    for (int i = 0; i < nInstances; ++i) {
      yval = dist(generator);
      for (int j = 0; j < nVars; ++j) {
        yval += X[i][j]*w[j];
      }
      (*y)[i] = (yval < static_cast<RealT>(0) ? static_cast<RealT>(-1)
              : (yval > static_cast<RealT>(0) ? static_cast<RealT>(1)
              : static_cast<RealT>(0)));
    }
    // set up objective
    ROL::Ptr<DualSVMObjective<RealT>>  obj = ROL::makePtr<DualSVMObjective<RealT>>(X,*y);
     
    // set up prox
    // <x,y> = c, x>0
    ROL::Ptr<ROL::StdVector<RealT>>               l = ROL::makePtr<ROL::StdVector<RealT>>(nInstances,0);
    ROL::Ptr<ROL::StdVector<RealT>>               x = ROL::makePtr<ROL::StdVector<RealT>>(nInstances,0);  
    ROL::Ptr<ROL::StdVector<RealT>>               xy = ROL::makePtr<ROL::StdVector<RealT>>(nInstances,0);  
    ROL::Ptr<ROL::Bounds<RealT>>                bnd = ROL::makePtr<ROL::Bounds<RealT>>(*l,true);
    ROL::Ptr<ROL::SingletonVector<RealT>>       lam = ROL::makePtr<ROL::SingletonVector<RealT>>(0); 
    ROL::Ptr<ROL::SingletonVector<RealT>>       res = ROL::makePtr<ROL::SingletonVector<RealT>>(0); 
    RealT c = 0.0; 
    ROL::Ptr<ROL::Constraint<RealT>>            con = ROL::makePtr<ROL::ScalarLinearConstraint<RealT>>(y, c);
    //c = lam->dual().clone(); 
    
    ROL::Ptr<ROL::PolyhedralProjection<RealT>> proj = ROL::PolyhedralProjectionFactory<RealT>(*x, x->dual(), bnd, con, *lam, *res, *parlist); 
    ROL::Ptr<DualSVMConstraint<RealT>> nobj         = ROL::makePtr<DualSVMConstraint<RealT>>(y, proj);
      
    ROL::Ptr<ROL::TypeP::STORMAlgorithm<RealT>> algo = ROL::makePtr<ROL::TypeP::STORMAlgorithm<RealT>>(*parlist);
    ROL::Ptr<ROL::TypeP::TrustRegionAlgorithm<RealT>> algo2 = ROL::makePtr<ROL::TypeP::TrustRegionAlgorithm<RealT>>(*parlist);
    
    bool checkDeriv = parlist->sublist("Problem").get("Check Derivatives",false);
    if ( checkDeriv ) {
      ROL::Ptr<ROL::StdVector<RealT>> dx = ROL::makePtr<ROL::StdVector<RealT>>(nInstances,0);
      ROL::Ptr<ROL::StdVector<RealT>> hx = ROL::makePtr<ROL::StdVector<RealT>>(nInstances,0);
      x->randomize(); dx->randomize(); hx->randomize();  
      //ROL::ValidateFunction<RealT> validate(1,13,20,11,true,*outStream);
      //ROL::ObjectiveCheck<RealT>::check(*obj,validate,*x);
      obj->checkGradient(*x,*dx,true,*outStream);
      obj->checkHessVec(*x,*dx,true,*outStream);
      obj->checkHessSym(*x,*dx,*hx,true,*outStream);
    }
    

    //std::clock_t timer = std::clock(); 
    algo2->run(*xy, *obj, *nobj, *outStream); // nonsmooth tr 
    algo->run(*x, *obj, *nobj, *outStream); // storm

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


