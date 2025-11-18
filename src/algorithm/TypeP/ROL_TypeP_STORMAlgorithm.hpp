// @HEADER
// *****************************************************************************
//               Rapid Optimization Library (ROL) Package
//
// Copyright 2014 NTESS and the ROL contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef ROL_STORMALGORITHM_HPP
#define ROL_STORMALGORITHM_HPP

#include "ROL_TypeP_TrustRegionAlgorithm.hpp"
#include "ROL_TypeP_Algorithm.hpp"
#include "ROL_TrustRegion_P_Types.hpp"
#include "ROL_TrustRegion_P.hpp"
#include "ROL_TrustRegionModel_U.hpp"
#include "ROL_TrustRegionUtilities.hpp"
#include "ROL_Types.hpp"

/** \class ROL::TypeP::TrustRegionAlgorithm
    \brief Provides an interface to run the storm trust-region algorithm.
*/

namespace ROL {
namespace TypeP {

template<typename Real>
class STORMAlgorithm : public TrustRegionAlgorithm<Real> {
private:
  // TRUST REGION INFORMATION
  Ptr<TrustRegion_P<Real>>      solver_; ///< Container for trust-region solver object 
  Ptr<TrustRegionModel_U<Real>> model_;  ///< Container for trust-region model
  ETrustRegionP                 algSelect_;
  
  //TRUST REGION PARAMETERS
  Real TRsafe_; ///< Safeguard size for numerically evaluating ratio (default: 1e2)
  Real eps_;    ///< Safeguard for numerically evaluating ratio

  // ITERATION FLAGS/INFORMATION
  TRUtils::ETRFlag TRflag_; ///< Trust-region exit flag
  int SPflag_;              ///< Subproblem solver termination flag
  int SPiter_;              ///< Subproblem solver iteration count

  // SECANT INFORMATION
  ESecant esec_;          ///< Secant type (default: Limited-Memory BFGS)
  bool useSecantPrecond_; ///< Flag to use secant as a preconditioner (default: false)
  bool useSecantHessVec_; ///< Flag to use secant as Hessian (default: false)

  // TRUNCATED CG INFORMATION
  Real tol1_; ///< Absolute tolerance for truncated CG (default: 1e-4)
  Real tol2_; ///< Relative tolerance for truncated CG (default: 1e-2)
  int maxit_; ///< Maximum number of CG iterations (default: 25)

  // ALGORITHM SPECIFIC PARAMETERS
  Real mu0_;       ///< Sufficient decrease parameter (default: 1e-2)
  //Real spexp_;     ///< Relative tolerance exponent for subproblem solve (default: 1, range: [1,2])

  // Nonmonotone PR 
  bool  useNM_;
  int   storageNM_;
  Real  rhoNM_;
  Real  sigmac_;//set sigmac/r to member variables
  Real  sigmar_;
  Real  fr_;
  Real  fc_;
  Real  fmin_;
  int   L_; 
  

  // Inexactness Parameters
  std::vector<bool> useInexact_;
  Real scale0_;
  Real scale1_;
  Real scale_;
  Real omega_;
  Real force_;
  int updateIter_;
  Real forceFactor_;
  Real gtol_;

  bool initProx_;
  Real t0_; 
  
  mutable int nhess_;  ///< Number of Hessian applications
  unsigned verbosity_; ///< Output level (default: 0)
  bool writeHeader_;   ///< Flag to write header at every iteration

public:
  STORMAlgorithm(ParameterList &list, const Ptr<Secant<Real>> &secant = nullPtr);

  using TrustRegionAlgorithm<Real>::initialize;  
  using TrustRegionAlgorithm<Real>::writeHeader;  
  using TrustRegionAlgorithm<Real>::writeOutput;  
  using Algorithm<Real>::state_;
  using Algorithm<Real>::status_;
  using Algorithm<Real>::pgstep;
  using Algorithm<Real>::run;

protected:

  // TRUST REGION PARAMETERS
  Real delMax_;     ///< Maximum trust-region radius (default: ROL_INF)
  Real eta0_;       ///< Step acceptance threshold (default: 0.05)
  Real eta1_;       ///< Radius decrease threshold (default: 0.05)
  Real eta2_;       ///< Radius increase threshold (default: 0.9)
  Real gamma0_;     ///< Radius decrease rate (negative rho) (default: 0.2)
  Real gamma1_;     ///< Radius decrease rate (positive rho) (default: 0.25)
  Real gamma2_;     ///< Radius increase rate (default: 5)
  bool interpRad_;  ///< Interpolate the trust-region radius if ratio is negative (default: false)
  
  void writeName( std::ostream& os ) const;
  
  virtual Real computeValue(Real inTol,
                    Real &outTol,
                    Real pRed,
                    Real &fold,
                    Real del,
                    int  iter,
                    const Vector<Real> &x,
                    const Vector<Real> &xold,
                    Objective<Real> &obj);

  virtual void computeGradient(const Vector<Real> &x,
                       Vector<Real> &g,
                       Vector<Real> &px,
                       Vector<Real> &dg,
                       Vector<Real> &step,
                       Real del,
                       Objective<Real> &sobj,
                       Objective<Real> &nobj,
                       bool accept,
                       Real &gtol,
                       Real &gnorm,
                       std::ostream &outStream = std::cout) const;


  virtual void stepUpdate(Vector<Real>             &x, 
                          Vector<Real>             &g,
                          Vector<Real>             &px,             
                          Vector<Real>             &dg, 
                          Objective<Real>          &sobj,
                          Objective<Real>          &nobj,
                          TrustRegionModel_U<Real> &model,
                          Real                     &rho,
                          Real                     &pRed,
                          Real                     &Ftrial,
                          Real                     &strial,
                          Real                     &ntrial,
                          bool                     accept,
                          Real                     &gtol,
                          Real                     &inTol,
                          Real                     &outTol,
                          Vector<Real>             &pwa1,
                          Vector<Real>             &dwa1,
                          std::ostream             &outStream = std::cout
      ) ;

}; // class ROL::TypeP::TrustRegionAlgorithm

} // namespace TypeP
} // namespace ROL

#include "ROL_TypeP_STORMAlgorithm_Def.hpp"

#endif
