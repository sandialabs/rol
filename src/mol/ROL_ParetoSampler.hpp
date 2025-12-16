
#ifndef ROL_PARETOSAMPLER_HPP
#define ROL_PARETOSAMPLER_HPP

#include <ctime>
#include <random>
#include <algorithm>

#include "ROL_MultiObjectiveFactory.hpp"
#include "ROL_UniformSimplexGenerator.hpp"
#include "ROL_Equispaced2dGenerator.hpp"

namespace ROL {

template<typename Real>
class ParetoSampler {
private:
  struct ParetoData {
  public:
    const Ptr<Vector<Real>> sol;
    const std::vector<Real> lam;
    const std::vector<Real> val;
    ParetoData(const Ptr<Vector<Real>>& x, const std::vector<Real>& l, const std::vector<Real>& v)
      : sol(x), lam(l), val(v) {}
  };

  const Ptr<BatchManager<Real>> bman_;
  std::vector<ParetoData> samples_;

public:
  ParetoSampler(const Ptr<BatchManager<Real>>& bman = nullPtr)
    : bman_(bman==nullPtr ? makePtr<BatchManager<Real>>() : bman) {}

  void run(const Ptr<MultiObjectiveFactory<Real>>& factory,
           ParameterList& parlist,
           std::ostream& outStream = std::cout,
           bool reset = false) {
    if (reset) samples_.clear();
    const Real zero(0), one(1);
    const unsigned nobj = factory->getNumObjectives();
    const int batchid = bman_->batchID();
    const int nbatch = bman_->numBatches();
    // Compute endpoints
    std::vector<Real> lam(nobj);
    std::vector<Ptr<Vector<Real>>> sol;
    std::vector<std::vector<Real>> val;
    factory->getEndPoints(sol,val,parlist,outStream);
    int frac   = nobj / nbatch;
    int rem    = nobj % nbatch;
    int N      = frac + ((batchid < rem) ? 1 : 0);
    int offset = 0;
    for (int i = 0; i < batchid; ++i) offset += frac + ((i < rem) ? 1 : 0);
    for (int i = 0; i < N; ++i) {
      lam.assign(nobj,zero);
      lam[offset+i] = one;
      ParetoData pd(sol[offset+i],lam,val[offset+i]);
      samples_.push_back(pd);
    }
    // Compute random samples
    const int nsamp = parlist.sublist("Multi-Objective").sublist("Pareto Sampler").get("Number of Points",10);
    const bool initGuess = parlist.sublist("Multi-Objective").sublist("Pareto Sampler").get("Warm Start",false);
    const bool useTrig = parlist.sublist("Multi-Objective").sublist("Pareto Sampler").get("Use Trigonometric Spacing",false);
    if (nobj > 1u) {
      Ptr<SampleGenerator<Real>> sampler;
      if (nobj == 2u) sampler = makePtr<Equispaced2dGenerator<Real>>(nsamp,bman_,useTrig);
      else            sampler = makePtr<UniformSimplexGenerator<Real>>(nsamp,nobj,bman_);
      auto x0 = factory->getSolution()->clone();
      x0->set(*factory->getSolution());
      std::vector<Real> fail(sampler->numMySamples(),zero);
      Real tottime(0);
      for (int i = 0; i < sampler->numMySamples(); ++i) {
        // Generate simplex sample
        lam = sampler->getMyPoint(i);
        // Solve scalarized problem
        auto problem = factory->getScalarProblem(lam,parlist,outStream,initGuess&&(i!=0),x0);
        //x0->randomize(1.0,2.0);
        problem->finalize(false,true,outStream);
        //problem->check(true,outStream,x0,0.1);
        auto solver = makePtr<Solver<Real>>(problem,parlist);
        auto timer = std::clock();
        solver->solve(outStream);
        Real time = static_cast<Real>(std::clock()-timer)/static_cast<Real>(CLOCKS_PER_SEC);
        outStream << "Optimization Time: " << time << " seconds" << std::endl;
        tottime += time;
        fail[i] = solver->getAlgorithmState()->statusFlag==EXITSTATUS_CONVERGED ? zero : one;
        // Store Pareto data
        auto x = factory->getSolution();
        auto val = factory->evaluateObjectiveVector(*x);
        ParetoData pd(x,lam,val);
        samples_.push_back(pd);
        x0->set(*x);
      }
      Real nfail = std::reduce(fail.begin(),fail.end()), gtime(0), gfail(0);
      bman_->sumAll(&tottime,&gtime,1);
      bman_->sumAll(&nfail,&gfail,1);
      if (bman_->batchID()==0) {
        outStream << std::endl;
        outStream << "ROL::ParetoSampler Performance Summary" << std::endl;
        outStream << "  Total Number of Samples:        " << nsamp << std::endl;
        outStream << "  Total Number of Batches:        " << nbatch << std::endl;
        outStream << "  Total Time:                     " << gtime << " seconds" << std::endl;
        outStream << "  Average Time Per Sample:        " << gtime / static_cast<Real>(nsamp) << " seconds" << std::endl;
        outStream << "  Average Time Per Batch:         " << gtime / static_cast<Real>(nbatch) << " seconds" << std::endl;
        outStream << "  Total Number of Failed Samples: " << gfail << std::endl;
        outStream << std::endl;
      }
    }
  }

  void print(std::string file) const {
    // Broadcast data to root batch
    const unsigned nobj = samples_[0].val.size();
    const unsigned nsamp = samples_.size();
    const unsigned lsize = nobj*nsamp;
    std::vector<Real> val(lsize);
    for (unsigned i=0; i<nsamp; ++i) {
      for (unsigned j=0; j<nobj; ++j) val[i*nobj+j] = samples_[i].val[j];
    }
    Real lcnt(nsamp), gcnt(0);
    bman_->sumAll(&lcnt,&gcnt,1);
    const unsigned nsampg = (unsigned)gcnt;
    const unsigned gsize = nobj*nsampg;
    std::vector<Real> valg(gsize);
    bman_->gather(&val[0],lsize,&valg[0],gsize,0);
    // Print to out stream
    if (bman_->batchID()==0) {
      std::ofstream os;
      os.open(file);
      os << std::scientific << std::setprecision(15);
      for (unsigned i=0; i<nsampg; ++i) {
        for (unsigned j=0; j<nobj; ++j) os << std::right << std::setw(25) << valg[i*nobj+j];
        os << std::endl;
      }
      os.close();
    }
  }

};

}

#endif
