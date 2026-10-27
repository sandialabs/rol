// @HEADER
// @HEADER

/*! \file  data_generator.cpp
    \brief Helper file to generate random data for the logistic example.
*/

#include <cmath>


template<class Real>
class LogisticDataGenerator {

private:
  Real sigmoid(Real z) {
    Real one(1);
    return one / (one + std::exp(-z));
  }

public:
  LogisticDataGenerator() {};

  void generate_and_save_data(int n_samples, std::vector<Real> params, std::string filename, std::ostream &outStream) {

    int n_features = params.size() - 1;
    Real one(1.);
    Real z(0);
    Real randFeature(0); 

    using seed_type = std::mt19937_64::result_type;
    seed_type const seed = 123;
    std::mt19937_64 eng{seed};
    std::normal_distribution<Real> normal(0., 1.);
    std::uniform_real_distribution<Real> uniform(0., 1.);
    std::ofstream file(filename);

    if (!file) {
      outStream << "Failed to open output file!\n";
    }
    else {
      for (int i=0; i<n_samples; ++i) {
        z += params[0];
        for (int j=0; j<n_features; ++j) {
          randFeature = normal(eng);
          file << randFeature << ' ';
          z += randFeature * params[j+1];
        }
        
        // Evaluate sigmoid(z)
        if (uniform(eng) < sigmoid(z)) {
          file << one;
        } else {
          file << -one;
        }
        file << '\n';
        z = 0;
      }
    }
    file.close();
  }
};
