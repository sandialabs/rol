#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <limits>
#include <random>
#include <numeric>

#include "ROL_Objective.hpp"
#include "ROL_StdVector.hpp"
#include "ROL_Solver.hpp"
#include "ROL_ParameterList.hpp"


struct Detection {
    double x;
    double y;
};

struct Assignment {
    int detection;
    int moth;
};


template <typename Real>
class SumToOneConstraint : public ROL::Constraint<Real> {
    typedef std::vector<Real>    vector;
    typedef ROL::Vector<Real>     V;
    typedef ROL::StdVector<Real>  SV;
  
private:
    int num_detections_;
    int num_moths_;

    ROL::Ptr<const vector> getVector(const V& x) {
        return dynamic_cast<const SV&>(x).getVector();
    }
    
    ROL::Ptr<vector> getVector(V& x) {
        return dynamic_cast<SV&>(x).getVector();
    }

public:
    SumToOneConstraint(int num_detections, int num_moths)
        : num_detections_(num_detections), num_moths_(num_moths) {}

        void value(ROL::Vector<Real> &c, const ROL::Vector<Real> &lam, Real &tol) override {
            vector &c_vec = *getVector(c);
            const vector &lam_vec = *getVector(lam);

            for (int i = 0; i < num_detections_; ++i) {
                Real sum = 0.0;
                for (int k = 0; k < num_moths_; ++k) {
                    sum += lam_vec[i * num_moths_ + k];
                }
                c_vec[i] = sum - Real(1.0);
            }
        }

        void applyJacobian(ROL::Vector<Real> &jv, const ROL::Vector<Real> &v,
                        const ROL::Vector<Real> &lam, Real &tol) override {
            vector &jv_vec = *getVector(jv);
            const vector &v_vec = *getVector(v);

            for (int i = 0; i < num_detections_; ++i) {
                Real sum = 0.0;
                for (int k = 0; k < num_moths_; ++k) {
                    sum += v_vec[i * num_moths_ + k];
                }
                jv_vec[i] = sum;
            }
        }

        void applyAdjointJacobian(ROL::Vector<Real> &ajv, const ROL::Vector<Real> &v,
                                const ROL::Vector<Real> &lam, Real &tol) override {
            vector &ajv_vec = *getVector(ajv);
            const vector &v_vec = *getVector(v);

            std::fill(ajv_vec.begin(), ajv_vec.end(), Real(0));

            for (int i = 0; i < num_detections_; ++i) {
                for (int k = 0; k < num_moths_; ++k) {
                    ajv_vec[i * num_moths_ + k] = v_vec[i];
                }
            }
        }

        void applyAdjointHessian(ROL::Vector<Real> &ahuv, const ROL::Vector<Real> &u,
                                const ROL::Vector<Real> &v, const ROL::Vector<Real> &lam,
                                Real &tol) override {
            ahuv.zero();
        }
};


// Define the custom ROL objective function for moth tracking
template <typename Real>
class MothObjective : public ROL::Objective<Real> {
private:
    std::vector<Detection> detections_;
    int num_moths_;

public:
    MothObjective(const std::vector<Detection>& detections, int num_moths)
        : detections_(detections), num_moths_(num_moths) {}

            
    Real value(const ROL::Vector<Real> &lam, Real &tol) override {
        
        auto& lambda_vec = *dynamic_cast<const ROL::StdVector<Real>&>(lam).getVector();

        std::vector<Real> x_vec(2 * num_moths_, Real{0});
        std::vector<Real> lambda_sums(num_moths_, Real{0});

        for (std::size_t d_idx = 0; d_idx < detections_.size(); ++d_idx) {
            const auto& d = detections_[d_idx];

            for (int k = 0; k < num_moths_; ++k) {
                Real lambda = lambda_vec[num_moths_ * d_idx + k];

                lambda_sums[k] += lambda;

                x_vec[2 * k]     += lambda * d.x;
                x_vec[2 * k + 1] += lambda * d.y;
            }
        }

        for (int k = 0; k < num_moths_; ++k) {
            if (lambda_sums[k] > Real{0}) {
                x_vec[2 * k]     /= lambda_sums[k];
                x_vec[2 * k + 1] /= lambda_sums[k];
            } else {
                x_vec[2 * k]     = Real{0};
                x_vec[2 * k + 1] = Real{0};
            }
        }

        Real total_error = 0.0;

        for (std::size_t d_idx = 0; d_idx < detections_.size(); ++d_idx) {
            const auto& d = detections_[d_idx];

            for (int k = 0; k < num_moths_; ++k) {
                Real lambda = lambda_vec[num_moths_ * d_idx + k];

                Real mx = x_vec[2 * k];
                Real my = x_vec[2 * k + 1];

                Real dist_sq = (mx - d.x) * (mx - d.x) + (my - d.y) * (my - d.y);

                total_error += lambda * dist_sq;
            }
        }

        return total_error;
    }


    void gradient(ROL::Vector<Real> &g, const ROL::Vector<Real> &lam, Real &tol) override {
        auto& g_vec = *dynamic_cast<ROL::StdVector<Real>&>(g).getVector();
        auto& lambda_vec = *dynamic_cast<const ROL::StdVector<Real>&>(lam).getVector();

        std::fill(g_vec.begin(), g_vec.end(), 0.0);

        std::vector<Real> x_vec = computeCentersFromLambda(lam);
        


        for (std::size_t d_idx = 0; d_idx < detections_.size(); ++d_idx) {
            const auto& d = detections_[d_idx];

            for (int k = 0; k < num_moths_; ++k) {

                Real mx = x_vec[2 * k];
                Real my = x_vec[2 * k + 1];

                Real dist_sq = (mx - d.x) * (mx - d.x) + (my - d.y) * (my - d.y);

                g_vec[num_moths_ * d_idx + k] = dist_sq;
            }
        }
        
    }

    // Compute final center locations for a given solution
    std::vector<Real> computeCentersFromLambda(const ROL::Vector<Real> &lam) const {
        auto& lambda_vec =
            *dynamic_cast<const ROL::StdVector<Real>&>(lam).getVector();

        std::vector<Real> x_vec(2 * num_moths_, Real{0});
        std::vector<Real> lambda_sums(num_moths_, Real{0});

        for (std::size_t d_idx = 0; d_idx < detections_.size(); ++d_idx) {
            const auto& d = detections_[d_idx];

            for (int k = 0; k < num_moths_; ++k) {
                Real lambda = lambda_vec[num_moths_ * d_idx + k];

                lambda_sums[k] += lambda;

                x_vec[2 * k]     += lambda * d.x;
                x_vec[2 * k + 1] += lambda * d.y;
            }
        }

        for (int k = 0; k < num_moths_; ++k) {
            if (lambda_sums[k] > Real{0}) {
                x_vec[2 * k]     /= lambda_sums[k];
                x_vec[2 * k + 1] /= lambda_sums[k];
            } else {
                x_vec[2 * k]     = Real{0};
                x_vec[2 * k + 1] = Real{0};
            }
        }

        return x_vec;
    }
   

    // Compute final assignments for a given solution 
    std::vector<Assignment> computeAssignmentsFromLambda(const ROL::Vector<Real> &lam) const {
        auto& lambda_vec = *dynamic_cast<const ROL::StdVector<Real>&>(lam).getVector();

        std::vector<Assignment> assignments;
 
        for (int d_idx = 0; d_idx < static_cast<int>(detections_.size()); ++d_idx) {
            int best_k = 0;
            Real best_lambda = lambda_vec[num_moths_ * d_idx];

            for (int k = 1; k < num_moths_; ++k) {
                Real lambda = lambda_vec[num_moths_ * d_idx + k];

                if (lambda > best_lambda) {
                    best_lambda = lambda;
                    best_k = k;
                }
            }

            assignments.push_back({d_idx + 1, best_k + 1});
        }

        return assignments;
    }
};


void configureDaiFletcherProjection(ROL::ParameterList &parlist,
                                                        int num_detections,
                                                        int num_moths) {

    auto &projlist = parlist.sublist("General").sublist("Polyhedral Projection");
        
    projlist.set("Number of Detections", num_detections);
    projlist.set("Number of Moths", num_moths);
    projlist.set("Target Sum", 1.0);
    projlist.set("Lower Bound", 0.0);
    projlist.set("Upper Bound", 1.0);
    projlist.set("Absolute Tolerance", 1e-12);
    projlist.set("Relative Tolerance", 1e-12);
    projlist.set("Iteration Limit", 100);

}


int main() {
    // Select user-defined options
    std::string opt_method = "Trust Region"; // Options: Line Search, Trust Region
    std::string init_method = "Cyclical"; // Options: Cyclical, Equal, Same, Random
    std::string proj_type = "Row-Wise-Dai-Fletcher"; // Options: Row-Wise-Dai-Fletcher, Dykstra, Semismooth Newton
    bool print_centers = true, print_assignments = true;
    int verbosity = 0; // Increase for more detailed output

    // Initialize detections
    std::vector<Detection> detections = {
        {-5.48062654e-02, 7.50455774e-02}, {2.67055376e-01, -1.48996101e-01}, {3.54633606e-01, -1.24620592e-01}, {-2.66806648e-01, 3.21574021e-01}, {3.01513814e-01, -2.05892000e-02}, {-2.52667107e-01, -4.52713231e-02}, {1.28824549e-01, -2.63916938e-01}, {-1.15246894e-01, 1.83534137e-01}, {5.98470611e-01, 2.55890063e-01}, {9.63280305e-01, 1.99847014e-01}, {7.48699618e-01, 4.49491399e-01}, {4.36479153e-01, 4.18606992e-01}, {-2.60848070e-02, -9.04756076e-02}, {1.06265641e+00, 5.59368933e-01}, {-2.01767764e-03, 4.19916898e-02}, {6.73397262e-01, 3.02172831e-01}, {5.22177408e-01, 6.62325388e-02}, {-2.59773910e-01, -4.96866643e-01}, {-1.48383931e-01, 2.32151624e-01}, 
        {2.44011592e-01, 1.33043840e-01}, {1.38679623e+00, 4.54101920e-01}, {1.76803382e+00, -2.69720341e-02}, {5.78068100e-01, 5.50578366e-01}, {1.19068829e+00, 7.37676263e-01}, {2.14723847e+00, 6.09387865e-01}, {8.45190269e-01, 4.11287623e-01}, {1.54203607e+00, 5.54348470e-01}, {1.64655184e+00, 4.82423911e-01}, {1.60754005e+00, 5.23256444e-01}, {8.02018224e-01, 7.71905541e-01}, {1.21150055e+00, 3.71273226e-01}, {1.02821374e+00, 1.83927656e-01}, {1.50835863e+00, 1.46815890e-01}, {1.46970889e+00, 9.78491945e-01}, {1.54959139e+00, -2.54945262e-01}, {6.53486665e-01, 1.94346275e-01}, {1.50348732e+00, 2.10728340e-01}, {1.23945534e+00, 1.00760991e+00}, {1.38830093e+00, 2.00069068e-01}, 
        {1.90168513e+00, 6.45816182e-01}, {1.43516995e+00, 1.77880583e-01}, {1.26611371e+00, 5.01057734e-01}, {1.14069463e+00, 8.25050035e-02}, {1.27779505e+00, 2.97867143e-01}, {1.25489058e+00, 9.66665485e-01}, {1.59587555e+00, 5.38856981e-01}, {1.79371590e+00, 5.09227784e-01}, {1.56653267e+00, 5.26876325e-01}, {1.86317567e+00, 8.92299555e-03}, {1.30181827e+00, -6.61041676e-01}, {1.23158110e+00, 9.09634941e-02}, {1.98851192e+00, 9.55154614e-01}, {1.45876894e+00, -3.90883810e-02}, {1.48134821e+00, 5.83284814e-01}, {1.39164928e+00, 2.90618424e-01}, {2.75802823e-01, 1.16339607e+00}, {8.98765702e-01, 1.64021854e+00}, {8.64618196e-01, 1.07381506e+00}, {7.53099051e-01, 1.03630403e+00}, {2.88293264e-01, 1.00406652e+00}, 
        {7.29604424e-01, 1.14926031e+00}, {1.15123684e+00, 9.35412516e-01}, {8.10163267e-01, 1.76361916e+00}, {9.52762170e-01, 2.20550968e+00}, {8.73193669e-01, 2.05543859e+00}, {6.25040126e-01, 1.74953461e+00}, {9.64260602e-01, 9.42941628e-01}, {-1.10943958e-01, 1.16205522e+00}, {6.88914262e-01, 1.26708668e+00}, {8.96515382e-01, 1.15718266e+00}, {4.45474204e-01, 8.11149484e-01}, {8.19994722e-01, 1.15663397e+00}, {1.10063477e+00, 1.25646421e+00}, {3.01753270e-01, 8.35436603e-01}, {3.36587820e-01, 9.69768589e-01}, {8.71799832e-01, 1.47304314e+00}, {7.48700176e-01, 1.49471698e+00}, {7.78956680e-01, 1.36847449e+00}, {1.18074205e+00, 1.61140941e+00}, {3.29348605e-01, 5.93640053e-01}, 
        {7.95988897e-01, 1.73179221e+00}, {5.99250887e-01, 5.19824282e-01}, {8.20420156e-01, 1.24358995e+00}, {8.82879743e-01, 2.05615527e+00}, {5.87737822e-01, 1.32258080e+00}, {6.84240054e-01, 1.65161296e+00}, {7.82283349e-01, 1.45688193e+00}, {1.21438211e+00, 1.43517275e+00}, {3.72404702e-01, 7.36742280e-01}, {5.30286966e-01, 1.19260479e+00}, {8.50765899e-01, 1.34895714e+00}, {5.08903284e-01, 1.02155366e+00}, {6.34979932e-01, 1.35671008e+00}, {1.06478890e-01, 1.03259781e+00}, {6.43535850e-01, 1.05373998e+00}, {7.25083385e-01, 1.96905997e+00}, {3.76487461e-01, 9.38324332e-01}, {2.51031354e-01, 1.27182702e+00}, {1.36797207e-01, 1.20051314e+00}, {3.28796542e-01, 1.76452787e+00}
    };
    
    for (auto &d : detections) {
        d.x = d.x * 10.0 + 10.0;
        d.y = d.y * 10.0 + 10.0;
    }
    
    int num_detections = detections.size();
    int num_moths = 3;

    // Create initial guess
    int num_lam = num_detections * num_moths;

    std::random_device rd; 
    std::mt19937 gen(12345); 
    std::uniform_int_distribution<> distr(0, 2); 

    auto initial_guess = std::make_shared<std::vector<double>>(num_lam, 0.0);
    
    if (init_method == "Equal") {
        initial_guess = std::make_shared<std::vector<double>>(num_lam, 1.0/num_moths);
    } else {
        for (std::size_t i = 0; i < detections.size(); ++i) {
            if (init_method == "Random") {
                (*initial_guess)[i*num_moths + distr(gen)] = 1.0;
            } else if (init_method == "Same") {
                (*initial_guess)[i*num_moths] = 1.0;
            } else {
                (*initial_guess)[i*num_moths + (i % num_moths)] = 1.0;
            }
        }
    }
    
    // Define problem and constraints
    auto lam = ROL::makePtr<ROL::StdVector<double>>(initial_guess);
    auto obj = ROL::makePtr<MothObjective<double>>(detections, num_moths);
    auto problem = ROL::makePtr<ROL::Problem<double>>(obj, lam);

    ROL::Ptr<ROL::Vector<double>> l = ROL::makePtr<ROL::StdVector<double>>(num_lam, 0.0);
    ROL::Ptr<ROL::Vector<double>> u = ROL::makePtr<ROL::StdVector<double>>(num_lam, 1.0);
    ROL::Ptr<ROL::BoundConstraint<double>> bnd = ROL::makePtr<ROL::Bounds<double>>(l,u);
    problem->addBoundConstraint(bnd);

    ROL::Ptr<ROL::Constraint<double>> linear_icon = ROL::makePtr<SumToOneConstraint<double>>(detections.size(), num_moths);
    ROL::Ptr<ROL::Vector<double>> linear_imul = ROL::makePtr<ROL::StdVector<double>>(detections.size(), 0.0);
    problem->addLinearConstraint("Sum to One", linear_icon, linear_imul);    

    // Configure parameter list
    ROL::ParameterList parlist;

    parlist.sublist("Step").set("Type", opt_method);
    parlist.sublist("General").set("Output Level", verbosity);
    parlist.sublist("Step").sublist(opt_method).sublist("Descent Method").set("Type", "Quasi-Newton");
    parlist.sublist("General").sublist("Polyhedral Projection").set("Type", proj_type);
    
    if (proj_type == "Row-Wise-Dai-Fletcher") {
        configureDaiFletcherProjection(parlist, num_detections, num_moths);
    }

    problem->setProjectionAlgorithm(parlist);
    problem->finalize(false, true, std::cout);

    // Solve problem
    ROL::Solver<double> solver(problem, parlist);
    solver.solve(std::cout);
    
    // Print optimized centers
    if (print_centers) {
        std::vector<double> centers = obj->computeCentersFromLambda(*lam);
        std::cout << "\nmoth,x,y\n";
        for (int k = 0; k < num_moths; ++k) {
            std::cout << (k + 1) << "," << centers[2 * k] << "," << centers[2 * k + 1] << "\n";
        }
    }
    
    // Print assignments
    if (print_assignments) {
        auto assignments = obj->computeAssignmentsFromLambda(*lam);
        std::cout << "\ndetection,moth\n";
        for (const auto& a : assignments) {
            std::cout << a.detection << "," << a.moth << "\n";
        }
    }

    return 0;
}