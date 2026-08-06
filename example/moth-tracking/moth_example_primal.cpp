#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <limits>
#include <random>
#include <numeric>
#include <set>

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

// Define the custom ROL objective function for moth tracking
template <typename Real>
class MothObjective : public ROL::Objective<Real> {
private:
    std::vector<Detection> detections_;
    int num_moths_;

public:
    MothObjective(const std::vector<Detection>& detections, int num_moths)
        : detections_(detections), num_moths_(num_moths) {}
            
    Real value(const ROL::Vector<Real> &x, Real &tol) override {
        
        auto& x_vec = *dynamic_cast<const ROL::StdVector<Real>&>(x).getVector();
        Real total_error = 0.0;
        for (const auto& d : detections_) {
            Real min_dist_sq = std::numeric_limits<Real>::max();

            for (int j = 0; j < num_moths_; ++j) {
                Real mx = x_vec[2 * j];
                Real my = x_vec[2 * j + 1];
                Real dist_sq = (mx - d.x) * (mx - d.x) + (my - d.y) * (my - d.y);
                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                }
            }
            total_error += min_dist_sq;
        }
        return total_error;

    }
    
    void gradient(ROL::Vector<Real> &g, const ROL::Vector<Real> &x, Real &tol) override {
        auto& x_vec = *dynamic_cast<const ROL::StdVector<Real>&>(x).getVector();
        auto& g_vec = *dynamic_cast<ROL::StdVector<Real>&>(g).getVector();

        std::fill(g_vec.begin(), g_vec.end(), 0.0);
        
        for (const auto& d : detections_) {
        
            Real min_dist_sq = std::numeric_limits<Real>::max();
            int closest_j = 0;

            for (int j = 0; j < num_moths_; ++j) {
                Real mx = x_vec[2 * j];
                Real my = x_vec[2 * j + 1];
                Real dist_sq = (mx - d.x) * (mx - d.x) + (my - d.y) * (my - d.y);
                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                    closest_j = j;
                }
            }

            g_vec[2 * closest_j]     += 2.0 * (x_vec[2 * closest_j] - d.x);
            g_vec[2 * closest_j + 1] += 2.0 * (x_vec[2 * closest_j + 1] - d.y);
        }
    }
    
    // Compute final assignments for a given solution 
    std::vector<Assignment> computeAssignments(const ROL::Vector<Real> &x) const {
        auto& x_vec = *dynamic_cast<const ROL::StdVector<Real>&>(x).getVector();
        std::vector<Assignment> assignments;

        for (int i = 0; i < static_cast<int>(detections_.size()); ++i) {
            const auto& d = detections_[i];
            Real min_dist_sq = std::numeric_limits<Real>::max();
            int closest_j = 0;

            for (int j = 0; j < num_moths_; ++j) {
                Real mx = x_vec[2 * j];
                Real my = x_vec[2 * j + 1];
                Real dist_sq = (mx - d.x) * (mx - d.x) + (my - d.y) * (my - d.y);
                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                    closest_j = j;
                }
            }

            assignments.push_back({i + 1, closest_j + 1}); // 1-based indexing
        }
        return assignments;
    }
};

void print_results(bool print_centers, 
                    bool print_assignments, 
                    ROL::Ptr<ROL::StdVector<double>> x, 
                    ROL::Ptr<MothObjective<double>> obj, 
                    int num_moths) {
    // Print optimized centers
    if (print_centers) {
        auto &x_vec = *x->getVector();
        std::cout << "\nmoth,x,y\n";
        for (int j = 0; j < num_moths; ++j) {
            std::cout << (j + 1) << ","
                    << x_vec[2 * j] << "," << x_vec[2 * j + 1] << "\n";
        }
    }   

    // Print assignments
    if (print_assignments) {
        std::cout << "\ndetection,moth\n";
        auto assignments = obj->computeAssignments(*x);
        for (const auto& a : assignments) {
            std::cout << a.detection << "," << a.moth << "\n";
        }
    } 
}


void runTrustRegionGridSearch(const std::vector<Detection> &detections, 
                                                            int num_moths, 
                                                            int verbosity, 
                                                            bool print_centers, 
                                                            bool print_assignments) {
    
    std::string method = "Trust Region";
    
    std::vector<double> init_rad = {4.0, 8.0, 12.0, 16.0};
    std::vector<double> rad_grow_rate = {1.5, 2.0, 2.5, 3.0};
    std::vector<double> rad_grow_thresh = {0.6, 0.7, 0.8, 0.9};
    
    int min_iter = 1000;
    int num_detections = detections.size();

    double best_init_rad = init_rad[0];
    double best_rad_grow_rate = rad_grow_rate[0];
    double best_rad_grow_thresh = rad_grow_thresh[0];
    
    for (std::size_t i = 0; i < init_rad.size(); ++i) {
        for (std::size_t j = 0; j < rad_grow_rate.size(); ++j) {
            for (std::size_t k = 0; k < rad_grow_thresh.size(); ++k) {
                
                // Create initial guess
                int num_z = 2 * num_moths;
                auto initial_guess = std::make_shared<std::vector<double>>(num_z, 0.0);

                for (int k = 0; k < num_moths; ++k) {
                    int idx = k * static_cast<int>(num_detections) / num_moths;
                    (*initial_guess)[2*k]     = detections[idx].x;
                    (*initial_guess)[2*k + 1] = detections[idx].y;
                }
                
                // Define problem 
                auto x = ROL::makePtr<ROL::StdVector<double>>(initial_guess);
                auto obj = ROL::makePtr<MothObjective<double>>(detections, num_moths);
                auto problem = ROL::makePtr<ROL::Problem<double>>(obj, x);

                // Configure parameter list
                ROL::ParameterList parlist;

                parlist.sublist("Step").set("Type", method);
                
                parlist.sublist("Step").sublist(method).set("Initial Radius", init_rad[i]);
                parlist.sublist("Step").sublist(method).set("Radius Growing Rate", rad_grow_rate[j]);
                parlist.sublist("Step").sublist(method).set("Radius Growing Threshold", rad_grow_thresh[k]);
             
                parlist.sublist("General").set("Output Level", verbosity);

                // Solve problem
                ROL::Solver<double> solver(problem, parlist);
                solver.solve(std::cout);

                int final_iter = solver.getAlgorithmState()->iter;
                double num_iter = solver.getAlgorithmState()->iter;

                std::cout << "\n\n" << init_rad[i] << ", " << rad_grow_rate[j] << ", " << rad_grow_thresh[k] << ", " << num_iter << "\n";

                if (num_iter < min_iter) {
                    min_iter = num_iter;
                    best_init_rad = init_rad[i];
                    best_rad_grow_rate = rad_grow_rate[j];
                    best_rad_grow_thresh = rad_grow_thresh[k];
                    
                    print_results(print_centers, print_assignments, x, obj, num_moths);
                }
                
            }
        }
    }

    // Print best solution
    std::cout << "MIN ITER: " << min_iter << "\n";
    std::cout << "INITIAL RADIUS: " << best_init_rad << "\n";
    std::cout << "RADIUS GROWING RATE: " << best_rad_grow_rate << "\n";
    std::cout << "RADIUS GROWING THRESH: " << best_rad_grow_thresh << "\n";
}


void runTrustRegionStepAcc(const std::vector<Detection> &detections, 
                                                        int num_moths, 
                                                        int verbosity, 
                                                        bool print_centers, 
                                                        bool print_assignments) {
    
    std::string method = "Trust Region";

    std::vector<double> step_acc_thresh = {0.1, 0.3, 0.5, 0.7, 0.9};

    int num_detections = detections.size();
                                                        
    for (std::size_t i = 0; i < step_acc_thresh.size(); ++i) {
        
        // Create initial guess
        int num_z = 2 * num_moths;
        auto initial_guess = std::make_shared<std::vector<double>>(num_z, 0.0);

        for (int k = 0; k < num_moths; ++k) {
            int idx = k * static_cast<int>(num_detections) / num_moths;
            (*initial_guess)[2*k]     = detections[idx].x;
            (*initial_guess)[2*k + 1] = detections[idx].y;
        }
        
        // Define problem 
        auto x = ROL::makePtr<ROL::StdVector<double>>(initial_guess);
        auto obj = ROL::makePtr<MothObjective<double>>(detections, num_moths);
        auto problem = ROL::makePtr<ROL::Problem<double>>(obj, x);

        // Configure parameter list
        ROL::ParameterList parlist;
        
        parlist.sublist("Step").set("Type", method);
        parlist.sublist("Step").sublist(method).set("Step Acceptance Threshold", step_acc_thresh[i]);
        parlist.sublist("General").set("Output Level", verbosity);

        // Solve problem
        ROL::Solver<double> solver(problem, parlist);
        solver.solve(std::cout);

        int final_iter = solver.getAlgorithmState()->iter;
        double num_iter = solver.getAlgorithmState()->iter;

        std::cout << "\n\n" << step_acc_thresh[i]  << ", " << num_iter << "\n";
        
        print_results(print_centers, print_assignments, x, obj, num_moths);
    }
}


void runLineSearch(const std::vector<Detection> &detections, 
                                                int num_moths, 
                                                int verbosity, 
                                                bool print_centers, 
                                                bool print_assignments) {
    
    std::string method = "Line Search";

    std::vector<std::string> line_search_method = {"Iteration Scaling", 
                                                    "Backtracking", 
                                                    "Path-Based Target Level",
                                                    "Cubic Interpolation",
                                                    "Bisection",
                                                    "Golden Section",
                                                    "Brent's"};

    int num_detections = detections.size();
                                                        
    for (std::size_t i = 0; i < line_search_method.size(); ++i) {
        
        // Create initial guess
        int num_z = 2 * num_moths;
        auto initial_guess = std::make_shared<std::vector<double>>(num_z, 0.0);

        for (int k = 0; k < num_moths; ++k) {
            int idx = k * static_cast<int>(num_detections) / num_moths;
            (*initial_guess)[2*k]     = detections[idx].x;
            (*initial_guess)[2*k + 1] = detections[idx].y;
        }
        
        // Define problem 
        auto x = ROL::makePtr<ROL::StdVector<double>>(initial_guess);
        auto obj = ROL::makePtr<MothObjective<double>>(detections, num_moths);
        auto problem = ROL::makePtr<ROL::Problem<double>>(obj, x);

        // Configure parameter list
        ROL::ParameterList parlist;
        
        parlist.sublist("Step").set("Type", method);
        parlist.sublist("Step").sublist(method).sublist("Line-Search Method").set("Type", line_search_method[i]);
        parlist.sublist("General").set("Output Level", verbosity);

        // Solve problem
        ROL::Solver<double> solver(problem, parlist);
        solver.solve(std::cout);

        int final_iter = solver.getAlgorithmState()->iter;
        double num_iter = solver.getAlgorithmState()->iter;

        std::cout << "\n\n" << line_search_method[i]  << ", " << num_iter << "\n";
        
        print_results(print_centers, print_assignments, x, obj, num_moths);
    }
}


int main() {
    // Select user-defined options
    std::string method = "Line Search"; // Other options: Line Search
    bool tr_grid_search = false; 
    bool print_centers = false, print_assignments = true;
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
    
    int num_moths = 3;

    if (tr_grid_search) {
        runTrustRegionGridSearch(detections, num_moths, verbosity, print_centers, print_assignments);
    } else if (method == "Trust Region") {
        runTrustRegionStepAcc(detections, num_moths, verbosity, print_centers, print_assignments);
    } else {
        runLineSearch(detections, num_moths, verbosity, print_centers, print_assignments); // define this
    }

    return 0;
}