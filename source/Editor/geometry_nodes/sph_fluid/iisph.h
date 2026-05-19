#pragma once 
#include "utils.h"
#include "particle_system.h"
#include "sph_base.h"
#include <Eigen/Dense>
#include <vector>

namespace USTC_CG::sph_fluid {

using namespace Eigen;

class IISPH : public SPHBase {
   public:
    IISPH() = default;
    IISPH(const MatrixXd& X, const Vector3d& box_min, const Vector3d& box_max);
    ~IISPH() = default;

    void step() override;
    void compute_pressure() override;

    double pressure_solve_iteration();
    void predict_advection();

    void reset() override;

    int& max_iter()
    {
        return max_iter_;
    }
    double& omega()
    {
        return omega_;
    }
    double& pressure_clamp()
    {
        return pressure_clamp_;
    }

   protected:
    void resize_solver_storage();

    int max_iter_ = 50;
    double omega_ = 0.5;
    double pressure_clamp_ = 1e5;
    int last_pressure_iterations_ = 0;
    double last_pressure_residual_ = 0.0;

    VectorXd predict_density_;
    VectorXd aii_;
    VectorXd Api_;  
    VectorXd last_pressure_;
    VectorXd pressure_old_;
    std::vector<Vector3d> pressure_acc_;
};
}  // namespace USTC_CG::node_sph_fluid
