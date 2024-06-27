/**
This file is part of ST-VIO.

Copyright (c) 2024, Max Planck Gesellschaft.
Developed by Haolong Li <haolong.li at tue dot mpg dot de>
Embodied Vision Group, Max Planck Institute for Intelligent Systems.
If you use this code, please cite the respective publication as
listed in README.md.

ST-VIO is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ST-VIO is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with ST-VIO.  If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once

#include "dynamics_vio/utils/utils.h"
#include "dynamics_vio/parameters/parameters.h"

#include <basalt/utils/imu_types.h>

namespace dynvio {

class ConstraintsFactor
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using Ptr = std::shared_ptr<ConstraintsFactor>;
    using Vec6 = Eigen::Matrix<double, 6, 1>;

    ConstraintsFactor(int64_t t_ns, double init_distance, double l_o_imu, double l_c_imu,
                      const Eigen::Vector3d& imu_forward);

    Vec6 residual(const PoseVelBiasState<double>& state,
                  const ExtrinsicState<double> &extr_state,
                  Eigen::Matrix<double, 3, 6>* d_res_d_state = nullptr,
                  Eigen::Matrix<double, 6, 6> *d_res_d_extr = nullptr) const;

    static void linearizeConstraints(const AbsOrderMap& aom, Eigen::MatrixXd& abs_H, Eigen::VectorXd& abs_b, double& constr_error, double &param_error,
                                     const SingleTrackParamOnline* param_init_prior, const SingleTrackParamOnline* param_marged_prior,
                                     const Eigen::aligned_map<int64_t, ParameterWithLin<SingleTrackParamOnline> > &param_states,
                                     const Eigen::aligned_map<int64_t, PoseVelBiasStateWithLin<double> > &frame_states,
                                     const Eigen::aligned_map<int64_t, ExtrinsicStateWithLin<double> > &extr_states,
                                     const Eigen::aligned_map<int64_t, ConstraintsFactor::Ptr> &constr_factors, const double &weight,
                                     const Eigen::Matrix<double, PARAM_SIZE, 1>& param_init_weight,
                                     const Eigen::Matrix<double, PARAM_SIZE, 1>& param_2nd_weight);

    static void computeConstraintsError(const AbsOrderMap& aom, double& constr_error, double &param_error,
                                        const SingleTrackParamOnline* param_init_prior, const SingleTrackParamOnline* param_marged_prior,
                                        const Eigen::aligned_map<int64_t, ParameterWithLin<SingleTrackParamOnline> > &param_states,
                                        const Eigen::aligned_map<int64_t, PoseVelBiasStateWithLin<double> > &frame_states,
                                        const Eigen::aligned_map<int64_t, ExtrinsicStateWithLin<double> > &extr_states,
                                        const Eigen::aligned_map<int64_t, ConstraintsFactor::Ptr> &constr_factors, const double& weight,
                                        const Eigen::Matrix<double, PARAM_SIZE, 1>& param_init_weight,
                                        const Eigen::Matrix<double, PARAM_SIZE, 1>& param_2nd_weight);

    int64_t get_t_ns() const;

private:
    int64_t t_ns;
    double init_distance;

    //displacement along forward direction, 0.041m is the distance between front wheel to t265 imu frame.
    double l_o_imu{0.041};
    //displacement along y direction, 0.0311m is distance between camera housing center to imu.
    double l_c_imu{0.0311};

    Eigen::Vector3d imu_forward{0, 0, 1};
};

}
