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
#include "dynamics_vio/dynamics_base.h"
#include "dynamics_vio/parameters/parameters.h"

#include <basalt/utils/imu_types.h>
#include <acado_toolkit.hpp>

namespace dynvio {

struct SingleTrackAcado{

    SingleTrackAcado(double mass, double wheel_base, double I_z, double l_front);
    ~SingleTrackAcado();

    ACADO::DifferentialEquation f;

    // acado state
    ACADO::DifferentialState trans_x, trans_y, yaw;
    ACADO::DifferentialState velocity_x, velocity_y, yaw_rate;
    ACADO::Parameter C_lat, Steering_rt, Throttle_f1, Throttle_f2, Throttle_res;
    ACADO::Control Cmd_linear, Cmd_angular;

    ACADO::IntermediateState F_x, F_front_y, F_rear_y;

private:
    //fixed parameters
    double mass;
    double wheel_base;
    double I_z;
    double l_front;
};


class SingleTrackModel:public DynamicsBase{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr = std::shared_ptr<SingleTrackModel>;
    using Mat56 = Eigen::Matrix<double, 5, 6>; // 6 for rot and vel
    using Mat63 = Eigen::Matrix<double, 6, 3>; // jacobian for bias
    using Mat66 = Eigen::Matrix<double, 6, 6>; // 6 for pose
    using Vec6 = Eigen::Matrix<double, 6, 1>;

    SingleTrackModel(int64_t start_t_ns, const Eigen::Vector3d& start_gyro, int64_t end_t_ns, const Eigen::Vector3d &end_gyro, const std::deque<Command> &command_history);

    static void linearizeDynamics(const SingleTrackAcado& singletrack_acado, const AbsOrderMap& aom, Eigen::MatrixXd& abs_H, Eigen::VectorXd& abs_b,
                                  double& dyn_error, double &extr_error, const Eigen::aligned_map<int64_t, ParameterWithLin<SingleTrackParamOnline> > &param_states,
                                  const Eigen::aligned_map<int64_t, PoseVelBiasStateWithLin<double>>& frame_states, const Eigen::aligned_map<int64_t, ExtrinsicStateWithLin<double> > &extr_states,
                                  const Eigen::aligned_map<int64_t, SingleTrackModel::Ptr>& dynamics_factors, const double &dyn_weight, const double &extr_rd_weight);
    static void computeDynamicsError(const SingleTrackAcado& singletrack_acado, const AbsOrderMap& aom, double &dyn_error, double &extr_error, const Eigen::aligned_map<int64_t, ParameterWithLin<SingleTrackParamOnline> > &param_states,
                                     const Eigen::aligned_map<int64_t, PoseVelBiasStateWithLin<double>>& frame_states, const Eigen::aligned_map<int64_t, ExtrinsicStateWithLin<double> > &extr_states,
                                     const Eigen::aligned_map<int64_t, SingleTrackModel::Ptr>& dynamics_factors, const double &dyn_weight, const double &extr_rd_weight);

    Vec6 residual(const SingleTrackAcado& singletrack_acado,
                  const PoseVelBiasState<double>& state0, const ExtrinsicState<double> &extr_state0,
                  const PoseVelBiasState<double> &state1, const ExtrinsicState<double> &extr_state1,
                  const SingleTrackParamOnline &param_state,
                  Eigen::Vector3d* pred_pose_state = nullptr,
                  Eigen::Vector3d* pred_vel_state = nullptr,
                  Mat66 *d_pred_d_state0 = nullptr,
                  Mat66* d_pred_d_extr0 = nullptr,
                  Mat66 *d_res_d_state0 = nullptr,
                  Mat63 *d_res_d_bias0 = nullptr,
                  Mat66 *d_res_d_extr0 = nullptr,
                  Mat56 *d_res_d_state1 = nullptr,
                  Eigen::Matrix3d *d_res_d_bias1 = nullptr,
                  Mat66* d_res_d_extr1 = nullptr,
                  Eigen::MatrixXd *d_res_d_param = nullptr) const;

    Vec6 residual(const SingleTrackAcado& singletrack_acado, const int64_t& curr_t_ns,
                  const PoseVelBiasState<double>& state0, const ExtrinsicState<double> &extr_state0,
                  const PoseVelBiasState<double> &state1, const ExtrinsicState<double> &extr_state1,
                  const SingleTrackParamOnline &param_state,
                  Eigen::Vector3d* pred_pose_state = nullptr,
                  Eigen::Vector3d* pred_vel_state = nullptr,
                  Mat66* d_pred_d_stateinit = nullptr,
                  Mat66* d_pred_d_extrinit = nullptr,
                  Mat66 *d_res_d_stateinit = nullptr,
                  Mat63 *d_res_d_biasinit = nullptr,
                  Mat66 *d_res_d_extrinit = nullptr,
                  Mat56* d_res_d_state1 = nullptr,
                  Eigen::Matrix3d *d_res_d_bias1 = nullptr,
                  Mat66* d_res_d_extr1 = nullptr,
                  Eigen::MatrixXd *d_res_d_param = nullptr) const;

    Vec6 residual_cmdstep(const SingleTrackAcado& singletrack_acado,
                          const SingleTrackParamOnline &param_state,
                          const Eigen::Vector3d& r_o0_o1,
                          const Eigen::Vector3d& t_o0_o1,
                          const Eigen::Vector3d& cg_end_lin,
                          const Eigen::Vector3d& cg_end_ang,
                          Eigen::Vector3d* pred_pose_state,
                          Eigen::Vector3d* pred_vel_state,
                          Mat66 *d_pred_d_stateinit,
                          Mat66 *d_pred_d_extrinit,
                          Mat66 *d_res_d_stateinit,
                          Mat63 *d_res_d_biasinit,
                          Mat66 *d_res_d_extrinit,
                          Eigen::MatrixXd *d_pred_d_param) const;

    Eigen::Vector3d get_start_gyro() const;
    std::deque<Command> get_command_window() const;

private:
    Eigen::Vector3d start_gyro;
    Eigen::Vector3d end_gyro;

    std::deque<Command> command_window;
    bool first_frame{true};
};
} // namespace dynvio
