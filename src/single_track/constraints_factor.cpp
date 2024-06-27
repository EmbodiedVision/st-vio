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

#include "dynamics_vio/single_track/constraints_factor.h"

namespace dynvio {

ConstraintsFactor::ConstraintsFactor(int64_t t_ns, double init_distance,
                                     double l_o_imu, double l_c_imu, const Eigen::Vector3d &imu_forward):t_ns(t_ns), init_distance(init_distance),
                                                    l_o_imu(l_o_imu), l_c_imu(l_c_imu),
                                                    imu_forward(imu_forward){}

int64_t ConstraintsFactor::get_t_ns() const
{
    return t_ns;
}

ConstraintsFactor::Vec6 ConstraintsFactor::residual(const PoseVelBiasState<double>& state,
                                                    const ExtrinsicState<double> &extr_state,
                                                    Eigen::Matrix<double, 3, 6>* d_res_d_state,
                                                    Eigen::Matrix<double, 6, 6> *d_res_d_extr) const{
    //plane angle constraint
    Sophus::SO3d R_w_o = state.T_w_i.so3() * extr_state.T_o_i.so3().inverse();

    Eigen::Vector3d z_base(0.0, 0.0, 1e1);
    Eigen::Vector3d z_ground = R_w_o * z_base;

    ConstraintsFactor::Vec6 res;
    res.setZero();
    res(0) = z_ground(0); //roll
    res(1) = z_ground(1); //pitch

    //z distance
    Eigen::Vector3d tmp = -(R_w_o * extr_state.T_o_i.translation()); //B
    Eigen::Vector3d t_w_o = tmp + state.T_w_i.translation();
    res(2) = t_w_o(2) - init_distance;

    //extrinsic yaw constraint, the z direction of imu frame is facing forward,
    //the yaw angle difference between base and imu should close to 0.
    Eigen::Vector3d forward_cg = extr_state.T_o_i.so3() * imu_forward;
    res(3) = forward_cg(1); //yaw

    res(4) = extr_state.T_o_i.translation()(0) - l_o_imu;
    res(5) = extr_state.T_o_i.translation()(1) - l_c_imu;
    if(d_res_d_state || d_res_d_extr){
        Eigen::Matrix<double, 2, 3> z_ground_hat_xy;
        z_ground_hat_xy << 0.0, -z_ground(2), z_ground(1),
                           z_ground(2), 0.0, -z_ground(0);
        Eigen::Matrix<double, 1, 3> tmp_hat_z(-tmp(1), tmp(0), 0.0);
        if(d_res_d_state){
            d_res_d_state->setZero();
            d_res_d_state->block<2, 3>(0, 3) = -z_ground_hat_xy;

            (*d_res_d_state)(2, 2) = 1.0;
            d_res_d_state->block<1, 3>(2, 3) = -tmp_hat_z;

        }
        if(d_res_d_extr){
            d_res_d_extr->setZero();
            Eigen::Matrix3d R_w_o_mat = R_w_o.matrix();
            d_res_d_extr->block<2, 3>(0, 3) = z_ground_hat_xy * R_w_o_mat;


            d_res_d_extr->block<1, 3>(2, 0) = - R_w_o_mat.bottomRows<1>();
            d_res_d_extr->block<1, 3>(2, 3) = tmp_hat_z * R_w_o_mat;

            Eigen::Matrix<double, 1, 3> forward_cg_hat;
            forward_cg_hat << forward_cg(2), 0.0, -forward_cg(0);
            d_res_d_extr->block<1, 3>(3, 3) = -forward_cg_hat;

            (*d_res_d_extr)(4, 0) = 1.0;
            (*d_res_d_extr)(5, 1) = 1.0;
        }
     }

    return  res;
}
} // namespace dynvio
