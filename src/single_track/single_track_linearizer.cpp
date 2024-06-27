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

#include "dynamics_vio/single_track/single_track.h"

namespace dynvio {

void SingleTrackModel::linearizeDynamics(const SingleTrackAcado& singletrack_acado, const AbsOrderMap &aom, Eigen::MatrixXd &abs_H,
                                         Eigen::VectorXd &abs_b, double &dyn_error, double &extr_error,
                                         const Eigen::aligned_map<int64_t, ParameterWithLin<SingleTrackParamOnline>> &param_states,
                                         const Eigen::aligned_map<int64_t, PoseVelBiasStateWithLin<double> > &frame_states,
                                         const Eigen::aligned_map<int64_t, ExtrinsicStateWithLin<double> > &extr_states,
                                         const Eigen::aligned_map<int64_t, SingleTrackModel::Ptr> &dynamics_factors,
                                         const double& dyn_weight, const double& extr_rd_weight){
    dyn_error = 0;
    extr_error = 0;
    bool all_state_linearized = false;
    int64_t init_t_ns = 0;
    int init_idx = -1;
    int state_init_idx = 0, extr_init_idx = 0;
    int bias_init_idx = 0;
    SingleTrackModel::Mat66 d_pred_d_stateinit;
    d_pred_d_stateinit.setZero();
    SingleTrackModel::Mat66 d_pred_d_extrinit;
    d_pred_d_extrinit.setZero();
    SingleTrackModel::Mat63 d_pred_d_biasinit;
    d_pred_d_biasinit.setZero();
    Eigen::MatrixXd d_pred_d_param;
    d_pred_d_param.setZero(6, PARAM_SIZE);
    Eigen::Vector3d pred_pose_state_lin, pred_vel_state_lin;
    Eigen::Vector3d pred_pose_state, pred_vel_state;

    const auto& param = param_states.begin()->second;
    const size_t param_idx = aom.abs_order_map.at(param_states.begin()->first).first + POSE_VEL_BIAS_SIZE + EXTR_SIZE;

    for(const auto& stamped_factor : dynamics_factors){
        SingleTrackModel::Ptr dynamics = stamped_factor.second;

        int64_t start_t = dynamics->get_start_t_ns();
        int64_t end_t = dynamics->get_end_t_ns();

        if (aom.abs_order_map.count(start_t) == 0 ||
        aom.abs_order_map.count(end_t) == 0)
        continue;

        if(init_idx < 0){
            init_t_ns = start_t;
            init_idx = aom.abs_order_map.at(start_t).first;
            state_init_idx = init_idx + 3; // start from rotation
            bias_init_idx = init_idx + POSE_VEL_SIZE;
            extr_init_idx = init_idx + POSE_VEL_BIAS_SIZE;

            BASALT_ASSERT(init_t_ns == param_states.begin()->first);
        }

        BASALT_ASSERT(init_idx <= aom.abs_order_map.at(start_t).first);
        const int state_end_idx = aom.abs_order_map.at(end_t).first + 3;
        const int bias_end_idx = state_end_idx - 3 + POSE_VEL_SIZE;
        const int extr_end_idx = state_end_idx - 3 + POSE_VEL_BIAS_SIZE;

        SingleTrackModel::Vec6 res;
        SingleTrackModel::Mat56 d_res_d_state1;
        d_res_d_state1.setZero();
        SingleTrackModel::Mat66 d_res_d_state_init, d_res_d_extrinit, d_res_d_extr1;
        d_res_d_state_init.setZero();
        d_res_d_extrinit.setZero();
        d_res_d_extr1.setZero();
        Eigen::Matrix3d d_res_d_bias1;
        d_res_d_bias1.setZero();

        // first state
        if(aom.abs_order_map.at(start_t).first == init_idx){
            auto& init_state = frame_states.at(init_t_ns);
            auto& init_extr = extr_states.at(init_t_ns);
            auto& end_state = frame_states.at(end_t);
            auto& end_extr = extr_states.at(end_t);
            res = dynamics->residual(singletrack_acado, init_state.getStateLin(), init_extr.getStateLin(),
                                     end_state.getStateLin(), end_extr.getStateLin(),
                                     param.getParamLin(), &pred_pose_state_lin, &pred_vel_state_lin,
                                     &d_pred_d_stateinit, &d_pred_d_extrinit,
                                     &d_res_d_state_init, &d_pred_d_biasinit, &d_res_d_extrinit,
                                     &d_res_d_state1, &d_res_d_bias1, &d_res_d_extr1,
                                     &d_pred_d_param);

            all_state_linearized = all_state_linearized || init_state.isLinearized() || init_extr.isLinearized() || end_state.isLinearized() ||
                    end_extr.isLinearized() || param.isLinearized();
            if(all_state_linearized)
                res = dynamics->residual(singletrack_acado, init_state.getState(), init_extr.getState(),
                                         end_state.getState(), end_extr.getState(),
                                         param.getParam(), &pred_pose_state, &pred_vel_state);
        }
        else{
            auto& init_state = frame_states.at(init_t_ns);
            auto& init_extr = extr_states.at(init_t_ns);
            auto& end_state = frame_states.at(end_t);
            auto& end_extr = extr_states.at(end_t);
            res = dynamics->residual(singletrack_acado, start_t, init_state.getStateLin(), init_extr.getStateLin(),
                                     end_state.getStateLin(), end_extr.getStateLin(),
                                     param.getParamLin(), &pred_pose_state_lin, &pred_vel_state_lin,
                                     &d_pred_d_stateinit, &d_pred_d_extrinit,
                                     &d_res_d_state_init, &d_pred_d_biasinit, &d_res_d_extrinit,
                                     &d_res_d_state1, &d_res_d_bias1, &d_res_d_extr1,
                                     &d_pred_d_param);

            if(init_state.isLinearized() || init_extr.isLinearized() || end_state.isLinearized() ||
               end_extr.isLinearized() || param.isLinearized() || all_state_linearized){
                BASALT_ASSERT(all_state_linearized);
                res = dynamics->residual(singletrack_acado, start_t, init_state.getState(), init_extr.getState(),
                                         end_state.getState(), end_extr.getState(),
                                         param.getParam(), &pred_pose_state, &pred_vel_state);
                all_state_linearized = all_state_linearized || init_state.isLinearized() || init_extr.isLinearized() || end_state.isLinearized() ||
                        end_extr.isLinearized() || param.isLinearized();
            }
        }

        dyn_error += 0.5 * res.transpose() * dyn_weight * res;
        // poses and velocity
        Eigen::Matrix<double, 6, 6> J_state0_wt = d_res_d_state_init.transpose() * dyn_weight;
        abs_H.block<6, 6>(state_init_idx, state_init_idx) += J_state0_wt * d_res_d_state_init;
        abs_b.segment<6>(state_init_idx) += J_state0_wt * res;

        Eigen::Matrix<double, 6, 5> J_state1_wt = d_res_d_state1.transpose() * dyn_weight;
        abs_H.block<6, 6>(state_end_idx, state_end_idx) += J_state1_wt * d_res_d_state1;
        abs_b.segment<6>(state_end_idx) += J_state1_wt * res.topRows<5>();

        // bias
        Eigen::Matrix<double, 3, 6> J_bias0_wt = d_pred_d_biasinit.transpose() * dyn_weight;
        abs_H.block<3, 3>(bias_init_idx, bias_init_idx) += J_bias0_wt * d_pred_d_biasinit;
        abs_b.segment<3>(bias_init_idx) += J_bias0_wt * res;

        Eigen::Matrix3d J_bias1_wt = d_res_d_bias1.transpose() * dyn_weight;
        abs_H.block<3, 3>(bias_end_idx, bias_end_idx) += J_bias1_wt * d_res_d_bias1;
        abs_b.segment<3>(bias_end_idx) += J_bias1_wt * res.tail<3>();

        // extr
        Eigen::Matrix<double, 6, 6> J_extr0_wt = d_res_d_extrinit.transpose() * dyn_weight;
        abs_H.block<6, 6>(extr_init_idx, extr_init_idx) += J_extr0_wt * d_res_d_extrinit;
        abs_b.segment<6>(extr_init_idx) += J_extr0_wt * res;

        Eigen::Matrix<double, 6, 6> J_extr1_wt = d_res_d_extr1.transpose() * dyn_weight;
        abs_H.block<6, 6>(extr_end_idx, extr_end_idx) += J_extr1_wt * d_res_d_extr1;
        abs_b.segment<6>(extr_end_idx) += J_extr1_wt * res;

        // param
        Eigen::MatrixXd J_param_wt = d_pred_d_param.transpose() * dyn_weight;
        abs_H.block<PARAM_SIZE, PARAM_SIZE>(param_idx, param_idx)
                += J_param_wt * d_pred_d_param;
        abs_b.segment<PARAM_SIZE>(param_idx) += J_param_wt * res;

        // off-diagonal state0 & state1`
        Eigen::Matrix<double, 6, 6> Hpose0pose1 = J_state0_wt.leftCols<5>() * d_res_d_state1;
        abs_H.block<6 ,6>(state_init_idx, state_end_idx) += Hpose0pose1;
        abs_H.block<6 ,6>(state_end_idx, state_init_idx) += Hpose0pose1.transpose();
        // off-diagonal state0 & bias0
        Eigen::Matrix<double, 6, 3> Hpose0bias0 = J_state0_wt * d_pred_d_biasinit;
        abs_H.block<6, 3>(state_init_idx, bias_init_idx) += Hpose0bias0;
        abs_H.block<3, 6>(bias_init_idx, state_init_idx) += Hpose0bias0.transpose();
        // off-diagonal state0 & extr0
        Eigen::Matrix<double, 6, 6> Hpose0extr0 = J_state0_wt * d_res_d_extrinit;
        abs_H.block<6, 6>(state_init_idx, extr_init_idx) += Hpose0extr0;
        abs_H.block<6, 6>(extr_init_idx, state_init_idx) += Hpose0extr0.transpose();
        // off-diagonal state0 & bias1
        Eigen::Matrix<double, 6, 3> Hpose0bias1 = J_state0_wt.rightCols<3>() * d_res_d_bias1;
        abs_H.block<6, 3>(state_init_idx, bias_end_idx) += Hpose0bias1;
        abs_H.block<3, 6>(bias_end_idx, state_init_idx) += Hpose0bias1.transpose();
        // off-diagonal state0 & extr1
        Eigen::Matrix<double, 6, 6> Hpose0extr1 = J_state0_wt * d_res_d_extr1;
        abs_H.block<6, 6>(state_init_idx, extr_end_idx) += Hpose0extr1;
        abs_H.block<6, 6>(extr_end_idx, state_init_idx) += Hpose0extr1.transpose();
        // off-diagonal state0 & param
        Eigen::Matrix<double, 6, PARAM_SIZE> Hpose0param = J_state0_wt * d_pred_d_param;
        abs_H.block<6, PARAM_SIZE>(state_init_idx, param_idx) += Hpose0param;
        abs_H.block<PARAM_SIZE, 6>(param_idx, state_init_idx) += Hpose0param.transpose();
        // off-diagonal state1 & bias0
        Eigen::Matrix<double, 6, 3> Hpose1bias0 = J_state1_wt * d_pred_d_biasinit.topRows<5>();
        abs_H.block<6, 3>(state_end_idx, bias_init_idx) += Hpose1bias0;
        abs_H.block<3, 6>(bias_init_idx, state_end_idx) += Hpose1bias0.transpose();
        // off-diagonal state1 & extr0
        Eigen::Matrix<double, 6, 6> Hpose1extr0 = J_state1_wt * d_res_d_extrinit.topRows<5>();
        abs_H.block<6, 6>(state_end_idx, extr_init_idx) += Hpose1extr0;
        abs_H.block<6, 6>(extr_init_idx, state_end_idx) += Hpose1extr0.transpose();
        // off-diagonal state1 & bias1
        Eigen::Matrix<double, 6, 3> Hpose1bias1 = J_state1_wt.rightCols<2>() * d_res_d_bias1.topRows<2>();
        abs_H.block<6, 3>(state_end_idx, bias_end_idx) += Hpose1bias1;
        abs_H.block<3, 6>(bias_end_idx, state_end_idx) += Hpose1bias1.transpose();
        // off-diagonal state1 & extr1
        Eigen::Matrix<double, 6, 6> Hpose1extr1 = J_state1_wt * d_res_d_extr1.topRows<5>();
        abs_H.block<6, 6>(state_end_idx, extr_end_idx) += Hpose1extr1;
        abs_H.block<6, 6>(extr_end_idx, state_end_idx) += Hpose1extr1.transpose();
        // off-diagonal state1 & param
        Eigen::Matrix<double, 6, PARAM_SIZE> Hpose1param = J_state1_wt * d_pred_d_param.topRows<5>();
        abs_H.block<6, PARAM_SIZE>(state_end_idx, param_idx) += Hpose1param;
        abs_H.block<PARAM_SIZE, 6>(param_idx, state_end_idx) += Hpose1param.transpose();
        // off-diagonal extr0 & bias0
        Eigen::Matrix<double, 6, 3> Hextr0bias0 = J_extr0_wt * d_pred_d_biasinit;
        abs_H.block<6, 3>(extr_init_idx, bias_init_idx) += Hextr0bias0;
        abs_H.block<3, 6>(bias_init_idx, extr_init_idx) += Hextr0bias0.transpose();
        // off-diagonal extr0 & bias1
        Eigen::Matrix<double, 6, 3> Hextr0bias1 = J_extr0_wt.rightCols<3>() * d_res_d_bias1;
        abs_H.block<6, 3>(extr_init_idx, bias_end_idx) += Hextr0bias1;
        abs_H.block<3, 6>(bias_end_idx, extr_init_idx) += Hextr0bias1.transpose();
        // off-diagonal extr0 & extr1
        Eigen::Matrix<double, 6, 6> Hextr0extr1 = J_extr0_wt * d_res_d_extr1;
        abs_H.block<6, 6>(extr_init_idx, extr_end_idx) += Hextr0extr1;
        abs_H.block<6, 6>(extr_end_idx, extr_init_idx) += Hextr0extr1.transpose();
        // off-diagonal extr0 & param
        Eigen::Matrix<double, 6, PARAM_SIZE> Hextr0param = J_extr0_wt * d_pred_d_param;
        abs_H.block<6, PARAM_SIZE>(extr_init_idx, param_idx) += Hextr0param;
        abs_H.block<PARAM_SIZE, 6>(param_idx, extr_init_idx) += Hextr0param.transpose();
        // off-diagonal extr1 & bias0
        Eigen::Matrix<double, 6, 3> Hextr1bias0 = J_extr1_wt * d_pred_d_biasinit;
        abs_H.block<6, 3>(extr_end_idx, bias_init_idx) += Hextr1bias0;
        abs_H.block<3, 6>(bias_init_idx, extr_end_idx) += Hextr1bias0.transpose();
        // off-diagonal extr1 & bias1
        Eigen::Matrix<double, 6, 3> Hextr1bias1 = J_extr1_wt.rightCols<3>() * d_res_d_bias1;
        abs_H.block<6, 3>(extr_end_idx, bias_end_idx) += Hextr1bias1;
        abs_H.block<3, 6>(bias_end_idx, extr_end_idx) += Hextr1bias1.transpose();
        // off-diagonal extr1 & param
        Eigen::Matrix<double, 6, PARAM_SIZE> Hextr1param = J_extr1_wt * d_pred_d_param;
        abs_H.block<6, PARAM_SIZE>(extr_end_idx, param_idx) += Hextr1param;
        abs_H.block<PARAM_SIZE, 6>(param_idx, extr_end_idx) += Hextr1param.transpose();
        // off-diagonal param & bias0
        Eigen::Matrix<double, PARAM_SIZE, 3> Hparambias0 = J_param_wt * d_pred_d_biasinit;
        abs_H.block<PARAM_SIZE, 3>(param_idx, bias_init_idx) += Hparambias0;
        abs_H.block<3, PARAM_SIZE>(bias_init_idx, param_idx) += Hparambias0.transpose();
        // off-diagonal param & bias1
        Eigen::Matrix<double, PARAM_SIZE, 3> Hparambias1 = J_param_wt.rightCols(3) * d_res_d_bias1;
        abs_H.block<PARAM_SIZE, 3>(param_idx, bias_end_idx) += Hparambias1;
        abs_H.block<3, PARAM_SIZE>(bias_end_idx, param_idx) += Hparambias1.transpose();
        // off-diagonal bias0 & bias1
        Eigen::Matrix3d Hbias0bias1 = J_bias0_wt.rightCols<3>() * d_res_d_bias1;
        abs_H.block<3, 3>(bias_init_idx, bias_end_idx) += Hbias0bias1;
        abs_H.block<3, 3>(bias_end_idx, bias_init_idx) += Hbias0bias1.transpose();

        // loose coupling
        // extr
        {
            auto& start_extr_state = extr_states.at(start_t);
            const int extr_start_idx = aom.abs_order_map.at(start_t).first + POSE_VEL_BIAS_SIZE;
            auto& end_extr_state = extr_states.at(end_t);

            double extr_weight = extr_rd_weight;
            Eigen::Matrix<double, 6, 1> res_extr;
            res_extr.head<3>() = end_extr_state.getState().T_o_i.translation() - start_extr_state.getState().T_o_i.translation();
            res_extr.tail<3>() = (start_extr_state.getState().T_o_i.so3().inverse() * end_extr_state.getState().T_o_i.so3()).log();

            Sophus::SO3d T_i0_o_Lin = start_extr_state.getStateLin().T_o_i.so3().inverse();
            Eigen::Vector3d extrLin_rot_diff = (T_i0_o_Lin * end_extr_state.getStateLin().T_o_i.so3()).log();
            Eigen::Matrix<double, 3, 3> J_left;
            Sophus::leftJacobianInvSO3(extrLin_rot_diff,J_left);

            Eigen::Matrix<double, 6, 6> d_diff_extr1;
            d_diff_extr1.setIdentity();
            d_diff_extr1.block<3, 3>(3, 3) = J_left * T_i0_o_Lin.matrix();

            Eigen::Matrix<double, 6, 6> J_diff_extr1weighted = d_diff_extr1.transpose() * extr_weight;
            Eigen::Matrix<double, 6, 6> H_diff_extr1extr1 = J_diff_extr1weighted * d_diff_extr1;
            abs_H.block<6, 6>(extr_start_idx, extr_start_idx) += H_diff_extr1extr1;
            abs_H.block<6, 6>(extr_start_idx, extr_end_idx) -= H_diff_extr1extr1;
            abs_H.block<6, 6>(extr_end_idx, extr_start_idx) -= H_diff_extr1extr1;
            abs_H.block<6, 6>(extr_end_idx, extr_end_idx) += H_diff_extr1extr1;

            abs_b.segment<6>(extr_start_idx) -= J_diff_extr1weighted * res_extr;
            abs_b.segment<6>(extr_end_idx) += J_diff_extr1weighted * res_extr;

            extr_error += 0.5 * res_extr.transpose() * extr_weight * res_extr;
        }
    } // loop over factors
}

void SingleTrackModel::computeDynamicsError(const SingleTrackAcado& singletrack_acado, const AbsOrderMap &aom, double &dyn_error, double &extr_error,
                                            const Eigen::aligned_map<int64_t, ParameterWithLin<SingleTrackParamOnline>> &param_states,
                                            const Eigen::aligned_map<int64_t, PoseVelBiasStateWithLin<double> > &frame_states,
                                            const Eigen::aligned_map<int64_t, ExtrinsicStateWithLin<double> > &extr_states,
                                            const Eigen::aligned_map<int64_t, SingleTrackModel::Ptr> &dynamics_factors,
                                            const double& dyn_weight, const double& extr_rd_weight){

    dyn_error = 0;
    extr_error = 0;
    int64_t init_t_ns = 0;
    int init_idx = -1;
    Eigen::Vector3d pred_pose_state, pred_vel_state;
    const auto& param = param_states.begin()->second;

    for(const auto& stamped_factor : dynamics_factors){

        SingleTrackModel::Ptr dynamics = stamped_factor.second;

        int64_t start_t = dynamics->get_start_t_ns();
        int64_t end_t = dynamics->get_end_t_ns();

        if (aom.abs_order_map.count(start_t) == 0 ||
            aom.abs_order_map.count(end_t) == 0)
            continue;

        if(init_idx < 0){
            init_idx = aom.abs_order_map.at(start_t).first;
            init_t_ns = start_t;
        }

        BASALT_ASSERT(init_idx <= aom.abs_order_map.at(start_t).first);

        Vec6 res;
        // first state
        if(aom.abs_order_map.at(start_t).first == init_idx){
            auto& init_state = frame_states.at(init_t_ns);
            auto& init_extr = extr_states.at(init_t_ns);
            auto& end_state = frame_states.at(end_t);
            auto& end_extr = extr_states.at(end_t);
            res = dynamics->residual(singletrack_acado, init_state.getState(), init_extr.getState(),
                                     end_state.getState(), end_extr.getState(),
                                     param.getParam(), &pred_pose_state, &pred_vel_state);
        }
        else{
            auto& init_state = frame_states.at(init_t_ns);
            auto& init_extr = extr_states.at(init_t_ns);
            auto& end_state = frame_states.at(end_t);
            auto& end_extr = extr_states.at(end_t);
            res = dynamics->residual(singletrack_acado, start_t, init_state.getState(), init_extr.getState(),
                                     end_state.getState(), end_extr.getState(),
                                     param.getParam(), &pred_pose_state, &pred_vel_state);
        }


        dyn_error += 0.5 * res.transpose() * dyn_weight * res;

        // loose coupling
        // extr
        {
            auto& start_extr_state = extr_states.at(start_t);
            auto& end_extr_state = extr_states.at(end_t);

            double extr_weight = extr_rd_weight;
            Eigen::Matrix<double, 6, 1> res_extr;
            res_extr.head<3>() = end_extr_state.getState().T_o_i.translation() - start_extr_state.getState().T_o_i.translation();
            res_extr.tail<3>() = (start_extr_state.getState().T_o_i.so3().inverse() * end_extr_state.getState().T_o_i.so3()).log();
            extr_error += 0.5 * res_extr.transpose() * extr_weight * res_extr;
        }

    } // loop over factors
}
} // namespace dynvio
