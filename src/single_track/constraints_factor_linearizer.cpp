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

void ConstraintsFactor::linearizeConstraints(const AbsOrderMap& aom, Eigen::MatrixXd& abs_H, Eigen::VectorXd& abs_b,
                                             double& constr_error, double &param_error,
                                             const dynvio::SingleTrackParamOnline* param_init_prior, const dynvio::SingleTrackParamOnline* param_marged_prior,
                                             const Eigen::aligned_map<int64_t, ParameterWithLin<SingleTrackParamOnline>> &param_states,
                                             const Eigen::aligned_map<int64_t, PoseVelBiasStateWithLin<double> > &frame_states,
                                             const Eigen::aligned_map<int64_t, ExtrinsicStateWithLin<double> > &extr_states,
                                             const Eigen::aligned_map<int64_t, ConstraintsFactor::Ptr> &constr_factors,
                                             const double& weight,
                                             const Eigen::Matrix<double, PARAM_SIZE, 1>& param_init_weight,
                                             const Eigen::Matrix<double, PARAM_SIZE, 1>& param_2nd_weight){

    constr_error = 0;
    for( const auto& stamped_factor : constr_factors){

        ConstraintsFactor::Ptr constraint = stamped_factor.second;
        int64_t t_ns = constraint->get_t_ns();

        if(aom.abs_order_map.count(t_ns) == 0) continue;

        const size_t state_idx = aom.abs_order_map.at(t_ns).first;
        const size_t extr_idx = state_idx + POSE_VEL_BIAS_SIZE;

        auto& state = frame_states.at(t_ns);
        auto& extr_state = extr_states.at(t_ns);

        Eigen::Matrix<double, 3, 6> d_res_d_state;
        d_res_d_state.setZero();
        Eigen::Matrix<double, 6, 6> d_res_d_extr;
        d_res_d_extr.setZero();

        ConstraintsFactor::Vec6 res = constraint->residual(state.getStateLin(), extr_state.getStateLin(), &d_res_d_state, &d_res_d_extr);

        if(state.isLinearized() || extr_state.isLinearized()){
            res = constraint->residual(state.getState(), extr_state.getState());
        }

        constr_error += 0.5 * res.transpose() * weight * res;

        //pose
        Eigen::Matrix<double, 6, 3> Jstateweighted_tp = d_res_d_state.transpose() * weight;
        abs_H.block<6, 6>(state_idx, state_idx) += Jstateweighted_tp * d_res_d_state;
        abs_b.segment<6>(state_idx) += Jstateweighted_tp * res.topRows(3);

        Eigen::Matrix<double, 6, 6> Jextrweighted_tp = d_res_d_extr.transpose() * weight;
        abs_H.block<6, 6>(extr_idx, extr_idx) += Jextrweighted_tp * d_res_d_extr;
        abs_b.segment<6>(extr_idx) += Jextrweighted_tp * res;

        Eigen::Matrix<double, 6, 6> H_state_extr = Jstateweighted_tp * d_res_d_extr.topRows(3);
        abs_H.block<6, 6>(state_idx, extr_idx) += H_state_extr;
        abs_H.block<6, 6>(extr_idx, state_idx) += H_state_extr.transpose();
    } // loop

    // param constraint
    const size_t param_idx = aom.abs_order_map.at(param_states.begin()->first).first + POSE_VEL_BIAS_SIZE + EXTR_SIZE;
    const auto& param_state = param_states.begin()->second;

    if(param_init_prior){
        Eigen::Matrix<double, PARAM_SIZE, 1> res_param;
        res_param << param_state.getParam().c_lat - param_init_prior->c_lat,
                param_state.getParam().steering_rt - param_init_prior->steering_rt,
                param_state.getParam().throttle_f1 - param_init_prior->throttle_f1,
                param_state.getParam().throttle_f2 - param_init_prior->throttle_f2,
                param_state.getParam().throttle_res - param_init_prior->throttle_res;

                abs_H.diagonal(). template segment<PARAM_SIZE>(param_idx) += param_init_weight;
        Eigen::Matrix<double, PARAM_SIZE, 1> res_param_scaled = param_init_weight.cwiseProduct(res_param);
        abs_b.segment<PARAM_SIZE>(param_idx) += res_param_scaled;


        param_error += 0.5 * res_param.transpose() * res_param_scaled;
    }

    if(param_marged_prior){
        Eigen::Matrix<double, PARAM_SIZE, 1> res_param;
        res_param << param_state.getParam().c_lat - param_marged_prior->c_lat,
                param_state.getParam().steering_rt - param_marged_prior->steering_rt,
                param_state.getParam().throttle_f1 - param_marged_prior->throttle_f1,
                param_state.getParam().throttle_f2 - param_marged_prior->throttle_f2,
                param_state.getParam().throttle_res - param_marged_prior->throttle_res;

                abs_H.diagonal(). template segment<PARAM_SIZE>(param_idx) += param_init_weight;
        Eigen::Matrix<double, PARAM_SIZE, 1> res_param_scaled = param_init_weight.cwiseProduct(res_param);
        abs_b.segment<PARAM_SIZE>(param_idx) += res_param_scaled;

        param_error += 0.5 * res_param.transpose() * res_param_scaled;
    }

    // 2nd param constraint
    {
        const auto& second_param_state = std::next(param_states.begin(), 1)->second;
        const size_t second_param_idx = param_idx + PARAM_SIZE + POSE_VEL_BIAS_SIZE + EXTR_SIZE;
        Eigen::Matrix<double, PARAM_SIZE, 1> res_param_2nd;;
        res_param_2nd << param_state.getParam().c_lat - second_param_state.getParam().c_lat,
                      param_state.getParam().steering_rt - second_param_state.getParam().steering_rt,
                      param_state.getParam().throttle_f1 - second_param_state.getParam().throttle_f1,
                      param_state.getParam().throttle_f2 - second_param_state.getParam().throttle_f2,
                      param_state.getParam().throttle_res - second_param_state.getParam().throttle_res;

                abs_H.diagonal(). template segment<PARAM_SIZE>(param_idx) += param_2nd_weight;
        abs_H.diagonal(). template segment<PARAM_SIZE>(second_param_idx) += param_2nd_weight;

        abs_H.block<PARAM_SIZE, PARAM_SIZE>(param_idx, second_param_idx).diagonal() -= param_2nd_weight;
        abs_H.block<PARAM_SIZE, PARAM_SIZE>(second_param_idx, param_idx).diagonal() -= param_2nd_weight;

        Eigen::Matrix<double, PARAM_SIZE, 1> res_param_2nd_scaled = param_2nd_weight.cwiseProduct(res_param_2nd);
        abs_b.segment<PARAM_SIZE>(param_idx) += res_param_2nd_scaled;
        abs_b.segment<PARAM_SIZE>(second_param_idx) -= res_param_2nd_scaled;


        param_error += 0.5 * res_param_2nd.transpose() * res_param_2nd_scaled;
    }

}

void ConstraintsFactor::computeConstraintsError(const AbsOrderMap &aom, double &constr_error, double &param_error,
                                                const dynvio::SingleTrackParamOnline* param_init_prior, const dynvio::SingleTrackParamOnline* param_marged_prior,
                                                const Eigen::aligned_map<int64_t, ParameterWithLin<SingleTrackParamOnline>> &param_states,
                                                const Eigen::aligned_map<int64_t, PoseVelBiasStateWithLin<double> > &frame_states, const Eigen::aligned_map<int64_t, ExtrinsicStateWithLin<double> > &extr_states,
                                                const Eigen::aligned_map<int64_t, ConstraintsFactor::Ptr> &constr_factors,
                                                const double& weight,
                                                const Eigen::Matrix<double, PARAM_SIZE, 1>& param_init_weight,
                                                const Eigen::Matrix<double, PARAM_SIZE, 1>& param_2nd_weight){
    constr_error = 0;

    for( const auto& stamped_factor : constr_factors){

        ConstraintsFactor::Ptr constraint = stamped_factor.second;
        int64_t t_ns = constraint->get_t_ns();

        if(aom.abs_order_map.count(t_ns) == 0) continue;


        auto& state = frame_states.at(t_ns);
        auto& extr_state = extr_states.at(t_ns);

        ConstraintsFactor::Vec6 res = constraint->residual(state.getState(), extr_state.getState());


        constr_error += 0.5 * res.transpose() * weight * res;
    } // loop

    // param constraint
    const auto& param_state = param_states.begin()->second;

    if(param_init_prior){
        Eigen::Matrix<double, PARAM_SIZE, 1> res_param;
        res_param << param_state.getParam().c_lat - param_init_prior->c_lat,
                param_state.getParam().steering_rt - param_init_prior->steering_rt,
                param_state.getParam().throttle_f1 - param_init_prior->throttle_f1,
                param_state.getParam().throttle_f2 - param_init_prior->throttle_f2,
                param_state.getParam().throttle_res - param_init_prior->throttle_res;

                Eigen::Matrix<double, PARAM_SIZE, 1> res_param_scaled = param_init_weight.cwiseProduct(res_param);
        param_error += 0.5 * res_param.transpose() * res_param_scaled;
    }
    if(param_marged_prior){
        Eigen::Matrix<double, PARAM_SIZE, 1> res_param;
        res_param << param_state.getParam().c_lat - param_marged_prior->c_lat,
                param_state.getParam().steering_rt - param_marged_prior->steering_rt,
                param_state.getParam().throttle_f1 - param_marged_prior->throttle_f1,
                param_state.getParam().throttle_f2 - param_marged_prior->throttle_f2,
                param_state.getParam().throttle_res - param_marged_prior->throttle_res;

                Eigen::Matrix<double, PARAM_SIZE, 1> res_param_scaled = param_init_weight.cwiseProduct(res_param);

        param_error += 0.5 * res_param.transpose() * res_param_scaled;
    }
    // 2nd param constraint
    {
        const auto& second_param_state = std::next(param_states.begin(), 1)->second;
        Eigen::Matrix<double, PARAM_SIZE, 1> res_param_2nd;;
        res_param_2nd << param_state.getParam().c_lat - second_param_state.getParam().c_lat,
                param_state.getParam().steering_rt - second_param_state.getParam().steering_rt,
                param_state.getParam().throttle_f1 - second_param_state.getParam().throttle_f1,
                param_state.getParam().throttle_f2 - second_param_state.getParam().throttle_f2,
                param_state.getParam().throttle_res - second_param_state.getParam().throttle_res;

                Eigen::Matrix<double, PARAM_SIZE, 1> res_param_2nd_scaled = param_2nd_weight.cwiseProduct(res_param_2nd);
        param_error += 0.5 * res_param_2nd.transpose() * res_param_2nd_scaled;
    }
}

} // namespace dynvio
