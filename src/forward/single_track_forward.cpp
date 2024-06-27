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

#include "dynamics_vio/forward/single_track_forward.h"

#include <acado_toolkit.hpp>

namespace dynvio {

SingleTrackForward::SingleTrackForward(const DynVIOConfig& dynvio_config, const std::vector<int64_t>& time_stamps,
                                       const std::vector<dynvio::Command>& cmd_data_out,
                                       const dynvio::SingleTrackAcado& singletrack_acado):
  cmd_history_size(dynvio_config.cmd_history_size), gt_timestamps(time_stamps), cmd_data_out(cmd_data_out),
  singletrack_acado(singletrack_acado){}

Sophus::SE3d SingleTrackForward::ComputePose(const size_t& start_indx,
                                             const size_t& steps,
                                             const Sophus::SE3d& init_pose,
                                             const Sophus::SE3d& init_extr,
                                             const Eigen::Vector3d& init_linvel,
                                             const Eigen::Vector3d& init_angvel,
                                             Eigen::aligned_map<int64_t, Sophus::SE3d>& poses){
    poses.clear();

    std::deque<Command> command_history;

    int64_t start_t_ns = gt_timestamps[start_indx];
    auto curr_cmd_itr = cmd_data_out.begin();

    poses[start_t_ns] = init_pose * init_extr.inverse(); // T_w_o_init
    Eigen::Vector3d last_pred_vel;
    Sophus::SO3d R_o_i = init_extr.so3();
    Eigen::Matrix3d t_o_i_hat = Sophus::SO3d::hat(init_extr.translation());
    Eigen::Vector3d cg_init_ang = R_o_i * init_angvel;
    Eigen::Vector3d cg_init_tmp = R_o_i * init_linvel;
    Eigen::Vector3d cg_init_lin = cg_init_tmp + t_o_i_hat * cg_init_ang;
    last_pred_vel(0) = cg_init_lin(0);
    last_pred_vel(1) = cg_init_lin(1);
    last_pred_vel(2) = cg_init_ang(2);

    Sophus::SE3d T_o_rel;
    BASALT_ASSERT(start_indx + steps < gt_timestamps.size());

    for(size_t i = 0; i < steps; i++){
        size_t indx = start_indx + i;
        auto cmd_itr = curr_cmd_itr;
        for(; cmd_itr!= cmd_data_out.end(); cmd_itr++){
                if(cmd_itr->t_ns < gt_timestamps[indx + 1]){
                    command_history.push_back(*cmd_itr);
                }else {
                    break;
                }
        }
        curr_cmd_itr = cmd_itr;

        // pop cmd_data in commd_history older than prev_frame->t_ns, but keep one
        if(command_history.size() > 1){
            for(auto it = command_history.begin(); it != std::prev(command_history.end(), 1);){
                if(it->t_ns < gt_timestamps[indx] && std::next(it, 1)->t_ns <= gt_timestamps[indx]){
                    it = command_history.erase(it);
                }else{
                    it++;
                }
            }
        }

        // command before t0
        Command prev_cmd = command_history[0];

        double frame_dt;
        // hard-coded, maximum command history size is 2
        if (command_history.size() == 1) {
            frame_dt = (gt_timestamps[indx + 1] - gt_timestamps[indx]) * 1e-9;
        } else{
            frame_dt = (command_history[1].t_ns - gt_timestamps[indx]) * 1e-9;
        }

        double init_state[6] = {0.0, 0.0, 0.0, last_pred_vel(0), last_pred_vel(1), last_pred_vel(2)};
        double param_list[PARAM_SIZE] = {c_lat, steering_rt, throttle_f1, throttle_f2, throttle_res};
        double control_input[2] = {prev_cmd.linear, prev_cmd.angular};

        ACADO::IntegratorRK45 integrator( singletrack_acado.f );
        ACADO::DVector end_states;

        integrator.set(ACADO::INTEGRATOR_PRINTLEVEL, ACADO::NONE);
        integrator.integrate(0.0, frame_dt, init_state, nullptr, param_list, control_input);
        integrator.getX(end_states);

        Sophus::SE3d T_pred_rel;
        T_pred_rel.so3() = Sophus::SO3d::exp(Eigen::Vector3d(0,0,end_states(2)));
        T_pred_rel.translation()(0) = end_states(0);
        T_pred_rel.translation()(1) = end_states(1);

        if(command_history.size() > 1){
            BASALT_ASSERT(command_history.size() == 2);
            // furthur prediction step
            double frame_dt2 = (gt_timestamps[indx + 1] - command_history[1].t_ns) * 1e-9;
            Command curr_cmd = command_history[1];

            double init_state2[6] = {0.0, 0.0, 0.0, end_states(3), end_states(4), end_states(5)};
            double control_input2[2] = {curr_cmd.linear, curr_cmd.angular};

            integrator.integrate(0.0, frame_dt2, init_state2, nullptr, param_list, control_input2);
            integrator.getX(end_states);

            Sophus::SE3d T_pred_rel2;
            T_pred_rel2.so3() = Sophus::SO3d::exp(Eigen::Vector3d(0,0,end_states(2)));
            T_pred_rel2.translation()(0) = end_states(0);
            T_pred_rel2.translation()(1) = end_states(1);

            T_pred_rel = T_pred_rel * T_pred_rel2;
        }

        //pose in o frame
        poses[gt_timestamps[indx + 1]] = poses.at(gt_timestamps[indx]) * T_pred_rel;
        last_pred_vel(0) = end_states(3);
        last_pred_vel(1) = end_states(4);
        last_pred_vel(2) = end_states(5);

        // rel
        T_o_rel = T_o_rel * T_pred_rel;
    }
    return T_o_rel;
}

void SingleTrackForward::UpdateParam(const SingleTrackParamOnline &param_state){
    c_lat = param_state.c_lat;
    steering_rt = param_state.steering_rt;
    throttle_f1 = param_state.throttle_f1;
    throttle_f2 = param_state.throttle_f2;
    throttle_res = param_state.throttle_res;
}
} // namespace dynvio
