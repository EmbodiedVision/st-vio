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

This file has been derived from the original file
basalt-mirror/src/vi_estimator/keypoint_vio.cpp
from the Basalt project (https://github.com/VladyslavUsenko/basalt-mirror/tree/cc6d896c47448958c8625ef766870e23d1fcd7ea).
The original file has been released under the following license:

=== BEGIN ORIGINAL FILE LICENSE ===
BSD 3-Clause License

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
=== END ORIGINAL FILE LICENSE ===
*/

#include "dynamics_vio/vio_estimator/vio_base.h"
#include "dynamics_vio/single_track/single_track.h"

#include <basalt/utils/assert.h>
#include <basalt/vi_estimator/keypoint_vio.h>
#include <basalt/optimization/accumulator.h>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <chrono>
#include <acado_toolkit.hpp>

namespace dynvio {
using namespace basalt;

template <class Param, class Model>
DynamicsVioEstimator<Param, Model>::DynamicsVioEstimator(const Eigen::Vector3d& g,
                                           const basalt::Calibration<double>& calib,
                                           const VioConfig& config,
                                           const DynVIOConfig& dynvio_config)
    :take_kf(true),
    frames_after_kf(0),
    g(g),
    initialized(false),
    config(config),
    lambda(config.vio_lm_lambda_min),
    min_lambda(config.vio_lm_lambda_min),
    max_lambda(config.vio_lm_lambda_max),
    lambda_vee(2),
    singletrack_acado(dynvio_config.mass, dynvio_config.wheel_base, dynvio_config.I_z,
                      dynvio_config.l_front){

    this->obs_std_dev = config.vio_obs_std_dev;
    this->huber_thresh = config.vio_obs_huber_thresh;
    this->calib = calib;
    this->dynvio_config = dynvio_config;

    // Setup marginalization
    state_size = POSE_VEL_BIAS_SIZE + EXTR_SIZE;
    state_param_size = state_size + PARAM_SIZE;
    marg_H.setZero(state_size, state_size);
    marg_b.setZero(state_size);

    // prior on position
    marg_H.diagonal(). template head<3>().setConstant(config.vio_init_pose_weight);
    // prior on yaw
    marg_H(5, 5) = config.vio_init_pose_weight;
    // we can assume the robot starts from static state.

    // small prior to avoid jumps in bias
    marg_H.diagonal().template segment<3>(9).array() = config.vio_init_ba_weight;
    marg_H.diagonal().template segment<3>(12).array() = config.vio_init_bg_weight;

    // prior on extr
    marg_H.diagonal().template segment<3>(15).array() = dynvio_config.extr_init_weight;
    marg_H.diagonal().template segment<2>(18).array() = 1; // roll, pitch
    marg_H.diagonal().template tail<1>().array() = dynvio_config.extr_init_weight; // yaw

    std::cout << "marg_H\n" << marg_H << std::endl;

    gyro_bias_weight = calib.gyro_bias_std.array().square().inverse();
    accel_bias_weight = calib.accel_bias_std.array().square().inverse();

    max_states = config.vio_max_states;
    max_kfs = config.vio_max_kfs;

    opt_started = false;

    vision_data_queue.set_capacity(10);
    imu_data_queue.set_capacity(300);
    cmd_data_queue.set_capacity(50);

    param_init_prior.reset(new Param(dynvio_config.c_lat_init,
                                     dynvio_config.steering_rt_init,
                                     dynvio_config.throttle_f1_init,
                                     dynvio_config.throttle_f2_init,
                                     dynvio_config.throttle_res_init));

    scaled_param_init_weight << 1 / dynvio_config.c_lat_init / dynvio_config.c_lat_init,
            1 / dynvio_config.steering_rt_init / dynvio_config.steering_rt_init,
            1 / dynvio_config.throttle_f1_init / dynvio_config.throttle_f1_init,
            1 / dynvio_config.throttle_f2_init / dynvio_config.throttle_f2_init,
            1 / dynvio_config.throttle_res_init / dynvio_config.throttle_res_init;
    scaled_param_init_weight *= dynvio_config.param_init_weight;

    scaled_param_prior_weight = scaled_param_init_weight;
    scaled_param_prior_weight *= dynvio_config.param_prior_weight;
}


template <class Param, class Model>
void DynamicsVioEstimator<Param, Model>::initialize(const Eigen::Vector3d& bg,
                                      const Eigen::Vector3d& ba) {

    auto proc_func = [&, bg, ba] {

        OpticalFlowResult::Ptr prev_frame, curr_frame;
        IntegratedImuMeasurement<double>::Ptr meas;

        const Eigen::Vector3d accel_cov =
            calib.dicrete_time_accel_noise_std().array().square();
        const Eigen::Vector3d gyro_cov =
            calib.dicrete_time_gyro_noise_std().array().square();

        ImuData<double>::Ptr data;
        imu_data_queue.pop(data);
        data->accel = calib.calib_accel_bias.getCalibrated(data->accel);
        data->gyro = calib.calib_gyro_bias.getCalibrated(data->gyro);

        /*****************dynamics*****************/
        typename Model::Ptr dynamics_factor;
        double z_g_w_init = 0;
        ConstraintsFactor::Ptr constraints_factor;

        Command::Ptr cmd_data;
        cmd_data_queue.pop(cmd_data);
        Eigen::Vector3d start_gyro, end_gyro;
        Sophus::SE3d last_pred;
        Eigen::Vector3d last_pred_vel;

        while(true){
            vision_data_queue.pop(curr_frame);

            if (config.vio_enforce_realtime) {
                // drop current frame if another frame is already in the queue.
                while (!vision_data_queue.empty()) vision_data_queue.pop(curr_frame);
            }

            if (!curr_frame.get()) {
                break;
            }

            if (!initialized) {

                while (data->t_ns < curr_frame->t_ns) {
                    imu_data_queue.pop(data);
                    if (!data.get()) break;
                    data->accel = calib.calib_accel_bias.getCalibrated(data->accel);
                    data->gyro = calib.calib_gyro_bias.getCalibrated(data->gyro);
                }

                /*****************dynamics*****************/
                //skip old cmd
                // actually not necessary
                if (cmd_data.get()) {
                    while (cmd_data->t_ns < curr_frame->t_ns) {
                        // here only keep the oldest cmd before t0 active frame
                        if(command_history.size() == 1)
                            command_history.pop_front();
                        command_history.push_back(*cmd_data);
                        cmd_data_queue.pop(cmd_data);
                        if (!cmd_data.get()) break;
                    }

                    if(command_history.empty())
                        std::cerr<<"Reset initial timestamp!"<<std::endl;
                }

                Eigen::Vector3d vel_w_i_init;
                vel_w_i_init.setZero();

                // robot: z up, x forward, T265 imu: y up, z forward
                T_w_i_init.setQuaternion(Eigen::Quaterniond::FromTwoVectors(
                    data->accel, Eigen::Vector3d::UnitZ()));
                T_w_i_init.so3() = T_w_i_init.so3() * dynvio_config.imu_forward.so3();

                last_state_t_ns = curr_frame->t_ns;
                imu_meas[last_state_t_ns] =
                    IntegratedImuMeasurement(last_state_t_ns, bg, ba);
                frame_states[last_state_t_ns] = PoseVelBiasStateWithLin<double>(
                    last_state_t_ns, T_w_i_init, vel_w_i_init, bg, ba, true);
                extr_states[last_state_t_ns] = ExtrinsicStateWithLin<double>(last_state_t_ns,
                                                                             dynvio_config.T_o_i_init ,true);
                param_states[last_state_t_ns] = ParameterWithLin<Param>(dynvio_config.c_lat_init,
                                                                        dynvio_config.steering_rt_init,
                                                                        dynvio_config.throttle_f1_init,
                                                                        dynvio_config.throttle_f2_init,
                                                                        dynvio_config.throttle_res_init,
                                                                        false);
                gyro_recording[last_state_t_ns] = data->gyro;
                // add initial constraint
                Sophus::SE3d T_w_o_init = T_w_i_init * dynvio_config.T_o_i_init.inverse();
                z_g_w_init = T_w_o_init.translation()(2);
                constraints_factor.reset(new ConstraintsFactor(last_state_t_ns, z_g_w_init,
                                                               dynvio_config.l_fw_imu + dynvio_config.l_front, dynvio_config.l_c_imu,
                                                               dynvio_config.imu_forward.translation()));
                constraints_factors[last_state_t_ns] = constraints_factor;

                marg_order.abs_order_map[last_state_t_ns] = std::make_pair(0, state_size);
                marg_order.total_size = state_size;
                marg_order.items = 1;

                std::cout << "Setting up filter: t_ns " << last_state_t_ns << std::endl;
                std::cout << "T_w_i\n" << T_w_i_init.matrix() << std::endl;
                std::cout << "vel_w_i " << vel_w_i_init.transpose() << std::endl;
                std::cout << "T_cg_i\n" << dynvio_config.T_o_i_init.matrix() << std::endl;

                initialized = true;
            } // initialize

            if(prev_frame){
                auto last_state = frame_states.at(last_state_t_ns);

                meas.reset(new IntegratedImuMeasurement<double>(
                    prev_frame->t_ns, last_state.getState().bias_gyro,
                    last_state.getState().bias_accel));

                while (data->t_ns <= prev_frame->t_ns) {
                    imu_data_queue.pop(data);
                    if (!data.get()) break;
                    data->accel = calib.calib_accel_bias.getCalibrated(data->accel);
                    data->gyro = calib.calib_gyro_bias.getCalibrated(data->gyro);
                }

                while (data->t_ns <= curr_frame->t_ns) {
                    if(meas->get_dt_ns() == 0){
                        start_gyro = data->gyro;
                    }
                    meas->integrate(*data, accel_cov, gyro_cov);
                    imu_data_queue.pop(data);
                    if (!data.get()) break;
                    data->accel = calib.calib_accel_bias.getCalibrated(data->accel);
                    data->gyro = calib.calib_gyro_bias.getCalibrated(data->gyro);
                }

                if (meas->get_start_t_ns() + meas->get_dt_ns() < curr_frame->t_ns) {
                    if (!data.get()) break;
                    int64_t tmp = data->t_ns;
                    data->t_ns = curr_frame->t_ns;
                    meas->integrate(*data, accel_cov, gyro_cov);
                    data->t_ns = tmp;
                }
                end_gyro = data->gyro;

                /*****************dynamics*****************/
                constraints_factor.reset(new ConstraintsFactor(curr_frame->t_ns, z_g_w_init,
                                                               dynvio_config.l_fw_imu + dynvio_config.l_front, dynvio_config.l_c_imu,
                                                               dynvio_config.imu_forward.translation()));

                if(cmd_data.get()){
                    while (cmd_data->t_ns < curr_frame->t_ns) {
                        command_history.push_back(*cmd_data);
                        cmd_data_queue.pop(cmd_data);
                        if(!cmd_data.get()) break;
                    }

                    // pop cmd_data in commd_history older than prev_frame->t_ns, but keep one
                    if(command_history.size() > 1){
                        for(auto it = command_history.begin(); it != std::prev(command_history.end(), 1);){
                            if(it->t_ns < prev_frame->t_ns && std::next(it, 1)->t_ns <= prev_frame->t_ns){
                                it = command_history.erase(it);
                            }else{
                                it++;
                            }
                        }
                    }

                    dynamics_factor.reset(new Model(
                        prev_frame->t_ns,
                        start_gyro,
                        curr_frame->t_ns,
                        end_gyro,
                        command_history));
                }
                gyro_recording[curr_frame->t_ns] =end_gyro;
            }
            measure(curr_frame, meas, dynamics_factor, constraints_factor);
            prev_frame = curr_frame;

        } // loop

        if (out_vis_queue) out_vis_queue->push(nullptr);
        if (out_marg_queue) out_marg_queue->push(nullptr);
        if (out_state_queue) out_state_queue->push(nullptr);
        if (out_extr_queue) out_extr_queue->push(nullptr);
        if (out_param_state_queue) out_param_state_queue->push(nullptr);
        if (out_gyro_queue) out_gyro_queue->push(nullptr);
        if (out_ba_var_queue) out_ba_var_queue->push(nullptr);
        finished = true;

        std::cout << "Finished VIOFilter " << std::endl;
    }; //proc_func

    processing_thread.reset(new std::thread(proc_func));
}

template <class Param, class Model>
bool DynamicsVioEstimator<Param, Model>::measure(const OpticalFlowResult::Ptr &opt_flow_meas,
                                   const IntegratedImuMeasurement<double>::Ptr &meas,
                                   const typename Model::Ptr &dynamics_factor,
                                   const ConstraintsFactor::Ptr & constraints_factor){

    if (meas.get()) {
    BASALT_ASSERT(frame_states[last_state_t_ns].getState().t_ns ==
                  meas->get_start_t_ns());
    BASALT_ASSERT(opt_flow_meas->t_ns ==
                  meas->get_dt_ns() + meas->get_start_t_ns());

    PoseVelBiasState next_state = frame_states.at(last_state_t_ns).getState();

    meas->predictState(frame_states.at(last_state_t_ns).getState(), g,
                       next_state);

    last_state_t_ns = opt_flow_meas->t_ns;
    next_state.t_ns = opt_flow_meas->t_ns;

    frame_states[last_state_t_ns] = next_state;
    imu_meas[meas->get_start_t_ns()] = *meas;

    /*****************dynamics*****************/
    // check if qualified to add dyanmics factor
    cur_window_start_t_ns = frame_states.begin()->first;
    cur_window_2nd_t_ns = std::next(frame_states.begin(), 1)->first;

    if(dynamics_factor.get()){
        if(opt_started && cur_window_start_ba_var.var.mean() < dynvio_config.ba_var_thr)
            adding_dynamics = true;
        else
            adding_dynamics = false;

        dynamics_factors[dynamics_factor->get_start_t_ns()] = dynamics_factor;

        ExtrinsicState next_extr_state = std::prev(extr_states.end(), 1)->second.getState();
        next_extr_state.t_ns = last_state_t_ns;
        extr_states[last_state_t_ns] = next_extr_state;

        param_states[last_state_t_ns] = ParameterWithLin<Param>(dynvio_config.c_lat_init,
                                                                dynvio_config.steering_rt_init,
                                                                dynvio_config.throttle_f1_init,
                                                                dynvio_config.throttle_f2_init,
                                                                dynvio_config.throttle_res_init,
                                                                false);

        if(constraints_factor){
            constraints_factors[last_state_t_ns] = constraints_factor;
        }

        if(adding_dynamics && is_marg_param){
            BASALT_ASSERT(param_states.at(cur_window_start_t_ns).isLinearized());
        }

        if(!adding_dynamics && is_marg_param){
            // remove param marg prior
            marg_order.total_size -= PARAM_SIZE;
            marg_order.abs_order_map.at(cur_window_start_t_ns).second -= PARAM_SIZE;

            marg_H.conservativeResize(marg_order.total_size, marg_order.total_size);
            marg_b.conservativeResize(marg_order.total_size);

            is_marg_param = false;
        }
    }
  }

  // save results
  prev_opt_flow_res[opt_flow_meas->t_ns] = opt_flow_meas;

  // Make new residual for existing keypoints
  int connected0 = 0;
  std::map<int64_t, int> num_points_connected;
  std::unordered_set<int> unconnected_obs0;
  for (size_t i = 0; i < opt_flow_meas->observations.size(); i++) {
    TimeCamId tcid_target(opt_flow_meas->t_ns, i);

    for (const auto& kv_obs : opt_flow_meas->observations[i]) {
      int kpt_id = kv_obs.first;

      if (lmdb.landmarkExists(kpt_id)) {
        const TimeCamId& tcid_host = lmdb.getLandmark(kpt_id).kf_id;

        KeypointObservation kobs;
        kobs.kpt_id = kpt_id;
        kobs.pos = kv_obs.second.translation().cast<double>();

        lmdb.addObservation(tcid_target, kobs);
        // obs[tcid_host][tcid_target].push_back(kobs);

        if (num_points_connected.count(tcid_host.frame_id) == 0) {
          num_points_connected[tcid_host.frame_id] = 0;
        }
        num_points_connected[tcid_host.frame_id]++;

        if (i == 0) connected0++;
      } else {
        if (i == 0) {
          unconnected_obs0.emplace(kpt_id);
        }
      }
    }
  }

  if (double(connected0) / (connected0 + unconnected_obs0.size()) <
          config.vio_new_kf_keypoints_thresh &&
      frames_after_kf > config.vio_min_frames_after_kf)
    take_kf = true;

  if (config.vio_debug) {
    std::cout << "connected0 " << connected0 << " unconnected0 "
              << unconnected_obs0.size() << std::endl;
  }

  if (take_kf) {
    // Triangulate new points from stereo and make keyframe for camera 0
    take_kf = false;
    frames_after_kf = 0;
    kf_ids.emplace(last_state_t_ns);

    TimeCamId tcidl(opt_flow_meas->t_ns, 0);

    int num_points_added = 0;
    for (int lm_id : unconnected_obs0) {
      // Find all observations
      std::map<TimeCamId, KeypointObservation> kp_obs;

      for (const auto& kv : prev_opt_flow_res) {
        for (size_t k = 0; k < kv.second->observations.size(); k++) {
          auto it = kv.second->observations[k].find(lm_id);
          if (it != kv.second->observations[k].end()) {
            TimeCamId tcido(kv.first, k);

            KeypointObservation kobs;
            kobs.kpt_id = lm_id;
            kobs.pos = it->second.translation().template cast<double>();

            // obs[tcidl][tcido].push_back(kobs);
            kp_obs[tcido] = kobs;
          }
        }
      }

      // triangulate
      bool valid_kp = false;
      const double min_triang_distance2 =
          config.vio_min_triangulation_dist * config.vio_min_triangulation_dist;
      for (const auto& kv_obs : kp_obs) {
        if (valid_kp) break;
        TimeCamId tcido = kv_obs.first;

        const Eigen::Vector2d p0 = opt_flow_meas->observations.at(0)
                                       .at(lm_id)
                                       .translation()
                                       .cast<double>();
        const Eigen::Vector2d p1 = prev_opt_flow_res[tcido.frame_id]
                                       ->observations[tcido.cam_id]
                                       .at(lm_id)
                                       .translation()
                                       .template cast<double>();

        Eigen::Vector4d p0_3d, p1_3d;
        bool valid1 = calib.intrinsics[0].unproject(p0, p0_3d);
        bool valid2 = calib.intrinsics[tcido.cam_id].unproject(p1, p1_3d);
        if (!valid1 || !valid2) continue;

        Sophus::SE3d T_i0_i1 =
            getPoseStateWithLin(tcidl.frame_id).getPose().inverse() *
            getPoseStateWithLin(tcido.frame_id).getPose();
        Sophus::SE3d T_0_1 =
            calib.T_i_c[0].inverse() * T_i0_i1 * calib.T_i_c[tcido.cam_id];

        if (T_0_1.translation().squaredNorm() < min_triang_distance2) continue;

        Eigen::Vector4d p0_triangulated =
            triangulate(p0_3d.head<3>(), p1_3d.head<3>(), T_0_1);

        if (p0_triangulated.array().isFinite().all() &&
            p0_triangulated[3] > 0 && p0_triangulated[3] < 3.0) {
          KeypointPosition kpt_pos;
          kpt_pos.kf_id = tcidl;
          kpt_pos.dir = StereographicParam<double>::project(p0_triangulated);
          kpt_pos.id = p0_triangulated[3];
          lmdb.addLandmark(lm_id, kpt_pos);

          num_points_added++;
          valid_kp = true;
        }
      }

      if (valid_kp) {
        for (const auto& kv_obs : kp_obs) {
          lmdb.addObservation(kv_obs.first, kv_obs.second);
        }
      }
    }

    num_points_kf[opt_flow_meas->t_ns] = num_points_added;
  } else {
    frames_after_kf++;
  }

  optimize();

  if(opt_started){

      if(out_ba_var_queue){
          AccelBiasVar::Ptr bavar_data = std::make_shared<AccelBiasVar>(cur_window_start_ba_var);
          bavar_data->t_ns = cur_window_start_t_ns;
          out_ba_var_queue->push(bavar_data);
      }

      if(adding_dynamics){
          if (out_param_state_queue){
              ParameterBase::Ptr param_data = std::make_shared<Param>(param_states.at(cur_window_start_t_ns).getParam());
              param_data->t_ns = cur_window_start_t_ns;
              out_param_state_queue->push(param_data);
          }

          if (out_extr_queue) {
              ExtrinsicState<double>::Ptr extr_data(new ExtrinsicState(extr_states.at(cur_window_start_t_ns).getState()));
              out_extr_queue->push(extr_data);
          }
      }

  }

  marginalize(num_points_connected);

  if (out_gyro_queue){
      Eigen::Vector3d gyro = gyro_recording.at(last_state_t_ns);
      std::shared_ptr<Eigen::Vector3d> data(new Eigen::Vector3d(gyro));
      out_gyro_queue->push(data);
  }

  if (out_state_queue) {
    PoseVelBiasState<double>::Ptr data(new PoseVelBiasState(frame_states.at(last_state_t_ns).getState()));
    out_state_queue->push(data);
  }

  if (out_vis_queue) {
    VioVisualizationData::Ptr data(new VioVisualizationData);

    data->t_ns = last_state_t_ns;

    for (const auto& kv : frame_states) {
      data->states.emplace_back(kv.second.getState().T_w_i);
    }

    for (const auto& kv : frame_poses) {
      data->frames.emplace_back(kv.second.getPose());
    }

    get_current_points(data->points, data->point_ids);

    data->projections.resize(opt_flow_meas->observations.size());
    computeProjections(data->projections);

    data->opt_flow_res = prev_opt_flow_res[last_state_t_ns];

    out_vis_queue->push(data);
  }

  last_processed_t_ns = last_state_t_ns;

  return true;
}

template <class Param, class Model>
void DynamicsVioEstimator<Param, Model>::computeProjections(
    std::vector<Eigen::aligned_vector<Eigen::Vector4d>>& data) const {
    for (const auto& kv : lmdb.getObservations()) {
        const TimeCamId& tcid_h = kv.first;

        for (const auto& obs_kv : kv.second) {
            const TimeCamId& tcid_t = obs_kv.first;

            if (tcid_t.frame_id != last_state_t_ns) continue;

            if (tcid_h != tcid_t) {
                PoseStateWithLin state_h = getPoseStateWithLin(tcid_h.frame_id);
                PoseStateWithLin state_t = getPoseStateWithLin(tcid_t.frame_id);

                Sophus::SE3d T_t_h_sophus =
                    computeRelPose(state_h.getPose(), calib.T_i_c[tcid_h.cam_id],
                                   state_t.getPose(), calib.T_i_c[tcid_t.cam_id]);

                Eigen::Matrix4d T_t_h = T_t_h_sophus.matrix();

                FrameRelLinData rld;

                std::visit(
                    [&](const auto& cam) {
                        for (size_t i = 0; i < obs_kv.second.size(); i++) {
                            const KeypointObservation& kpt_obs = obs_kv.second[i];
                            const KeypointPosition& kpt_pos =
                                lmdb.getLandmark(kpt_obs.kpt_id);

                            Eigen::Vector2d res;
                            Eigen::Vector4d proj;

                            linearizePoint(kpt_obs, kpt_pos, T_t_h, cam, res, nullptr,
                                           nullptr, &proj);

                            proj[3] = kpt_obs.kpt_id;
                            data[tcid_t.cam_id].emplace_back(proj);
                        }
                    },
                    calib.intrinsics[tcid_t.cam_id].variant);

            } else {
                // target and host are the same
                // residual does not depend on the pose
                // it just depends on the point

                std::visit(
                    [&](const auto& cam) {
                        for (size_t i = 0; i < obs_kv.second.size(); i++) {
                            const KeypointObservation& kpt_obs = obs_kv.second[i];
                            const KeypointPosition& kpt_pos =
                                lmdb.getLandmark(kpt_obs.kpt_id);

                            Eigen::Vector2d res;
                            Eigen::Vector4d proj;

                            linearizePoint(kpt_obs, kpt_pos, Eigen::Matrix4d::Identity(),
                                           cam, res, nullptr, nullptr, &proj);

                            proj[3] = kpt_obs.kpt_id;
                            data[tcid_t.cam_id].emplace_back(proj);
                        }
                    },
                    calib.intrinsics[tcid_t.cam_id].variant);
            }
        }
    }
}

template <class Param, class Model>
void DynamicsVioEstimator<Param, Model>::optimize(){
    if (config.vio_debug) {
        std::cout << "=================================" << std::endl;
    }

    if (opt_started || frame_states.size() >= max_states) {
        // Optimize
        opt_started = true;

        AbsOrderMap aom;

        // poses
        for (const auto& kv : frame_poses) {
            aom.abs_order_map[kv.first] = std::make_pair(aom.total_size, POSE_SIZE);

            // Check that we have the same order as marginalization
            BASALT_ASSERT(marg_order.abs_order_map.at(kv.first) ==
                          aom.abs_order_map.at(kv.first));

            aom.total_size += POSE_SIZE;
            aom.items++;
        }

        // states and extr or param
        for (const auto& kv : frame_states) {
            size_t tmp_size;
            if(kv.first <= cur_window_2nd_t_ns && adding_dynamics)
                tmp_size = state_param_size;
            else
                tmp_size = state_size;

            aom.abs_order_map[kv.first] =
                    std::make_pair(aom.total_size, tmp_size);

            // Check that we have the same order as marginalization
            if (aom.items < marg_order.abs_order_map.size()){
                if(!is_marg_param && adding_dynamics){
                    BASALT_ASSERT(marg_order.abs_order_map.at(kv.first).first ==
                                  aom.abs_order_map.at(kv.first).first);
                    BASALT_ASSERT(marg_order.abs_order_map.at(kv.first).second ==
                                  aom.abs_order_map.at(kv.first).second - PARAM_SIZE);
                }
                else{
                    BASALT_ASSERT(marg_order.abs_order_map.at(kv.first) ==
                                  aom.abs_order_map.at(kv.first));
                }
            }
            aom.total_size += tmp_size;
            aom.items++;
        }

        for (int iter = 0; iter < config.vio_max_iterations; iter++) {
            auto t1 = std::chrono::high_resolution_clock::now();
            double rld_error;
            Eigen::aligned_vector<RelLinData> rld_vec;
            linearizeHelper(rld_vec, lmdb.getObservations(), rld_error);

            BundleAdjustmentBase::LinearizeAbsReduce<DenseAccumulator<double>> lopt(
                aom);

            tbb::blocked_range<Eigen::aligned_vector<RelLinData>::iterator> range(
                rld_vec.begin(), rld_vec.end());

            tbb::parallel_reduce(range, lopt);

            double marg_prior_error = 0;
            double imu_error = 0, bg_error = 0, ba_error = 0;
            KeypointVioEstimator::linearizeAbsIMU(aom, lopt.accum.getH(), lopt.accum.getB(), imu_error,
                            bg_error, ba_error, frame_states, imu_meas,
                            gyro_bias_weight, accel_bias_weight, g);

            /*****************dynamics*****************/
            double dyn_error = 0;
            double extr_error = 0;
            double constraint_error = 0;
            double param_error = 0;
            if(adding_dynamics){
                Model::linearizeDynamics(singletrack_acado, aom, lopt.accum.getH(), lopt.accum.getB(), dyn_error, extr_error, param_states,
                                         frame_states, extr_states, dynamics_factors,
                                         dynvio_config.dynamics_weight, dynvio_config.extr_rd_weight);
                if(is_marg_param)
                    ConstraintsFactor::linearizeConstraints(aom, lopt.accum.getH(), lopt.accum.getB(), constraint_error, param_error,
                                                            nullptr, param_marged_prior.get(), param_states, frame_states, extr_states, constraints_factors,
                                                            dynvio_config.constraint_weight, scaled_param_init_weight, scaled_param_prior_weight);

                else
                    ConstraintsFactor::linearizeConstraints(aom, lopt.accum.getH(), lopt.accum.getB(), constraint_error, param_error,
                                                            param_init_prior.get(), nullptr, param_states, frame_states, extr_states, constraints_factors,
                                                            dynvio_config.constraint_weight, scaled_param_init_weight, scaled_param_prior_weight);
            }

            linearizeMargPriorDyn(marg_order, marg_H, marg_b, aom, lopt.accum.getH(),
                               lopt.accum.getB(), marg_prior_error);

            double error_total =
                rld_error + imu_error + marg_prior_error + ba_error + bg_error +
                dyn_error + extr_error + param_error + constraint_error;

            if (config.vio_debug)
                std::cout << "[LINEARIZE] Error: " << error_total << " num points "
                          << std::endl;


            lopt.accum.setup_solver();
            Eigen::VectorXd Hdiag = lopt.accum.Hdiagonal();

            bool converged = false;

            // Gauss-Newton
            Eigen::VectorXd Hdiag_lambda = (Hdiag * min_lambda).cwiseMax(min_lambda);

            Eigen::MatrixXd b_appd(aom.total_size, 4); // 1 for b, 3 for accel bias
            b_appd.setZero();
            b_appd.col(0) = lopt.accum.getB();
            auto tmp_accel_indx = aom.abs_order_map.at(frame_states.begin()->first).first + POSE_VEL_BIAS_SIZE - 3;
            for(size_t tmp_i = 0; tmp_i < 3; tmp_i++){
                b_appd(tmp_accel_indx + tmp_i, tmp_i + 1) = 1.;
            }

            Eigen::MatrixXd HH = lopt.accum.getH();
            HH.diagonal() += Hdiag_lambda;
            Eigen::MatrixXd inc_appd = HH.ldlt().solve(b_appd);
            Eigen::VectorXd inc= inc_appd.col(0);

            double max_inc = inc.array().abs().maxCoeff();
            if (max_inc < 1e-4) converged = true;

            // apply increment to poses
            for (auto& kv : frame_poses) {
                int idx = aom.abs_order_map.at(kv.first).first;
                kv.second.applyInc(-inc.segment<POSE_SIZE>(idx));
            }

            // apply increment to states
            for (auto& kv : frame_states) {
                int idx = aom.abs_order_map.at(kv.first).first;
                kv.second.applyInc(-inc.segment<POSE_VEL_BIAS_SIZE>(idx));

                // save ba variance
                if(kv.first == cur_window_start_t_ns){
                    auto& ba_var = cur_window_start_ba_var.var;
                    for(size_t tmp_i = 0; tmp_i < 3; tmp_i++){
                        ba_var(tmp_i) = inc_appd(tmp_accel_indx + tmp_i, tmp_i + 1);
                    }
                }
            }

            // apply increment to extrinsics
            for (auto& extr : extr_states) {
                int idx = aom.abs_order_map.at(extr.first).first + POSE_VEL_BIAS_SIZE;
                extr.second.applyInc(-inc.segment<EXTR_SIZE>(idx));

                // apply increment to parameters
                if(adding_dynamics && extr.first <= cur_window_2nd_t_ns){
                    idx += EXTR_SIZE;
                    param_states.at(extr.first).applyInc(-inc.segment(idx, PARAM_SIZE));
                } // applly inc to parameters
            }

            // Update points
            tbb::blocked_range<size_t> keys_range(0, rld_vec.size());
            auto update_points_func = [&](const tbb::blocked_range<size_t>& r) {
                for (size_t i = r.begin(); i != r.end(); ++i) {
                    const auto& rld = rld_vec[i];
                    updatePoints(aom, rld, inc);
                }
            };
            tbb::parallel_for(keys_range, update_points_func);


            if (config.vio_debug) {
                double after_update_marg_prior_error = 0;
                double after_update_vision_error = 0, after_update_imu_error = 0,
                       after_bg_error = 0, after_ba_error = 0;

                computeError(after_update_vision_error);
                KeypointVioEstimator::computeImuError(aom, after_update_imu_error, after_bg_error,
                                after_ba_error, frame_states, imu_meas,
                                gyro_bias_weight, accel_bias_weight, g);
                /*****************dynamics*****************/
                double after_update_dyn_error = 0;
                double after_update_extr_error = 0;
                double after_update_constraint_error = 0;
                double after_update_param_error = 0;

                if(adding_dynamics){
                    Model::computeDynamicsError(singletrack_acado, aom, after_update_dyn_error, after_update_extr_error,
                                                param_states, frame_states, extr_states, dynamics_factors,
                                                dynvio_config.dynamics_weight, dynvio_config.extr_rd_weight);
                    if(is_marg_param)
                        ConstraintsFactor::computeConstraintsError(aom, after_update_constraint_error, after_update_param_error,
                                                                   nullptr, param_marged_prior.get(), param_states, frame_states, extr_states, constraints_factors,
                                                                   dynvio_config.constraint_weight, scaled_param_init_weight, scaled_param_prior_weight);
                    else
                        ConstraintsFactor::computeConstraintsError(aom, after_update_constraint_error, after_update_param_error,
                                                                   param_init_prior.get(), nullptr, param_states, frame_states, extr_states, constraints_factors,
                                                                   dynvio_config.constraint_weight, scaled_param_init_weight, scaled_param_prior_weight);
                }


                computeMargPriorErrorDyn(marg_order, marg_H, marg_b,
                                      after_update_marg_prior_error);

                double after_error_total =
                    after_update_vision_error + after_update_imu_error +
                    after_update_marg_prior_error + after_bg_error + after_ba_error +
                    after_update_dyn_error + after_update_extr_error + after_update_param_error + after_update_constraint_error;


                double error_diff = error_total - after_error_total;

                auto t2 = std::chrono::high_resolution_clock::now();

                auto elapsed =
                    std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);

                std::cout << "iter " << iter
                          << " before_update_error: vision: " << rld_error
                          << " imu: " << imu_error << " bg_error: " << bg_error
                          << " ba_error: " << ba_error << " dynamics error: " << dyn_error
                          << " constraint error: "<< constraint_error
                          << " marg_prior: " << marg_prior_error
                          << " total: " << error_total << std::endl;

                std::cout << "iter " << iter << "  after_update_error: vision: "
                          << after_update_vision_error
                          << " imu: " << after_update_imu_error
                          << " bg_error: " << after_bg_error
                          << " ba_error: " << after_ba_error
                          << " dynamics error: " << after_update_dyn_error
                          << " constraint error: "<< after_update_constraint_error
                          << " marg prior: " << after_update_marg_prior_error
                          << " total: " << after_error_total << " error_diff "
                          << error_diff << " time : " << elapsed.count()
                          << "(us),  num_states " << frame_states.size()
                          << " num_poses " << frame_poses.size() << std::endl;


                if (after_error_total > error_total) {
                    std::cout << "increased error after update!!!" << std::endl;
                }

            }

            if (iter == config.vio_filter_iteration) {
                filterOutliers(config.vio_outlier_threshold, 4);
            }

            if (converged) break;

        }

    }

    if (config.vio_debug) {
        std::cout << "=================================" << std::endl;
    }
}

template<class Param, class Model>
void DynamicsVioEstimator<Param, Model>::marginalize(

    const std::map<int64_t, int>& num_points_connected) {
    if (!opt_started) return;

    if (frame_poses.size() > max_kfs || frame_states.size() >= max_states) {
        // Marginalize
        const int states_to_remove = frame_states.size() - max_states + 1;

        auto it = frame_states.cbegin();
        for (int i = 0; i < states_to_remove; i++) it++;
        int64_t last_state_to_marg = it->first;

        AbsOrderMap aom;
        // remove all frame_poses that are not kfs
        std::set<int64_t> poses_to_marg;
        for (const auto& kv : frame_poses) {
            aom.abs_order_map[kv.first] = std::make_pair(aom.total_size, POSE_SIZE);

            if (kf_ids.count(kv.first) == 0) poses_to_marg.emplace(kv.first);

            // Check that we have the same order as marginalization
            if (aom.items < marg_order.abs_order_map.size()){
                BASALT_ASSERT(marg_order.abs_order_map.at(kv.first).first ==
                              aom.abs_order_map.at(kv.first).first);

                BASALT_ASSERT(marg_order.abs_order_map.at(kv.first).second ==
                              aom.abs_order_map.at(kv.first).second);
            }

            aom.total_size += POSE_SIZE;
            aom.items++;
        }

        std::set<int64_t> states_to_marg_vel_bias_extr_param;
        std::set<int64_t> states_to_marg_all;
        BASALT_ASSERT(frame_states.size() == param_states.size());
        BASALT_ASSERT(frame_states.size() == extr_states.size());
        for (const auto& kv : frame_states) {
            if (kv.first > last_state_to_marg) break;

            if (kv.first != last_state_to_marg) {
                if (kf_ids.count(kv.first) > 0) {
                    states_to_marg_vel_bias_extr_param.emplace(kv.first);
                } else {
                    states_to_marg_all.emplace(kv.first);
                }
            }

            size_t tmp_size;
            if(kv.first <= cur_window_2nd_t_ns && adding_dynamics)
                tmp_size = state_param_size;
            else
                tmp_size = state_size;
            aom.abs_order_map[kv.first] =
                std::make_pair(aom.total_size, tmp_size);

            // Check that we have the same order as marginalization
            if (aom.items < marg_order.abs_order_map.size()){
                if(!is_marg_param && adding_dynamics){
                    BASALT_ASSERT(marg_order.abs_order_map.at(kv.first).first ==
                                  aom.abs_order_map.at(kv.first).first);
                    BASALT_ASSERT(marg_order.abs_order_map.at(kv.first).second ==
                                  aom.abs_order_map.at(kv.first).second - PARAM_SIZE);
                }
                else{
                    BASALT_ASSERT(marg_order.abs_order_map.at(kv.first) ==
                                  aom.abs_order_map.at(kv.first));
                }

            }

            aom.total_size += tmp_size;
            aom.items++;
        }

        auto kf_ids_all = kf_ids;
        std::set<int64_t> kfs_to_marg;
        while (kf_ids.size() > max_kfs && !states_to_marg_vel_bias_extr_param.empty()) {
            int64_t id_to_marg = -1;

            {
                std::vector<int64_t> ids;
                for (int64_t id : kf_ids) {
                    if(frame_states.count(id) > 0)
                        continue; //tmp_change
                    ids.push_back(id);
                }

                for (size_t i = 0; i < ids.size() - 2; i++) {
                    if (num_points_connected.count(ids[i]) == 0 ||
                        (num_points_connected.at(ids[i]) / num_points_kf.at(ids[i]) <
                         0.05)) {
                        id_to_marg = ids[i];
                        break;
                    }
                }
            }

            if (id_to_marg < 0) {
                std::vector<int64_t> ids;
                for (int64_t id : kf_ids) {
                    if(frame_states.count(id) > 0)
                        continue; //tmp_change
                    ids.push_back(id);
                }

                int64_t last_kf = *kf_ids.crbegin();
                double min_score = std::numeric_limits<double>::max();
                int64_t min_score_id = -1;

                for (size_t i = 0; i < ids.size() - 2; i++) {
                    double denom = 0;
                    for (size_t j = 0; j < ids.size() - 2; j++) {
                        denom += 1 / ((frame_poses.at(ids[i]).getPose().translation() -
                                       frame_poses.at(ids[j]).getPose().translation())
                                          .norm() +
                                      1e-5);
                    }

                    double score =
                        std::sqrt(
                            (frame_poses.at(ids[i]).getPose().translation() -
                             frame_states.at(last_kf).getState().T_w_i.translation())
                                .norm()) *
                        denom;

                    if (score < min_score) {
                        min_score_id = ids[i];
                        min_score = score;
                    }
                }

                id_to_marg = min_score_id;
            }

            kfs_to_marg.emplace(id_to_marg);
            poses_to_marg.emplace(id_to_marg);

            kf_ids.erase(id_to_marg);
        }


        if (config.vio_debug) {
            std::cout << "states_to_remove " << states_to_remove << std::endl;
            std::cout << "poses_to_marg.size() " << poses_to_marg.size() << std::endl;
            std::cout << "states_to_marg.size() " << states_to_marg_all.size()
                      << std::endl;
            std::cout << "states_to_marg_vel_bias.size() "
                      << states_to_marg_vel_bias_extr_param.size() << std::endl;
            std::cout << "kfs_to_marg.size() " << kfs_to_marg.size() << std::endl;
        }

        size_t asize = aom.total_size;

        double marg_prior_error;
        double imu_error, bg_error, ba_error;
        double dyn_error;
        double extr_error;
        double param_error;
        double constraint_error;

        DenseAccumulator accum;
        accum.reset(asize);

        {
            // Linearize points
            Eigen::aligned_map<
                TimeCamId, Eigen::aligned_map<
                    TimeCamId, Eigen::aligned_vector<KeypointObservation>>>
                obs_to_lin;

            for (auto it = lmdb.getObservations().cbegin();
                 it != lmdb.getObservations().cend();) {
                if (kfs_to_marg.count(it->first.frame_id) > 0) {
                    for (auto it2 = it->second.cbegin(); it2 != it->second.cend();
                         ++it2) {
                        if (it2->first.frame_id <= last_state_to_marg)
                            obs_to_lin[it->first].emplace(*it2);
                    }
                }
                ++it;
            }

            double rld_error;
            Eigen::aligned_vector<RelLinData> rld_vec;

            linearizeHelper(rld_vec, obs_to_lin, rld_error);

            for (auto& rld : rld_vec) {
                rld.invert_keypoint_hessians();

                Eigen::MatrixXd rel_H;
                Eigen::VectorXd rel_b;
                linearizeRel(rld, rel_H, rel_b);

                linearizeAbs(rel_H, rel_b, rld, aom, accum);
            }
        }

        KeypointVioEstimator::linearizeAbsIMU(aom, accum.getH(), accum.getB(), imu_error, bg_error,
                        ba_error, frame_states, imu_meas, gyro_bias_weight,
                        accel_bias_weight, g);

        if(adding_dynamics){
            Model::linearizeDynamics(singletrack_acado, aom, accum.getH(), accum.getB(), dyn_error, extr_error,
                                     param_states, frame_states, extr_states, dynamics_factors,
                                     dynvio_config.dynamics_weight, dynvio_config.extr_rd_weight);
            if(is_marg_param)
                ConstraintsFactor::linearizeConstraints(aom, accum.getH(), accum.getB(), constraint_error, param_error,
                                                        nullptr, param_marged_prior.get(), param_states, frame_states, extr_states, constraints_factors,
                                                        dynvio_config.constraint_weight, scaled_param_init_weight, scaled_param_prior_weight);
            else
                ConstraintsFactor::linearizeConstraints(aom, accum.getH(), accum.getB(), constraint_error, param_error,
                                                        param_init_prior.get(), nullptr, param_states, frame_states, extr_states, constraints_factors,
                                                        dynvio_config.constraint_weight, scaled_param_init_weight, scaled_param_prior_weight);
        }

        linearizeMargPriorDyn(marg_order, marg_H, marg_b, aom, accum.getH(),
                           accum.getB(), marg_prior_error);

        // Save marginalization prior
        if (out_marg_queue && !kfs_to_marg.empty()) {
            // int64_t kf_id = *kfs_to_marg.begin();

            {
                MargData::Ptr m(new MargData);
                m->aom = aom;
                m->abs_H = accum.getH();
                m->abs_b = accum.getB();
                m->frame_poses = frame_poses;
                m->frame_states = frame_states;
                m->kfs_all = kf_ids_all;
                m->kfs_to_marg = kfs_to_marg;
                m->use_imu = true;

                for (int64_t t : m->kfs_all) {
                    m->opt_flow_res.emplace_back(prev_opt_flow_res.at(t));
                }

                out_marg_queue->push(m);
            }
        }

        std::set<int> idx_to_keep, idx_to_marg;
        for (const auto& kv : aom.abs_order_map) {
            if (kv.second.second == POSE_SIZE) {
                int start_idx = kv.second.first;
                if (poses_to_marg.count(kv.first) == 0) {
                    for (size_t i = 0; i < POSE_SIZE; i++)
                        idx_to_keep.emplace(start_idx + i);
                } else {
                    for (size_t i = 0; i < POSE_SIZE; i++)
                        idx_to_marg.emplace(start_idx + i);
                }

            } else if(kv.second.second == (int)state_size){
                // state
                int start_idx = kv.second.first;
                if (states_to_marg_all.count(kv.first) > 0) {
                    for (size_t i = 0; i < state_size; i++)
                        idx_to_marg.emplace(start_idx + i);
                } else if (states_to_marg_vel_bias_extr_param.count(kv.first) > 0) {
                    for (size_t i = 0; i < POSE_SIZE; i++)
                      idx_to_keep.emplace(start_idx + i);
                    for (size_t i = POSE_SIZE; i < state_size; i++)
                      idx_to_marg.emplace(start_idx + i);
                } else {
                    BASALT_ASSERT(kv.first == last_state_to_marg);
                    for (size_t i = 0; i < state_size; i++)
                        idx_to_keep.emplace(start_idx + i);
                }
            }
            else{
                BASALT_ASSERT(kv.second.second == (int)state_param_size);
                // state
                int start_idx = kv.second.first;
                if (states_to_marg_all.count(kv.first) > 0) {
                    for (size_t i = 0; i < state_param_size; i++)
                        idx_to_marg.emplace(start_idx + i);
                } else if (states_to_marg_vel_bias_extr_param.count(kv.first) > 0) {
                    for (size_t i = 0; i < POSE_SIZE; i++)
                        idx_to_keep.emplace(start_idx + i);
                    for (size_t i = POSE_SIZE; i < state_param_size; i++)
                        idx_to_marg.emplace(start_idx + i);
                } else {
                    BASALT_ASSERT(kv.first == last_state_to_marg);
                    for (size_t i = 0; i < state_param_size; i++)
                        idx_to_keep.emplace(start_idx + i);
                }
            }
        }

        if (config.vio_debug) {
            std::cout << "keeping " << idx_to_keep.size() << " marg "
                      << idx_to_marg.size() << " total " << asize << std::endl;
            std::cout << "last_state_to_marg " << last_state_to_marg
                      << " frame_poses " << frame_poses.size() << " frame_states "
                      << frame_states.size() << std::endl;
        }

        Eigen::MatrixXd marg_H_new;
        Eigen::VectorXd marg_b_new;
        marginalizeHelper(accum.getH(), accum.getB(), idx_to_keep, idx_to_marg,
                          marg_H_new, marg_b_new);


        {
            BASALT_ASSERT(frame_states.at(last_state_to_marg).isLinearized() ==
                          false);
            frame_states.at(last_state_to_marg).setLinTrue();
            extr_states.at(last_state_to_marg).setLinTrue();
            if(adding_dynamics)
                param_states.at(last_state_to_marg).setLinTrue();
        }

        for (const int64_t id : states_to_marg_all) {
            frame_states.erase(id);
            extr_states.erase(id);
            imu_meas.erase(id);
            prev_opt_flow_res.erase(id);
            gyro_recording.erase(id);
            param_marged_prior.reset(new Param(param_states.at(id).getParam()));
            param_states.erase(id);

            constraints_factors.erase(id);
            dynamics_factors.erase(id);
        }

        for (const int64_t id : states_to_marg_vel_bias_extr_param) {
            const PoseVelBiasStateWithLin<double>& state = frame_states.at(id);
            PoseStateWithLin pose(state);
            frame_poses[id] = pose;

            frame_states.erase(id);
            extr_states.erase(id);
            imu_meas.erase(id);
            gyro_recording.erase(id);
            param_marged_prior.reset(new Param(param_states.at(id).getParam()));
            param_states.erase(id);

            constraints_factors.erase(id);
            dynamics_factors.erase(id);
        }

        BASALT_ASSERT(states_to_marg_all.count(cur_window_start_t_ns) > 0 ||
                      states_to_marg_vel_bias_extr_param.count(cur_window_start_t_ns) > 0);

        for (const int64_t id : poses_to_marg) {
            frame_poses.erase(id);
            prev_opt_flow_res.erase(id);
        }

        lmdb.removeKeyframes(kfs_to_marg, poses_to_marg, states_to_marg_all);

        AbsOrderMap marg_order_new;
        for (const auto& kv : frame_poses) {
            marg_order_new.abs_order_map[kv.first] =
                std::make_pair(marg_order_new.total_size, POSE_SIZE);

            marg_order_new.total_size += POSE_SIZE;
            marg_order_new.items++;
        }
        {
            size_t tmp_size;
            if(adding_dynamics)
                tmp_size = state_param_size;
            else
                tmp_size = state_size;

            marg_order_new.abs_order_map[last_state_to_marg] =
                std::make_pair(marg_order_new.total_size, tmp_size);
            marg_order_new.total_size += tmp_size;
            marg_order_new.items++;
        }

        marg_H = marg_H_new;
        marg_b = marg_b_new;
        marg_order = marg_order_new;

        Eigen::VectorXd delta;
        computeDeltaDyn(marg_order, delta);
        marg_b -= marg_H * delta;

        if (config.vio_debug) {
            std::cout << "marginalizaon done!!" << std::endl;
            std::cout << "=================================" << std::endl;
        }

        if(adding_dynamics && !is_marg_param)
            is_marg_param = true; // tell optimier we have paramter marg prior
    }
}

template<class Param, class Model>
void DynamicsVioEstimator<Param, Model>::linearizeMargPriorDyn(const AbsOrderMap &marg_order, const Eigen::MatrixXd &marg_H, const Eigen::VectorXd &marg_b,
                                                 const AbsOrderMap &aom, Eigen::MatrixXd &abs_H, Eigen::VectorXd &abs_b, double &marg_prior_error) const{
    // Assumed to be in the top left corner

    BASALT_ASSERT(size_t(marg_H.cols()) == marg_order.total_size);

    // Check if the order of variables is the same.
    for (const auto& kv : marg_order.abs_order_map)
        if(!is_marg_param && adding_dynamics && kv.second.second != POSE_SIZE)
            BASALT_ASSERT(aom.abs_order_map.at(kv.first).second - PARAM_SIZE == kv.second.second);
        else
            BASALT_ASSERT(aom.abs_order_map.at(kv.first).second == kv.second.second);
    size_t marg_size = marg_order.total_size;
    abs_H.topLeftCorner(marg_size, marg_size) += marg_H;

    Eigen::VectorXd delta;
    computeDeltaDyn(marg_order, delta);

    abs_b.head(marg_size) += marg_b;
    abs_b.head(marg_size) += marg_H * delta;

    marg_prior_error = 0.5 * delta.transpose() * marg_H * delta;
    marg_prior_error += delta.transpose() * marg_b;
}

template<class Param, class Model>
void DynamicsVioEstimator<Param, Model>::computeMargPriorErrorDyn(const AbsOrderMap &marg_order, const Eigen::MatrixXd &marg_H, const Eigen::VectorXd &marg_b, double &marg_prior_error) const{
    BASALT_ASSERT(size_t(marg_H.cols()) == marg_order.total_size);

    Eigen::VectorXd delta;
    computeDeltaDyn(marg_order, delta);

    marg_prior_error = 0.5 * delta.transpose() * marg_H * delta;
    marg_prior_error += delta.transpose() * marg_b;
}


template<class Param, class Model>
void DynamicsVioEstimator<Param, Model>::computeDeltaDyn(const AbsOrderMap &marg_order, Eigen::VectorXd &delta) const{
    size_t marg_size = marg_order.total_size;
    delta.setZero(marg_size);
    for (const auto& kv : marg_order.abs_order_map) {
        if (kv.second.second == POSE_SIZE) {
            BASALT_ASSERT(frame_poses.at(kv.first).isLinearized());
            delta.segment<POSE_SIZE>(kv.second.first) =
                frame_poses.at(kv.first).getDelta();
        } else if (kv.second.second == (int)state_size) {
            BASALT_ASSERT(frame_states.at(kv.first).isLinearized());
            delta.segment<POSE_VEL_BIAS_SIZE>(kv.second.first) =
                frame_states.at(kv.first).getDelta();
            delta.segment<EXTR_SIZE>(kv.second.first + POSE_VEL_BIAS_SIZE) = extr_states.at(kv.first).getDelta();
        } else if (kv.second.second == (int)state_param_size){
            // delta.tail<PARAM_SIZE>() = param_states.at(kv.first).getDelta();
            BASALT_ASSERT(param_states.at(kv.first).isLinearized());
            delta.segment<POSE_VEL_BIAS_SIZE>(kv.second.first) =
                    frame_states.at(kv.first).getDelta();
            delta.segment<EXTR_SIZE>(kv.second.first + POSE_VEL_BIAS_SIZE) = extr_states.at(kv.first).getDelta();
            delta.segment<PARAM_SIZE>(kv.second.first + state_size) = param_states.begin()->second.getDelta();
        } else{
            BASALT_ASSERT(false);
        }
    }
}

template<class Param, class Model>
void DynamicsVioEstimator<Param, Model>::initialize(int64_t t_ns, const Sophus::SE3d &T_w_i,
                                      const Eigen::Vector3d &vel_w_i,
                                      const Eigen::Vector3d &bg, const Eigen::Vector3d &ba){
    UNUSED(t_ns);
    UNUSED(T_w_i);
    UNUSED(vel_w_i);
    UNUSED(bg);
    UNUSED(ba);
}

template class DynamicsVioEstimator<SingleTrackParamOnline, SingleTrackModel>;
} //namespace dynvio
