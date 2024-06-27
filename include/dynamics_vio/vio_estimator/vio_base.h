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
basalt-mirror/include/vi_estimator/keypoint_vio.h
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

#pragma once

#include "dynamics_vio/dynamics_base.h"
#include "dynamics_vio/utils/utils.h"
#include "dynamics_vio/parameters/parameters.h"
#include "dynamics_vio/single_track/constraints_factor.h"
#include "dynamics_vio/single_track/single_track.h"

#include <memory>
#include <thread>

#include <Eigen/Dense>
#include <sophus/se3.hpp>

#include <basalt/imu/preintegration.h>
#include <basalt/io/dataset_io.h>
#include <basalt/utils/assert.h>
#include <basalt/utils/test_utils.h>
#include <basalt/camera/generic_camera.hpp>
#include <basalt/camera/stereographic_param.hpp>
#include <basalt/utils/sophus_utils.hpp>

#include <basalt/vi_estimator/ba_base.h>
#include <basalt/vi_estimator/vio_estimator.h>

#include <iostream>

namespace dynvio {
using namespace basalt;

struct AccelBiasVar{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr = std::shared_ptr<AccelBiasVar>;
    int64_t t_ns{0};
    Eigen::Vector3d var;
};

template<class Param, class Model>
class DynamicsVioEstimator : public VioEstimatorBase,
                             public BundleAdjustmentBase{

public:
    typedef std::shared_ptr<DynamicsVioEstimator<Param, Model>> Ptr;

    DynamicsVioEstimator(const Eigen::Vector3d& g,
                         const basalt::Calibration<double>& calib,
                         const VioConfig& config,
                         const DynVIOConfig &dynvio_config);

    ~DynamicsVioEstimator() { processing_thread->join(); }

    tbb::concurrent_bounded_queue<Command::Ptr> cmd_data_queue;
    tbb::concurrent_bounded_queue<ParameterBase::Ptr>* out_param_state_queue;
    tbb::concurrent_bounded_queue<ExtrinsicState<double>::Ptr>* out_extr_queue;
    tbb::concurrent_bounded_queue<std::shared_ptr<Eigen::Vector3d>>* out_gyro_queue;
    tbb::concurrent_bounded_queue<AccelBiasVar::Ptr>* out_ba_var_queue;
    bool opt_started;
    bool adding_dynamics{false};
    bool is_marg_param{false};

    void initialize(const Eigen::Vector3d& bg, const Eigen::Vector3d& ba);
    void initialize(int64_t t_ns, const Sophus::SE3d& T_w_i,
                    const Eigen::Vector3d& vel_w_i,
                    const Eigen::Vector3d& bg,
                    const Eigen::Vector3d& ba);

    bool measure(const OpticalFlowResult::Ptr& opt_flow_meas,
                 const IntegratedImuMeasurement<double>::Ptr& meas,
                 const typename Model::Ptr& dynamics_factor, const ConstraintsFactor::Ptr &constraints_factor);

    void marginalize(const std::map<int64_t, int>& num_points_connected);

    void optimize();

    void computeProjections(
        std::vector<Eigen::aligned_vector<Eigen::Vector4d>>& res) const;

    const Sophus::SE3d& getT_w_i_init() { return T_w_i_init; }

    const SingleTrackAcado& GetODERef(){return singletrack_acado;}

private:
    void linearizeMargPriorDyn(const AbsOrderMap &marg_order, const Eigen::MatrixXd &marg_H,
                            const Eigen::VectorXd &marg_b, const AbsOrderMap &aom,
                            Eigen::MatrixXd &abs_H, Eigen::VectorXd &abs_b, double &marg_prior_error) const;

    void computeMargPriorErrorDyn(const AbsOrderMap& marg_order, const Eigen::MatrixXd& marg_H,
                               const Eigen::VectorXd& marg_b, double& marg_prior_error) const;


    void computeDeltaDyn(const AbsOrderMap& marg_order,
                         Eigen::VectorXd& delta) const;

    bool take_kf;
    int frames_after_kf;
    std::set<int64_t> kf_ids;

    int64_t last_state_t_ns;

    Eigen::aligned_map<int64_t, IntegratedImuMeasurement<double>> imu_meas;

    const Eigen::Vector3d g;

    // Input
    Eigen::aligned_map<int64_t, OpticalFlowResult::Ptr> prev_opt_flow_res;

    std::map<int64_t, int> num_points_kf;

    // Marginalization
    AbsOrderMap marg_order;
    Eigen::MatrixXd marg_H;
    Eigen::VectorXd marg_b;

    Eigen::Vector3d gyro_bias_weight, accel_bias_weight;

    size_t max_states;
    size_t max_kfs;

    Sophus::SE3d T_w_i_init;

    bool initialized;

    VioConfig config;

    double lambda, min_lambda, max_lambda, lambda_vee;
    SingleTrackAcado singletrack_acado;

    std::shared_ptr<std::thread> processing_thread;
    DynVIOConfig dynvio_config;

    Eigen::aligned_map<int64_t, typename Model::Ptr> dynamics_factors;
    Eigen::aligned_map<int64_t, ConstraintsFactor::Ptr> constraints_factors;

    std::deque<Command> command_history;
    std::map<int64_t, Command> rbf_cmds;
    std::map<int64_t, Command> avg_cmds;
    std::map<int64_t, Command> raw_cmds;

    size_t state_size{0};
    size_t state_param_size{0};
    Eigen::aligned_map<int64_t, ExtrinsicStateWithLin<double>> extr_states;
    Eigen::aligned_map<int64_t, ParameterWithLin<Param>> param_states;
    typename Param::Ptr param_init_prior, param_marged_prior;
    Eigen::aligned_map<int64_t, Eigen::Vector3d> gyro_recording;

    int64_t cur_window_start_t_ns{0};
    int64_t cur_window_2nd_t_ns{0};
    AccelBiasVar cur_window_start_ba_var;

    Eigen::Matrix<double, PARAM_SIZE, 1> scaled_param_init_weight;
    Eigen::Matrix<double, PARAM_SIZE, 1> scaled_param_prior_weight;
};
} // namespace dynvio
