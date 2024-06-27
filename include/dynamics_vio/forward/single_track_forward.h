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

#include "dynamics_vio/io/dyn_vio_io.h"
#include "dynamics_vio/utils/utils.h"
#include "dynamics_vio/parameters/parameters.h"
#include "dynamics_vio/single_track/single_track.h"

namespace dynvio {

class SingleTrackForward{
public:
    SingleTrackForward(const DynVIOConfig& dynvio_config, const std::vector<int64_t>& time_stamps,
                       const std::vector<dynvio::Command>& cmd_data_out,
                       const dynvio::SingleTrackAcado& singletrack_acado);

    Sophus::SE3d ComputePose(const size_t &start_indx, const size_t &steps, const Sophus::SE3d& init_pose, const Sophus::SE3d& init_extr, 
                     const Eigen::Vector3d &init_linvel, const Eigen::Vector3d &init_angvel,
                     Eigen::aligned_map<int64_t, Sophus::SE3d> &poses);

    void UpdateParam(const SingleTrackParamOnline &param_state);

private:
    size_t cmd_history_size{1};
    std::vector<int64_t> gt_timestamps;
    std::vector<Command> cmd_data_out;

    double c_lat{30};  //distantfe between front wheel and cg, m
    double steering_rt{-0.5}; //steering ratio [0,1]->[0,0.28]grad
    double throttle_f1{3};
    double throttle_f2{0};
    double throttle_res{0};

    Sophus::SE3d T_o_i;
    const dynvio::SingleTrackAcado& singletrack_acado;
};
} // namespace dynvio
