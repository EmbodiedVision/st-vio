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

#include "dynamics_vio/utils/utils.h"

#include <basalt/serialization/eigen_io.h>
#include <fstream>
#include <cereal/archives/json.hpp>
#include <cereal/cereal.hpp>

namespace dynvio {

DynVIOConfig::DynVIOConfig(){
    cmd_history_size = 10;
    extr_init_weight = 1e3;
    param_init_weight = 1e2;

    //crawler parameters
    wheel_base = 0.31; //m
    mass = 6.4; //kg
    I_z = 0.15;
    l_fw_imu = 0.041;
    l_c_imu = 0.031;
    l_front = 0.15;

    c_lat_init = 40;
    steering_rt_init = -0.8;
    throttle_f1_init = 10;
    throttle_f2_init = 10;
    throttle_res_init = 10;

    dynamics_weight = 1.0;
    constraint_weight = 1.0;
    param_prior_weight = 1.0;
    extr_rd_weight = 1.0;

    ba_var_thr = 7e-4;
}

void DynVIOConfig::save(const std::string &filename){
    std::ofstream os(filename);

    {
        cereal::JSONOutputArchive archive(os);
        archive(*this);
    }
    os.close();
}

void DynVIOConfig::load(const std::string& filename){
    std::ifstream is(filename);

    {
        cereal::JSONInputArchive archive(is);
        archive(*this);
    }
    is.close();
}
} //namespace dynvio

namespace  cereal {
using namespace basalt;
template <class Archive>
void serialize(Archive& ar, dynvio::DynVIOConfig& dynvio_config) {

    ar(CEREAL_NVP(dynvio_config.cmd_history_size));
    ar(CEREAL_NVP(dynvio_config.extr_init_weight));
    ar(CEREAL_NVP(dynvio_config.param_init_weight));

    ar(CEREAL_NVP(dynvio_config.T_o_i_init));

    ar(CEREAL_NVP(dynvio_config.wheel_base));
    ar(CEREAL_NVP(dynvio_config.mass));
    ar(CEREAL_NVP(dynvio_config.I_z));
    ar(CEREAL_NVP(dynvio_config.l_fw_imu));
    ar(CEREAL_NVP(dynvio_config.l_c_imu));
    ar(CEREAL_NVP(dynvio_config.l_front));
    ar(CEREAL_NVP(dynvio_config.imu_forward));

    ar(CEREAL_NVP(dynvio_config.c_lat_init));
    ar(CEREAL_NVP(dynvio_config.steering_rt_init));
    ar(CEREAL_NVP(dynvio_config.throttle_f1_init));
    ar(CEREAL_NVP(dynvio_config.throttle_f2_init));
    ar(CEREAL_NVP(dynvio_config.throttle_res_init));

    ar(CEREAL_NVP(dynvio_config.dynamics_weight));
    ar(CEREAL_NVP(dynvio_config.constraint_weight));
    ar(CEREAL_NVP(dynvio_config.param_prior_weight));
    ar(CEREAL_NVP(dynvio_config.extr_rd_weight));

    ar(CEREAL_NVP(dynvio_config.ba_var_thr));
}
} // namespace cereal


