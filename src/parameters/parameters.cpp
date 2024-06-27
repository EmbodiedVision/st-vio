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

#include "dynamics_vio/parameters/parameters.h"
#include <basalt/imu/imu_types.h>

namespace dynvio {
using namespace basalt;

void SingleTrackParamOnline::applyInc(const Eigen::VectorXd &inc){
    BASALT_ASSERT(inc.rows() == PARAM_SIZE);

    c_lat += inc(0);
    steering_rt += inc(1);
    throttle_f1 += inc(2);
    throttle_f2 += inc(3);
    throttle_res += inc(4);
}
} //namespace dynvio
