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

SingleTrackAcado::SingleTrackAcado(double mass, double wheel_base, double I_z, double l_front):
                                    mass(mass), wheel_base(wheel_base), I_z(I_z), l_front(l_front){


    // throttle to accel mapping
    ACADO::IntermediateState exp_2scaled_vx = exp(20 * velocity_x);
    ACADO::IntermediateState tanh_scaled_vx = (exp_2scaled_vx - 1) / (exp_2scaled_vx + 1);
    ACADO::IntermediateState resistance = tanh_scaled_vx * Throttle_res;
    ACADO::IntermediateState power_train = Throttle_f1 * Cmd_linear - Throttle_f2 * velocity_x;
    ACADO::IntermediateState power_train_reg = 0.202 * power_train + 2.335 * (log(1 + exp(power_train)) - log(2));
    F_x = power_train_reg - resistance;

    ACADO::IntermediateState tmp_v_front_y =  velocity_y + l_front * yaw_rate;
    ACADO::IntermediateState cos_steering = cos(Steering_rt * Cmd_angular);
    ACADO::IntermediateState sin_steering = sin(Steering_rt * Cmd_angular);

    ACADO::IntermediateState tmp_front_vy = (velocity_x * cos_steering + tmp_v_front_y * sin_steering);
    F_front_y = C_lat * atan((velocity_x * sin_steering - tmp_v_front_y * cos_steering)/ (log(exp(2 * tmp_front_vy) + 1) - tmp_front_vy));
    F_rear_y = C_lat* atan(((this->wheel_base - this->l_front) * yaw_rate - velocity_y) / (log(exp(2 * velocity_x) + 1) - velocity_x));


    f << dot(trans_x) == velocity_x - yaw_rate * trans_y;
    f << dot(trans_y) == velocity_y + yaw_rate * trans_x;
    f << dot(yaw) == yaw_rate;

    //linear
    f << dot(velocity_x) == (F_x - F_front_y * sin(Steering_rt * Cmd_angular)) / this->mass
                                + velocity_y * yaw_rate;

    f << dot(velocity_y) == (F_front_y * cos_steering + F_rear_y) / this->mass
                                - velocity_x * yaw_rate;
    //angular
    f << dot(yaw_rate) == (this->l_front * F_front_y * cos_steering
                           - (this->wheel_base - this->l_front) * F_rear_y) / this->I_z;
}

SingleTrackAcado::~SingleTrackAcado(){
    velocity_x.clearStaticCounters();
    C_lat.clearStaticCounters();
    F_x.clearStaticCounters();
    Cmd_linear.clearStaticCounters();
}
} // namespace dynvio
