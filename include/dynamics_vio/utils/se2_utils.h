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

#include <sophus/se2.hpp>

namespace dynvio {
inline Eigen::Vector2d planar_rotate(const double& theta, const Eigen::Vector2d& p){
    double real = std::cos(theta);
    double imag = std::sin(theta);
    return Eigen::Vector2d(real * p[0] - imag * p[1],
                           imag * p[0] + real * p[1]);
}

inline Eigen::Vector3d logSE2(const Eigen::Vector2d trans_xy, const double yaw,
                              Eigen::Matrix2d& V_inv){

    Sophus::SO2d yaw_so2(yaw);
    double halftheta = 0.5 * yaw;
    double halftheta_by_tan_of_halftheta;
    Eigen::Vector2d z =
        yaw_so2.unit_complex();  // cos(yaw) + i * sin(yaw)
    double real_minus_one = z.x() - 1.0;

    if (std::abs(real_minus_one) < Sophus::Constants<double>::epsilon()) {
        halftheta_by_tan_of_halftheta = 1.0 - 1.0 / 12.0 * yaw * yaw;
    } else {
        halftheta_by_tan_of_halftheta = -(halftheta * z.y()) / (real_minus_one);
    }
    V_inv << halftheta_by_tan_of_halftheta, halftheta, -halftheta,
        halftheta_by_tan_of_halftheta;

    Eigen::Vector3d se2_log;
    se2_log.head<2>() = V_inv * trans_xy;
    se2_log(2) = yaw;

    return se2_log;
}

inline Eigen::Vector2d J_se2xy_theta(const Eigen::Vector2d trans_xy, const double yaw){
    Eigen::Vector2d J_se2xy_theta;

    Sophus::SO2d yaw_so2(yaw);
    Eigen::Vector2d z =
        yaw_so2.unit_complex();  // cos(yaw) + i * sin(yaw)
    double real_minus_one = z.x() - 1.0;

    if (std::abs(real_minus_one) < 1e-10) {
        J_se2xy_theta(0) = trans_xy(1) / 2.0 - yaw * trans_xy(0) / 6.0;
        J_se2xy_theta(1) = -trans_xy(0) / 2.0 - yaw * trans_xy(1) / 6.0;
    } else {
        double sin_sq = z.y() * z.y();
        double real_minus_one_sq = real_minus_one * real_minus_one;
        J_se2xy_theta(0) =
            ((-trans_xy(0) * z.y() - trans_xy(0) * z.x() * yaw) /
                 real_minus_one -
             trans_xy(0) * sin_sq * yaw / real_minus_one_sq +
             trans_xy(1)) /
            2.0;
        J_se2xy_theta(1) =
            ((-trans_xy(1) * z.y() - trans_xy(1) * z.x() * yaw) /
                 real_minus_one -
             trans_xy(1) * sin_sq * yaw / real_minus_one_sq -
             trans_xy(0)) /
            2.0;
    }

    return J_se2xy_theta;
}
} //namesapce dynvio
