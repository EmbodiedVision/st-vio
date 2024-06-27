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

#include "dynamics_vio/utils/utils.h"

#include <basalt/io/dataset_io.h>

namespace dynvio {
using namespace basalt;

class VioDataset {
public:
    virtual ~VioDataset(){};

    virtual size_t get_num_cams() const = 0;

    virtual std::vector<int64_t> &get_image_timestamps() = 0;

    virtual const Eigen::aligned_vector<AccelData> &get_accel_data() const = 0;
    virtual const Eigen::aligned_vector<GyroData> &get_gyro_data() const = 0;
    virtual const std::vector<int64_t> &get_odom_timestamps() const = 0;
    virtual const std::vector<int64_t> &get_gt_timestamps() const = 0;
    virtual const Eigen::aligned_vector<Sophus::SE3d> &get_gt_pose_data()
        const = 0;
    virtual const Eigen::aligned_vector<Sophus::SE3d> &get_odom_pose_data()
        const = 0;
    virtual const Eigen::aligned_vector<Eigen::Vector3d> &get_odom_linvel_data() const = 0;
    virtual const Eigen::aligned_vector<Eigen::Vector3d> &get_odom_angvel_data() const = 0;
    virtual int64_t get_mocap_to_imu_offset_ns() const = 0;
    virtual std::vector<ImageData> get_image_data(int64_t t_ns) = 0;

    virtual const std::vector<Command> &get_cmd_data() const = 0;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

typedef std::shared_ptr<VioDataset> VioDatasetPtr;

class DatasetIoInterface {
public:
    virtual void read(const std::string &path) = 0;
    virtual void reset() = 0;
    virtual dynvio::VioDatasetPtr get_data() = 0;

    virtual ~DatasetIoInterface(){};
};

typedef std::shared_ptr<DatasetIoInterface> DatasetIoInterfacePtr;

class DatasetIoFactory {
public:
    static DatasetIoInterfacePtr getDatasetIo(const std::string &dataset_type, double start_time = 0.0, double end_time =0.0);
};
} // namespace dynvio
