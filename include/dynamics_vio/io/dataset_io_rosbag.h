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
basalt-mirror/include/basalt/io/dataset_io_rosbag.h
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

#include "dynamics_vio/io/dyn_vio_io.h"

#include <mutex>
#include <optional>

// Hack to access private functions
#define private public
#include <rosbag/bag.h>
#include <rosbag/view.h>
#undef private

#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <mavros_msgs/ManualControl.h>

#include <basalt/utils/filesystem.h>

namespace dynvio {

class RosbagVioDataset : public dynvio::VioDataset {
    std::shared_ptr<rosbag::Bag> bag;
    std::mutex m;

    size_t num_cams;

    std::vector<int64_t> image_timestamps;

    // vector of images for every timestamp
    // assumes vectors size is num_cams for every timestamp with null pointers for
    // missing frames
    std::unordered_map<int64_t, std::vector<std::optional<rosbag::IndexEntry>>>
        image_data_idx;

    Eigen::aligned_vector<AccelData> accel_data;
    Eigen::aligned_vector<GyroData> gyro_data;
    std::vector<Command> cmd_data;

    std::vector<int64_t> odom_timestamps;
    Eigen::aligned_vector<Sophus::SE3d> odom_pose_data;
    Eigen::aligned_vector<Eigen::Vector3d> odom_linvel_data;
    Eigen::aligned_vector<Eigen::Vector3d> odom_angvel_data;

    std::vector<int64_t> gt_timestamps;  // ordered gt timestamps
    Eigen::aligned_vector<Sophus::SE3d> gt_pose_data;

    int64_t mocap_to_imu_offset_ns;

public:
    ~RosbagVioDataset() {}

    size_t get_num_cams() const { return num_cams; }

    std::vector<int64_t> &get_image_timestamps() { return image_timestamps; }

    const std::vector<Command> &get_cmd_data() const { return cmd_data; }
    const std::vector<int64_t> &get_odom_timestamps() const {
        return odom_timestamps;
    }
    const Eigen::aligned_vector<Sophus::SE3d> &get_odom_pose_data() const {
        return odom_pose_data;
    }
    const Eigen::aligned_vector<Eigen::Vector3d> &get_odom_linvel_data() const {
        return odom_linvel_data;
    }
    const Eigen::aligned_vector<Eigen::Vector3d> &get_odom_angvel_data() const {
        return odom_angvel_data;
    }
    const Eigen::aligned_vector<AccelData> &get_accel_data() const {
        return accel_data;
    }
    const Eigen::aligned_vector<GyroData> &get_gyro_data() const {
        return gyro_data;
    }
    const std::vector<int64_t> &get_gt_timestamps() const {
        return gt_timestamps;
    }
    const Eigen::aligned_vector<Sophus::SE3d> &get_gt_pose_data() const {
        return gt_pose_data;
    }

    int64_t get_mocap_to_imu_offset_ns() const { return mocap_to_imu_offset_ns; }

    std::vector<ImageData> get_image_data(int64_t t_ns) {
        std::vector<ImageData> res(num_cams);

        auto it = image_data_idx.find(t_ns);

        if (it != image_data_idx.end())
            for (size_t i = 0; i < num_cams; i++) {
                ImageData &id = res[i];

                if (!it->second[i].has_value()) continue;

                m.lock();
                sensor_msgs::ImageConstPtr img_msg =
                    bag->instantiateBuffer<sensor_msgs::Image>(*it->second[i]);
                m.unlock();

                //        std::cerr << "img_msg->width " << img_msg->width << "
                //        img_msg->height "
                //                  << img_msg->height << std::endl;

                id.img.reset(
                    new ManagedImage<uint16_t>(img_msg->width, img_msg->height));

                if (!img_msg->header.frame_id.empty() &&
                    std::isdigit(img_msg->header.frame_id[0])) {
                    id.exposure = std::stol(img_msg->header.frame_id) * 1e-9;
                } else {
                    id.exposure = -1;
                }

                if (img_msg->encoding == "mono8") {
                    const uint8_t *data_in = img_msg->data.data();
                    uint16_t *data_out = id.img->ptr;

                    for (size_t i = 0; i < img_msg->data.size(); i++) {
                        int val = data_in[i];
                        val = val << 8;
                        data_out[i] = val;
                    }

                } else if (img_msg->encoding == "mono16") {
                    std::memcpy(id.img->ptr, img_msg->data.data(), img_msg->data.size());
                } else {
                    std::cerr << "Encoding " << img_msg->encoding << " is not supported."
                              << std::endl;
                    std::abort();
                }
            }

        return res;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    friend class RosbagIO;
};

class RosbagIO : public dynvio::DatasetIoInterface {
public:
    RosbagIO(double start_time = 0.0, double end_time = 0.0)
        : start_time(start_time), end_time(end_time) {}

    void read(const std::string &path) {
        if (!fs::exists(path))
            std::cerr << "No dataset found in " << path << std::endl;

        data.reset(new RosbagVioDataset);

        data->bag.reset(new rosbag::Bag);
        data->bag->open(path, rosbag::bagmode::Read);

        // start from s
        rosbag::View full_view(*data->bag);
        ros::Time initial_time = full_view.getBeginTime();
        initial_time += ros::Duration(start_time);

        ros::Time stop_time = full_view.getEndTime();
        stop_time -= ros::Duration(end_time);
        rosbag::View view(*data->bag, initial_time, stop_time);

        // get topics
        std::vector<const rosbag::ConnectionInfo *> connection_infos =
            view.getConnections();

        std::set<std::string> cam_topics;
        std::string imu_topic;
        std::string mocap_topic;
        std::string point_topic;
        std::string cmd_topic;
        std::string odom_topic;

        for (const rosbag::ConnectionInfo *info : connection_infos) {
            if (info->datatype == std::string("sensor_msgs/Image")&&
                info->topic.find("image", 0)!=std::string::npos) {
                cam_topics.insert(info->topic);
            } else if (info->datatype == std::string("sensor_msgs/Imu") &&
                       info->topic.rfind("/fcu", 0) != 0) {
                imu_topic = info->topic;
            } else if (info->datatype ==
                           std::string("geometry_msgs/TransformStamped") ||
                       info->datatype == std::string("geometry_msgs/PoseStamped")) {
                mocap_topic = info->topic;
            } else if (info->datatype == std::string("geometry_msgs/PointStamped")) {
                point_topic = info->topic;
            } else if (info->datatype == std::string("mavros_msgs/ManualControl")){
                cmd_topic = info->topic;
            } else if (info->datatype == std::string("nav_msgs/Odometry")){
                odom_topic = info->topic;
            }
        }

        std::cout << "cmd_topic: " << cmd_topic << std::endl;
        std::cout << "odom topic: " << odom_topic <<std::endl;
        std::cout << "imu_topic: " << imu_topic << std::endl;
        std::cout << "mocap_topic: " << mocap_topic << std::endl;
        std::cout << "cam_topics: ";
        for (const std::string &s : cam_topics) std::cout << s << " ";
        std::cout << std::endl;

        std::map<std::string, int> topic_to_id;
        int idx = 0;
        for (const std::string &s : cam_topics) {
            topic_to_id[s] = idx;
            idx++;
        }

        data->num_cams = cam_topics.size();

        int num_msgs = 0;

        int64_t min_time = std::numeric_limits<int64_t>::max();
        int64_t max_time = std::numeric_limits<int64_t>::min();

        std::vector<geometry_msgs::TransformStampedConstPtr> odom_pose_msgs;
        std::vector<geometry_msgs::TwistStampedConstPtr> odom_twist_msgs;
        std::vector<geometry_msgs::TransformStampedConstPtr> mocap_msgs;
        std::vector<geometry_msgs::PointStampedConstPtr> point_msgs;

        std::vector<int64_t>
            system_to_imu_offset_vec;  // t_imu = t_system + system_to_imu_offset
        std::vector<int64_t> system_to_mocap_offset_vec;  // t_mocap = t_system +
            // system_to_mocap_offset

        std::set<int64_t> image_timestamps;

        int64_t init_cmd_t_ns = std::numeric_limits<int64_t>::max();

        for (const rosbag::MessageInstance &m : view) {
            const std::string &topic = m.getTopic();

            if (odom_topic == topic){
                nav_msgs::OdometryConstPtr odom_msg = m.instantiate<nav_msgs::Odometry>();

                geometry_msgs::TransformStampedPtr odom_pose_msg(
                    new geometry_msgs::TransformStamped);
                odom_pose_msg->header = odom_msg->header;
                odom_pose_msg->transform.rotation = odom_msg->pose.pose.orientation;
                odom_pose_msg->transform.translation.x =
                    odom_msg->pose.pose.position.x;
                odom_pose_msg->transform.translation.y =
                    odom_msg->pose.pose.position.y;
                odom_pose_msg->transform.translation.z =
                    odom_msg->pose.pose.position.z;
                odom_pose_msgs.push_back(odom_pose_msg);

                geometry_msgs::TwistStampedPtr odom_twist_msg(
                    new geometry_msgs::TwistStamped);
                odom_twist_msg->header = odom_msg->header;
                odom_twist_msg->twist = odom_msg->twist.twist;
                odom_twist_msgs.push_back(odom_twist_msg);
            }

            if (cmd_topic == topic) {
                mavros_msgs::ManualControlConstPtr control_msg =
                    m.instantiate<mavros_msgs::ManualControl>();
                int64_t time = control_msg->header.stamp.toNSec();
                data->cmd_data.emplace_back();
                data->cmd_data.back().t_ns = time;
                data->cmd_data.back().linear = control_msg->z * 1e-3; // scale to [-1, 1] mavros
                data->cmd_data.back().angular = control_msg->y * 1e-3;

                if(time < init_cmd_t_ns)
                    init_cmd_t_ns = time;
            }

            if (cam_topics.find(topic) != cam_topics.end()) {
                sensor_msgs::ImageConstPtr img_msg =
                    m.instantiate<sensor_msgs::Image>();
                int64_t timestamp_ns = img_msg->header.stamp.toNSec();

                if( timestamp_ns < init_cmd_t_ns)
                    continue; //drop images smaller than first cmd.

                auto &img_vec = data->image_data_idx[timestamp_ns];
                if (img_vec.size() == 0) img_vec.resize(data->num_cams);

                img_vec[topic_to_id.at(topic)] = m.index_entry_;
                image_timestamps.insert(timestamp_ns);
                min_time = std::min(min_time, timestamp_ns);
                max_time = std::max(max_time, timestamp_ns);
            }

            if (imu_topic == topic) {
                sensor_msgs::ImuConstPtr imu_msg = m.instantiate<sensor_msgs::Imu>();
                int64_t time = imu_msg->header.stamp.toNSec();

                data->accel_data.emplace_back();
                data->accel_data.back().timestamp_ns = time;
                data->accel_data.back().data = Eigen::Vector3d(
                    imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y,
                    imu_msg->linear_acceleration.z);

                data->gyro_data.emplace_back();
                data->gyro_data.back().timestamp_ns = time;
                data->gyro_data.back().data = Eigen::Vector3d(
                    imu_msg->angular_velocity.x, imu_msg->angular_velocity.y,
                    imu_msg->angular_velocity.z);

                min_time = std::min(min_time, time);
                max_time = std::max(max_time, time);

                int64_t msg_arrival_time = m.getTime().toNSec();
                system_to_imu_offset_vec.push_back(time - msg_arrival_time);
            }

            if (mocap_topic == topic) {
                geometry_msgs::TransformStampedConstPtr mocap_msg =
                    m.instantiate<geometry_msgs::TransformStamped>();

                // Try different message type if instantiate did not work
                if (!mocap_msg) {
                    geometry_msgs::PoseStampedConstPtr mocap_pose_msg =
                        m.instantiate<geometry_msgs::PoseStamped>();

                    geometry_msgs::TransformStampedPtr mocap_new_msg(
                        new geometry_msgs::TransformStamped);
                    mocap_new_msg->header = mocap_pose_msg->header;
                    mocap_new_msg->transform.rotation = mocap_pose_msg->pose.orientation;
                    mocap_new_msg->transform.translation.x =
                        mocap_pose_msg->pose.position.x;
                    mocap_new_msg->transform.translation.y =
                        mocap_pose_msg->pose.position.y;
                    mocap_new_msg->transform.translation.z =
                        mocap_pose_msg->pose.position.z;

                    mocap_msg = mocap_new_msg;
                }

                int64_t time = mocap_msg->header.stamp.toNSec();

                mocap_msgs.push_back(mocap_msg);

                int64_t msg_arrival_time = m.getTime().toNSec();
                system_to_mocap_offset_vec.push_back(time - msg_arrival_time);
            }

            if (point_topic == topic) {
                geometry_msgs::PointStampedConstPtr mocap_msg =
                    m.instantiate<geometry_msgs::PointStamped>();

                int64_t time = mocap_msg->header.stamp.toNSec();

                point_msgs.push_back(mocap_msg);

                int64_t msg_arrival_time = m.getTime().toNSec();
                system_to_mocap_offset_vec.push_back(time - msg_arrival_time);
            }

            num_msgs++;
        }

        data->image_timestamps.clear();
        data->image_timestamps.insert(data->image_timestamps.begin(),
                                      image_timestamps.begin(),
                                      image_timestamps.end());

        if (system_to_mocap_offset_vec.size() > 0) {
            int64_t system_to_imu_offset =
                system_to_imu_offset_vec[system_to_imu_offset_vec.size() / 2];

            int64_t system_to_mocap_offset =
                system_to_mocap_offset_vec[system_to_mocap_offset_vec.size() / 2];

            data->mocap_to_imu_offset_ns =
                system_to_imu_offset - system_to_mocap_offset;
        }

        data->gt_pose_data.clear();
        data->gt_timestamps.clear();

        if (!mocap_msgs.empty())
            for (size_t i = 0; i < mocap_msgs.size() - 1; i++) {
                auto mocap_msg = mocap_msgs[i];

                int64_t time = mocap_msg->header.stamp.toNSec();

                Eigen::Quaterniond q(
                    mocap_msg->transform.rotation.w, mocap_msg->transform.rotation.x,
                    mocap_msg->transform.rotation.y, mocap_msg->transform.rotation.z);

                Eigen::Vector3d t(mocap_msg->transform.translation.x,
                                  mocap_msg->transform.translation.y,
                                  mocap_msg->transform.translation.z);

                int64_t timestamp_ns = time + data->mocap_to_imu_offset_ns;
                data->gt_timestamps.emplace_back(timestamp_ns);
                data->gt_pose_data.emplace_back(q, t);
            }

        data->odom_pose_data.clear();
        data->odom_timestamps.clear();
        if (!odom_pose_msgs.empty())
            for (size_t i = 0; i < odom_pose_msgs.size() - 1; i++) {
                auto odom_pose_msg = odom_pose_msgs[i];

                int64_t time = odom_pose_msg->header.stamp.toNSec();

                Eigen::Quaterniond q(
                    odom_pose_msg->transform.rotation.w, odom_pose_msg->transform.rotation.x,
                    odom_pose_msg->transform.rotation.y, odom_pose_msg->transform.rotation.z);

                Eigen::Vector3d t(odom_pose_msg->transform.translation.x,
                                  odom_pose_msg->transform.translation.y,
                                  odom_pose_msg->transform.translation.z);


                data->odom_timestamps.emplace_back(time);
                data->odom_pose_data.emplace_back(q, t);
            }

        if(!odom_twist_msgs.empty())
            for(size_t i = 0; i < odom_twist_msgs.size() -1 ; i++){
                auto odom_twist_msg = odom_twist_msgs[i];

                Eigen::Vector3d linvel(odom_twist_msg->twist.linear.x,
                                       odom_twist_msg->twist.linear.y,
                                       odom_twist_msg->twist.linear.z);

                Eigen::Vector3d angvel(odom_twist_msg->twist.angular.x,
                                       odom_twist_msg->twist.angular.y,
                                       odom_twist_msg->twist.angular.z);

                data->odom_linvel_data.emplace_back(linvel);
                data->odom_angvel_data.emplace_back(angvel);
            }


        if (!point_msgs.empty())
            for (size_t i = 0; i < point_msgs.size() - 1; i++) {
                auto point_msg = point_msgs[i];

                int64_t time = point_msg->header.stamp.toNSec();

                Eigen::Vector3d t(point_msg->point.x, point_msg->point.y,
                                  point_msg->point.z);

                int64_t timestamp_ns = time;  // + data->mocap_to_imu_offset_ns;
                data->gt_timestamps.emplace_back(timestamp_ns);
                data->gt_pose_data.emplace_back(Sophus::SO3d(), t);
            }

        std::cout << "Total number of messages: " << num_msgs << std::endl;
        std::cout << "Image size: " << data->image_data_idx.size() << std::endl;

        std::cout << "Min time: " << min_time << " max time: " << max_time
                  << " mocap to imu offset: " << data->mocap_to_imu_offset_ns
                  << std::endl;

        std::cout << "Number of mocap poses: " << data->gt_timestamps.size()
                  << std::endl;
    }

    void reset() { data.reset(); }

    VioDatasetPtr get_data() {
        // return std::dynamic_pointer_cast<VioDataset>(data);
        return data;
    }

private:
    std::shared_ptr<RosbagVioDataset> data;
    double start_time;
    double end_time;
};

}  // namespace dynvio
