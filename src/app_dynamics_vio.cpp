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
basalt-mirror/src/vio.cpp
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

#include "dynamics_vio/io/dataset_io_rosbag.h"
#include "dynamics_vio/vio_estimator/vio_base.h"
#include "dynamics_vio/parameters/parameters.h"
#include "dynamics_vio/single_track/single_track.h"
#include "dynamics_vio/forward/single_track_forward.h"

#include <iostream>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <thread>

#include <sophus/se3.hpp>

#include <tbb/concurrent_queue.h>
#include <tbb/global_control.h>

#include <pangolin/display/image_view.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/image/image.h>
#include <pangolin/pangolin.h>
#include <pangolin/image/image_io.h>
#include <pangolin/image/typed_image.h>

#include <CLI/CLI.hpp>

#include <basalt/io/marg_data_io.h>
#include <basalt/calibration/calibration.hpp>
#include <basalt/utils/vis_utils.h>
#include <boost/archive/binary_oarchive.hpp>
#include <basalt/serialization/headers_serialization.h>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>

// GUI functions
const uint8_t vio_color[3]{129, 134, 74};
const uint8_t calib_color[3]{214, 39, 40};
const uint8_t init_color[3]{31, 119, 180};
void draw_image_overlay(pangolin::View& v, size_t cam_id);
void draw_scene(pangolin::View& view);
void load_data(const std::string& calib_path);
bool next_step();
bool prev_step();
void draw_plots();
void alignButton();
void alignDeviceButton();
void saveTrajectoryButton();

// Pangolin variables
constexpr int UI_WIDTH = 200;

using Button = pangolin::Var<std::function<void(void)>>;

pangolin::DataLog imu_data_log, vio_data_log, error_data_log;
pangolin::Plotter* plotter;

pangolin::Var<int> show_frame("ui.show_frame", 0, 0, 1500);

pangolin::Var<bool> show_flow("ui.show_flow", false, false, true);
pangolin::Var<bool> show_obs("ui.show_obs", true, false, true);
pangolin::Var<bool> show_ids("ui.show_ids", false, false, true);

pangolin::Var<bool> show_est_param("ui.show_est_param", true, false, true);

pangolin::Var<bool> show_est_pos("ui.show_est_pos", true, false, true);
pangolin::Var<bool> show_est_vel("ui.show_est_vel", false, false, true);
pangolin::Var<bool> show_est_bg("ui.show_est_bg", false, false, true);
pangolin::Var<bool> show_est_ba("ui.show_est_ba", false, false, true);

pangolin::Var<bool> show_gt("ui.show_gt", false, false, true);

Button next_step_btn("ui.next_step", &next_step);
Button prev_step_btn("ui.prev_step", &prev_step);

pangolin::Var<bool> continue_btn("ui.continue", true, false, true);
pangolin::Var<bool> continue_fast("ui.continue_fast", false, false, true);

Button align_se3_btn("ui.align_se3", &alignButton);

pangolin::Var<bool> euroc_fmt("ui.euroc_fmt", false, false, true);
pangolin::Var<bool> tum_rgbd_fmt("ui.tum_rgbd_fmt", true, false, true);
pangolin::Var<bool> kitti_fmt("ui.kitti_fmt", false, false, true);
Button save_traj_btn("ui.save_traj", &saveTrajectoryButton);
bool save_gt = false;
bool save_pred = false;
bool save_ba_var = false;
bool record = false;

pangolin::Var<bool> follow("ui.follow", true, false, true);

pangolin::OpenGlRenderState camera;

// Visualization variables
std::unordered_map<int64_t, basalt::VioVisualizationData::Ptr> vis_map;

tbb::concurrent_bounded_queue<basalt::VioVisualizationData::Ptr> out_vis_queue;
tbb::concurrent_bounded_queue<basalt::PoseVelBiasState<double>::Ptr> out_state_queue;
tbb::concurrent_bounded_queue<dynvio::ExtrinsicState<double>::Ptr> out_extr_queue;
tbb::concurrent_bounded_queue<dynvio::ParameterBase::Ptr> out_param_state_queue;
tbb::concurrent_bounded_queue<std::shared_ptr<Eigen::Vector3d>> out_gyro_queue;
tbb::concurrent_bounded_queue<dynvio::AccelBiasVar::Ptr> out_ba_var_queue;

std::vector<int64_t> vio_t_ns;
std::vector<int64_t> vio_dyn_t_ns;

// Eigen::aligned_vector<Eigen::Vector3d> vio_bg;
Eigen::aligned_vector<Eigen::Vector3d> vio_ba;
Eigen::aligned_vector<Eigen::Vector3d> vio_vel_w_i;
Eigen::aligned_vector<Eigen::Vector3d> vio_gyro_calib;
Eigen::aligned_vector<Eigen::Vector3d> vio_t_w_i;
Eigen::aligned_vector<Sophus::SE3d> vio_T_w_i;
Eigen::aligned_vector<Sophus::SE3d> vio_T_o_i;
Eigen::aligned_map<int64_t, dynvio::SingleTrackParamOnline> param_map;
Eigen::aligned_map<int64_t, Eigen::Vector3d> ba_var_map;

std::vector<int64_t> gt_t_ns;
Eigen::aligned_vector<Eigen::Vector3d> gt_t_w_i;
Eigen::aligned_vector<Sophus::SE3d> gt_T_w_i;

std::string marg_data_path;
size_t last_frame_processed = 0;
tbb::concurrent_unordered_map<int64_t, int, std::hash<int64_t>> timestamp_to_id;

std::mutex m;
std::condition_variable cv;
bool step_by_step = false;

// VIO variables
basalt::Calibration<double> calib;

dynvio::VioDatasetPtr vio_dataset;
dynvio::DynVIOConfig dynvio_config;
basalt::VioConfig vio_config;
basalt::OpticalFlowBase::Ptr opt_flow_ptr;

std::string output_dir;

dynvio::DynamicsVioEstimator<dynvio::SingleTrackParamOnline, dynvio::SingleTrackModel>::Ptr dyn_vio;

// Feed functions
void feed_images() {
    std::cout << "Started input_data thread " << std::endl;

    for (size_t i = 0; i < vio_dataset->get_image_timestamps().size(); i++) {

        if (step_by_step) {
            std::unique_lock<std::mutex> lk(m);
            cv.wait(lk);
        }

        basalt::OpticalFlowInput::Ptr data(new basalt::OpticalFlowInput);

        data->t_ns = vio_dataset->get_image_timestamps()[i];
        data->img_data = vio_dataset->get_image_data(data->t_ns);

        timestamp_to_id[data->t_ns] = i;

        opt_flow_ptr->input_queue.push(data);
    }

    // Indicate the end of the sequence
    opt_flow_ptr->input_queue.push(nullptr);

    std::cout << "Finished input_data thread " << std::endl;
}

void feed_imu(tbb::concurrent_bounded_queue<basalt::ImuData<double>::Ptr>& imu_data_queue) {
    for (size_t i = 0; i < vio_dataset->get_gyro_data().size(); i++) {
        basalt::ImuData<double>::Ptr data(new basalt::ImuData<double>);
        data->t_ns = vio_dataset->get_gyro_data()[i].timestamp_ns;

        data->accel = vio_dataset->get_accel_data()[i].data;
        data->gyro = vio_dataset->get_gyro_data()[i].data;

        imu_data_queue.push(data);
    }
    imu_data_queue.push(nullptr);
}

void feed_cmd(tbb::concurrent_bounded_queue<dynvio::Command::Ptr>& cmd_data_queue) {

    for (size_t i = 0; i < vio_dataset->get_cmd_data().size(); i++) {
        dynvio::Command::Ptr data(new dynvio::Command);
        data->t_ns = vio_dataset->get_cmd_data()[i].t_ns;
        data->linear = vio_dataset->get_cmd_data()[i].linear;
        data->angular = vio_dataset->get_cmd_data()[i].angular;

        cmd_data_queue.push(data);
    }
    cmd_data_queue.push(nullptr);
}

void feed_cmd(std::vector<dynvio::Command>& cmd_data_vec) {

    for (size_t i = 0; i < vio_dataset->get_cmd_data().size(); i++) {
        dynvio::Command data;
        data.t_ns = vio_dataset->get_cmd_data()[i].t_ns;
        data.linear = vio_dataset->get_cmd_data()[i].linear;
        data.angular = vio_dataset->get_cmd_data()[i].angular;

        cmd_data_vec.push_back(data);
    }
}

int main(int argc, char** argv)
{
    bool show_gui = true;
    bool print_queue = false;
    bool terminate = false;
    std::string cam_calib_path;
    std::string dataset_path;
    std::string config_path;
    std::string dynvio_config_path;
    std::string result_path;
    std::string trajectory_fmt;
    int num_threads = 0;
    double start_time = 0.0;
    double end_time = 0.0;
    CLI::App app{"Dynamics VIO"};

    Eigen::aligned_vector<Eigen::Vector3d> tmp_param_history;

    app.add_option("--show-gui", show_gui, "Show GUI");
    app.add_option("--cam-calib", cam_calib_path,
                   "Ground-truth camera calibration used for simulation.")
        ->required();

    app.add_option("--dataset-path", dataset_path, "Path to dataset.")
        ->required();

    app.add_option("--marg-data", marg_data_path,
                   "Path to folder where marginalization data will be stored.");

    app.add_option("--print-queue", print_queue, "Print queue.");
    app.add_option("--config-path", config_path, "Path to config file.");
    app.add_option("--dynvio-config-path", dynvio_config_path, "Path to dynvio config file.");
    app.add_option("--result-path", result_path,
                   "Path to result file where the system will write RMSE ATE.");
    app.add_option("--num-threads", num_threads, "Number of threads.");
    app.add_option("--step-by-step", step_by_step, "Path to config file.");
    app.add_option("--save-trajectory", trajectory_fmt,
                   "Save trajectory. Supported formats <tum, euroc, kitti>");

    app.add_option("--start-time", start_time, "Start time offset of dataset.");
    app.add_option("--end-time", end_time, "End time offset of dataset.");
    app.add_option("--save-gt", save_gt, "Save ground truth poses.");
    app.add_option("--save-pred", save_pred, "Save predicted poses.");
    app.add_option("--output-dir", output_dir, "Save ground truth poses.");
    app.add_option("--save-bavar", save_ba_var, "Save ba variance.");

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    }

    // global thread limit is in effect until global_control object is destroyed
    std::unique_ptr<tbb::global_control> tbb_global_control;
    if (num_threads > 0) {
        tbb_global_control = std::make_unique<tbb::global_control>(
            tbb::global_control::max_allowed_parallelism, num_threads);
    }

    if (!config_path.empty()) {
        vio_config.load(config_path);

        if (vio_config.vio_enforce_realtime) {
            vio_config.vio_enforce_realtime = false;
            std::cout
                << "The option vio_config.vio_enforce_realtime was enabled, "
                   "but it should only be used with the live executables (supply "
                   "images at a constant framerate). This executable runs on the "
                   "datasets and processes images as fast as it can, so the option "
                   "will be disabled. "
                << std::endl;
        }
    }

    if(!dynvio_config_path.empty()){
        dynvio_config.load(dynvio_config_path);
    }

    load_data(cam_calib_path);

    {
        dynvio::DatasetIoInterfacePtr dataset_io =
            dynvio::DatasetIoFactory::getDatasetIo("bag",start_time, end_time);

        dataset_io->read(dataset_path);

        vio_dataset = dataset_io->get_data();

        show_frame.Meta().range[1] = vio_dataset->get_image_timestamps().size() - 1;
        show_frame.Meta().gui_changed = true;

        opt_flow_ptr =
            basalt::OpticalFlowFactory::getOpticalFlow(vio_config, calib);

        if(!vio_dataset->get_gt_pose_data().empty()){
            for (size_t i = 0; i < vio_dataset->get_gt_pose_data().size(); i++) {
                gt_t_ns.push_back(vio_dataset->get_gt_timestamps()[i]);
                gt_T_w_i.push_back(vio_dataset->get_gt_pose_data()[i]);
                gt_t_w_i.push_back(vio_dataset->get_gt_pose_data()[i].translation());
            }
        }
        else {
            std::cout << "No ground truth available!" << std::endl;
        }
    }

    const int64_t start_t_ns = vio_dataset->get_image_timestamps().front();
    {
        dyn_vio.reset(new dynvio::DynamicsVioEstimator<dynvio::SingleTrackParamOnline, dynvio::SingleTrackModel>
                      (basalt::constants::g, calib, vio_config, dynvio_config));
        dyn_vio->initialize(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

        opt_flow_ptr->output_queue = &dyn_vio->vision_data_queue;
        if (show_gui) dyn_vio->out_vis_queue = &out_vis_queue;
        dyn_vio->out_state_queue = &out_state_queue;
        dyn_vio->out_param_state_queue = &out_param_state_queue;
        dyn_vio->out_extr_queue = &out_extr_queue;
        dyn_vio->out_gyro_queue = &out_gyro_queue;
        dyn_vio->out_ba_var_queue = &out_ba_var_queue;
    }

    basalt::MargDataSaver::Ptr marg_data_saver;

    if (!marg_data_path.empty()) {
        marg_data_saver.reset(new basalt::MargDataSaver(marg_data_path));
        dyn_vio->out_marg_queue = &marg_data_saver->in_marg_queue;

        // Save gt.
        {
            std::string p = marg_data_path + "/gt.cereal";
            std::ofstream os(p, std::ios::binary);

            {
                cereal::BinaryOutputArchive archive(os);
                archive(gt_t_ns);
                archive(gt_t_w_i);
            }
            os.close();
        }
    }

    vio_data_log.Clear();

    std::thread t1(&feed_images);
    std::thread t2([&]{feed_imu(dyn_vio->imu_data_queue);});
    std::shared_ptr<std::thread> t_cmd;
    t_cmd.reset(new std::thread([&]{feed_cmd(dyn_vio->cmd_data_queue);}));


    std::shared_ptr<std::thread> t3;

    if (show_gui)
        t3.reset(new std::thread([&]() {
            basalt::VioVisualizationData::Ptr data;

            while (true) {
                out_vis_queue.pop(data);

                if (data.get()) {
                    vis_map[data->t_ns] = data;
                } else {
                    break;
                }
            }

            std::cout << "Finished t3" << std::endl;
        }));

    std::thread t4([&]() {
        basalt::PoseVelBiasState<double>::Ptr data;
        dynvio::ParameterBase::Ptr param_data;
        dynvio::ExtrinsicState<double>::Ptr extr_data;
        std::shared_ptr<Eigen::Vector3d> gyro_data;
        dynvio::AccelBiasVar::Ptr bg_var_data;

        bool pop_state = true;
        bool pop_extr = true;
        bool pop_param = true;
        bool pop_ba_var = true;

        while (true) {
            if(pop_state){
                out_state_queue.pop(data);
                out_gyro_queue.pop(gyro_data);
                if(data.get()){
                    int64_t t_ns = data->t_ns;
                    Sophus::SE3d T_w_i = data->T_w_i;
                    Eigen::Vector3d vel_w_i = data->vel_w_i;
                    Eigen::Vector3d bg = data->bias_gyro;
                    Eigen::Vector3d ba = data->bias_accel;

                    vio_t_ns.emplace_back(data->t_ns);
                    vio_t_w_i.emplace_back(T_w_i.translation());
                    vio_T_w_i.emplace_back(T_w_i);
                    // vio_bg.emplace_back(bg);
                    vio_ba.emplace_back(ba);
                    vio_vel_w_i.emplace_back(vel_w_i);

                    if(gyro_data.get()){
                        Eigen::Vector3d gyro_calib = *gyro_data - bg;
                        vio_gyro_calib.emplace_back(gyro_calib);
                    }

                    if (show_gui) {
                        std::vector<float> vals;
                        vals.push_back((t_ns - start_t_ns) * 1e-9);

                        for (int i = 0; i < 3; i++) vals.push_back(vel_w_i[i]);
                        for (int i = 0; i < 3; i++) vals.push_back(T_w_i.translation()[i]);
                        for (int i = 0; i < 3; i++) vals.push_back(bg[i]);
                        for (int i = 0; i < 3; i++) vals.push_back(ba[i]);

                        vio_data_log.Log(vals);
                    }
                }
                else{
                    pop_state = false;
                }
            }

            if(dyn_vio->opt_started){
                if(pop_ba_var){
                    out_ba_var_queue.pop(bg_var_data);
                    if(bg_var_data.get()){
                        ba_var_map[bg_var_data->t_ns] = bg_var_data->var;
                    }
                    else{
                        pop_ba_var = false;
                    }
                }
            }

            if(dyn_vio->adding_dynamics){
                if(pop_extr){
                    out_extr_queue.pop(extr_data);
                    if(extr_data.get()){
                        vio_T_o_i.emplace_back(extr_data->T_o_i);
                        vio_dyn_t_ns.emplace_back(extr_data->t_ns);
                    }
                    else{
                        pop_extr = false;
                    }
                }

                if(pop_param){
                    out_param_state_queue.pop(param_data);
                    if(param_data.get()){
                        auto param = std::dynamic_pointer_cast<dynvio::SingleTrackParamOnline>(param_data);
                        // param_vec.push_back(*param);
                        param_map[param->t_ns] = *param;
                    }
                    else{
                        pop_param = false;
                    }
                }
            }

            if (!pop_state) break;
        }

        std::cout << "Finished t4" << std::endl;
    });

    std::shared_ptr<std::thread> t5;

    if (print_queue) {
        t5.reset(new std::thread([&]() {
            while (!terminate) {
                std::cout << "opt_flow_ptr->input_queue "
                          << opt_flow_ptr->input_queue.size()
                          << " opt_flow_ptr->output_queue "
                          << opt_flow_ptr->output_queue->size() << " out_state_queue "
                          << out_state_queue.size()
                          << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }));
    }

    auto time_start = std::chrono::high_resolution_clock::now();

    if (show_gui) {
        pangolin::CreateWindowAndBind("Main", 1800, 1000);

        glEnable(GL_DEPTH_TEST);

        pangolin::View& main_display = pangolin::CreateDisplay().SetBounds(
            0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0);

        pangolin::View& img_view_display = pangolin::CreateDisplay()
                                               .SetBounds(0.4, 1.0, 0.0, 0.4)
                                               .SetLayout(pangolin::LayoutEqual);

        pangolin::View& plot_display = pangolin::CreateDisplay().SetBounds(
            0.0, 0.4, pangolin::Attach::Pix(UI_WIDTH), 1.0);

        plotter = new pangolin::Plotter(&imu_data_log, 0.0, 100, -10.0, 10.0, 0.01f,
                                        0.01f);
        plot_display.AddDisplay(*plotter);

        pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0,
                                              pangolin::Attach::Pix(UI_WIDTH));

        std::vector<std::shared_ptr<pangolin::ImageView>> img_view;
        while (img_view.size() < calib.intrinsics.size()) {
            std::shared_ptr<pangolin::ImageView> iv(new pangolin::ImageView);

            size_t idx = img_view.size();
            img_view.push_back(iv);

            img_view_display.AddDisplay(*iv);
            iv->extern_draw_function =
                std::bind(&draw_image_overlay, std::placeholders::_1, idx);
        }

        Eigen::Vector3d cam_p(-0.5, -3, -5);
        cam_p = dyn_vio->getT_w_i_init().so3() * calib.T_i_c[0].so3() * cam_p;

        camera = pangolin::OpenGlRenderState(
            pangolin::ProjectionMatrix(640, 480, 400, 400, 320, 240, 0.001, 2000),
            pangolin::ModelViewLookAt(cam_p[0], cam_p[1], cam_p[2], 0, 0, 0,
                                      pangolin::AxisZ));

        pangolin::View& display3D =
            pangolin::CreateDisplay()
                .SetAspect(-640 / 480.0)
                .SetBounds(0.4, 1.0, 0.4, 1.0)
                .SetHandler(new pangolin::Handler3D(camera));

        display3D.extern_draw_function = draw_scene;

        main_display.AddDisplay(img_view_display);
        main_display.AddDisplay(display3D);

        while (!pangolin::ShouldQuit()) {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            if (follow) {
                size_t frame_id = show_frame;
                int64_t t_ns = vio_dataset->get_image_timestamps()[frame_id];
                auto it = vis_map.find(t_ns);

                if (it != vis_map.end()) {
                    Sophus::SE3d T_w_i;
                    if (!it->second->states.empty()) {
                        T_w_i = it->second->states.back();
                    } else if (!it->second->frames.empty()) {
                        T_w_i = it->second->frames.back();
                    }
                    T_w_i.so3() = Sophus::SO3d();

                    camera.Follow(T_w_i.matrix());
                }
            }

            display3D.Activate(camera);
            glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

            img_view_display.Activate();

            if (show_frame.GuiChanged()) {
                for (size_t cam_id = 0; cam_id < calib.intrinsics.size(); cam_id++) {
                    size_t frame_id = static_cast<size_t>(show_frame);
                    int64_t timestamp = vio_dataset->get_image_timestamps()[frame_id];

                    std::vector<basalt::ImageData> img_vec =
                        vio_dataset->get_image_data(timestamp);

                    pangolin::GlPixFormat fmt;
                    fmt.glformat = GL_LUMINANCE;
                    fmt.gltype = GL_UNSIGNED_SHORT;
                    fmt.scalable_internal_format = GL_LUMINANCE16;

                    if (img_vec[cam_id].img.get())
                        img_view[cam_id]->SetImage(
                            img_vec[cam_id].img->ptr, img_vec[cam_id].img->w,
                            img_vec[cam_id].img->h, img_vec[cam_id].img->pitch, fmt);
                }

                draw_plots();
            }

            if (show_est_vel.GuiChanged() || show_est_pos.GuiChanged() ||
                show_est_ba.GuiChanged() || show_est_bg.GuiChanged() ||
                show_est_param.GuiChanged()) {
                draw_plots();
            }

            if (euroc_fmt.GuiChanged()) {
                euroc_fmt = true;
                tum_rgbd_fmt = false;
                kitti_fmt = false;
            }

            if (tum_rgbd_fmt.GuiChanged()) {
                tum_rgbd_fmt = true;
                euroc_fmt = false;
                kitti_fmt = false;
            }

            if (kitti_fmt.GuiChanged()) {
                kitti_fmt = true;
                euroc_fmt = false;
                tum_rgbd_fmt = false;
            }

                  if (record) {
                    main_display.RecordOnRender(
                        "ffmpeg:[fps=50,bps=80000000,unique_filename]///tmp/"
                        "vio_screencap.avi");
                    record = false;
                  }

            pangolin::FinishFrame();

            if (continue_btn) {
                if (!next_step())
                    std::this_thread::sleep_for(std::chrono::milliseconds(66));
                std::this_thread::sleep_for(std::chrono::milliseconds(66));
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(66));
            }

            if (continue_fast) {
                int64_t t_ns = dyn_vio->last_processed_t_ns;
                if (timestamp_to_id.count(t_ns)) {
                    show_frame = timestamp_to_id[t_ns];
                    show_frame.Meta().gui_changed = true;
                }

                if (dyn_vio->finished) {
                    continue_fast = false;
                }
            }
        }
    }

    terminate = true;

    t1.join();
    t2.join();
    if (t_cmd.get()) t_cmd->join();
    if (t3.get()) t3->join();
    t4.join();
    if (t5.get()) t5->join();

    auto time_end = std::chrono::high_resolution_clock::now();

    if (!trajectory_fmt.empty()) {
        std::cout << "Saving trajectory..." << std::endl;

        if (trajectory_fmt == "kitti") {
            kitti_fmt = true;
            euroc_fmt = false;
            tum_rgbd_fmt = false;
        }
        if (trajectory_fmt == "euroc") {
            euroc_fmt = true;
            kitti_fmt = false;
            tum_rgbd_fmt = false;
        }
        if (trajectory_fmt == "tum") {
            tum_rgbd_fmt = true;
            euroc_fmt = false;
            kitti_fmt = false;
        }

        saveTrajectoryButton();
    }

    if (!result_path.empty()) {
        double error = basalt::alignSVD(vio_t_ns, vio_t_w_i, gt_t_ns, gt_t_w_i);

        auto exec_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            time_end - time_start);

        std::ofstream os(result_path);
        {
            cereal::JSONOutputArchive ar(os);
            ar(cereal::make_nvp("rms_ate", error));
            ar(cereal::make_nvp("num_frames",
                                vio_dataset->get_image_timestamps().size()));
            ar(cereal::make_nvp("exec_time_ns", exec_time_ns.count()));
        }
        os.close();
    }

    return 0;
}

void draw_image_overlay(pangolin::View& v, size_t cam_id) {
    UNUSED(v);

    //  size_t frame_id = show_frame;
    //  basalt::TimeCamId tcid =
    //      std::make_pair(vio_dataset->get_image_timestamps()[frame_id],
    //      cam_id);

    size_t frame_id = show_frame;
    auto it = vis_map.find(vio_dataset->get_image_timestamps()[frame_id]);

    if (show_obs) {
        glLineWidth(1.0);
        glColor3f(1.0, 0.0, 0.0);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        if (it != vis_map.end() && cam_id < it->second->projections.size()) {
            const auto& points = it->second->projections[cam_id];

            if (points.size() > 0) {
                double min_id = points[0][2], max_id = points[0][2];

                for (const auto& points2 : it->second->projections)
                    for (const auto& p : points2) {
                        min_id = std::min(min_id, p[2]);
                        max_id = std::max(max_id, p[2]);
                    }

                for (const auto& c : points) {
                    const float radius = 6.5;

                    float r, g, b;
                    getcolor(c[2] - min_id, max_id - min_id, b, g, r);
                    glColor3f(r, g, b);

                    pangolin::glDrawCirclePerimeter(c[0], c[1], radius);

                    if (show_ids)
                        pangolin::GlFont::I().Text("%d", int(c[3])).Draw(c[0], c[1]);
                }
            }

            glColor3f(1.0, 0.0, 0.0);
            pangolin::GlFont::I()
                .Text("Tracked %d points", points.size())
                .Draw(5, 20);
        }
    }

    if (show_flow) {
        glLineWidth(1.0);
        glColor3f(1.0, 0.0, 0.0);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        if (it != vis_map.end()) {
            const Eigen::aligned_map<basalt::KeypointId, Eigen::AffineCompact2f>&
                kp_map = it->second->opt_flow_res->observations[cam_id];

            for (const auto& kv : kp_map) {
                Eigen::MatrixXf transformed_patch =
                    kv.second.linear() * opt_flow_ptr->patch_coord;
                transformed_patch.colwise() += kv.second.translation();

                for (int i = 0; i < transformed_patch.cols(); i++) {
                    const Eigen::Vector2f c = transformed_patch.col(i);
                    pangolin::glDrawCirclePerimeter(c[0], c[1], 0.5f);
                }

                const Eigen::Vector2f c = kv.second.translation();

                if (show_ids)
                    pangolin::GlFont::I().Text("%d", kv.first).Draw(5 + c[0], 5 + c[1]);
            }

            pangolin::GlFont::I()
                .Text("%d opt_flow patches", kp_map.size())
                .Draw(5, 20);
        }
    }
}

void draw_scene(pangolin::View& view) {
    UNUSED(view);
    view.Activate(camera);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

    glPointSize(3);
    glColor3f(1.0, 0.0, 0.0);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    {
//    glColor3ubv(cam_color);
        glLineWidth(2.0);
        glColor3ubv(vio_color);
        if (!vio_t_w_i.empty()) {
            BASALT_ASSERT_STREAM((int) vio_t_w_i.size() >= show_frame,
                                 "vio_t_w_i size: " << vio_t_w_i.size() << " show frame num: " << show_frame);
            Eigen::aligned_vector<Eigen::Vector3d> sub_vio(
                    vio_t_w_i.begin(), vio_t_w_i.begin() + show_frame);
            pangolin::glDrawLineStrip(sub_vio);
        }
    }

    glColor3ubv(gt_color);
    if (show_gt) pangolin::glDrawLineStrip(gt_t_w_i);

    size_t frame_id = show_frame;
    int64_t t_ns = vio_dataset->get_image_timestamps()[frame_id];
    auto it = vis_map.find(t_ns);

    if (it != vis_map.end()) {
        for (size_t i = 0; i < calib.T_i_c.size(); i++)
            if (!it->second->states.empty()) {
                render_camera((it->second->states.back() * calib.T_i_c[i]).matrix(),
                              2.0f, cam_color, 0.1f);
            } else if (!it->second->frames.empty()) {
                render_camera((it->second->frames.back() * calib.T_i_c[i]).matrix(),
                              2.0f, cam_color, 0.1f);
            }

        for (const auto& p : it->second->states)
            for (size_t i = 0; i < calib.T_i_c.size(); i++)
                render_camera((p * calib.T_i_c[i]).matrix(), 2.0f, state_color, 0.1f);

        for (const auto& p : it->second->frames)
            for (size_t i = 0; i < calib.T_i_c.size(); i++)
                render_camera((p * calib.T_i_c[i]).matrix(), 2.0f, pose_color, 0.1f);

        glColor3ubv(pose_color);
        pangolin::glDrawPoints(it->second->points);
    }

    pangolin::glDrawAxis(Sophus::SE3d().matrix(), 1.0);
}

void load_data(const std::string& calib_path) {
    std::ifstream os(calib_path, std::ios::binary);

    if (os.is_open()) {
        cereal::JSONInputArchive archive(os);
        archive(calib);
        std::cout << "Loaded camera with " << calib.intrinsics.size() << " cameras"
                  << std::endl;

    } else {
        std::cerr << "could not load camera calibration " << calib_path
                  << std::endl;
        std::abort();
    }
}

bool next_step() {
    if (show_frame < int(vio_dataset->get_image_timestamps().size()) - 1) {
        show_frame = show_frame + 1;
        show_frame.Meta().gui_changed = true;
        cv.notify_one();
        return true;
    } else {
        return false;
    }
}

bool prev_step() {
    if (show_frame > 1) {
        show_frame = show_frame - 1;
        show_frame.Meta().gui_changed = true;
        return true;
    } else {
        return false;
    }
}

void draw_plots() {
    plotter->ClearSeries();
    plotter->ClearMarkers();

    if (show_est_pos) {
        plotter->AddSeries("$0", "$4", pangolin::DrawingModeLine,
                           pangolin::Colour::Red(), "position x", &vio_data_log);
        plotter->AddSeries("$0", "$5", pangolin::DrawingModeLine,
                           pangolin::Colour::Green(), "position y", &vio_data_log);
        plotter->AddSeries("$0", "$6", pangolin::DrawingModeLine,
                           pangolin::Colour::Blue(), "position z", &vio_data_log);
    }

    if (show_est_vel) {
        plotter->AddSeries("$0", "$1", pangolin::DrawingModeLine,
                           pangolin::Colour::Red(), "velocity x", &vio_data_log);
        plotter->AddSeries("$0", "$2", pangolin::DrawingModeLine,
                           pangolin::Colour::Green(), "velocity y", &vio_data_log);
        plotter->AddSeries("$0", "$3", pangolin::DrawingModeLine,
                           pangolin::Colour::Blue(), "velocity z", &vio_data_log);
    }

    if (show_est_bg) {
        plotter->AddSeries("$0", "$7", pangolin::DrawingModeLine,
                           pangolin::Colour::Red(), "gyro bias x", &vio_data_log);
        plotter->AddSeries("$0", "$8", pangolin::DrawingModeLine,
                           pangolin::Colour::Green(), "gyro bias y", &vio_data_log);
        plotter->AddSeries("$0", "$9", pangolin::DrawingModeLine,
                           pangolin::Colour::Blue(), "gyro bias z", &vio_data_log);
    }

    if (show_est_ba) {
        plotter->AddSeries("$0", "$10", pangolin::DrawingModeLine,
                           pangolin::Colour::Red(), "accel bias x", &vio_data_log);
        plotter->AddSeries("$0", "$11", pangolin::DrawingModeLine,
                           pangolin::Colour::Green(), "accel bias y",
                           &vio_data_log);
        plotter->AddSeries("$0", "$12", pangolin::DrawingModeLine,
                           pangolin::Colour::Blue(), "accel bias z", &vio_data_log);
    }

    double t = vio_dataset->get_image_timestamps()[show_frame] * 1e-9;
    plotter->AddMarker(pangolin::Marker::Vertical, t, pangolin::Marker::Equal,
                       pangolin::Colour::White());
}

void alignButton() { basalt::alignSVD(vio_t_ns, vio_t_w_i, gt_t_ns, gt_t_w_i); }

void saveTrajectoryButton() {
    if(!dynvio::fs::is_directory(dynvio::fs::status(output_dir))){
        dynvio::fs::create_directory(output_dir);
    }
    if (tum_rgbd_fmt) {
        std::string trajectory_txt = output_dir + "/stamped_traj_estimate.txt";
        std::ofstream os(trajectory_txt);

        os << "# timestamp tx ty tz qx qy qz qw" << std::endl;

        for (size_t i = 0; i < vio_t_ns.size(); i++) {
            const Sophus::SE3d& pose = vio_T_w_i[i];
            os << std::scientific << std::setprecision(18) << vio_t_ns[i] * 1e-9
               << " " << pose.translation().x() << " " << pose.translation().y()
               << " " << pose.translation().z() << " " << pose.unit_quaternion().x()
               << " " << pose.unit_quaternion().y() << " "
               << pose.unit_quaternion().z() << " " << pose.unit_quaternion().w()
               << std::endl;
        }

        os.close();

        std::cout
            << "Saved trajectory in TUM RGB-D Dataset format in stamped_traj_estimate.txt"
            << std::endl;

        // save ba var
        if(save_ba_var && !ba_var_map.empty() && !vio_ba.empty()){
            std::string ba_var_txt = output_dir + "/ba_var.txt";
            std::ofstream ba_os(ba_var_txt);
            ba_os << "# timestamp bax bax_var bay bay_var baz baz_var" << std::endl;

            for(int i = 0; i < vio_t_ns.size(); i++){
                if(vio_t_ns[i] >= ba_var_map.begin()->first && vio_t_ns[i]<= std::prev(ba_var_map.end(), 1)->first){
                    ba_os << std::scientific << std::setprecision(18) << vio_t_ns[i] * 1e-9 << " "
                    << vio_ba[i].x() << " " << ba_var_map.at(vio_t_ns[i]).x() << " "
                    << vio_ba[i].y() << " " << ba_var_map.at(vio_t_ns[i]).y() << " "
                    << vio_ba[i].z() << " " << ba_var_map.at(vio_t_ns[i]).z() << std::endl;
                }
            }
            ba_os.close();
        }

        // save gt
        if (save_gt && !gt_T_w_i.empty()) {
            std::string gt_trajectory_txt = output_dir + "/stamped_groundtruth.txt";
            std::ofstream gt_os(gt_trajectory_txt);
            gt_os << "# timestamp tx ty tz qx qy qz qw" << std::endl;

            for (size_t i = 0; i < gt_T_w_i.size(); i++) {
                int64_t t_ns = gt_t_ns[i];
                const Sophus::SE3d& pose = gt_T_w_i[i];
                gt_os << std::scientific << std::setprecision(18) << t_ns * 1e-9 << " "
                      << pose.translation().x() << " " << pose.translation().y() << " "
                      << pose.translation().z() << " " << pose.unit_quaternion().x() << " "
                      << pose.unit_quaternion().y() << " " << pose.unit_quaternion().z()
                      << " " << pose.unit_quaternion().w() << std::endl;
            }
            gt_os.close();
        }

        // save extr
        if(!vio_T_o_i.empty()){
            std::string extr_txt = output_dir + "/extrinsic_poses.txt";
            std::ofstream extr_os(extr_txt);

            extr_os << "# timestamp tx ty tz qx qy qz qw" << std::endl;

            for(size_t i = 0; i < vio_T_o_i.size(); i++) {
                const Sophus::SE3d& pose = vio_T_o_i[i];
                extr_os << std::scientific << std::setprecision(18) << vio_dyn_t_ns[i] * 1e-9
                   << " " << pose.translation().x() << " " << pose.translation().y()
                   << " " << pose.translation().z() << " " << pose.unit_quaternion().x()
                   << " " << pose.unit_quaternion().y() << " "
                   << pose.unit_quaternion().z() << " " << pose.unit_quaternion().w()
                   << std::endl;
            }
            extr_os.close();
        }

        // save param
        if(!param_map.empty()){
            BASALT_ASSERT(param_map.size() == vio_T_o_i.size());
            std::string param_txt = output_dir + "/param.txt";
            std::ofstream param_os(param_txt);

            param_os << "# timestamp c_lat c_lat_var steer_rt steer_rt_var thr_1 thr_1_var thr_2 thr_2_var thr_res thr_res_var" << std::endl;
            for(const auto& param_state : param_map){
                const auto& param = param_state.second;
                param_os << std::scientific << std::setprecision(18) << param_state.first * 1e-9
                         << " " << param.c_lat << " " << param.var[0]
                         << " " << param.steering_rt << " " << param.var[1]
                         << " " << param.throttle_f1 << " " << param.var[2]
                         << " " << param.throttle_f2 << " " << param.var[3]
                         << " " << param.throttle_res << " " << param.var[4]
                        << std::endl;
            }

            param_os.close();
        }

        // save prediction
        if(save_pred){
            std::string pred_output_dir = output_dir +  "/pred";
            if(!dynvio::fs::is_directory(dynvio::fs::status(pred_output_dir))){
                dynvio::fs::create_directory(pred_output_dir);
            }

            std::vector<dynvio::Command> cmd_data_vec;
            feed_cmd(cmd_data_vec);
            BASALT_ASSERT(vio_dyn_t_ns.size() == vio_T_o_i.size());
            dynvio::SingleTrackForward sim(dynvio_config, vio_t_ns, cmd_data_vec, dyn_vio->GetODERef());
            // create raw param
            dynvio::SingleTrackParamOnline raw_param(dynvio_config.c_lat_init, dynvio_config.steering_rt_init,
                                                     dynvio_config.throttle_f1_init, dynvio_config.throttle_f2_init,
                                                     dynvio_config.throttle_res_init);

            size_t steps_array[5] = {10, 20, 50, 100, 300};

            for(const auto& steps :  steps_array) {
                std::string calib_file_name = "/calib_pred" + std::toString(steps) + ".txt";
                std::ofstream pred_calib_file(pred_output_dir + calib_file_name);
                pred_calib_file << "# timestart timeend tx ty tz qx qy qz qw" << std::endl;

                std::string raw_file_name = "/raw_pred" + std::toString(steps) + ".txt";
                std::ofstream pred_raw_file(pred_output_dir + raw_file_name);
                pred_raw_file << "# timestart timeend tx ty tz qx qy qz qw" << std::endl;

                for(size_t i = 0; i < vio_dyn_t_ns.size(); i++){
                    Eigen::aligned_map<int64_t, Sophus::SE3d> calib_poses, init_poses;

                    int vio_id = timestamp_to_id[vio_dyn_t_ns[i]];
                    if(vio_id + steps >= vio_t_ns.size() || vio_t_ns[vio_id + steps] > vio_dyn_t_ns.back())
                        break;

                    const Sophus::SE3d& T_w_i_start = vio_T_w_i[vio_id];
                    Eigen::Vector3d vel_i_start = T_w_i_start.so3().inverse() * vio_vel_w_i[vio_id];
                    const Eigen::Vector3d calib_gyro_start = vio_gyro_calib[vio_id];
                    const Sophus::SE3d& T_o_i_start = vio_T_o_i[i];

                    // calib
                    sim.UpdateParam(param_map[vio_dyn_t_ns[i]]);
                    Sophus::SE3d rel_pose = sim.ComputePose(vio_id, steps, T_w_i_start, T_o_i_start,
                                                            vel_i_start, calib_gyro_start, calib_poses);

                    // save calib
                    pred_calib_file << std::scientific << std::setprecision(18) << vio_dyn_t_ns[i] * 1e-9 << " " << vio_t_ns[vio_id + steps] * 1e-9
                                    << " " << rel_pose.translation().x() << " " << rel_pose.translation().y()
                                    << " " << rel_pose.translation().z() << " " << rel_pose.unit_quaternion().x()
                                    << " " << rel_pose.unit_quaternion().y() << " "
                                    << rel_pose.unit_quaternion().z() << " " << rel_pose.unit_quaternion().w()
                                    << std::endl;

                    // uncalib
                    sim.UpdateParam(raw_param);
                    Sophus::SE3d uncalib_rel_pose = sim.ComputePose(vio_id, steps, T_w_i_start, dynvio_config.T_o_i_init,
                                                                    vel_i_start, calib_gyro_start, init_poses);
                    // save uncalib
                    pred_raw_file << std::scientific << std::setprecision(18) << vio_dyn_t_ns[i] * 1e-9 << " " << vio_t_ns[vio_id + steps] * 1e-9
                                  << " " << uncalib_rel_pose.translation().x() << " " << uncalib_rel_pose.translation().y()
                                  << " " << uncalib_rel_pose.translation().z() << " " << uncalib_rel_pose.unit_quaternion().x()
                                  << " " << uncalib_rel_pose.unit_quaternion().y() << " "
                                  << uncalib_rel_pose.unit_quaternion().z() << " " << uncalib_rel_pose.unit_quaternion().w()
                                  << std::endl;
                }
                pred_calib_file.close();
                pred_raw_file.close();
            }
        }
    } else if (euroc_fmt) {
        std::ofstream os("trajectory.csv");

        os << "#timestamp [ns],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w "
              "[],q_RS_x [],q_RS_y [],q_RS_z []"
           << std::endl;

        for (size_t i = 0; i < vio_t_ns.size(); i++) {
            const Sophus::SE3d& pose = vio_T_w_i[i];
            os << std::scientific << std::setprecision(18) << vio_t_ns[i] << ","
               << pose.translation().x() << "," << pose.translation().y() << ","
               << pose.translation().z() << "," << pose.unit_quaternion().w() << ","
               << pose.unit_quaternion().x() << "," << pose.unit_quaternion().y()
               << "," << pose.unit_quaternion().z() << std::endl;
        }

        std::cout << "Saved trajectory in Euroc Dataset format in trajectory.csv"
                  << std::endl;
    } else {
        std::ofstream os("trajectory_kitti.txt");

        for (size_t i = 0; i < vio_t_ns.size(); i++) {
            Eigen::Matrix<double, 3, 4> mat = vio_T_w_i[i].matrix3x4();
            os << std::scientific << std::setprecision(12) << mat.row(0) << " "
               << mat.row(1) << " " << mat.row(2) << " " << std::endl;
        }

        os.close();

        std::cout
            << "Saved trajectory in KITTI Dataset format in trajectory_kitti.txt"
            << std::endl;
    }


} // main
