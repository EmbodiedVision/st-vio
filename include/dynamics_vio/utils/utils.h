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

#include <basalt/calibration/calibration.hpp>
#include <basalt/imu/imu_types.h>
#include <cereal/cereal.hpp>

namespace dynvio {
using namespace basalt;

struct DynVIOConfig{

    DynVIOConfig();
    void load(const std::string& filename);
    void save(const std::string& filename);

    size_t cmd_history_size;

    Sophus::SE3d T_o_i_init;
    double extr_init_weight;
    double param_init_weight;

    //crawler paramters
    double wheel_base;
    double mass;
    double I_z;

    double l_fw_imu;
    double l_c_imu;
    double l_front;
    Sophus::SE3d imu_forward;

    double c_lat_init;
    double steering_rt_init;
    double throttle_f1_init;
    double throttle_f2_init;
    double throttle_res_init;

    double dynamics_weight;
    double constraint_weight;
    double param_prior_weight;
    double extr_rd_weight;

    double ba_var_thr;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

constexpr size_t EXTR_SIZE = 6;

template <class Scalar_>
struct ExtrinsicState{
    using Ptr = std::shared_ptr<ExtrinsicState>;

    using VecN = Eigen::Matrix<Scalar_, EXTR_SIZE, 1>;
    using SE3 = Sophus::SE3<Scalar_>;

    ExtrinsicState():t_ns(0){}
    ExtrinsicState(int64_t t_ns, const SE3& T_o_i):t_ns(t_ns),T_o_i(T_o_i){}

    void applyInc(const VecN& inc){
        PoseState<Scalar_>::incPose(inc.template segment<6>(0), T_o_i);
    }

    VecN diff(const ExtrinsicState<Scalar_>& other){
        VecN res;
        res.template segment<3>(0) =
            other.T_o_i.translation() - this->T_o_i.translation();
        res.template segment<3>(3) =
            (other.T_o_i.so3() * this->T_o_i.so3().inverse()).log();

        return res;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int64_t t_ns;
    Sophus::SE3d T_o_i;
};

template <typename Scalar_>
struct ExtrinsicStateWithLin{

    using Scalar = Scalar_;
    using VecN = typename ExtrinsicState<Scalar>::VecN;
    using SE3 = typename ExtrinsicState<Scalar>::SE3;
    ExtrinsicStateWithLin() {
        linearized = false;
        delta.setZero();
    }

    ExtrinsicStateWithLin(int64_t t_ns, const SE3& T_o_i,bool linearized):
        linearized(linearized),state_linearized(t_ns, T_o_i){
        delta.setZero();
        state_current = state_linearized;
    }

    ExtrinsicStateWithLin(const ExtrinsicState<Scalar>& other)
        : linearized(false), state_linearized(other) {
        delta.setZero();
        state_current = other;
    }

    void setLinFalse() {
        linearized = false;
        delta.setZero();
    }

    void setLinTrue() {
        linearized = true;
        BASALT_ASSERT(delta.isApproxToConstant(0));
        state_current = state_linearized;
    }

    void applyInc(const VecN& inc) {
        if (!linearized) {
            state_linearized.applyInc(inc);
        } else {
            delta += inc;
            state_current = state_linearized;
            state_current.applyInc(delta);
        }
    }

    inline const ExtrinsicState<Scalar>& getState() const {
        if (!linearized) {
            return state_linearized;
        } else {
            return state_current;
        }
    }

    inline const ExtrinsicState<Scalar>& getStateLin() const {
        return state_linearized;
    }

    inline bool isLinearized() const { return linearized; }
    inline const VecN& getDelta() const { return delta; }
    inline int64_t getT_ns() const { return state_linearized.t_ns; }

    inline void backup() {
        backup_delta = delta;
        backup_state_linearized = state_linearized;
        backup_state_current = state_current;
    }

    inline void restore() {
        delta = backup_delta;
        state_linearized = backup_state_linearized;
        state_current = backup_state_current;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    bool linearized;
    VecN delta;
    ExtrinsicState<Scalar> state_linearized, state_current;

    VecN backup_delta;
    ExtrinsicState<Scalar> backup_state_linearized, backup_state_current;
    friend class cereal::access;

    template <class Archive>
    void serialize(Archive& ar) {
        ar(state_linearized.T_o_i);
        ar(state_current.T_o_i);
        ar(delta);
        ar(linearized);
        ar(state_linearized.t_ns);
    }
};


struct Command
{
    using Ptr = std::shared_ptr<Command>;

    int64_t t_ns;
    double linear{0}; // throttle
    double angular{0};// steering


    Command():t_ns(0),linear(0),angular(0){}
    Command(int64_t t_ns, double linear, double angular):t_ns(t_ns),linear(linear),angular(angular){}

    Command operator +(const Command& cmd) const {
        Command cmd_new;
        cmd_new.linear = this->linear + cmd.linear;
        cmd_new.angular = this->angular + cmd.angular;

        return cmd_new;
    }

    Command operator *(double f) const{
        Command cmd_new;
        cmd_new.linear = this->linear * f;
        cmd_new.angular = this->angular * f;
        return cmd_new;
    }

    Command operator /(double f) const{
        Command cmd_new;
        cmd_new.linear = this->linear / f;
        cmd_new.angular = this->angular / f;
        return cmd_new;
    }

    void setZero(){
        t_ns = 0;
        linear = 0;
        angular = 0;
    }
};
} // namespace dynvio

