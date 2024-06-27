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
#include <basalt/imu/imu_types.h>

namespace dynvio {

class ParameterBase{

public:
    using Ptr = std::shared_ptr<ParameterBase>;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    virtual void applyInc(const Eigen::VectorXd& inc) = 0;
    virtual ~ParameterBase(){}
    int64_t t_ns;
};

constexpr size_t PARAM_SIZE = 5;
struct SingleTrackParamOnline : public ParameterBase
{
    using Ptr = std::shared_ptr<SingleTrackParamOnline>;

    SingleTrackParamOnline(): c_lat(20), steering_rt(0.28), throttle_f1(10),
                              throttle_f2(10), throttle_res(10),
                              var{0, 0, 0, 0, 0,}{}
    SingleTrackParamOnline(double c_lat, double steering_rt,
                           double throttle_f1, double throttle_f2, double throttle_res):
                           c_lat(c_lat), steering_rt(steering_rt),
                           throttle_f1(throttle_f1),
                           throttle_f2(throttle_f2),
                           throttle_res(throttle_res),
                           var{0, 0, 0, 0, 0} {}

    void applyInc(const Eigen::VectorXd &inc);

    double c_lat;
    double steering_rt;
    double throttle_f1;
    double throttle_f2;
    double throttle_res;

    std::vector<double> var;
};

template<class Param>
struct ParameterWithLin
{
    using Ptr = std::shared_ptr<ParameterWithLin<Param>>;
    ParameterWithLin(){
        linearized = false;
        delta.setZero(PARAM_SIZE);
    }

    ParameterWithLin(double c_lat, double steering_rt,
                     double throttle_f1, double throttle_f2, double throttle_res,
                     bool linearized):param_linearized(c_lat, steering_rt,
                     throttle_f1, throttle_f2, throttle_res),
                     linearized(linearized){
        delta.setZero(PARAM_SIZE);
        param_current = param_linearized;
    }
    ParameterWithLin(const Param& other):param_linearized(other){
        linearized = false;
        delta.setZero(PARAM_SIZE);
        param_current = other;
    }

    void setLinTrue(){
        linearized = true;
        BASALT_ASSERT(delta.isApproxToConstant(0));
        param_current = param_linearized;
    }

    void applyInc(const Eigen::VectorXd& inc){
        if (!linearized) {
            param_linearized.applyInc(inc);
        } else {
            delta += inc;
            param_current = param_linearized;
            param_current.applyInc(delta);
        }
    }

    void backup(){
        backup_delta = delta;
        backup_param_linearized = param_linearized;
        backup_param_current = param_current;
    }

    void restore(){
        delta = backup_delta;
        param_linearized = backup_param_linearized;
        param_current = backup_param_current;
    }

    inline const Param& getParam() const {
        if (!linearized) {
            return param_linearized;
        } else {
            return param_current;
        }
    }

    inline Param& getParam() {
        if (!linearized) {
            return param_linearized;
        } else {
            return param_current;
        }
    }

    inline const Param& getParamLin() const {
        return param_linearized;
    }

    inline Param& getParamLin() {
        return param_linearized;
    }

    void setLinFalse(){
        linearized = false;
        delta.setZero(PARAM_SIZE);
    }
    inline bool isLinearized() const { return linearized; }
    inline const Eigen::VectorXd& getDelta() const { return delta; }

    int64_t param_indx{-1};
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    Param param_linearized;
    bool linearized;

    Param param_current;

    Param backup_param_current;
    Param backup_param_linearized;

    Eigen::VectorXd delta;
    Eigen::VectorXd backup_delta;
};
} // namespace dynvio
