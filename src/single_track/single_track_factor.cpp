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
#include "dynamics_vio/utils/se2_utils.h"

#include <chrono>
namespace dynvio{

SingleTrackModel::SingleTrackModel(int64_t start_t_ns, const Eigen::Vector3d& start_gyro,
                                   int64_t end_t_ns, const Eigen::Vector3d& end_gyro,
                                   const std::deque<Command> &command_history):
    DynamicsBase(start_t_ns, end_t_ns),start_gyro(start_gyro),end_gyro(end_gyro),
    command_window(command_history){}

SingleTrackModel::Vec6 SingleTrackModel::residual(const SingleTrackAcado& singletrack_acado,
                                           const PoseVelBiasState<double>& state0,
                                           const ExtrinsicState<double> &extr_state0,
                                           const PoseVelBiasState<double> &state1,
                                           const ExtrinsicState<double> &extr_state1,
                                           const SingleTrackParamOnline &param_state,
                                           Eigen::Vector3d* pred_pose_state,
                                           Eigen::Vector3d* pred_vel_state,
                                           Mat66 *d_pred_d_state0,
                                           Mat66 *d_pred_d_extr0,
                                           Mat66 *d_res_d_state0,
                                           Mat63 *d_res_d_bias0,
                                           Mat66 *d_res_d_extr0,
                                           Mat56 *d_res_d_state1,
                                           Eigen::Matrix3d *d_res_d_bias1,
                                           Mat66 *d_res_d_extr1,
                                           Eigen::MatrixXd* d_pred_d_param) const{

    Eigen::Vector3d start_gyro_calib = start_gyro - state0.bias_gyro;
    Eigen::Vector3d end_gyro_calib = end_gyro - state1.bias_gyro;

    const Sophus::SE3d& T_w_i0 = state0.T_w_i;
    const Sophus::SE3d& T_w_i1 = state1.T_w_i;
    Sophus::SO3d R_o_i0 = extr_state0.T_o_i.so3();
    Eigen::Matrix3d t_o_i0_hat = Sophus::SO3d::hat(extr_state0.T_o_i.translation());
    Sophus::SO3d R_o0_w = R_o_i0 * T_w_i0.so3().inverse();
    Sophus::SO3d R_o_i1 = extr_state1.T_o_i.so3();
    Eigen::Matrix3d t_o_i1_hat = Sophus::SO3d::hat(extr_state1.T_o_i.translation());
    Sophus::SO3d R_o1_w = R_o_i1 * T_w_i1.so3().inverse();

    Sophus::SO3d R_o0_o1 = R_o0_w * R_o1_w.inverse();
    Eigen::Vector3d r_o0_o1 = R_o0_o1.log();
    Eigen::Vector3d trans_tmp = -(R_o0_o1 * extr_state1.T_o_i.translation());
    Eigen::Vector3d trans_tmp2 = R_o0_w * (T_w_i1.translation() - T_w_i0.translation()) + trans_tmp;
    Eigen::Vector3d t_o0_o1 = trans_tmp2 + extr_state0.T_o_i.translation();

    // command before t0
    Command prev_cmd = command_window[0];
    // check if there is another command in command window
    double frame_dt;
    if (command_window.size() == 1) {
        frame_dt = (state1.t_ns - start_t_ns) * 1e-9;
    } else{
        frame_dt = (command_window[1].t_ns - start_t_ns) * 1e-9;
    }

    //initial body twist in cg frame
    Eigen::Vector3d cg_init_ang = R_o_i0 * start_gyro_calib;
    Eigen::Vector3d cg_init_tmp = R_o0_w * state0.vel_w_i;
    Eigen::Vector3d cg_init_lin = cg_init_tmp + t_o_i0_hat * cg_init_ang; //notice state.vel_lin is NOT linear twist! vel_w = R_w_i * twist_lin_i;

    //end body twist in cg frame
    Eigen::Vector3d cg_end_ang = R_o_i1 * end_gyro_calib;
    Eigen::Vector3d cg_end_tmp = R_o1_w * state1.vel_w_i;
    Eigen::Vector3d cg_end_lin = cg_end_tmp + t_o_i1_hat * cg_end_ang;


    double init_state[6] = {0., 0., 0., cg_init_lin(0), cg_init_lin(1), cg_init_ang(2)};
    double param_list[PARAM_SIZE] = {param_state.c_lat, param_state.steering_rt,
                                     param_state.throttle_f1, param_state.throttle_f2, param_state.throttle_res};
    double control_input[2] = {prev_cmd.linear, prev_cmd.angular};

    ACADO::IntegratorRK45 integrator( singletrack_acado.f );
    ACADO::DVector end_states;

    integrator.set(ACADO::INTEGRATOR_PRINTLEVEL, ACADO::NONE);
    integrator.freezeAll();
    integrator.integrate(0.0, frame_dt, init_state, nullptr, param_list, control_input);
    integrator.getX(end_states);

    // predicted pose
    if(pred_pose_state){
        (*pred_pose_state) = end_states.head<3>();
    }
    // predicted vel
    if(pred_vel_state){
        (*pred_vel_state) = end_states.tail<3>();
    }
    Vec6 residuals;
    if(command_window.size() == 1){
        residuals.head<2>() = end_states.head<2>() - t_o0_o1.head<2>();
        residuals(2) = end_states(2) - r_o0_o1(2);
        residuals(3) = end_states(3) - cg_end_lin(0); //vx
        residuals(4) = end_states(4) - cg_end_lin(1); //vy
        residuals(5) = end_states(5)- cg_end_ang(2); //omega
    }

    //jacobian
    if (d_pred_d_state0 || d_res_d_state0 || d_res_d_state1 || d_pred_d_param || d_pred_d_extr0 || d_res_d_extr0 || d_res_d_extr1){

        // Jacobian of res wrt. compensated states
        Eigen::Matrix<double, 2, 3> tmp = Sophus::SO3d::hat(trans_tmp).topRows<2>();    // B
        Eigen::Matrix<double, 2, 3> tmp2 = Sophus::SO3d::hat(trans_tmp2).topRows<2>();  // A
        Eigen::Matrix3d J;
        Sophus::leftJacobianInvSO3(r_o0_o1, J);
        Eigen::Matrix<double, 1, 3> J_yaw = J.row(2);
        Eigen::Matrix3d cg_init_tmp_hat = Sophus::SO3d::hat(cg_init_tmp);
        Eigen::Matrix<double, 2, 3> cg_end_tmp_hat = Sophus::SO3d::hat(cg_end_tmp).topRows<2>();
        Eigen::Matrix3d R_o0_w_mat = R_o0_w.matrix();
        Eigen::Matrix<double, 1, 3> J_yaw_state = J_yaw * R_o0_w_mat;

        Eigen::Matrix<double, 6, 3> J_init;
        //derivative of residual tran_x, tran_y yaw wrt. parameters
        if(d_pred_d_param){
            d_pred_d_param->setZero();
            ACADO::DVector seed(6);
            // TODO: parallize
            for(int i = 0; i < 6; i++){
                seed.setZero();
                seed(i) = 1.0;

                ACADO::DVector D_p(PARAM_SIZE);
                ACADO::DVector D_state_init(6);
                integrator.setBackwardSeed(1, seed);
                integrator.integrateSensitivities();
                integrator.getBackwardSensitivities(D_state_init,D_p,ACADO::emptyVector,ACADO::emptyVector,1); // w.r.t. x0,p,u,w
                J_init.row(i) = D_state_init.tail<3>();

                d_pred_d_param->block<1, PARAM_SIZE>(i, 0) = D_p;
            }
        }

        if(d_pred_d_extr0){
            d_pred_d_extr0->setZero();
            Eigen::Matrix3d cg_init_ang_hat = Sophus::SO3d::hat(cg_init_ang);
            Eigen::Matrix<double, 2, 3> J_linvel_extrtrans = -cg_init_ang_hat.topRows(2);
            Eigen::Matrix<double, 2, 3> J_linvel_extrrot = -cg_init_tmp_hat.topRows(2) - t_o_i0_hat.topRows(2) * cg_init_ang_hat;
            Eigen::Matrix<double, 1, 3> J_angvel_extrrot = -cg_init_ang_hat.row(2);
            // J acado pose, vel wrt. extr0_trans
            d_pred_d_extr0->block<6, 3>(0, 0) = J_init.block<6, 2>(0, 0) * J_linvel_extrtrans;
            // J acado pose, vel wrt. extr0_rot
            d_pred_d_extr0->block<6, 3>(0, 3) = J_init.block<6, 2>(0, 0) * J_linvel_extrrot + J_init.block<6, 1>(0, 2) * J_angvel_extrrot;

            if(d_res_d_extr0){
                (*d_res_d_extr0) = (*d_pred_d_extr0);
                d_res_d_extr0->block<2, 3>(0, 0) -= Eigen::Matrix3d::Identity().topRows(2);
                d_res_d_extr0->block<2, 3>(0, 3) += tmp2;
                d_res_d_extr0->block<1, 3>(2, 3) -= J_yaw;
            }
        }
        if(d_res_d_extr1){
            // Jacobain of end velocity wrt. end extr
            d_res_d_extr1->setZero();
            Eigen::Matrix3d R_o0_o1_mat = R_o0_o1.matrix();
            // J vio pose wrt. extr1
            d_res_d_extr1->block<2, 3>(0, 0) = R_o0_o1_mat.topRows<2>();
            d_res_d_extr1->block<2, 3>(0, 3) = -tmp * R_o0_o1_mat;
            d_res_d_extr1->block<1, 3>(2, 3) = J_yaw * R_o0_o1_mat;
            // J vio vel wrt. extr1
            Eigen::Matrix3d cg_end_ang_hat = Sophus::SO3d::hat(cg_end_ang);
            d_res_d_extr1->block<2, 3>(3, 0) = cg_end_ang_hat.topRows<2>();
            d_res_d_extr1->block<2, 3>(3, 3) = cg_end_tmp_hat + t_o_i1_hat.topRows(2) * cg_end_ang_hat;
            d_res_d_extr1->block<1, 3>(5, 3) = cg_end_ang_hat.row(2);
        }

        if(d_pred_d_state0){
            // Jacobian of linear velocity wrt. rotation and vel.
            d_pred_d_state0->setZero();
            // J acado pose, vel wrt. state0_rot
            d_pred_d_state0->block<6, 3>(0, 0) = J_init.block<6, 2>(0, 0) * cg_init_tmp_hat.topRows<2>() * R_o0_w_mat;
            // J acado pose, vel wrt. state0_vel
            d_pred_d_state0->block<6, 3>(0, 3) = J_init.block<6, 2>(0, 0) * R_o0_w_mat.topRows<2>();
            if(d_res_d_state0){
                (*d_res_d_state0) = (*d_pred_d_state0);
                d_res_d_state0->block<2, 3>(0, 0) += R_o0_w_mat.topRows<2>();
                d_res_d_state0->block<2, 3>(0, 3) -= tmp2 * R_o0_w_mat;
                d_res_d_state0->block<1, 3>(2, 3) += J_yaw_state;
            }
        }

        if(d_res_d_bias0){
            // d_res_d_bias0 is d_pred_d_bias0
            d_res_d_bias0->setZero();
            Eigen::Matrix3d R_o_i0_mat = R_o_i0.matrix();
            Eigen::Matrix<double, 2, 3> J_linvel_bias = -t_o_i0_hat.topRows(2) * R_o_i0_mat;
            Eigen::Matrix<double, 1, 3> J_angvel_bias = -R_o_i0_mat.row(2);
            (*d_res_d_bias0) = J_init.block<6, 2>(0, 0) * J_linvel_bias + J_init.block<6, 1>(0, 2) * J_angvel_bias;

        }

        if(d_res_d_state1){
            d_res_d_state1->setZero();
            // pose res
            d_res_d_state1->block<2, 3>(0, 0) = -R_o0_w_mat.topRows<2>();
            d_res_d_state1->block<2, 3>(0, 3) = tmp * R_o0_w_mat;
            d_res_d_state1->block<1, 3>(2, 3) = -J_yaw_state;

            Eigen::Matrix3d R_o1_w_mat = R_o1_w.matrix();
            // vel res
            d_res_d_state1->block<2, 3>(3, 0) = -cg_end_tmp_hat* R_o1_w_mat;
            d_res_d_state1->block<2, 3>(3, 3) = -R_o1_w_mat.topRows<2>();
        }

        if(d_res_d_bias1){
            d_res_d_bias1->setZero();
            Eigen::Matrix3d R_o_i1_mat = R_o_i1.matrix();
            // vel residual
            d_res_d_bias1->block<2, 3>(0, 0) = t_o_i1_hat.topRows<2>() * R_o_i1_mat;
            d_res_d_bias1->row(2) = R_o_i1_mat.row(2);
        }
    } // jacobians
    // command appears in the middle of the frames
    if(command_window.size() > 1){
        residuals = residual_cmdstep(singletrack_acado,
                                     param_state, r_o0_o1, t_o0_o1, cg_end_lin, cg_end_ang,
                                     pred_pose_state, pred_vel_state,
                                     d_pred_d_state0, d_pred_d_extr0, d_res_d_state0, d_res_d_bias0,
                                     d_res_d_extr0, d_pred_d_param);

    }

    return residuals;
}

SingleTrackModel::Vec6 SingleTrackModel::residual(const SingleTrackAcado& singletrack_acado, const int64_t& curr_t_ns,
                                           const PoseVelBiasState<double>& state0,
                                           const ExtrinsicState<double> &extr_state0,
                                           const PoseVelBiasState<double> &state1,
                                           const ExtrinsicState<double> &extr_state1,
                                           const SingleTrackParamOnline &param_state,
                                           Eigen::Vector3d* pred_pose_state,
                                           Eigen::Vector3d* pred_vel_state,
                                           Mat66 *d_pred_d_stateinit,
                                           Mat66 *d_pred_d_extrinit,
                                           Mat66 *d_res_d_stateinit,
                                           Mat63 *d_res_d_biasinit,
                                           Mat66 *d_res_d_extrinit,
                                           Mat56 *d_res_d_state1,
                                           Eigen::Matrix3d *d_res_d_bias1,
                                           Mat66* d_res_d_extr1,
                                           Eigen::MatrixXd *d_pred_d_param) const {
    Eigen::Vector3d end_gyro_calib = end_gyro - state1.bias_gyro;

    //BASALT_ASSERT(curr_t_ns == start_t_ns);
    //BASALT_ASSERT(end_t_ns == state1.t_ns);

    const Sophus::SE3d& T_w_i1 = state1.T_w_i;
    Sophus::SO3d R_o_i1 = extr_state1.T_o_i.so3();
    Eigen::Matrix3d t_o_i1_hat = Sophus::SO3d::hat(extr_state1.T_o_i.translation());
    Sophus::SO3d R_o1_w = R_o_i1 * T_w_i1.so3().inverse();

    const Sophus::SE3d& T_w_i0 = state0.T_w_i;
    Sophus::SO3d R_o0_w = extr_state0.T_o_i.so3() * T_w_i0.so3().inverse();
    Sophus::SO3d R_o0_o1 = R_o0_w * R_o1_w.inverse();
    Eigen::Vector3d r_o0_o1 = R_o0_o1.log();
    Eigen::Vector3d trans_tmp = -(R_o0_o1 * extr_state1.T_o_i.translation());
    Eigen::Vector3d trans_tmp2 = R_o0_w * (T_w_i1.translation() - T_w_i0.translation()) + trans_tmp;
    Eigen::Vector3d t_o0_o1 = trans_tmp2 + extr_state0.T_o_i.translation();

    Command prev_cmd = command_window[0];

    // check if there is another command in command window
    double frame_dt;
    if (command_window.size() == 1) {
        frame_dt = (state1.t_ns - curr_t_ns) * 1e-9;
    } else{
        frame_dt = (command_window[1].t_ns - curr_t_ns) * 1e-9;
    }

    //end body twist in cg frame
    Eigen::Vector3d cg_end_ang = R_o_i1 * end_gyro_calib;
    Eigen::Vector3d cg_end_tmp = R_o1_w * state1.vel_w_i;
    Eigen::Vector3d cg_end_lin = cg_end_tmp + t_o_i1_hat * cg_end_ang;

    double init_state[6] = {0., 0., 0., (*pred_vel_state)(0), (*pred_vel_state)(1), (*pred_vel_state)(2)};
    double param_list[PARAM_SIZE] = {param_state.c_lat, param_state.steering_rt,
                                                                 param_state.throttle_f1, param_state.throttle_f2,
                                                                 param_state.throttle_res};
    double control_input[2] = {prev_cmd.linear, prev_cmd.angular};

    ACADO::IntegratorRK45 integrator( singletrack_acado.f );
    ACADO::DVector end_states;

    integrator.set(ACADO::INTEGRATOR_PRINTLEVEL, ACADO::NONE);
    integrator.freezeAll();
    integrator.integrate(0.0, frame_dt, init_state, nullptr, param_list, control_input);
    integrator.getX(end_states);

    // predicted pose
    Sophus::SO2d rot_last((*pred_pose_state)(2));
    Eigen::Vector2d se2_rot_p = rot_last * end_states.head<2>();

    pred_pose_state->head<2>() += se2_rot_p;
    (*pred_pose_state)(2) += end_states(2);

    // predicted vel
    (*pred_vel_state) = end_states.tail<3>();


    Vec6 residuals;

    if(command_window.size() == 1){
        residuals.head<2>() = pred_pose_state->head<2>() - t_o0_o1.head<2>();
        residuals(2) = (*pred_pose_state)(2) - r_o0_o1(2);
        residuals(3) = end_states(3) - cg_end_lin(0); //vx
        residuals(4) = end_states(4) - cg_end_lin(1); //vy
        residuals(5) = end_states(5)- cg_end_ang(2); //omega
    }


    //jacobian
    if (d_pred_d_stateinit || d_pred_d_extrinit || d_res_d_stateinit || d_res_d_extrinit || d_res_d_state1 || d_pred_d_param || d_res_d_extr1){

        Eigen::Matrix<double, 2, 3> tmp = Sophus::SO3d::hat(trans_tmp).topRows<2>();    // B
        Eigen::Matrix<double, 2, 3> tmp2 = Sophus::SO3d::hat(trans_tmp2).topRows<2>();  // A
        Eigen::Matrix3d J;
        Sophus::leftJacobianInvSO3(r_o0_o1, J);
        Eigen::Matrix<double, 1, 3> J_yaw = J.row(2);
        Eigen::Matrix<double, 2, 3> cg_end_tmp_hat = Sophus::SO3d::hat(cg_end_tmp).topRows<2>();
        Eigen::Matrix3d R_o0_w_mat = R_o0_w.matrix();
        Eigen::Matrix<double, 1, 3> J_yaw_state = J_yaw * R_o0_w_mat;

        Eigen::Matrix2d J_predtrans_acadotrans = rot_last.matrix();
        Eigen::Vector2d J_predtrans_lastrot(-se2_rot_p(1), se2_rot_p(0));

        Eigen::Matrix<double, 6, 3> J_init;
        //derivative of residual tran_x, tran_y yaw wrt. parameters
        if(d_pred_d_param){
            Eigen::MatrixXd d_pred_d_param_last = *d_pred_d_param; // copy of last J_pred_param
            d_pred_d_param->setZero();
            ACADO::DVector seed(6);
            for(int i = 0; i < 6; i++){
                seed.setZero();
                seed(i) = 1.0;

                ACADO::DVector D_p(PARAM_SIZE);
                ACADO::DVector D_state_init(6);
                integrator.setBackwardSeed(1, seed);
                integrator.integrateSensitivities();
                integrator.getBackwardSensitivities(D_state_init,D_p,ACADO::emptyVector,ACADO::emptyVector,1); // w.r.t. x0,p,u,w
                J_init.row(i) = D_state_init.tail<3>();

                d_pred_d_param->block<1, PARAM_SIZE>(i, 0) = D_p;
            }
            (*d_pred_d_param) += J_init * d_pred_d_param_last.bottomRows(3);

            // d_pred_d_param = d_res_d_param
            d_pred_d_param->topRows(2) = J_predtrans_acadotrans * d_pred_d_param->topRows(2);
            d_pred_d_param->topRows(2) += J_predtrans_lastrot * d_pred_d_param_last.row(2);
            d_pred_d_param->topRows(3) += d_pred_d_param_last.topRows(3);
        }

        if(d_pred_d_extrinit){
            if(d_res_d_extrinit){
                d_res_d_extrinit->setZero();
                (*d_res_d_extrinit) = J_init * d_pred_d_extrinit->bottomRows<3>();
                d_res_d_extrinit->topRows<2>() = J_predtrans_acadotrans * d_res_d_extrinit->topRows<2>();
                d_res_d_extrinit->topRows<2>() += J_predtrans_lastrot * d_pred_d_extrinit->row(2);
                d_res_d_extrinit->topRows<3>() += d_pred_d_extrinit->topRows<3>();
                (*d_pred_d_extrinit) = (*d_res_d_extrinit);

                d_res_d_extrinit->block<2, 3>(0, 0) -= Eigen::Matrix3d::Identity().topRows(2);
                d_res_d_extrinit->block<2, 3>(0, 3) += tmp2;
                d_res_d_extrinit->block<1, 3>(2, 3) -= J_yaw;
            }
        }

        if(d_res_d_extr1){
            // Jacobain of end velocity wrt. end extr
            d_res_d_extr1->setZero();
            Eigen::Matrix3d R_o0_o1_mat = R_o0_o1.matrix();
            // J vio pose wrt. extr1
            d_res_d_extr1->block<2, 3>(0, 0) = R_o0_o1_mat.topRows<2>();
            d_res_d_extr1->block<2, 3>(0, 3) = -tmp * R_o0_o1_mat;
            d_res_d_extr1->block<1, 3>(2, 3) = J_yaw * R_o0_o1_mat;
            // J vio vel wrt. extr1
            Eigen::Matrix3d cg_end_ang_hat = Sophus::SO3d::hat(cg_end_ang);
            d_res_d_extr1->block<2, 3>(3, 0) = cg_end_ang_hat.topRows<2>();
            d_res_d_extr1->block<2, 3>(3, 3) = cg_end_tmp_hat + t_o_i1_hat.topRows(2) * cg_end_ang_hat;
            d_res_d_extr1->block<1, 3>(5, 3) = cg_end_ang_hat.row(2);
        }

        if(d_pred_d_stateinit){
            if(d_res_d_stateinit){
                d_res_d_stateinit->setZero();
                (*d_res_d_stateinit) = J_init * d_pred_d_stateinit->bottomRows<3>();
                d_res_d_stateinit->topRows<2>() = J_predtrans_acadotrans * d_res_d_stateinit->topRows<2>();
                d_res_d_stateinit->topRows<2>() += J_predtrans_lastrot * d_pred_d_stateinit->row(2);
                d_res_d_stateinit->topRows<3>() += d_pred_d_stateinit->topRows<3>();
                (*d_pred_d_stateinit) = (*d_res_d_stateinit);

                d_res_d_stateinit->block<2, 3>(0, 0) += R_o0_w_mat.topRows<2>();
                d_res_d_stateinit->block<2, 3>(0, 3) -= tmp2 * R_o0_w_mat;
                d_res_d_stateinit->block<1, 3>(2, 3) += J_yaw_state;
            }
        }

        if(d_res_d_biasinit){
            // d_res_d_biasinit is d_pred_d_biasinit
            Mat63 d_res_d_biasinit_tmp = J_init * d_res_d_biasinit->bottomRows<3>();
            d_res_d_biasinit_tmp.topRows<2>() = J_predtrans_acadotrans * d_res_d_biasinit_tmp.topRows<2>();
            d_res_d_biasinit_tmp.topRows<2>() += J_predtrans_lastrot * d_res_d_biasinit->row(2);
            d_res_d_biasinit_tmp.topRows<3>() += d_res_d_biasinit->topRows<3>();
            (*d_res_d_biasinit) = d_res_d_biasinit_tmp;
        }

        if(d_res_d_state1){
            d_res_d_state1->setZero();
            // pose res
            d_res_d_state1->block<2, 3>(0, 0) = -R_o0_w_mat.topRows<2>();
            d_res_d_state1->block<2, 3>(0, 3) = tmp * R_o0_w_mat;
            d_res_d_state1->block<1, 3>(2, 3) = -J_yaw_state;

            Eigen::Matrix3d R_o1_w_mat = R_o1_w.matrix();
            // vel res
            d_res_d_state1->block<2, 3>(3, 0) = -cg_end_tmp_hat* R_o1_w_mat;
            d_res_d_state1->block<2, 3>(3, 3) = -R_o1_w_mat.topRows<2>();
        }

        if(d_res_d_bias1){
            d_res_d_bias1->setZero();
            Eigen::Matrix3d R_o_i1_mat = R_o_i1.matrix();
            // vel residual
            d_res_d_bias1->block<2, 3>(0, 0) = t_o_i1_hat.topRows<2>() * R_o_i1_mat;
            d_res_d_bias1->row(2) = R_o_i1_mat.row(2);
        }
    } //jacobians

    if(command_window.size() > 1){
        residuals = residual_cmdstep(singletrack_acado,
                                     param_state, r_o0_o1, t_o0_o1, cg_end_lin, cg_end_ang,
                                     pred_pose_state, pred_vel_state,
                                     d_pred_d_stateinit, d_pred_d_extrinit, d_res_d_stateinit, d_res_d_biasinit,
                                     d_res_d_extrinit, d_pred_d_param);

    }

    return residuals;
}

    SingleTrackModel::Vec6 SingleTrackModel::residual_cmdstep(const SingleTrackAcado& singletrack_acado,
                                                            const SingleTrackParamOnline &param_state,
                                                            const Eigen::Vector3d& r_o0_o1,
                                                            const Eigen::Vector3d& t_o0_o1,
                                                            const Eigen::Vector3d& cg_end_lin,
                                                            const Eigen::Vector3d& cg_end_ang,
                                                            Eigen::Vector3d* pred_pose_state,
                                                            Eigen::Vector3d* pred_vel_state,
                                                            Mat66 *d_pred_d_stateinit,
                                                            Mat66 *d_pred_d_extrinit,
                                                            Mat66 *d_res_d_stateinit,
                                                            Mat63 *d_res_d_biasinit,
                                                            Mat66 *d_res_d_extrinit,
                                                            Eigen::MatrixXd *d_pred_d_param) const {
        Command curr_cmd = command_window[1];

        double init_state[6] = {0., 0., 0., (*pred_vel_state)(0), (*pred_vel_state)(1), (*pred_vel_state)(2)};
        double param_list[PARAM_SIZE] = {param_state.c_lat, param_state.steering_rt,
                                                                 param_state.throttle_f1, param_state.throttle_f2,
                                                                 param_state.throttle_res};
        double control_input[2] = {curr_cmd.linear, curr_cmd.angular};

        ACADO::IntegratorRK45 integrator(singletrack_acado.f);
        ACADO::DVector end_states;

        double frame_dt = (end_t_ns - curr_cmd.t_ns) * 1e-9;
        integrator.set(ACADO::INTEGRATOR_PRINTLEVEL, ACADO::NONE);
        integrator.freezeAll();
        integrator.integrate(0.0, frame_dt, init_state, nullptr, param_list, control_input);
        integrator.getX(end_states);

        // predicted pose
        Sophus::SO2d rot_last((*pred_pose_state)(2));
        Eigen::Vector2d se2_rot_p = rot_last * end_states.head<2>();
        pred_pose_state->head<2>() += se2_rot_p;
        (*pred_pose_state)(2) += end_states(2);

        // predicted vel
        (*pred_vel_state) = end_states.tail<3>();

        Vec6 residuals;
        residuals.head<2>() = pred_pose_state->head<2>() - t_o0_o1.head<2>();
        residuals(2) = (*pred_pose_state)(2) - r_o0_o1(2);
        residuals(3) = end_states(3) - cg_end_lin(0); //vx
        residuals(4) = end_states(4) - cg_end_lin(1); //vy
        residuals(5) = end_states(5) - cg_end_ang(2); //omega

        // jacobians
        if (d_pred_d_stateinit || d_pred_d_extrinit || d_res_d_stateinit || d_res_d_extrinit || d_pred_d_param) {
            Eigen::Matrix2d J_predtrans_acadotrans = rot_last.matrix();
            Eigen::Vector2d J_predtrans_lastrot(-se2_rot_p(1), se2_rot_p(0));

            Eigen::Matrix<double, 6, 3> J_init;
            //derivative of residual tran_x, tran_y yaw wrt. parameters
            if (d_pred_d_param) {
                Eigen::MatrixXd d_pred_d_param_last = *d_pred_d_param; // copy of last J_pred_param
                d_pred_d_param->setZero();
                ACADO::DVector seed(6);
                for (int i = 0; i < 6; i++) {
                    seed.setZero();
                    seed(i) = 1.0;

                    ACADO::DVector D_p(PARAM_SIZE);
                    ACADO::DVector D_state_init(6);
                    integrator.setBackwardSeed(1, seed);
                    integrator.integrateSensitivities();
                    integrator.getBackwardSensitivities(D_state_init, D_p, ACADO::emptyVector, ACADO::emptyVector,
                                                        1); // w.r.t. x0,p,u,w
                    J_init.row(i) = D_state_init.tail<3>();

                    d_pred_d_param->block<1, PARAM_SIZE>(i, 0) = D_p;
                }
                (*d_pred_d_param) += J_init * d_pred_d_param_last.bottomRows(3);

                // d_pred_d_param = d_res_d_param
                d_pred_d_param->topRows(2) = J_predtrans_acadotrans * d_pred_d_param->topRows(2);
                d_pred_d_param->topRows(2) += J_predtrans_lastrot * d_pred_d_param_last.row(2);
                d_pred_d_param->topRows(3) += d_pred_d_param_last.topRows(3);
            }

            if (d_pred_d_extrinit) {
                if (d_res_d_extrinit) {
                    // remove d_pred_d_extrinit part from d_res_d_extrinit
                    (*d_res_d_extrinit) -= (*d_pred_d_extrinit);

                    SingleTrackModel::Mat66 d_res_d_extrinit_tmp = J_init * d_pred_d_extrinit->bottomRows<3>();
                    d_res_d_extrinit_tmp.topRows<2>() = J_predtrans_acadotrans * d_res_d_extrinit_tmp.topRows<2>();
                    d_res_d_extrinit_tmp.topRows<2>() += J_predtrans_lastrot * d_pred_d_extrinit->row(2);
                    d_res_d_extrinit_tmp.topRows<3>() += d_pred_d_extrinit->topRows<3>();
                    (*d_pred_d_extrinit) = d_res_d_extrinit_tmp;

                    (*d_res_d_extrinit) += (*d_pred_d_extrinit);
                }
            }

            if (d_pred_d_stateinit) {
                if (d_res_d_stateinit) {
                    // remove d_pred_d_stateinit part from d_res_d_stateinit
                    (*d_res_d_stateinit) -= (*d_pred_d_stateinit);

                    SingleTrackModel::Mat66 d_res_d_stateinit_tmp = J_init * d_pred_d_stateinit->bottomRows<3>();
                    d_res_d_stateinit_tmp.topRows<2>() = J_predtrans_acadotrans * d_res_d_stateinit_tmp.topRows<2>();
                    d_res_d_stateinit_tmp.topRows<2>() += J_predtrans_lastrot * d_pred_d_stateinit->row(2);
                    d_res_d_stateinit_tmp.topRows<3>() += d_pred_d_stateinit->topRows<3>();
                    (*d_pred_d_stateinit) = d_res_d_stateinit_tmp;

                    (*d_res_d_stateinit) += (*d_pred_d_stateinit);
                }
            }

            if (d_res_d_biasinit) {
                // d_res_d_biasinit is d_pred_d_biasinit
                Mat63 d_res_d_biasinit_tmp = J_init * d_res_d_biasinit->bottomRows<3>();
                d_res_d_biasinit_tmp.topRows<2>() = J_predtrans_acadotrans * d_res_d_biasinit_tmp.topRows<2>();
                d_res_d_biasinit_tmp.topRows<2>() += J_predtrans_lastrot * d_res_d_biasinit->row(2);
                d_res_d_biasinit_tmp.topRows<3>() += d_res_d_biasinit->topRows<3>();
                (*d_res_d_biasinit) = d_res_d_biasinit_tmp;
            }


        } // end jacobians

        return residuals;
    }

Eigen::Vector3d SingleTrackModel::get_start_gyro() const {
    return start_gyro;
}

std::deque<Command> SingleTrackModel::get_command_window() const {
    return command_window;
}
} //namespace dynvio
