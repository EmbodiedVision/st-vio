# This file is part of ST-VIO.
#
# Copyright (c) 2024, Max Planck Gesellschaft.
# Developed by Haolong Li <haolong.li at tue dot mpg dot de>
# Embodied Vision Group, Max Planck Institute for Intelligent Systems.
# If you use this code, please cite the respective publication as
# listed in README.md.
#
# ST-VIO is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ST-VIO is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ST-VIO.  If not, see <https://www.gnu.org/licenses/>.


import os
import sys
sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 './rpg_trajectory_evaluation/scripts'))

import argparse

import numpy as np

import add_path
from trajectory import Trajectory
import compute_trajectory_errors as traj_err
from fn_constants import kNsToEstFnMapping, kNsToMatchFnMapping, kFnExt
from multiple_traj_errors import MulTrajError
import trajectory_utils as tu

import transformations as tf
import results_writer as res_writer

def load_poses(results_dir, nm_es="stamped_groundtruth.txt",
                start_t_sec=-float('inf'), end_t_sec=float('inf')):
    fn_es = os.path.join(results_dir, nm_es)
    data_es = np.loadtxt(fn_es)
    t_es = []
    p_es = []
    q_es = []
    for d in data_es:
        if d[0] < start_t_sec or d[0] > end_t_sec:
            continue
        t_es.append(d[0])
        p_es.append(d[1:4])
        q_es.append(d[4:8])
    t_es = np.array(t_es)
    p_es = np.array(p_es)
    q_es = np.array(q_es)

    return t_es, p_es, q_es


def load_pred(results_dir, nm_es="pred.txt",
                 start_t_sec=-float('inf'), end_t_sec=float('inf')):
    fn_es = os.path.join(results_dir, nm_es)
    data_es = np.loadtxt(fn_es)
    t_start = []
    t_end = []
    p_es = []
    q_es = []
    for d in data_es:
        if d[0] < start_t_sec or d[0] > end_t_sec:
            continue
        t_start.append(d[0])
        t_end.append(d[1])
        p_es.append(d[2:5])
        q_es.append(d[5:9])
    t_start = np.array(t_start)
    t_end = np.array(t_end)
    p_es = np.array(p_es)
    q_es = np.array(q_es)

    return t_start, t_end, p_es, q_es

def compute_pred_error(pred_t_start, pred_t_end, step, 
                       pred_p, pred_q,
                       gt_t, gt_p, gt_q, 
                       t_extr = None, extr_q = None, extr_p = None):
    errors = []
    es_idx = 0
    for idx in range(len(gt_t)):
        if(es_idx < len(pred_t_start) and gt_t[idx] != pred_t_start[es_idx]):
            continue

        if gt_t[idx] > pred_t_start[-1]:
            break

        assert pred_t_start[es_idx] == gt_t[idx], 'Same timestamp! {} {} {} {}'.format(pred_t_start[es_idx], es_idx, gt_t[idx], idx)
        assert pred_t_end[es_idx] == gt_t[idx + step]

        T_gt1 = tu.get_rigid_body_trafo(gt_q[idx, :], gt_p[idx, :])
        T_gt2 = tu.get_rigid_body_trafo(gt_q[idx + step, :], gt_p[idx + step, :])

        T_error = np.eye(4)

        T_gt11_gt22 = np.dot(np.linalg.inv(T_gt1), T_gt2)
        T_pred1_pred2 = tu.get_rigid_body_trafo(pred_q[es_idx, :], pred_p[es_idx, :])

        if extr_q is not None and extr_p is not None:
            assert t_extr[es_idx] == gt_t[idx]
            assert t_extr[es_idx + step] == gt_t[idx + step]
            T_extr1 = tu.get_rigid_body_trafo(extr_q[es_idx, :], extr_p[es_idx, :])
            T_extr2 = tu.get_rigid_body_trafo(extr_q[es_idx + step, :], extr_p[es_idx + step, :])
            new_gt1_gt2 = np.dot(T_extr1, T_gt11_gt22)
            new_gt1_gt2 = np.dot(new_gt1_gt2, np.linalg.inv(T_extr2))

            T_error = np.dot(np.linalg.inv(new_gt1_gt2), T_pred1_pred2)
        else:
            T_error = np.dot(np.linalg.inv(T_gt11_gt22), T_pred1_pred2)

        errors.append(T_error)
        es_idx += 1

    error_trans_norm = []
    e_rot = []

    for e in errors:
        tn = np.linalg.norm(e[0:3, 3])
        error_trans_norm.append(tn)
        e_rot.append(tu.compute_angle(e))

    return np.array(error_trans_norm), np.array(e_rot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Analyze trajectory estimate in a folder.''')
    parser.add_argument(
        'root_dir', type=str,
        help="Root directory.")

    args = parser.parse_args()
    assert os.path.exists(args.root_dir)

    predict_dir = os.path.join(args.root_dir, 'pred')
    assert os.path.exists(predict_dir)

    t_gt_sp = None
    p_gt_sp = None
    q_gt_sp = None

    t_gt_sp, p_gt_sp, q_gt_sp = load_poses(args.root_dir, nm_es="stamped_groundtruth.txt")
    parent_dir = os.path.split(predict_dir)[0]
    t_extr_sp, p_extr_sp, q_extr_sp = load_poses(parent_dir, nm_es="extrinsic_poses.txt")

    pred_calib_trans_all = []
    pred_calib_rot_all = []

    pred_raw_trans_all = []
    pred_raw_rot_all = []

    steps = [10, 20, 50, 100, 300]

    output_dir = os.path.join(predict_dir, 'results')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for step in steps:
        pred_calib_step = "calib_pred" + str(step)
        pred_raw_step = "raw_pred" + str(step)

        pred_calib_start, pred_calib_end, pred_calib_p, pred_calib_q = load_pred(predict_dir, pred_calib_step + '.txt')
        pred_calib_trans_e, pred_calib_rot_e = compute_pred_error(pred_calib_start, pred_calib_end, step,
                                                                  pred_calib_p, pred_calib_q,
                                                                  t_gt_sp, p_gt_sp, q_gt_sp, 
                                                                  t_extr_sp, q_extr_sp, p_extr_sp)
        pred_calib_stats_trans = res_writer.compute_statistics(pred_calib_trans_e)
        pred_calib_stats_rot = res_writer.compute_statistics(pred_calib_rot_e)
        pred_calib_trans_all.extend(pred_calib_trans_e.tolist())
        pred_calib_rot_all.extend(pred_calib_rot_e.tolist())
        calib_step_out_path = os.path.join(output_dir, pred_calib_step + '.yaml')
        res_writer.update_and_save_stats(
            pred_calib_stats_trans, 'trans',
            calib_step_out_path)
        res_writer.update_and_save_stats(
            pred_calib_stats_rot, 'rot',
            calib_step_out_path)

        pred_raw_start, pred_raw_end, pred_raw_p, pred_raw_q = load_pred(predict_dir, pred_raw_step + '.txt')
        pred_raw_trans_e, pred_raw_rot_e = compute_pred_error(pred_raw_start, pred_raw_end, step, 
                                                              pred_raw_p, pred_raw_q, 
                                                              t_gt_sp, p_gt_sp, q_gt_sp, 
                                                              t_extr_sp, q_extr_sp, p_extr_sp)
        pred_raw_stats_trans = res_writer.compute_statistics(pred_raw_trans_e)
        pred_raw_stats_rot = res_writer.compute_statistics(pred_raw_rot_e)
        pred_raw_trans_all.extend(pred_raw_trans_e.tolist())
        pred_raw_rot_all.extend(pred_raw_rot_e.tolist())
        raw_step_out_path = os.path.join(output_dir, pred_raw_step + '.yaml')
        res_writer.update_and_save_stats(
            pred_raw_stats_trans, 'trans',
            raw_step_out_path)
        res_writer.update_and_save_stats(
            pred_raw_stats_rot, 'rot',
            raw_step_out_path)

    pred_calib_stats_trans_all = res_writer.compute_statistics(pred_calib_trans_all)
    pred_calib_stats_rot_all = res_writer.compute_statistics(pred_calib_rot_all)
    calib_all_out_path = os.path.join(output_dir, 'pred_calib_all.yaml')
    res_writer.update_and_save_stats(
        pred_calib_stats_trans_all, 'trans',
        calib_all_out_path)
    res_writer.update_and_save_stats(
        pred_calib_stats_rot_all, 'rot',
        calib_all_out_path)

    pred_raw_stats_trans_all = res_writer.compute_statistics(pred_raw_trans_all)
    pred_raw_stats_rot_all = res_writer.compute_statistics(pred_raw_rot_all)
    raw_all_out_path = os.path.join(output_dir, 'pred_raw_all.yaml')
    res_writer.update_and_save_stats(
        pred_raw_stats_trans_all, 'trans',
        raw_all_out_path)
    res_writer.update_and_save_stats(
        pred_raw_stats_rot_all, 'rot',
        raw_all_out_path)
