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
#
# This file has been derived from the original file
# rpg_trajectory_evaluation/scripts/analyze_trajectory_single.py
# from the rpg_trajectory_evaluation project (https://github.com/uzh-rpg/rpg_trajectory_evaluation).
# The original file has been released under the MIT License (see https://github.com/uzh-rpg/rpg_trajectory_evaluation/blob/master/LICENSE).


import os
import sys
sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 './rpg_trajectory_evaluation/scripts'))

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
from colorama import init, Fore

import add_path
from trajectory import Trajectory
import plot_utils as pu
from fn_constants import kNsToEstFnMapping, kNsToMatchFnMapping, kFnExt
from multiple_traj_errors import MulTrajError
from metrics import kRelMetrics, kRelMetricLables
import results_writer as res_writer

init(autoreset=True)
rc('font', **{'family': 'serif', 'serif': ['Cardo']})
rc('text', usetex=True)

FORMAT = '.pdf'


def analyze_multiple_trials(results_dir, est_type, n_trials,
                            recalculate_errors=False,
                            preset_boxplot_distances=[],
                            preset_boxplot_percentages=[0.1, 0.2, 0.3, 0.4, 0.5],
                            compute_odometry_error=True):
    traj_list = []
    mt_error = MulTrajError()
    for trial_i in range(n_trials):
        if n_trials == 1:
            suffix = ''
        else:
            suffix = str(trial_i)
        print(Fore.RED+"### Trial {0} ###".format(trial_i))

        match_base_fn = kNsToMatchFnMapping[est_type]+suffix+'.'+kFnExt

        if recalculate_errors:
            Trajectory.remove_cached_error(results_dir,
                                           est_type, suffix)
            Trajectory.remove_files_in_save_dir(results_dir, est_type,
                                                match_base_fn)
        traj = Trajectory(
            results_dir, est_type=est_type, suffix=suffix,
            nm_est=kNsToEstFnMapping[est_type] + suffix + '.'+kFnExt,
            nm_matches=match_base_fn,
            preset_boxplot_distances=preset_boxplot_distances,
            preset_boxplot_percentages=preset_boxplot_percentages)
        if traj.data_loaded:
            traj.compute_absolute_error()
            if compute_odometry_error:
                traj.compute_relative_errors()
        if traj.success:
            traj.cache_current_error()
            traj.write_errors_to_yaml()

        # overall rel erro
        overall_rel_errors = {}
        for et in kRelMetrics:
            overall_rel_errors[et] = []
        for dist in traj.rel_errors:
          cur_err = traj.rel_errors[dist]
          for et in kRelMetrics:
            overall_rel_errors[et].extend(cur_err[et])          

        rel_err_overall_stats_fn = os.path.join(
            traj.saved_results_dir, 'relative_err_overall' + '.yaml')
        for et, label in zip(kRelMetrics, kRelMetricLables):
            res_writer.update_and_save_stats(
                res_writer.compute_statistics(overall_rel_errors[et]), label,
                rel_err_overall_stats_fn)

        if traj.success and not preset_boxplot_distances:
            print("Save the boxplot distances for next trials.")
            preset_boxplot_distances = traj.preset_boxplot_distances

        if traj.success:
            mt_error.addTrajectoryError(traj, trial_i)
            traj_list.append(traj)
        else:
            print("Trials {0} fails, will not count.".format(trial_i))
    mt_error.summary()
    mt_error.updateStatistics()
    return traj_list, mt_error


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Analyze trajectory estimate in a folder.''')
    parser.add_argument(
        'result_dir', type=str,
        help="Folder containing the groundtruth and the estimate.")
    parser.add_argument(
        '--est_types', nargs="*", type=str,
        default=['traj_est'])

    args = parser.parse_args()

    assert os.path.exists(args.result_dir)

    for est_type in args.est_types:
        assert est_type in kNsToEstFnMapping
        assert est_type in kNsToMatchFnMapping

    print(Fore.YELLOW + "=== Summary ===")
    print(Fore.YELLOW +
          "Going to analyze the results in {0}.".format(args.result_dir))
    print(Fore.YELLOW +
          "Will analyze estimate types: {0}".format(args.est_types))

    n_trials = 1

    for est_type_i in args.est_types:
        print(Fore.RED +
              "#### Processing error type {0} ####".format(est_type_i))
        mt_error = MulTrajError()
        traj_list, mt_error = analyze_multiple_trials(
            args.result_dir, est_type_i, n_trials, False)
        if not traj_list:
            print("No success runs.")

        if n_trials > 1:
            print(">>> Save results for multiple runs in {0}...".format(
                mt_error.save_results_dir))
            mt_error.saveErrors()
            mt_error.cache_current_error()

        print(Fore.GREEN +
              "#### Done processing error type {0} ####".format(est_type_i))
    import subprocess as s
    s.call(['notify-send', 'rpg_trajectory_evaluation finished',
            'results in: {0}'.format(os.path.abspath(args.result_dir))])
