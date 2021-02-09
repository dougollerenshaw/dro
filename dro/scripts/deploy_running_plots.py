import os
import argparse
import sys
import time
import pandas as pd
import numpy as  np
from visual_behavior import database as db
import visual_behavior.data_access.loading as loading
sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/src/')
from pbstools import pbstools  # NOQA E402

parser = argparse.ArgumentParser(description='run sdk validation')
parser.add_argument('--env', type=str, default='visual_behavior', metavar='name of conda environment to use')

job_dir = r"/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/cluster_jobs/job_records"

job_settings = {'queue': 'braintv',
                'mem': '4g',
                'walltime': '0:5:00',
                'ppn': 1,
                }

if __name__ == "__main__":
    args = parser.parse_args()
    python_executable = "{}/.conda/envs/{}/bin/python".format(os.path.expanduser('~'), args.env)
    print('python executable = {}'.format(python_executable))
    python_file = "{}/code/dro/dro/scripts/make_running_sample_plots.py".format(os.path.expanduser('~'))

    # experiments_table = loading.get_filtered_ophys_experiment_table()

    # session_ids = experiments_table['ophys_session_id'].unique()

    savedir = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/session_plots/running_smoothing_check/new/cause_for_concern_new'
    session_ids = [
        995115654,
        1044379245,
        937682841,
        848891498,
        964887530,
        870352564,
        993738515,
        874323406,
        968900812,
        954800562,
        825682144,
        919041767,
        1004824790,
        842752650,
        994883056,
        955775716,
        951556788,
        1006419624,
        895588149,
        990464099,
        993727065,
        1071202230,
        1042505240,
        797078933,
        973701907,
        993231283,
        855353286,
        890086402,
        851438454,
        858863712,
        877497698,
        856295914,
        875259383,
        873653940,
        962821424,
        853773937,
        1050929040,
        884613038,
        931326814,
        904771513,
        848253761,
        849600749,
        906968227,
        994731371,
        871526950,
        857202651,
        853416014,
        940145217,
        966603493,
        971852110,
        843363571,
        940775208,
        829521794,
        907177554,
        889179793,
        882351065,
        923705570,
        843871375,
        876522267,
        888190799,
        908441202,
        929686773
    ]

    for session_id in session_ids:

        print('starting cluster job for {}'.format(session_id))
        job_title = 'osid_{}_running_plot'.format(session_id)
        pbstools.PythonJob(
            python_file,
            python_executable,
            python_args="--osid {} --savedir {}".format(session_id, savedir),
            jobname=job_title,
            jobdir=job_dir,
            **job_settings
        ).run(dryrun=False)
        time.sleep(0.001)