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

parser = argparse.ArgumentParser(description='make running plots')
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
    python_file = "{}/code/dro/dro/scripts/2021.02.03_make_running_sample_plots.py".format(os.path.expanduser('~'))

    experiments_table = loading.get_filtered_ophys_experiment_table()

    session_ids = experiments_table['ophys_session_id'].unique()

    for session_id in session_ids:

        print('starting cluster job for {}'.format(session_id))
        job_title = 'osid_{}_running_plot'.format(session_id)
        pbstools.PythonJob(
            python_file,
            python_executable,
            python_args="--osid {}".format(session_id),
            jobname=job_title,
            jobdir=job_dir,
            **job_settings
        ).run(dryrun=False)
        time.sleep(0.001)
