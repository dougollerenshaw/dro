import os
import pandas as pd
import numpy as np

import sys
sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/src')

from pbstools.pbstools import PythonJob 
# python_file = r"/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/src/licking_behavior/model_fitting_script.py"
python_file = r"/home/dougo/code/scripts/2019.12.10_make_eyetracking_summary_plots.py"
jobdir = '/allen/programs/braintv/workgroups/nc-ophys/Doug/cluster_jobs'
job_settings = {'queue': 'braintv',
                'mem': '4g',
                'walltime': '0:20:00',
                'ppn':1,
                'jobdir': jobdir,
                }

manifest = pd.read_csv('/allen/programs/braintv/workgroups/nc-ophys/Doug/2019.12.10.sessions_with_tracking.csv')

for ind_row, row in manifest.iterrows():
    osid = row['ophys_session_id']
    print('sending job for osid={}'.format(osid))
    PythonJob(
        python_file,
        python_executable='/allen/programs/braintv/workgroups/nc-ophys/Doug/.conda/envs/visual_behavior/bin/python3.7', # path to conda environment that has the correct python version, and all needed packages
        python_args=osid,
        conda_env=None,
        jobname = 'eye_tracking_summary_{}'.format(osid),
        **job_settings
    ).run(dryrun=False)
