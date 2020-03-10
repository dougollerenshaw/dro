import os
import pandas as pd
import numpy as np

import sys
sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/src/pbstools')

from pbstools import PythonJob 

def deploy_job(
    python_file,
    python_executable = '/allen/programs/braintv/workgroups/nc-ophys/Doug/.conda/envs/visual_behavior/bin/python3.7',
    python_args = [],
    jobname = 'unnamed'
    ):

    jobdir = '/allen/programs/braintv/workgroups/nc-ophys/Doug/cluster_jobs'
    job_settings = {'queue': 'braintv',
                    'mem': '4g',
                    'walltime': '0:20:00',
                    'ppn':1,
                    'jobdir': jobdir,
                    }
        
    PythonJob(
        python_file,
        python_executable=python_executable, # path to conda environment that has the correct python version, and all needed packages
        python_args=python_args,
        conda_env=None,
        jobname = jobname,
        **job_settings
    ).run(dryrun=False)

if __name__ == "__main__":
    manifest = pd.read_csv('/allen/programs/braintv/workgroups/nc-ophys/Doug/2019.12.10.sessions_with_tracking.csv')
    for ind_row, row in manifest.iterrows():
        osid = row['ophys_session_id']
        print('sending job for osid={}'.format(osid))

        deploy_job(
            python_file = '/home/dougo/code/dro/dro/scripts/2019.12.11_make_individual_image_response_plots.py',
            python_args = osid,
            jobname = '{}'.format(osid)
        )