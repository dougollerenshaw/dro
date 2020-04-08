import os
import pandas as pd
import numpy as np

import sys
sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/src/pbstools')

from pbstools import PythonJob 

def deploy_job(
    python_file,
    python_executable = '~/.conda/envs/visual_behavior/bin/python3.7',
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
    for i in range(10):
        print('sending job for i={}'.format(i))

        deploy_job(
            python_file = '/home/dougo/code/dro/dro/scripts/make_a_plot.py',
            jobname = 'test_job_{}'.format(i)
        )