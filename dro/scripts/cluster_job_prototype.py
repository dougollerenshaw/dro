import os
import pandas as pd
import numpy as np

import sys
sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/pbstools')

from pbstools.pbstools import PythonJob 
# python_file = r"/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/src/licking_behavior/model_fitting_script.py"
python_file = r"/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/src/SWDB_2019/DynamicBrain/plot_lick_raster_one_session.py"
jobdir = '/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/cluster_jobs/20190729_plot_licks_rasters'
job_settings = {'queue': 'braintv',
                'mem': '4g',
                'walltime': '0:20:00',
                'ppn':1,
                'jobdir': jobdir,
                }

manifest = pd.read_csv('visual_behavior_data_manifest.csv')

for ind_row, row in manifest.iterrows():
    experiment_id = row['ophys_experiment_id']

    PythonJob(
        python_file,
        python_executable='/home/alex.piet/codebase/miniconda3/envs/visbeh/bin/python', # path to conda environment that has the correct python version, and all needed packages
        python_args=experiment_id,
        conda_env=None,
        jobname = 'lick_rasters_{}'.format(experiment_id),
        **job_settings
    ).run(dryrun=False)
