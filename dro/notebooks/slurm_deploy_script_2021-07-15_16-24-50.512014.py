# This temporary file that will execute a given python script using slurm

# imports
from simple_slurm import Slurm
import os

python_path = PYTHON_PATH
stdout_location = STDOUT_LOCATION
python_command = PYTHON_COMMAND

# instantiate a Slurm object
slurm = Slurm(
    cpus_per_task=CPUS_PER_TASK,
    job_name=JOB_NAME,
    partition=PARTITION,
    mem=MEM,
    walltime=WALLTIME,
    output=f'{stdout_location}/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
)

# call the `sbatch` command to run the job
slurm.sbatch('{} {}'.format(
        python_path,
        python_command,
    )
)