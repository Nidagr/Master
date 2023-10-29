#!/bin/bash
#SBATCH --job-name="networkunit_128e"
#SBATCH --account=icei-hbp-2020-0005
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --partition=batch


module load GCC/11.2.0 OpenMPI/4.1.2 mpi4py/3.1.3 Python SciPy-bundle Seaborn 
source /p/home/jusers/groenbekk1/jusuf/envs/nunit/bin/activate
export GIT_PYTHON_REFRESH=quiet
srun python ws_equal_1_128.py
