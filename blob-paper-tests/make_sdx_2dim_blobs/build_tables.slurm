#!/bin/bash
#SBATCH --job-name=build_2dim_tables
#SBATCH --output=${BLOB_TEST_DIR}/build_2dim_slurm_out/out.%a.out
#SBATCH --time=7-0
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-1000
arrayNum="${SLURM_ARRAY_TASK_ID}"
source ${BLOB_TEST_DIR}/.venv/bin/activate
python build_tables.py $arrayNum