#!/bin/bash
#SBATCH --job-name=all_stats
#SBATCH --output=${BLOB_TEST_DIR}/all_tests_slurm_out/out.%a.out
#SBATCH --time=7-0
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-1000
arrayNum="${SLURM_ARRAY_TASK_ID}"
source ${BLOB_TEST_DIR}/.venv/bin/activate
python ${BLOB_TEST_CODE}/blob-paper-tests/all_stat_tests/compute_one.py $arrayNum