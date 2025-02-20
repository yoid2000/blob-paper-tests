#!/bin/bash
#SBATCH --job-name=all_stats
#SBATCH --output=/INS/syndiffix/work/paul/blob_tests/all_stats_slurm_out/out.%a.out
#SBATCH --time=7-0
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-4000
arrayNum="${SLURM_ARRAY_TASK_ID}"
source /INS/syndiffix/work/paul/blob_tests/.venv/bin/activate
python /INS/syndiffix/work/paul/github/blob-paper-tests/blob-paper-tests/all_stat_tests/compute_one.py $arrayNum