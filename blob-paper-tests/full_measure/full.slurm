#!/bin/bash
#SBATCH --job-name=full_measure
#SBATCH --output=/INS/syndiffix/work/paul/blob_tests/full_measure_out/out.%a.out
#SBATCH --time=7-0
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-1000
arrayNum="${SLURM_ARRAY_TASK_ID}"
source /INS/syndiffix/work/paul/blob_tests/.venv/bin/activate
python /INS/syndiffix/work/paul/github/blob-paper-tests/blob-paper-tests/full_measure/compute_one.py $arrayNum