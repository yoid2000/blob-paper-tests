The code here is used to generate syndiffix blobs with up to 2 columns per synthetic table

the command `python build_tables.py <job_num>` generates the synthetic tables for the dataset whose index is `<job_num>` when alphabetically sorted.

The file `build_tables.slurm` is a slurm script that can be used to run the `build_tables.py` job. (You may need to replace the env variables with the corresponding full paths).

The command `python check_tables.py` checks to see that every column pair has been produced.