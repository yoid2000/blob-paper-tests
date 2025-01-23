import os
import json
import sys
import pandas as pd
from syndiffix import SyndiffixBlobBuilder

# check to see if a command line argument was passed
job_num = 0
if len(sys.argv) == 2:
    # check to see if the argument is an integer
    if sys.argv[1].isdigit():
        job_num = int(sys.argv[1])
    else:
        print("Usage: python build_tables.py [job_num]")
        sys.exit(1)

print(f"Running with job number: {job_num}")

blob_code_path = '.'
# check if there is an environment variable called BLOB_TEST_DIR
if 'BLOB_TEST_DIR' in os.environ:
    blob_path = os.environ['BLOB_TEST_DIR']
    blob_code_path = os.environ['BLOB_TEST_CODE']
else:
    blob_path = os.path.join(os.path.dirname(os.getcwd()), 'blob_tests')

print(f"Blob path: {blob_path}")

os.makedirs(blob_path, exist_ok=True)
datasets_path = os.path.join(blob_path, 'datasets')
# make directory 'datasets' if it doesn't exist
os.makedirs(datasets_path, exist_ok=True)
sdx_2dim_path = os.path.join(datasets_path, 'sdx_2dim')
os.makedirs(sdx_2dim_path, exist_ok=True)
kaggle_parquet_path = os.path.join(datasets_path, 'kaggle_parquet')
# make sure kaggle_parquet_path exists
if not os.path.exists(kaggle_parquet_path):
    print(f"Error: {kaggle_parquet_path} does not exist")
    print("You need to gather the Kaggle datasets first")
    sys.exit(1)

filenames = os.listdir(kaggle_parquet_path)
filenames.sort()
if len(filenames) < job_num+1:
    print(f"Error: job number {job_num} is out of range")
    sys.exit(1)
filename = filenames[job_num]
file_path = os.path.join(kaggle_parquet_path, filename)

blob_name = filename.replace('.parquet', '')
blob_dir_path = os.path.join(sdx_2dim_path, blob_name)
# check if a directory at blob_dir_path already exists
if os.path.exists(blob_dir_path):
    blob_full_path = os.path.join(blob_dir_path, blob_name + 'sdxblob.zip')
    if os.path.exists(blob_full_path):
        print(f"Skipping {blob_full_path} already exists")
        sys.exit(1)

print(f"Build synthetic tables for:\n{file_path}")

df_orig = pd.read_parquet(file_path)
print(df_orig.head())
print(df_orig.columns)
# remove .parquet from filename
os.makedirs(blob_dir_path, exist_ok=True)

print(f"Make blob for {blob_name} at {blob_dir_path}")
sbb = SyndiffixBlobBuilder(blob_name=blob_name,
                           path_to_dir=blob_dir_path,
                           max_cluster_size=2)
sbb.write(df_raw=df_orig)