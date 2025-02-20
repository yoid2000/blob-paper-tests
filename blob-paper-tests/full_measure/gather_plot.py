import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def file_is_in_kaggle(file, kaggle_files):
    # go through each name in kaggle_files. If file begins with name, return True
    for name in kaggle_files:
        if file.startswith(name):
            return True 
    return False

blob_code_path = '.'
# check if there is an environment variable called BLOB_TEST_DIR
if 'BLOB_TEST_DIR' in os.environ:
    blob_path = os.environ['BLOB_TEST_DIR']
    blob_code_path = os.environ['BLOB_TEST_CODE']
else:
    blob_path = os.path.join(os.path.dirname(os.getcwd()), 'blob_tests')

print(f"Blob path: {blob_path}")

results_path = os.path.join(blob_path, 'results')
kaggle_path = os.path.join(blob_path, 'datasets', 'kaggle_parquet')
all_results_path = os.path.join(results_path, 'full_measure')
results_file_path = os.path.join(results_path, 'full_measure.parquet')
os.makedirs('real_results', exist_ok=True)

if os.path.exists(results_file_path):
    # read file as df
    df = pd.read_parquet(results_file_path)
else:
    # read all of the base kaggle file names in kaggle_path that end with .parquet, strip off the suffix, and put in a list called kaggle_files
    kaggle_files = [f[:-8] for f in os.listdir(kaggle_path) if f.endswith('.parquet')]
    # read all parquet files in all_results_path and sub directories
    # and gather the results in a single dataframe
    num_files_read = 0
    df = pd.DataFrame()
    for root, dirs, files in os.walk(all_results_path):
        for file in files:
            if file.endswith('.parquet'):
                if not file_is_in_kaggle(file, kaggle_files):
                    print(f"File {file} not in kaggle, skip")
                    continue
                file_path = os.path.join(root, file)
                df_temp = pd.read_parquet(file_path)
                df = pd.concat([df, df_temp])
                num_files_read += 1
                if num_files_read % 100 == 0:
                    print(f"Read {num_files_read} files")
    df = df.reset_index(drop=True)
    df.to_parquet(results_file_path)

print(df.head())
