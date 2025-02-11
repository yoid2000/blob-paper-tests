import os
import pandas as pd

blob_code_path = '.'
# check if there is an environment variable called BLOB_TEST_DIR
if 'BLOB_TEST_DIR' in os.environ:
    blob_path = os.environ['BLOB_TEST_DIR']
    blob_code_path = os.environ['BLOB_TEST_CODE']
else:
    blob_path = os.path.join(os.path.dirname(os.getcwd()), 'blob_tests')

print(f"Blob path: {blob_path}")

results_path = os.path.join(blob_path, 'results')
all_results_path = os.path.join(results_path, 'all_tests')

results_file_path = os.path.join(all_results_path, 'all_results.parquet')
if os.path.exists(results_file_path):
    # read file as df
    df = pd.read_parquet(results_file_path)
else:
    # read all parquet files in all_results_path and sub directories
    # and gather the results in a single dataframe
    df = pd.DataFrame()
    for root, dirs, files in os.walk(all_results_path):
        for file in files:
            if file.endswith('.parquet'):
                file_path = os.path.join(root, file)
                df_temp = pd.read_parquet(file_path)
                df = pd.concat([df, df_temp])
print(df.head())
print(f"Number of rows: {len(df)}")
df.to_parquet(os.path.join(all_results_path, 'all_results.parquet'))