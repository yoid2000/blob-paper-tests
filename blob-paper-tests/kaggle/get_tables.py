import kaggle
import json
import os
import shutil
import pandas as pd
import pprint
pp = pprint.PrettyPrinter(indent=4)

min_bytes_per_cell = 1.0
max_bytes_per_cell = 2.0
min_file_size = 200000
max_file_size = 500000
min_rows = 10000
max_rows = 250000
min_columns = 5
max_columns = 50

num_tables_to_gather = 1000

blob_code_path = '.'
# check if there is an environment variable called BLOB_TEST_DIR
if 'BLOB_TEST_DIR' in os.environ:
    blob_path = os.environ['BLOB_TEST_DIR']
    blob_code_path = os.environ['BLOB_TEST_CODE']
else:
    blob_path = os.path.join(os.path.dirname(os.getcwd()), 'blob_tests')

os.makedirs(blob_path, exist_ok=True)
print(f"Blob path: {blob_path}")

# make directory 'datasets' if it doesn't exist
datasets_path = os.path.join(blob_path, 'datasets')
os.makedirs(datasets_path, exist_ok=True)
kaggle_parquet_path = os.path.join(datasets_path, 'kaggle_parquet')
os.makedirs(kaggle_parquet_path, exist_ok=True)
temp_path = os.path.join(datasets_path, 'temp')
os.makedirs(temp_path, exist_ok=True)

status_path = os.path.join(datasets_path, 'status.json')
if os.path.exists(status_path):
    with open(status_path, 'r') as f:
        status = json.load(f)
else:
    status = {
        'last_page': 1,
    }

# count the number of files in kaggle_parquet_path
num_gathered_tables = 0
for file in os.listdir(kaggle_parquet_path):
    if file.endswith('.parquet'):
        num_gathered_tables += 1
last_page = status['last_page']

# Authenticate using kaggle.json
def authenticate_kaggle():
    kaggle.api.authenticate()

# List all datasets (you can customize the search query as needed)
def list_datasets(page=1, min_size=None, max_size=None, search=None):
    return kaggle.api.dataset_list(search=search, page=page, min_size=min_size, max_size=max_size)

authenticate_kaggle()

def is_datetime_column(series):
    if series.dtype == 'object':  # Check if the column is of type object (string)
        try:
            # Attempt to convert the entire series to datetime
            converted_series = pd.to_datetime(series, errors='raise')
            # Check if the conversion was successful for all elements
            return not converted_series.isnull().any()
        except (ValueError, TypeError):
            return False
    return False

while num_gathered_tables < num_tables_to_gather:
    datasets = list_datasets(page=last_page, min_size = min_file_size, max_size = max_file_size)
    if len(datasets) == 0:
        break
    # Iterate through datasets and download metadata
    for dataset in datasets:
        # remove all files and directories in temp_path
        shutil.rmtree(temp_path)
        os.makedirs(temp_path, exist_ok=True)
        owner, datasetName = dataset.ref.split('/')
        file_name = owner + '_slash_' + datasetName
        try:
            metadata = kaggle.api.metadata_get(owner, datasetName)
        except Exception as e:
            print(f"An error occurred: {e}")
            metadata = None  # or handle the error in another way
        if metadata is None:
            continue
        print(f"Downloading file: {dataset}")
        kaggle.api.dataset_download_files(f'{owner}/{datasetName}', path=temp_path, unzip=True)
        #ensure that there is only one file in temp_path, and that it is a csv file
        files = os.listdir(temp_path)
        if len(files) != 1:
            print(f"Skipping {file_name} because it has {len(files)} files")
            continue
        if not files[0].endswith('.csv'):
            print(f"Skipping {file_name} because it is not a csv file")
            continue
        # read the file into a dataframe

        try:
            df = pd.read_csv(os.path.join(temp_path, files[0]))
        except Exception as e:
            print(f"An error occurred while reading the CSV file: {e}")
            continue
        # make sure df has at least min_rows rows and min_columns columns
        if df.shape[0] < min_rows or df.shape[0] > max_rows or df.shape[1] < min_columns or df.shape[1] > max_columns:
            print(f"Skipping {file_name} because it has {df.shape[0]} rows and {df.shape[1]} columns")
            continue
        num_nan_rows = len(df[df.isnull().any(axis=1)])
        if num_nan_rows > len(df)/2:
            print(f"Skipping {file_name} because it has {num_nan_rows} rows with NaN values")
            continue
        for column in df.columns:
            if is_datetime_column(df[column]):
                df[column] = pd.to_datetime(df[column]).dt.tz_localize(None)

        path_df = os.path.join(kaggle_parquet_path, file_name + '.parquet')
        print(f"Saving {file_name} to {path_df}")
        # save df as a parquet file at path_df
        try:
            df.to_parquet(path_df)
        except Exception as e:
            print(f"An error occurred while writing to parquet: {e}")
            continue
        num_gathered_tables += 1
    # Update status
    last_page += 1
    status['last_page'] = last_page
    with open(status_path, 'w') as f:
        json.dump(status, f, indent=4)