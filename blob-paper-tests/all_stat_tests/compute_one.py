import pandas as pd
import os
import sys
import itertools
import pprint
from syndiffix import SyndiffixBlobReader
from common.stat_tests import StatTests

pp = pprint.PrettyPrinter(indent=4)

dataset_types = ['orig', 'syn']
test_types = [
    'mutual_information',
    'distance_correlation',
]
num_tests = len(test_types)

def get_df(sbr, df, cols):
    df_temp = df
    if sbr is not None:
        try:
            df_temp = sbr.read(columns=cols)
        except Exception as e:
            #traceback.print_exc()
            print(f"Error: {e}")
            print(f"Error: couldn't read {blob_name} '{col1}' '{col2}'")
            pp.pprint(sbr.catalog.catalog.keys())
            return None
    return df_temp

def check_pairs(file_path, blob_name, blob_dir_path, test_type, dataset_type):
    print(f"Running {test_type} on {blob_name} with dataset type {dataset_type}")
    # This is the original dataset
    sbr = None
    df = None
    if dataset_type == 'orig':
        df = pd.read_parquet(file_path)
        cols = sorted(list(df.columns))
    else:
        try:
            sbr = SyndiffixBlobReader(blob_name=blob_name,
                                    path_to_dir=blob_dir_path,
                                    force=False)
        except Exception as e:
            print(f"Error: {e}")
            return
        cols = sorted(sbr.col_names_all)
    all_combs = list(itertools.combinations(cols, 2))
    force_zero_cols = []
    for col in cols:
        df_temp = get_df(sbr, df, [col])
        # Check if there is only one value in df_temp[col]
        if len(df_temp[col].unique()) == 1:
            force_zero_cols.append(col)
    results = []
    for col1, col2 in all_combs:
        print(f"Try columns {col1} and {col2}")
        df_temp = get_df(sbr, df, [col1, col2])
        # Get the number of distinct values in col1 and col2
        num_distinct_col1 = len(df_temp[col1].unique())
        num_distinct_col2 = len(df_temp[col2].unique())
        stat_tests = StatTests(df_temp, col1, col2)
        if col1 in force_zero_cols or col2 in force_zero_cols:
            result = {'score': 0.0, 'elapsed_time': None}
            forced = 1
        else:
            result = stat_tests.run_stat_test(test_type)
            forced = 0
        if result is not None:
            results.append({
                'dataset_type': dataset_type,
                'test_type': test_type,
                'blob_name': blob_name,
                'col1': col1,
                'col2': col2,
                'n_dist_col1': num_distinct_col1,
                'n_dist_col2': num_distinct_col2,
                'd_type_col1': str(df_temp[col1].dtype),
                'd_type_col2': str(df_temp[col2].dtype),
                'score': result['score'],
                'forced_zero': forced,
                'elapsed_time': result['elapsed_time']
            })
    df_results = pd.DataFrame(results)
    return df_results

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
kaggle_parquet_path = os.path.join(datasets_path, 'kaggle_parquet')
if not os.path.exists(kaggle_parquet_path):
    print(f"Error: {kaggle_parquet_path} does not exist")
    print("You need to gather the Kaggle datasets first")
    sys.exit(1)
sdx_2dim_path = os.path.join(datasets_path, 'sdx_2dim')
if not os.path.exists(sdx_2dim_path):
    print(f"Error: {sdx_2dim_path} does not exist")
    print("You need to build 2dim syndiffix tables first")
    sys.exit(1)

filenames = os.listdir(kaggle_parquet_path)
# Sort filenames according to the size of the file. This somewhat avoids files
# with more columns (only somewhat)
filenames = sorted(filenames, key=lambda x: os.path.getsize(os.path.join(kaggle_parquet_path, x)))
total_jobs = len(filenames) * num_tests

dataset_type_index = job_num % len(dataset_types)
dataset_type = dataset_types[dataset_type_index]
new_job_num = job_num // len(dataset_types)
filenames_index = new_job_num // num_tests
test_index = new_job_num % num_tests
if filenames_index >= len(filenames) or test_index >= num_tests:
    print(f"Error: bad job number {job_num}: new_job_num {new_job_num}, num_filenames={len(filenames)}, filenames_index={filenames_index}, test_index={test_index}, num_tests={num_tests}")
    sys.exit(1)
filename = filenames[filenames_index]
test_type = test_types[test_index]
file_path = os.path.join(kaggle_parquet_path, filename)

blob_name = filename.replace('.parquet', '')
blob_dir_path = os.path.join(sdx_2dim_path, blob_name)
# check if a directory at blob_dir_path already exists
if not os.path.exists(blob_dir_path):
    print(f"Error: {blob_dir_path} does not exist")
    sys.exit(1)
blob_full_path = os.path.join(blob_dir_path, blob_name + '.sdxblob.zip')
if not os.path.exists(blob_full_path):
    print(f"Error {blob_full_path} does not exist")
    sys.exit(1)

results_path = os.path.join(blob_path, 'results')
os.makedirs(results_path, exist_ok=True)
all_results_path = os.path.join(results_path, 'all_tests')
os.makedirs(all_results_path, exist_ok=True)
results_filename = os.path.join(all_results_path, f'{blob_name}__{test_type}__{dataset_type}.parquet')
# check if results_filename exists, and if so, exit
if os.path.exists(results_filename):
    print(f"Skip job: {results_filename} already exists")
    sys.exit(1)

df_results = check_pairs(file_path, blob_name, blob_dir_path, test_type, dataset_type)
if df_results is None:
    print("Error: check_pairs failed to return a result")
    sys.exit(1)
# write results to a parquet file
df_results.to_parquet(results_filename, index=False)
