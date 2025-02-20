import pandas as pd
import os
import sys
import itertools
import pprint
from syndiffix import SyndiffixBlobReader
from common.stat_tests import StatTests

pp = pprint.PrettyPrinter(indent=4)

def get_syn_dfs(sbr, col1, col2):
    return_dfs = [None, None, None]
    for i, cols in enumerate([[col1], [col2], [col1, col2]]):
        try:
            df_temp = sbr.read(columns=cols)
        except Exception as e:
            #traceback.print_exc()
            print(f"Error: {e}")
            print(f"Error: couldn't read {blob_name} '{col1}' '{col2}'")
            pp.pprint(sbr.catalog.catalog.keys())
            return None
        return_dfs[i] = df_temp
    return return_dfs[2], return_dfs[0], return_dfs[1]

def check_pairs(blob_name, blob_dir_path):
    print(f"Running {blob_name}")
    # This is the original dataset
    sbr = None
    try:
        sbr = SyndiffixBlobReader(blob_name=blob_name,
                                path_to_dir=blob_dir_path,
                                force=False)
    except Exception as e:
        print(f"Error: {e}")
        return
    cols = sorted(sbr.col_names_all)
    all_combs = list(itertools.combinations(cols, 2))
    results = []
    for col1, col2 in all_combs:
        print(f"Try columns {col1} and {col2}")
        try:
            df_temp = sbr.read(columns=[col1,col2])
        except Exception as e:
            #traceback.print_exc()
            print(f"Error: {e}")
            print(f"Error: couldn't read {blob_name} '{col1}' '{col2}'")
            pp.pprint(sbr.catalog.catalog.keys())
            return None
        stat_tests = StatTests(df_temp, col1, col2)
        _ = stat_tests.run_full_measure()
        result = stat_tests.get_full_measure_stats()
        result['col1'] = col1
        result['col2'] = col2
        result['type_col1'] = str(df_temp[col1].dtype)
        result['type_col2'] = str(df_temp[col2].dtype)
        result['dataset'] = blob_name
        results.append(result)
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
if job_num >= len(filenames):
    print(f"Error: job_num {job_num} is too large")
    sys.exit(1)

filename = filenames[job_num]

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
all_results_path = os.path.join(results_path, 'full_measure')
os.makedirs(all_results_path, exist_ok=True)
results_filename = os.path.join(all_results_path, f'{blob_name}.parquet')
# check if results_filename exists, and if so, exit
if os.path.exists(results_filename):
    print(f"Skip job: {results_filename} already exists")
    sys.exit(1)

df_results = check_pairs(blob_name, blob_dir_path)
if df_results is None:
    print("Error: check_pairs failed to return a result")
    sys.exit(1)
# write results to a parquet file
df_results.to_parquet(results_filename, index=False)
