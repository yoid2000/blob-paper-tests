import os
import sys
import itertools
import traceback
import shutil
import subprocess
import stat
import pandas as pd
from syndiffix import SyndiffixBlobReader
import pprint
pp = pprint.PrettyPrinter(indent=4)

def check_file(filename):
    file_path = os.path.join(kaggle_parquet_path, filename)
    blob_name = filename.replace('.parquet', '')
    blob_dir_path = os.path.join(sdx_2dim_path, blob_name)
    blob_temp_path = os.path.join(blob_dir_path, '.sdx_blob_' + blob_name)
    if os.path.exists(blob_temp_path):
        shutil.rmtree(blob_temp_path)
    if not os.path.exists(blob_dir_path):
        print(f"Error: {blob_dir_path} does not exist")
        return
    blob_full_path = os.path.join(blob_dir_path, blob_name + '.sdxblob.zip')
    if not os.path.exists(blob_full_path):
        print(f"Error: {blob_full_path} does not exist")
        return
    # This is the original dataset
    df_orig = pd.read_parquet(file_path)
    try:
        sbr = SyndiffixBlobReader(blob_name=blob_name,
                                  path_to_dir=blob_dir_path,
                                  force=True)
    except Exception as e:
        print(f"Error: {e}")
        return
    cols_org = sorted(list(df_orig.columns))
    cols_sbr = sorted(sbr.col_names_all)
    if cols_org != cols_sbr:
        print(f"Error: {blob_name} columns do not match")
        print(cols_org)
        print(cols_sbr)
        return
    all_combs = list(itertools.combinations(cols_org, 2))
    for col1, col2 in all_combs:
        try:
            _ = sbr.read(columns=[col1, col2])
        except Exception as e:
            #traceback.print_exc()
            print(f"Error: {e}")
            print(f"Error: couldn't read {blob_name} '{col1}' '{col2}'")
            pp.pprint(sbr.catalog.catalog.keys())
            quit()

#do_these_checks = []
do_these_checks = ['ahmedwadood_slash_adulttest.parquet']

# check if there is an environment variable called BLOB_TEST_DIR
if 'BLOB_TEST_DIR' in os.environ:
    blob_path = os.environ['BLOB_TEST_DIR']
    blob_code_path = os.environ['BLOB_TEST_CODE']
else:
    blob_path = os.path.join(os.path.dirname(os.getcwd()), 'blob_tests')

print(f"Blob path: {blob_path}")

datasets_path = os.path.join(blob_path, 'datasets')
kaggle_parquet_path = os.path.join(datasets_path, 'kaggle_parquet')
# make sure kaggle_parquet_path exists
if not os.path.exists(kaggle_parquet_path):
    print(f"Error: {kaggle_parquet_path} does not exist")
    print("You need to gather the Kaggle datasets first")
    sys.exit(1)
sdx_2dim_path = os.path.join(datasets_path, 'sdx_2dim')
if not os.path.exists(sdx_2dim_path):
    print(f"Error: {sdx_2dim_path} does not exist")
    print("You need to run build_tables.py first")
    sys.exit(1)

if len(do_these_checks) > 0:
    for this_check in do_these_checks:
        check_file(this_check)
    sys.exit(0)

filenames = os.listdir(kaggle_parquet_path)
for filename in filenames:
    check_file(filename)