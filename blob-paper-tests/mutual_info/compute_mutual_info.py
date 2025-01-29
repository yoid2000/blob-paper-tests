import pandas as pd
import os
import sys
import itertools
import pprint
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.metrics.cluster import entropy
from syndiffix import SyndiffixBlobReader

pp = pprint.PrettyPrinter(indent=4)

def preprocess_column(column):
    if pd.api.types.is_numeric_dtype(column):
        # Treat numeric columns as continuous
        return column
    elif pd.api.types.is_datetime64_any_dtype(column):
        # Treat datetime columns as continuous by converting to numeric
        return column.astype('int64')
    else:
        # Treat other columns as categorical
        le = LabelEncoder()
        return le.fit_transform(column)

def compute_mutual_information(df_in, col1, col2):
    df = df_in[[col1, col2]].dropna()
    col1_processed = preprocess_column(df[col1])
    col2_processed = preprocess_column(df[col2])

    # Discretize continuous columns
    if pd.api.types.is_numeric_dtype(col1_processed):
        if isinstance(col1_processed, pd.Series):
            col1_processed = col1_processed.values
        col1_processed = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform').fit_transform(col1_processed.reshape(-1, 1)).flatten()

    if pd.api.types.is_numeric_dtype(col2_processed):
        if isinstance(col2_processed, pd.Series):
            col2_processed = col2_processed.values
        col2_processed = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform').fit_transform(col2_processed.reshape(-1, 1)).flatten()

    # Compute mutual information
    mi = mutual_info_score(col1_processed, col2_processed)

    # Compute entropy for normalization
    h_col1 = entropy(col1_processed)
    h_col2 = entropy(col2_processed)
    
    # Check for zero entropy to avoid division by zero
    if h_col1 == 0 or h_col2 == 0:
        return 0  # or handle this case as needed
        
    normalized_mi = mi / min(h_col1, h_col2)
    
    return normalized_mi

def check_pairs(file_path, blob_name, blob_dir_path):
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
    results = []
    for col1, col2 in all_combs:
        try:
            df_syn = sbr.read(columns=[col1, col2])
        except Exception as e:
            #traceback.print_exc()
            print(f"Error: {e}")
            print(f"Error: couldn't read {blob_name} '{col1}' '{col2}'")
            pp.pprint(sbr.catalog.catalog.keys())
            return
        mi_syn = compute_mutual_information(df_syn, col1, col2)
        mi_orig = compute_mutual_information(df_orig, col1, col2)
        # get the number of discinct values in col1 and col2 for both df_syn and df_orig
        num_unique_syn_col1 = df_syn[col1].nunique()
        num_unique_syn_col2 = df_syn[col2].nunique()
        num_unique_orig_col1 = df_orig[col1].nunique()
        num_unique_orig_col2 = df_orig[col2].nunique()
        # get the column type for col1 and col2 for both df_syn and df_orig
        col1_type_syn = str(df_syn[col1].dtype)
        col2_type_syn = str(df_syn[col2].dtype)
        col1_type_orig = str(df_orig[col1].dtype)
        col2_type_orig = str(df_orig[col2].dtype)
        results.append({'blob_name':blob_name,
                        'column1':col1,
                        'column2':col2,
                        'mi_syn':mi_syn,
                        'mi_orig':mi_orig,
                        'num_unique_syn_col1':num_unique_syn_col1,
                        'num_unique_syn_col2':num_unique_syn_col2,
                        'num_unique_orig_col1':num_unique_orig_col1,
                        'num_unique_orig_col2':num_unique_orig_col2,
                        'col1_type_syn':col1_type_syn,
                        'col2_type_syn':col2_type_syn,
                        'col1_type_orig':col1_type_orig,
                        'col2_type_orig':col2_type_orig})
    # convert results to a dataframe
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
filenames.sort()
if len(filenames) < job_num+1:
    print(f"Error: job number {job_num} is out of range")
    sys.exit(1)
filename = filenames[job_num]
file_path = os.path.join(kaggle_parquet_path, filename)

blob_name = filename.replace('.parquet', '')
blob_dir_path = os.path.join(sdx_2dim_path, blob_name)
# check if a directory at blob_dir_path already exists
print(f"Checking if {blob_dir_path} exists")
if not os.path.exists(blob_dir_path):
    print(f"Error: {blob_dir_path} does not exist")
    sys.exit(1)
blob_full_path = os.path.join(blob_dir_path, blob_name + '.sdxblob.zip')
if not os.path.exists(blob_full_path):
    print(f"Error {blob_full_path} does not exist")
    sys.exit(1)

df_results = check_pairs(file_path, blob_name, blob_dir_path)
results_path = os.path.join(blob_path, 'results')
os.makedirs(results_path, exist_ok=True)
mi_results_path = os.path.join(results_path, 'mutual_info')
os.makedirs(mi_results_path, exist_ok=True)
results_filename = os.path.join(mi_results_path, f'{blob_name}.parquet')
# write results to a parquet file
df_results.to_parquet(results_filename, index=False)