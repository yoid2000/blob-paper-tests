import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from check_onehot import find_onehot_encoded_sets

def get_stats(parquet_path, blob_path):

    max_rows = 250000
    max_columns = 50
    stats = []
    info = {'datasets': [], 'stats': {}}
    # Iterate over each file in the directory
    for filename in os.listdir(parquet_path):
        if filename.endswith('.parquet'):
            file_path = os.path.join(parquet_path, filename)
            
            # Read the .parquet file as a dataframe
            df = pd.read_parquet(file_path)
            
            # Get the number of rows and columns
            num_rows = len(df)
            num_columns = len(df.columns)
            # if the number of rows or columns exceeds the maximum, remove this file
            if num_rows > max_rows or num_columns > max_columns:
                os.remove(file_path)
                continue

            # Due to a bug in syndiffix, we need to remove datasets where an illegal filename
            # character is present in the column name
            illegal_chars = ['<','>',':','"','/','\\','|','?','*',]
            found_bad_col = False
            for column in df.columns:
                if any(char in column for char in illegal_chars):
                    found_bad_col = True
                    break
            if found_bad_col:
                os.remove(file_path)
                continue

            # count the number of rows where any column is NaN
            num_nan_rows = len(df[df.isnull().any(axis=1)])
            if num_nan_rows > num_rows/2:
                os.remove(file_path)
                continue

            onehot_sets = find_onehot_encoded_sets(df)
            if len(onehot_sets) > 0:
                print(f"Found one-hot encoded sets in {filename} with {len(df.columns)} columns:")
                print(onehot_sets)
                os.remove(file_path)
                continue

            # determine if any columns are datetime
            has_datetime = False
            col_info = []
            for column in df.columns:
                # get the column type
                col_type = df[column].dtype
                # get the number of distinct values in the column
                num_unique = df[column].nunique()
                # count the number of rows with NaN values for this column
                num_nan = int(df[column].isnull().sum())
                col_info.append({'column': column, 'type': str(col_type), 'num_unique': num_unique, 'num_nan': num_nan})
                if pd.api.types.is_datetime64_any_dtype(df[column]):
                    df[column] = pd.to_datetime(df[column]).dt.tz_localize(None)
                    has_datetime = True
            # Add the stats to the array
            stats.append((num_rows, num_columns, has_datetime))
            # write the possibly-modified dataframe back to the parquet file
            df.to_parquet(file_path)

            # strip off the .parquet extension
            kaggle_dataset = filename[:-8]
            # change '_slash_' to '/'
            kaggle_dataset = kaggle_dataset.replace('_slash_', '/')
            info['datasets'].append({'dataset': kaggle_dataset, 'rows': num_rows, 'columns': num_columns, 'num_nan_rows': num_nan_rows, 'num_non_nan_rows': num_rows - num_nan_rows, 'datetime': has_datetime, 'col_info': col_info})
    
    # Convert the stats to a DataFrame for easier plotting
    df_stats = pd.DataFrame(stats, columns=['Rows', 'Columns', 'HasDatetime'])

    info['stats'] = {
        'num_datasets': int(len(df_stats)),
        'num_with_datetime': int(len(df_stats[df_stats['HasDatetime'] == True])),
        'max_rows': int(df_stats['Rows'].max()),
        'min_rows': int(df_stats['Rows'].min()),
        'max_columns': int(df_stats['Columns'].max()),
        'min_columns': int(df_stats['Columns'].min()),
        'avg_rows': float(df_stats['Rows'].mean()),
        'avg_columns': float(df_stats['Columns'].mean()),
        'stdev_rows': float(df_stats['Rows'].std()),
        'stdev_columns': float(df_stats['Columns'].std()),
    }

    # Save the info to a JSON file
    with open(os.path.join(blob_path, 'datasets_info.json'), 'w') as f:
        json.dump(info, f, indent=4)

    # Plot the stats as a scatterplot
    plt.scatter(df_stats['Rows'], df_stats['Columns'], c=df_stats['HasDatetime'].map({True: 'blue', False: 'red'}))
    plt.xlabel('Rows')
    plt.ylabel('Columns')
    plt.title('Scatterplot of Rows vs Columns in Parquet Files')
    plt.tight_layout()
    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='HasDatetime=True'),
                        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='HasDatetime=False')])
    plt.savefig(os.path.join(blob_path, 'rows_cols.png'))
    plt.close()

    # Plot the stats as a scatterplot, log scale
    plt.scatter(df_stats['Rows'], df_stats['Columns'], c=df_stats['HasDatetime'].map({True: 'blue', False: 'red'}))
    plt.xlabel('Rows')
    plt.ylabel('Columns')
    plt.title('Scatterplot of Rows vs Columns in Parquet Files')
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='HasDatetime=True'),
                        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='HasDatetime=False')])
    plt.savefig(os.path.join(blob_path, 'rows_cols_log.png'))
    plt.close()

blob_code_path = '.'
# check if there is an environment variable called BLOB_TEST_DIR
if 'BLOB_TEST_DIR' in os.environ:
    blob_path = os.environ['BLOB_TEST_DIR']
    blob_code_path = os.environ['BLOB_TEST_CODE']
else:
    blob_path = os.path.join(os.path.dirname(os.getcwd()), 'blob_tests')

print(f"Blob path: {blob_path}")

# make directory 'datasets' if it doesn't exist
datasets_path = os.path.join(blob_path, 'datasets')
kaggle_parquet_path = os.path.join(datasets_path, 'kaggle_parquet')

# Example usage
get_stats(kaggle_parquet_path, blob_path)