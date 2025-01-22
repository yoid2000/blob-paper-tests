import os
import pandas as pd

def process_parquet_files(kaggle_parquet_path_in, kaggle_parquet_path_out):
    # Helper function to check if a column can be converted to datetime
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

    # Iterate over each file in the directory
    for filename in os.listdir(kaggle_parquet_path_in):
        if filename.endswith('.parquet'):
            file_path_in = os.path.join(kaggle_parquet_path_in, filename)
            file_path_out = os.path.join(kaggle_parquet_path_out, filename)
            
            # Read the .parquet file as a dataframe
            df = pd.read_parquet(file_path_in)
            
            # Check each column to see if it could be a datetime column
            for column in df.columns:
                if is_datetime_column(df[column]):
                    print("###################################################################")
                    print(f'Converting {column} to datetime in {filename}')
                    df[column] = pd.to_datetime(df[column])

            # Write the modified dataframe back to the .parquet file
            df.to_parquet(file_path_out)
            print(f'Processed {filename}')

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
kaggle_parquet_path_in = os.path.join(datasets_path, 'kaggle_parquet')
kaggle_parquet_path_out = os.path.join(datasets_path, 'kaggle_parquet_out')
os.makedirs(kaggle_parquet_path_out, exist_ok=True)

# Example usage
process_parquet_files(kaggle_parquet_path_in, kaggle_parquet_path_out)