import os
import zipfile
import pandas as pd
import pprint
pp = pprint.PrettyPrinter(indent=4)

# Directory containing the zip files
directory = 'kaggle_tables'

results = []

# Iterate over each file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.zip'):
        zip_path = os.path.join(directory, filename)
        
        # Print the zip file name
        print(f'Zip file: {filename}')
        
        # Print the size of the zip file in bytes
        zip_size = os.path.getsize(zip_path)
        print(f'Size: {zip_size} bytes')
        
        # Unzip the file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(directory)
            csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
        
        if len(csv_files) == 1:
            # Read the CSV file as a dataframe
            csv_path = os.path.join(directory, csv_files[0])
            df = pd.read_csv(csv_path)
            size_per_cell = zip_size / df.size
            print(f'Rows: {len(df)}, Columns: {len(df.columns)}')
            print(f'Size per cell: {size_per_cell} bytes')
            results.append({'file':filename, 'zip_size':zip_size, 'rows':len(df), 'columns':len(df.columns), 'size_per_cell':size_per_cell})

        # Delete the CSV file
        for csv_file in csv_files:
            csv_path = os.path.join(directory, csv_file)
            os.remove(csv_path)

# Sort results by rows
results = sorted(results, key=lambda x: x['rows'], reverse=True)
pp.pprint(results)