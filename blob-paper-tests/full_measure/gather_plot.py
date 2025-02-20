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

def plot_reverse_scatter(df):
    ''' Make a scatter plot and a boxplot. The y axis is spearman. The y axis consists of two columns,
        reverse_col1 in blue, and reverse_col2 in red. The second subplot contains two horizontally oriented
        boxplots, one for reverse_col1 values and one for reverse_col2 values.
    '''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))

    # Scatter plot
    sns.scatterplot(data=df, y='spearman', x='reverse_spearman_col1', ax=ax1, color='blue', alpha=0.4)
    sns.scatterplot(data=df, y='spearman', x='reverse_spearman_col2', ax=ax1, color='red', alpha=0.4)
    # Make a legend for the two colors
    ax1.legend(['col1', 'col2'])
    ax1.set_ylabel('Spearman Rank')
    ax1.set_xlabel('Reverse Spearman Rank')

    # Prepare data for boxplots
    boxplot_data = pd.DataFrame({
        'value': pd.concat([df['reverse_col1'], df['reverse_col2']], ignore_index=True),
        'variable': ['reverse_col1'] * len(df) + ['reverse_col2'] * len(df)
    })

    # Boxplots
    sns.boxplot(data=boxplot_data, x='value', y='variable', ax=ax2, palette=['blue', 'red'], orient='h')
    ax2.set_xlabel('Spearman - Reversed')
    ax2.set_ylabel('')
    ax2.set_yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join('real_results', 'reverse.png'))
    plt.savefig(os.path.join('real_results', 'reverse.pdf'))
    plt.close()


print(df.head())
print(df.shape)
print(df['spearman'].describe())
df['reverse_col1'] = df['spearman'] - df['reverse_spearman_col1']
print(df['reverse_col1'].describe())
df['reverse_col2'] = df['spearman'] - df['reverse_spearman_col2']
print(df['reverse_col2'].describe())

p_cols = ['spearman', 'reverse_spearman_col1', 'reverse_col1', 'col1', 'col2', 'dataset']
# Print columns p_cols from the the 5 rows where reverse_col1 is the largest
print(df.nsmallest(5, 'reverse_col1')[p_cols].to_string())
      

''' When there are relatively few distinct numeric values and a few of them dominate, then
    it is pretty easy for the reverse spearman thing to give a false signal in either
    direction, just because a few dominant values accidently happen to line up. But in these
    cases, we should be getting a high chi_square, so we can use that instead of spearman.
'''

plot_reverse_scatter(df)