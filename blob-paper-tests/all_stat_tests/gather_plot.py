import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

test_types = [
    'mutual_information',
    'distance_correlation',
]

def create_merged_dataframe(df):
    """
    Create a new DataFrame where each row is a unique combination of 'test_type', 'blob_name', 'col1', and 'col2'.
    The new DataFrame has columns 'score_orig', 'score_syn', 'elapsed_time_orig', and 'elapsed_time_syn' containing the scores and elapsed times for 'dataset_type=orig' and 'dataset_type=syn'.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame containing the columns 'test_type', 'blob_name', 'col1', 'col2', 'dataset_type', 'score', and 'elapsed_time'.
    
    Returns:
    pd.DataFrame: Merged DataFrame with columns 'test_type', 'blob_name', 'col1', 'col2', 'score_orig', 'score_syn', 'elapsed_time_orig', and 'elapsed_time_syn'.
    """
    # Pivot the DataFrame to separate the scores for 'dataset_type=orig' and 'dataset_type=syn'
    score_pivot_df = df.pivot_table(index=['test_type', 'blob_name', 'col1', 'col2'], columns='dataset_type', values='score')
    
    # Rename the columns for clarity
    score_pivot_df = score_pivot_df.rename(columns={'orig': 'score_orig', 'syn': 'score_syn'})
    
    # Pivot the DataFrame to separate the elapsed times for 'dataset_type=orig' and 'dataset_type=syn'
    elapsed_time_pivot_df = df.pivot_table(index=['test_type', 'blob_name', 'col1', 'col2'], columns='dataset_type', values='elapsed_time')
    
    # Rename the columns for clarity
    elapsed_time_pivot_df = elapsed_time_pivot_df.rename(columns={'orig': 'elapsed_time_orig', 'syn': 'elapsed_time_syn'})
    
    # Merge the score and elapsed time pivot tables
    df_merged = pd.merge(score_pivot_df, elapsed_time_pivot_df, on=['test_type', 'blob_name', 'col1', 'col2'])
    
    # Reset the index to convert the MultiIndex to columns
    df_merged = df_merged.reset_index()
    
    return df_merged

def plot_sorted_scores(df):
    """
    Create 15 subplots organized as 3 rows and 5 columns.
    In each subplot, there are two lines, one for score_orig and one for score_syn.
    The y axis is score. Before plotting, sort the df by score_orig ascending.
    The x axis for each subplot is the row index after sorting, and has no tick labels.
    The first subplot includes all rows (x axis labeled "All").
    The second subplot is blank. The remaining 13 subplots contain the rows for each of the test_types,
    and the x label is the test_type.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame containing the columns 'test_type', 'score_orig', and 'score_syn'.
    """
    # Sort the DataFrame by score_orig ascending
    df_sorted = df.sort_values(by='score_orig').reset_index(drop=True)
    
    # Create the subplots
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    axes = axes.flatten()
    
    # Plot the first subplot with all rows
    axes[0].plot(df_sorted.index, df_sorted['score_syn'], label='score_syn', color='red')
    axes[0].plot(df_sorted.index, df_sorted['score_orig'], label='score_orig', color='blue')
    axes[0].set_title('All')
    axes[0].set_xlabel('All')
    axes[0].set_ylabel('Score')
    axes[0].legend()
    
    # Plot the remaining subplots for each test_type
    test_types = df['test_type'].unique()
    for i, test_type in enumerate(test_types):
        ax = axes[i + 1]
        df_test_type = df_sorted[df_sorted['test_type'] == test_type]
        ax.plot(df_test_type.index, df_test_type['score_syn'], label='score_syn', color='red')
        ax.plot(df_test_type.index, df_test_type['score_orig'], label='score_orig', color='blue')
        ax.set_xlabel(test_type)
        ax.set_ylabel('Score')
        ax.legend()
        ax.set_xticks([])  # Remove x-axis tick labels
    
    # Remove x-axis tick labels for all subplots
    for ax in axes:
        ax.set_xticks([])
    
    plt.tight_layout()
    plt.savefig(os.path.join('real_results', 'sorted_scores.png'))
    plt.savefig(os.path.join('real_results', 'sorted_scores.pdf'))
    plt.close()

def plot_basic_per_test(df):
    ''' Make a seaborn plot with two subplots, arranged horizontally.
        In both subplots, the y axis is the set of test_types
        For each test_type, there is a boxplot
        In the first subplot, the x axis for the boxplots is score
        In the second subplot, the x axis for the boxplots is elapsed_time
    '''
    fig, ax = plt.subplots(1, 3, figsize=(11, 4))
    
    sns.boxplot(x='score_syn', y='test_type', data=df, ax=ax[0], hue='test_type', order=test_types, legend=False)
    sns.boxplot(x='score_diff', y='test_type', data=df, ax=ax[1], hue='test_type', order=test_types, legend=False)
    sns.boxplot(x='elapsed_time_syn', y='test_type', data=df, ax=ax[2], hue='test_type', order=test_types, legend=False)
        
    ax[1].set_yticks([])
    ax[1].set_ylabel('')
    ax[2].set_yticks([])
    ax[2].set_ylabel('')
    ax[0].set_ylabel('')
    ax[2].set_xscale('log')
    # Set the x-axis label for the first subplot
    ax[0].set_xlabel('Score (syn only)')
    ax[1].set_xlabel('Score Diff (orig - syn)')
    ax[2].set_xlabel('Elapsed Time (log scale, syn only)')
    plt.tight_layout()
    plt.savefig(os.path.join('real_results', 'basic_per_test.png'))
    plt.savefig(os.path.join('real_results', 'basic_per_test.pdf'))
    plt.close()

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

results_file_path = os.path.join(results_path, 'all_results.parquet')
if os.path.exists(results_file_path):
    # read file as df
    df = pd.read_parquet(results_file_path)
else:
    # read all parquet files in all_results_path and sub directories
    # and gather the results in a single dataframe
    num_files_read = 0
    df = pd.DataFrame()
    for root, dirs, files in os.walk(all_results_path):
        for file in files:
            if file.endswith('.parquet'):
                file_path = os.path.join(root, file)
                df_temp = pd.read_parquet(file_path)
                df = pd.concat([df, df_temp])
                num_files_read += 1
                if num_files_read % 100 == 0:
                    print(f"Read {num_files_read} files")
    df = df.reset_index(drop=True)
    df.to_parquet(results_file_path)

print(df.head())
print(f"Number of rows: {len(df)}")
print(f"Count of each test type:")
print(df['test_type'].value_counts())
# For each value of test_type, show the count of rows where score > 1.0
df_filtered = df[df['score'] > 1.0]
print(df_filtered.groupby('test_type').size())
# Some distance_correlation tests have score > 1.0, so for now let's just clean them out
df = df[df['score'] <= 1.0]
# Remove all rows where the combination of 'blob_name', 'col1', 'col2', and
# 'dataset_type' does not have 13 rows
df = df.groupby(['blob_name', 'col1', 'col2', 'dataset_type']).filter(lambda x: len(x) == len(test_types))

df_merged = create_merged_dataframe(df)
# make a new column called 'score_diff' that is the difference between score_orig and score_syn
df_merged['score_diff'] = df_merged['score_orig'] - df_merged['score_syn']
print(df_merged.describe())

# Ok, this represents an apples-to-apples comparison of the different stat tests

print(df_merged.columns)

plot_sorted_scores(df_merged)
plot_basic_per_test(df_merged)