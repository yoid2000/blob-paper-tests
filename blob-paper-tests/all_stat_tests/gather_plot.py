import os
import numpy as np
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
    The new DataFrame has columns for each specified column in the input DataFrame, separated by 'dataset_type=orig' and 'dataset_type=syn'.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame containing the columns 'test_type', 'blob_name', 'col1', 'col2', 'dataset_type', and the specified columns.
    
    Returns:
    pd.DataFrame: Merged DataFrame with columns for each specified column, separated by 'dataset_type=orig' and 'dataset_type=syn'.
    """
    # Partition the DataFrame into two DataFrames based on dataset_type
    df_orig = df[df['dataset_type'] == 'orig'].drop(columns=['dataset_type']).add_suffix('_orig')
    df_syn = df[df['dataset_type'] == 'syn'].drop(columns=['dataset_type']).add_suffix('_syn')
    
    # Join the two DataFrames on the specified columns
    df_merged = df_orig.merge(df_syn, left_on=['test_type_orig', 'blob_name_orig', 'col1_orig', 'col2_orig'],
                              right_on=['test_type_syn', 'blob_name_syn', 'col1_syn', 'col2_syn'])
    
    # Drop duplicate columns from the join
    df_merged = df_merged.drop(columns=['test_type_syn', 'blob_name_syn', 'col1_syn', 'col2_syn'])
    
    # Rename columns to remove suffixes from the join keys
    df_merged = df_merged.rename(columns={
        'test_type_orig': 'test_type',
        'blob_name_orig': 'blob_name',
        'col1_orig': 'col1',
        'col2_orig': 'col2'
    })
    
    return df_merged

def plot_sorted_scores(df):
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

def plot_mi_vs_dc_orig(df):
    df_mi = df[df['test_type'] == 'mutual_information']
    df_dc = df[df['test_type'] == 'distance_correlation']
    df_mi_dc = df_mi.merge(df_dc, on=['blob_name', 'col1', 'col2'])
    
    # Calculate the differences between score_orig_y and score_orig_x
    df_mi_dc['score_diff'] = df_mi_dc['score_orig_y'] - df_mi_dc['score_orig_x']
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # First subplot: scatter plot
    ax[0].scatter(df_mi_dc['score_orig_x'], df_mi_dc['score_orig_y'])
    ax[0].set_xlabel('MI score orig')
    ax[0].set_ylabel('DC score orig')
    ax[0].set_title('Scatter Plot of MI vs DC Scores')
    
    # Second subplot: boxplot of score differences
    ax[1].boxplot(df_mi_dc['score_diff'])
    ax[1].set_ylabel('Score Difference (DC - MI)')
    ax[1].set_title('Boxplot of Score Differences')
    
    plt.tight_layout()
    plt.savefig(os.path.join('real_results', 'mi_vs_dc_orig.png'))
    plt.savefig(os.path.join('real_results', 'mi_vs_dc_orig.pdf'))
    plt.close()

def plot_orig_syn_diff(df):
    # Calculate the required differences
    df['max_score_diff'] = df[['score_orig_mi', 'score_orig_dc']].max(axis=1) - df[['score_syn_mi', 'score_syn_dc']].max(axis=1)
    df['min_score_diff'] = df[['score_orig_mi', 'score_orig_dc']].min(axis=1) - df[['score_syn_mi', 'score_syn_dc']].min(axis=1)
    df['score_diff_mi_dc_orig'] = df['score_orig_mi'] - df['score_orig_dc']
    df['score_diff_mi_dc_syn'] = df['score_syn_mi'] - df['score_syn_dc']
    
    # Sort the differences
    max_score_diff_sorted = df['max_score_diff'].sort_values().values
    min_score_diff_sorted = df['min_score_diff'].sort_values().values
    score_diff_mi_sorted = df['score_diff_mi'].sort_values().values
    score_diff_dc_sorted = df['score_diff_dc'].sort_values().values
    score_diff_mi_dc_orig_sorted = df['score_diff_mi_dc_orig'].sort_values().values
    score_diff_mi_dc_syn_sorted = df['score_diff_mi_dc_syn'].sort_values().values
    
    # Calculate absolute values and set minimum to 0.001
    max_score_diff_abs = np.maximum(np.abs(max_score_diff_sorted), 0.001)
    min_score_diff_abs = np.maximum(np.abs(min_score_diff_sorted), 0.001)
    score_diff_mi_abs = np.maximum(np.abs(score_diff_mi_sorted), 0.001)
    score_diff_dc_abs = np.maximum(np.abs(score_diff_dc_sorted), 0.001)
    score_diff_mi_dc_orig_abs = np.maximum(np.abs(score_diff_mi_dc_orig_sorted), 0.001)
    score_diff_mi_dc_syn_abs = np.maximum(np.abs(score_diff_mi_dc_syn_sorted), 0.001)
    
    # Create the plot
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    
    # First subplot: line plots with dots at the first and last data points
    lines = [
        (max_score_diff_sorted, 'Max(MI,DC) Orig-Syn'),
        (min_score_diff_sorted, 'Min(MI,DC) Orig-Syn'),
        (score_diff_mi_sorted, 'MI Orig-Syn'),
        (score_diff_dc_sorted, 'DC Orig-Syn'),
        (score_diff_mi_dc_orig_sorted, 'MI-DC Orig'),
        (score_diff_mi_dc_syn_sorted, 'MI-DC Syn')
    ]
    
    # Third subplot: boxplots
    ax[0].boxplot([score_diff_mi_dc_syn_sorted, score_diff_mi_dc_orig_sorted, score_diff_dc_sorted, score_diff_mi_sorted, min_score_diff_sorted, max_score_diff_sorted], vert=False)
    ax[0].set_yticklabels(['MI-DC Syn', 'MI-DC Orig', 'DC Orig-Syn', 'MI Orig-Syn', 'Min(MI,DC) Orig-Syn', 'Max(MI,DC) Orig-Syn'])
    
    
    for line_data, label in lines:
        line, = ax[1].plot(line_data, label=label)
        ax[1].plot(0, line_data[0], 'o', color=line.get_color())  # First data point
        ax[1].plot(len(line_data) - 1, line_data[-1], 'o', color=line.get_color())  # Last data point

    ax[1].set_xlabel('')
    ax[1].set_xticks([])
    ax[1].legend()
    
    # Second subplot: line plots with absolute values and log scale
    abs_lines = [
        (max_score_diff_abs, 'Max(MI,DC) Orig-Syn'),
        (min_score_diff_abs, 'Min(MI,DC) Orig-Syn'),
        (score_diff_mi_abs, 'MI Orig-Syn'),
        (score_diff_dc_abs, 'DC Orig-Syn'),
        (score_diff_mi_dc_orig_abs, 'MI-DC Orig'),
        (score_diff_mi_dc_syn_abs, 'MI-DC Syn')
    ]
    
    for line_data, label in abs_lines:
        line, = ax[2].plot(line_data, label=label)
        ax[2].plot(0, line_data[0], 'o', color=line.get_color())  # First data point
        ax[2].plot(len(line_data) - 1, line_data[-1], 'o', color=line.get_color())  # Last data point

    ax[2].set_yscale('log')
    ax[2].set_xlabel('')
    ax[2].set_xticks([])

    plt.tight_layout()
    plt.savefig(os.path.join('real_results', 'orig_syn_diff.png'))
    plt.savefig(os.path.join('real_results', 'orig_syn_diff.pdf'))
    plt.close()

def plot_basic_per_test(df):
    fig, ax = plt.subplots(3, 1, figsize=(4, 8))
    
    sns.boxplot(x='score_syn', y='test_type', data=df, ax=ax[0], hue='test_type', order=test_types, legend=False)
    sns.boxplot(x='score_diff', y='test_type', data=df, ax=ax[1], hue='test_type', order=test_types, legend=False)
    sns.boxplot(x='elapsed_time_syn', y='test_type', data=df, ax=ax[2], hue='test_type', order=test_types, legend=False)
    
    # Custom y-tick labels
    y_tick_labels = {
        'mutual_information': 'Mutual \n Information',
        'distance_correlation': 'Distance \n Correlation'
    }
    
    for i in range(3):
        # Set y-ticks explicitly
        ax[i].set_yticks([0, 1])
        # Set custom y-tick labels
        ax[i].set_yticklabels([y_tick_labels.get(label.get_text(), label.get_text()) for label in ax[i].get_yticklabels()])
    
    ax[0].set_ylabel('')
    ax[1].set_ylabel('')
    ax[2].set_ylabel('')
    ax[2].set_xscale('log')
    ax[0].set_xlabel('Score (syn only)')
    ax[1].set_xlabel('Score Diff (orig - syn)')
    ax[2].set_xlabel('Elapsed Time (log scale, syn only)')
    plt.tight_layout()
    plt.savefig(os.path.join('real_results', 'basic_per_test.png'))
    plt.savefig(os.path.join('real_results', 'basic_per_test.pdf'))
    plt.close()

def plot_by_num_distinct(df):
    # Define the ranges
    ranges = [[0, 10], [10, 20], [20, 50], [50, 100], [100, 200], [200, 100000]]
    range_labels = ['0-10', '10-20', '20-50', '50-100', '100-200', '200-100000']
    
    # Create a new column for range labels
    def get_range_label(row):
        for r, label in zip(ranges, range_labels):
            if r[0] <= row['n_dist_col1'] <= r[1] and r[0] <= row['n_dist_col2'] <= r[1]:
                return label
        return None
    
    df['range_label'] = df.apply(get_range_label, axis=1)
    
    # Filter out rows with no range label
    df = df[df['range_label'].notna()]
    
    # Create the plot
    fig, ax = plt.subplots(2, 2, figsize=(7, 5))
    
    sns.boxplot(x='score', y='range_label', data=df[df['test_type'] == 'distance_correlation'], ax=ax[0, 0], order=range_labels)
    ax[0, 0].set_xlabel('Score (Distance Correlation, No Text)')
    ax[0, 0].set_ylabel('')
    ax[0, 0].set_xlim(-0.05, 1.05)
    
    df_neither_object = df[(df['test_type'] == 'mutual_information') & (df['d_type_col1'] != 'object') & (df['d_type_col2'] != 'object')]
    sns.boxplot(x='score', y='range_label', data=df_neither_object, ax=ax[0, 1], order=range_labels)
    ax[0, 1].set_xlabel('Score (Mutual Information, No Text)')
    ax[0, 1].set_ylabel('')
    ax[0, 1].set_yticks([])
    ax[0, 1].set_xlim(-0.05, 1.05)
    
    df_both_object = df[(df['test_type'] == 'mutual_information') & (df['d_type_col1'] == 'object') & (df['d_type_col2'] == 'object')]
    sns.boxplot(x='score', y='range_label', data=df_both_object, ax=ax[1, 0], order=range_labels)
    ax[1, 0].set_xlabel('Score (Mutual Information, Both Text)')
    ax[1, 0].set_ylabel('')
    ax[1, 0].set_xlim(-0.05, 1.05)
    
    df_one_object = df[(df['test_type'] == 'mutual_information') & ((df['d_type_col1'] == 'object') | (df['d_type_col2'] == 'object')) & ~((df['d_type_col1'] == 'object') & (df['d_type_col2'] == 'object'))]
    sns.boxplot(x='score', y='range_label', data=df_one_object, ax=ax[1, 1], order=range_labels)
    ax[1, 1].set_xlabel('Score (Mutual Information, One Text)')
    ax[1, 1].set_ylabel('')
    ax[1, 1].set_yticks([])
    ax[1, 1].set_xlim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join('real_results', 'by_num_distinct.png'))
    plt.savefig(os.path.join('real_results', 'by_num_distinct.pdf'))
    plt.close()

def split_and_join_on_test_type(df):
    """
    Split the DataFrame based on test_type, then join the resulting DataFrames on blob_name, col1, and col2.
    The remaining columns should be appended with '_mi' if mutual_information, and with '_dc' if distance_correlation.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame containing the specified columns.
    
    Returns:
    pd.DataFrame: Joined DataFrame with columns appended with appropriate suffixes.
    """
    # Split the DataFrame based on test_type
    df_mi = df[df['test_type'] == 'mutual_information'].drop(columns=['test_type']).add_suffix('_mi')
    df_dc = df[df['test_type'] == 'distance_correlation'].drop(columns=['test_type']).add_suffix('_dc')
    
    # Join the DataFrames on blob_name, col1, and col2
    df_joined = df_mi.merge(df_dc, left_on=['blob_name_mi', 'col1_mi', 'col2_mi'],
                            right_on=['blob_name_dc', 'col1_dc', 'col2_dc'])
    
    # Drop duplicate columns from the join
    df_joined = df_joined.drop(columns=['blob_name_dc', 'col1_dc', 'col2_dc'])
    
    # Rename columns to remove suffixes from the join keys
    df_joined = df_joined.rename(columns={
        'blob_name_mi': 'blob_name',
        'col1_mi': 'col1',
        'col2_mi': 'col2'
    })
    return df_joined


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
merged_results_file_path = os.path.join(results_path, 'all_results_merged.parquet')
merged_mi_dc_results_file_path = os.path.join(results_path, 'all_results_merged_mi_dc.parquet')
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
print("Number of rows per test_type where score > 1.0")
print(df_filtered.groupby('test_type').size())
# Some distance_correlation tests have score > 1.0, so for now let's just clean them out
if len(df_filtered) > 0:
    print(f"Removing {len(df_filtered)} rows with score > 1.0")
    df = df[df['score'] <= 1.0]
# Remove all rows where the combination of 'blob_name', 'col1', 'col2', and
# 'dataset_type' does not have both measures
print(f"Prior to removing rows that do not have both measures there are {len(df)} rows")
df_both = df.groupby(['blob_name', 'col1', 'col2', 'dataset_type']).filter(lambda x: len(x) == len(test_types))
print(f"After removing rows there are {len(df_both)} rows")
# Add a new column 'cat'. Set cat to 0 if score < 0.1, to 1 if score between 0.1 and 0.25, and 2 otherwise
df_both['cat'] = pd.cut(df_both['score'], bins=[-1, 0.1, 0.25, 1], labels=[0, 1, 2])

df_merged = create_merged_dataframe(df_both)
# make a new column called 'score_diff' that is the difference between score_orig and score_syn
df_merged['score_diff'] = df_merged['score_orig'] - df_merged['score_syn']
df_merged['distinct_diff_col1'] = df_merged['n_dist_col1_orig'] - df_merged['n_dist_col1_syn']
df_merged['distinct_diff_col2'] = df_merged['n_dist_col2_orig'] - df_merged['n_dist_col2_syn']
pd.set_option('display.max_columns', None)
for column in df_merged.columns:
    print(df_merged[column].describe())

if False:
    print("Rows with largest distinct_diff_col1:")
    print(df_merged.nlargest(10, 'distinct_diff_col1'))
    print("Rows with smallest distinct_diff_col1:")
    print(df_merged.nsmallest(10, 'distinct_diff_col1'))

    print("Rows with largest score_diff:")
    print(df_merged.nlargest(10, 'score_diff'))
    print("Rows with smallest score_diff:")
    print(df_merged.nsmallest(10, 'score_diff'))

df_merged.to_parquet(merged_results_file_path)

df_merged_mi_dc = split_and_join_on_test_type(df_merged)
df_merged_mi_dc.to_parquet(merged_mi_dc_results_file_path)
df_merged_mi_dc['max_score_syn'] = df_merged_mi_dc[['score_syn_mi', 'score_syn_dc']].max(axis=1)
df_merged_mi_dc['max_score_orig'] = df_merged_mi_dc[['score_orig_mi', 'score_orig_dc']].max(axis=1)
df_merged_mi_dc['min_score_syn'] = df_merged_mi_dc[['score_syn_mi', 'score_syn_dc']].min(axis=1)
df_merged_mi_dc['min_score_orig'] = df_merged_mi_dc[['score_orig_mi', 'score_orig_dc']].min(axis=1)

# in df_merged_mi_dc, count the number of rows for every combination of columns cat_orig_mi, cat_syn_mi, cat_orig_dc, and cat_syn_dc
pd.set_option('display.max_rows', 100)
print("Count of rows for every combination of cat_orig_mi and cat_syn_mi:")
print(df_merged_mi_dc.groupby(['cat_orig_mi', 'cat_syn_mi']).size())
print("Count of rows for every combination of cat_orig_dc and cat_syn_dc:")
print(df_merged_mi_dc.groupby(['cat_orig_dc', 'cat_syn_dc']).size())


print("Initial collected data columns:")
print(df.columns)
print("Columns after merge:")
print(df_merged.columns)
print("Columns after MI/DC split and join:")
print(df_merged_mi_dc.columns)

plot_by_num_distinct(df)
plot_basic_per_test(df_merged)
plot_orig_syn_diff(df_merged_mi_dc)
plot_mi_vs_dc_orig(df_merged)
plot_sorted_scores(df_merged)