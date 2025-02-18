import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

test_types = [
    'mutual_information',
    'distance_correlation',
]

def compute_order_error(mi_id_list, dc_id_list):
    # Compute the order error
    errors = []
    for mi_id in mi_id_list:
        mi_index = mi_id_list.index(mi_id)
        dc_index = dc_id_list.index(mi_id)
        errors.append(abs(mi_index - dc_index))
    return errors

def plot_order_error(df, measure_type):
    median_errors = []
    avg_errors = []
    norm_median_errors = []
    norm_avg_errors = []
    for index, row in df.iterrows():
        # Create dataframes from the lists
        mi_df = pd.DataFrame({
            'value': row[f'{measure_type}_syn_mi_list'],
            'id': row[f'{measure_type}_syn_mi_id_list']
        })
        dc_df = pd.DataFrame({
            'value': row[f'{measure_type}_syn_dc_list'],
            'id': row[f'{measure_type}_syn_dc_id_list']
        })
        
        # Sort the dataframes
        mi_df = mi_df.sort_values(by=['value', 'id'], ascending=[False, True])
        dc_df = dc_df.sort_values(by=['value', 'id'], ascending=[False, True])
        
        # Compare the resulting id lists
        mi_id_list = mi_df['id'].tolist()
        dc_id_list = dc_df['id'].tolist()
        
        # Compute the order error
        errors = compute_order_error(mi_id_list, dc_id_list)
        avg_errors.append(np.mean(errors))
        median_errors.append(np.median(errors))
        norm_avg_errors.append(np.mean(errors) / len(errors))
        norm_median_errors.append(np.median(errors) / len(errors))
    
    # Create a DataFrame for both the avg_error and median_errors
    error_df = pd.DataFrame({
        'Average': avg_errors,
        'Median': median_errors
    })
    norm_error_df = pd.DataFrame({
        'Average': norm_avg_errors,
        'Median': norm_median_errors
    })
    # Create a plot with two subplots, one for error_df and one for norm_error_df
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    # Plot error_df as two boxplots, horizontally oriented in the first subplot
    sns.boxplot(data=error_df, orient='h', ax=ax[0])
    # set x axis to log scale
    ax[0].set_xscale('log')
    ax[0].set_xlabel(f'{measure_type} Order Error MI versus DC (log scale)')
    # Plot norm_error_df as two boxplots, horizontally oriented in the second subplot
    sns.boxplot(data=norm_error_df, orient='h', ax=ax[1])
    ax[1].set_xlabel(f'Normalized {measure_type} Order Error MI versus DC')
    plt.tight_layout()
    plt.savefig(os.path.join('real_results', f'{measure_type}_order_error_boxplot.png'))
    plt.savefig(os.path.join('real_results', f'{measure_type}_order_error_boxplot.pdf'))
    plt.close()

def mean_absolute_error(list1, list2):
    return np.mean(np.abs(np.array(list1) - np.array(list2)))

def mismatch_fraction(list1, list2):
    return np.mean(np.array(list1) != np.array(list2))

def plot_cat_compare(df):
    # Compute the error between the two lists for each row
    mae_errors = []
    mismatch_fractions = []
    for index, row in df.iterrows():
        mae_error = mean_absolute_error(row['cat_syn_mi_list'], row['cat_syn_dc_list'])
        mismatch_fraction_value = mismatch_fraction(row['cat_syn_mi_list'], row['cat_syn_dc_list'])
        if False and mismatch_fraction_value > 0.95:
            # print the blob_name, col1, col2, 'cat_syn_mi_list, and 'cat_syn_dc_list'
            print(f"{row['count']}:{row['blob_name']}\n{row['cat_syn_mi_list']}\n{row['cat_syn_dc_list']}")
        mae_errors.append(mae_error)
        mismatch_fractions.append(mismatch_fraction_value)
    
    # Create DataFrames for the errors and mismatch fractions
    mae_error_df = pd.DataFrame({'MAE': mae_errors})
    mismatch_fraction_df = pd.DataFrame({'Mismatch Fraction': mismatch_fractions})
    
    # Create the plot with 2 subplots
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot the MAE errors as a box plot
    sns.boxplot(data=mae_error_df, y='MAE', ax=ax[0])
    ax[0].set_title('Mean Absolute Error')
    ax[0].set_ylabel('MAE')
    
    # Plot the mismatch fractions as a box plot
    sns.boxplot(data=mismatch_fraction_df, y='Mismatch Fraction', ax=ax[1])
    ax[1].set_title('Mismatch Fraction')
    ax[1].set_ylabel('Mismatch Fraction')
    
    plt.tight_layout()
    plt.savefig(os.path.join('real_results', 'cat_compare_boxplot.png'))
    plt.savefig(os.path.join('real_results', 'cat_compare_boxplot.pdf'))
    plt.close()

def set_id(df):
    # Create a mapping for blob_name to blob_id
    blob_name_to_id = {name: idx + 1 for idx, name in enumerate(df['blob_name'].unique())}
    df['blob_id'] = df['blob_name'].map(blob_name_to_id)
    
    # Create a mapping for (col1, col2) pairs to col_pair_id
    col_pair_to_id = {pair: idx + 1 for idx, pair in enumerate(df[['col1', 'col2']].drop_duplicates().apply(tuple, axis=1))}
    df['col_pair_id'] = df.apply(lambda row: col_pair_to_id[(row['col1'], row['col2'])], axis=1)
    
    # Create the blob_col_id column
    df['blob_col_id'] = df['blob_id'].astype(str) + '_' + df['col_pair_id'].astype(str)
    return df

import pandas as pd

def make_listed_data(df):
    # Define the columns to be processed
    columns_to_process = [
        'score_orig_mi', 'score_orig_dc', 'cat_orig_mi', 'cat_orig_dc',
        'score_syn_mi', 'score_syn_dc', 'cat_syn_mi', 'cat_syn_dc'
    ]
    
    new_rows = []
    grouped = df.groupby('blob_name')
    for blob_name, group in grouped:
        new_row = {'blob_name': blob_name}
        for column in columns_to_process:
            # Create a temporary DataFrame with the column and blob_col_id
            temp_df = group[[column, 'blob_col_id']].copy()
            # Sort the temporary DataFrame by the column values from largest to smallest
            temp_df = temp_df.sort_values(by=column, ascending=False)
            # Generate the lists for the sorted values and blob_col_id
            sorted_values = temp_df[column].tolist()
            sorted_ids = temp_df['blob_col_id'].tolist()
            new_row[f'{column}_list'] = sorted_values
            new_row[f'{column}_id_list'] = sorted_ids
        new_row['count'] = len(new_row['score_orig_mi_list'])
        new_rows.append(new_row)
    df_new = pd.DataFrame(new_rows)
    return df_new


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
    df_merged = df_orig.merge(df_syn, left_on=['test_type_orig', 'blob_name_orig', 'col1_orig', 'col2_orig', 'blob_id_orig', 'col_pair_id_orig', 'blob_col_id_orig'],
                              right_on=['test_type_syn', 'blob_name_syn', 'col1_syn', 'col2_syn', 'blob_id_syn', 'col_pair_id_syn', 'blob_col_id_syn'])
    
    # Drop duplicate columns from the join
    df_merged = df_merged.drop(columns=['test_type_syn', 'blob_name_syn', 'col1_syn', 'col2_syn', 'blob_id_syn', 'col_pair_id_syn', 'blob_col_id_syn'])
    
    # Rename columns to remove suffixes from the join keys
    df_merged = df_merged.rename(columns={
        'test_type_orig': 'test_type',
        'blob_name_orig': 'blob_name',
        'col1_orig': 'col1',
        'col2_orig': 'col2',
        'blob_id_orig': 'blob_id',
        'col_pair_id_orig': 'col_pair_id',
        'blob_col_id_orig': 'blob_col_id'
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

    lines = {
        'Max(MI,DC) Orig-Syn': {'sort': max_score_diff_sorted, 'abs': max_score_diff_abs}, 
        'Min(MI,DC) Orig-Syn': {'sort': min_score_diff_sorted, 'abs': min_score_diff_abs}, 
        'MI Orig-Syn': {'sort': score_diff_mi_sorted, 'abs': score_diff_mi_abs}, 
        'DC Orig-Syn': {'sort': score_diff_dc_sorted, 'abs': score_diff_dc_abs}, 
        'MI-DC Orig': {'sort': score_diff_mi_dc_orig_sorted, 'abs': score_diff_mi_dc_orig_abs}, 
        'MI-DC Syn': {'sort': score_diff_mi_dc_syn_sorted, 'abs': score_diff_mi_dc_syn_abs}, 
    }
    
    # Create the plot
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    
    # First subplot: boxplots
    ax[0].boxplot([score_diff_mi_dc_syn_sorted, score_diff_mi_dc_orig_sorted, score_diff_dc_sorted, score_diff_mi_sorted, min_score_diff_sorted, max_score_diff_sorted], vert=False)
    ax[0].set_yticklabels(['MI-DC Syn', 'MI-DC Orig', 'DC Orig-Syn', 'MI Orig-Syn', 'Min(MI,DC) Orig-Syn', 'Max(MI,DC) Orig-Syn'])
    
    # Second subplot, lines with dots at the edges
    for label, dat in lines.items():
        line, = ax[1].plot(dat['sort'], label=label)
        ax[1].plot(0, dat['sort'][0], 'o', color=line.get_color())  # First data point
        ax[1].plot(len(dat['sort']) - 1, dat['sort'][-1], 'o', color=line.get_color())  # Last data point
    ax[1].set_xlabel('')
    ax[1].set_xticks([])
    ax[1].legend()
    
    for label, dat in lines.items():
        line, = ax[2].plot(dat['abs'], label=label)
        ax[2].plot(0, dat['abs'][0], 'o', color=line.get_color())  # First data point
        ax[2].plot(len(dat['abs']) - 1, dat['abs'][-1], 'o', color=line.get_color())  # Last data point
    ax[2].set_yscale('log')
    ax[2].set_xlabel('')
    ax[2].set_xticks([])

    plt.tight_layout()
    plt.savefig(os.path.join('real_results', 'all_diff.png'))
    plt.savefig(os.path.join('real_results', 'all_diff.pdf'))
    plt.close()

    def run_double_plot(use_labels, file_name):
        fig, ax = plt.subplots(2, figsize=(3, 4))
        for label in use_labels:
            line, = ax[0].plot(lines[label]['sort'], label=label)
            ax[0].plot(0, lines[label]['sort'][0], 'o', color=line.get_color())  # First data point
            ax[0].plot(len(lines[label]['sort']) - 1, lines[label]['sort'][-1], 'o', color=line.get_color())  # Last data point
        ax[0].set_xlabel('Sorted by score difference')
        ax[0].set_xticks([])
        ax[0].legend()
        
        for label in use_labels:
            line, = ax[1].plot(lines[label]['abs'], label=label)
            ax[1].plot(0, lines[label]['abs'][0], 'o', color=line.get_color())  # First data point
            ax[1].plot(len(lines[label]['abs']) - 1, lines[label]['abs'][-1], 'o', color=line.get_color())  # Last data point
        ax[1].set_yscale('log')
        ax[1].set_xlabel('Absolute value after sorting')
        ax[1].set_xticks([])
        plt.tight_layout()
        plt.savefig(os.path.join('real_results', f'{file_name}.png'))
        plt.savefig(os.path.join('real_results', f'{file_name}.pdf'))
        plt.close()

    runs_stuff = [
        {'labels': ['MI Orig-Syn', 'DC Orig-Syn',], 'file_name': 'orig_syn_diff'},
        {'labels': ['MI-DC Orig', 'MI-DC Syn',], 'file_name': 'mi_dc_diff'},
    ]
    for run in runs_stuff:
        run_double_plot(run['labels'], run['file_name'])

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
    fig, ax = plt.subplots(3, 2, figsize=(8, 7))
    
    df_neither_object = df[(df['test_type'] == 'distance_correlation') & (df['d_type_col1'] != 'object') & (df['d_type_col2'] != 'object')]
    sns.boxplot(x='score', y='range_label', data=df_neither_object, ax=ax[0, 0], order=range_labels)
    ax[0, 0].set_xlabel('Score (Distance Correlation, No Text)')
    ax[0, 0].set_ylabel('')
    ax[0, 0].set_xlim(-0.05, 1.05)

    df_neither_object = df[(df['test_type'] == 'mutual_information') & (df['d_type_col1'] != 'object') & (df['d_type_col2'] != 'object')]
    sns.boxplot(x='score', y='range_label', data=df_neither_object, ax=ax[0, 1], order=range_labels)
    ax[0, 1].set_xlabel('Score (Mutual Information, No Text)')
    ax[0, 1].set_ylabel('')
    ax[0, 1].set_yticks([])
    ax[0, 1].set_xlim(-0.05, 1.05)
    
    df_both_object = df[(df['test_type'] == 'distance_correlation') & (df['d_type_col1'] == 'object') & (df['d_type_col2'] == 'object')]
    sns.boxplot(x='score', y='range_label', data=df_both_object, ax=ax[1, 0], order=range_labels)
    ax[1, 0].set_xlabel('Score (Distance Correlation, Both Text)')
    ax[1, 0].set_ylabel('')
    ax[1, 0].set_xlim(-0.05, 1.05)
    
    df_both_object = df[(df['test_type'] == 'mutual_information') & (df['d_type_col1'] == 'object') & (df['d_type_col2'] == 'object')]
    sns.boxplot(x='score', y='range_label', data=df_both_object, ax=ax[1, 1], order=range_labels)
    ax[1, 1].set_xlabel('Score (Mutual Information, Both Text)')
    ax[1, 1].set_ylabel('')
    ax[0, 1].set_yticks([])
    ax[1, 1].set_xlim(-0.05, 1.05)
    
    df_one_object = df[(df['test_type'] == 'distance_correlation') & ((df['d_type_col1'] == 'object') | (df['d_type_col2'] == 'object')) & ~((df['d_type_col1'] == 'object') & (df['d_type_col2'] == 'object'))]
    sns.boxplot(x='score', y='range_label', data=df_one_object, ax=ax[2, 0], order=range_labels)
    ax[2, 0].set_xlabel('Score (Distance Correlation, One Text)')
    ax[2, 0].set_ylabel('')
    ax[2, 0].set_xlim(-0.05, 1.05)
    
    df_one_object = df[(df['test_type'] == 'mutual_information') & ((df['d_type_col1'] == 'object') | (df['d_type_col2'] == 'object')) & ~((df['d_type_col1'] == 'object') & (df['d_type_col2'] == 'object'))]
    sns.boxplot(x='score', y='range_label', data=df_one_object, ax=ax[2, 1], order=range_labels)
    ax[2, 1].set_xlabel('Score (Mutual Information, One Text)')
    ax[2, 1].set_ylabel('')
    ax[2, 1].set_yticks([])
    ax[2, 1].set_xlim(-0.05, 1.05)
    
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
    df_joined = df_joined.drop(columns=['blob_name_dc', 'col1_dc', 'col2_dc', 'blob_id_dc', 'col_pair_id_dc', 'blob_col_id_dc'])
    
    # Rename columns to remove suffixes from the join keys
    df_joined = df_joined.rename(columns={
        'blob_name_mi': 'blob_name',
        'col1_mi': 'col1',
        'col2_mi': 'col2',
        'blob_id_mi': 'blob_id',
        'col_pair_id_mi': 'col_pair_id',
        'blob_col_id_mi': 'blob_col_id',
    })
    return df_joined

def plot_score_list_lines(df, tag):
    # Define the columns to be plotted
    columns_to_plot = [
        'score_orig_mi_list', 'score_syn_mi_list', 'score_orig_dc_list', 'score_syn_dc_list',
        'cat_orig_mi_list', 'cat_syn_mi_list', 'cat_orig_dc_list', 'cat_syn_dc_list'
    ]
    
    # Create the plot
    fig, ax = plt.subplots(4, 2, figsize=(12, 16))
    ax = ax.flatten()  # Flatten the 2D array of axes to 1D for easier indexing
    
    for i, column in enumerate(columns_to_plot):
        for row in df[column]:
            ax[i].plot(range(1, len(row) + 1), row)
        ax[i].set_title(f'{column} ({tag})')
        ax[i].set_xscale('log')
        ax[i].set_xlabel('')
        ax[i].set_ylabel('Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join('real_results', f'score_list_lines_{tag}.png'))
    plt.savefig(os.path.join('real_results', f'score_list_lines_{tag}.pdf'))
    plt.close()

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
all_results_path = os.path.join(results_path, 'all_tests')

results_file_path = os.path.join(results_path, 'all_results.parquet')
processed_results_file_path = os.path.join(results_path, 'all_results_processed.parquet')
merged_results_file_path = os.path.join(results_path, 'all_results_merged.parquet')
merged_mi_dc_results_file_path = os.path.join(results_path, 'all_results_merged_mi_dc.parquet')
listed_mi_dc_results_file_path = os.path.join(results_path, 'all_results_listed_mi_dc.parquet')
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

print(df.head())
df = set_id(df)
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
df.to_parquet(processed_results_file_path)
# Remove all rows where the combination of 'blob_name', 'col1', 'col2', and
# 'dataset_type' does not have both measures
print(f"Prior to removing rows that do not have both measures there are {len(df)} rows")
df_both = df.groupby(['blob_name', 'col1', 'col2', 'dataset_type']).filter(lambda x: len(x) == len(test_types))
print(f"After removing rows there are {len(df_both)} rows")
# Add a new column 'cat'. Set cat to 0 if score < 0.1, to 1 if score between 0.1 and 0.25, and 2 otherwise
df_both['cat'] = pd.cut(df_both['score'], bins=[-10, 0.1, 0.25, 0.6, 1], labels=[0, 1, 2, 3])

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
df_mi_dc_listed = make_listed_data(df_merged_mi_dc)
# sort df_mi_dc_listed by count ascending
df_mi_dc_listed = df_mi_dc_listed.sort_values(by='count')
df_mi_dc_listed.to_parquet(listed_mi_dc_results_file_path)

# for every row in df_mi_dc_listed, print blob_name where the first 50 entries in score_syn_mi_list are 1.0
print("Blob names where the first 50 entries in score_syn_mi_list are 1.0:")
bad_blob_names = df_mi_dc_listed[df_mi_dc_listed['score_syn_mi_list'].apply(lambda x: x[:50] == [1.0] * 50)]['blob_name']
# For each blob_name in bad_blob_names, list col1 and col2 from 20 rows in df_merged_mi_dc with that blob_name where score_syn_mi == 1.0
for blob_name in bad_blob_names:
    print(f"{blob_name}:")
    print(df_merged_mi_dc[(df_merged_mi_dc['blob_name'] == blob_name) & (df_merged_mi_dc['score_syn_mi'] == 1.0)].head(20)[['col1', 'col2', 'score_syn_mi']])

print("Initial collected data columns:")
print(df.columns)
print("Columns after merge:")
print(df_merged.columns)
print("Columns after MI/DC split and join:")
print(df_merged_mi_dc.columns)
print("Columns after making listed data:")
print(df_mi_dc_listed.columns)

plot_order_error(df_mi_dc_listed, 'score')
plot_order_error(df_mi_dc_listed, 'cat')
plot_cat_compare(df_mi_dc_listed)
plot_score_list_lines(df_mi_dc_listed, 'all')
# call plot_score_list_lines with the first 100 rows of df_mi_dc_listed
plot_score_list_lines(df_mi_dc_listed.head(100), 'first_100')
# call plot_score_list_lines with the last 100 rows of df_mi_dc_listed
plot_score_list_lines(df_mi_dc_listed.tail(100), 'last_100')
plot_orig_syn_diff(df_merged_mi_dc)
plot_by_num_distinct(df)
plot_basic_per_test(df_merged)
plot_mi_vs_dc_orig(df_merged)
plot_sorted_scores(df_merged)