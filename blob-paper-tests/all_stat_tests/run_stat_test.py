import pandas as pd
import numpy as np
import pprint
import os
import seaborn as sns
import matplotlib.pyplot as plt
from stat_tests import StatTests

'''
This creates simpl 2-column dataframes with increasing amounts of randomness, and therefore
decreasing amounts of correlation, and runs a series of statistical tests on them.
'''

pp = pprint.PrettyPrinter(indent=4)
num_rows = 5000

test_types = [
    'mutual_information',
    'pearson_correlation',
    'spearman_rank_correlations',
    'distance_correlation',
    'chi_square',
    'linear_regression',
    'decision_tree_regressor',
    'random_forest_regressor',
    'gradient_boosting_regressor',
    'support_vector_regressor',
    'decision_tree_classifier',
    'gradient_boosting_classifier',
    'random_forest_classifier',
]
test_colors = {
    'mutual_information': 'blue',
    'pearson_correlation': 'blue',
    'spearman_rank_correlations': 'blue',
    'distance_correlation': 'blue',
    'chi_square': 'blue',
    'linear_regression': 'green',
    'decision_tree_regressor': 'green',
    'random_forest_regressor': 'green',
    'gradient_boosting_regressor': 'green',
    'support_vector_regressor': 'green',
    'decision_tree_classifier': 'red',
    'gradient_boosting_classifier': 'red',
    'random_forest_classifier': 'red',
}

# Define dataframes for each test type
basic_cols = {
    'cat': pd.DataFrame({'col': np.random.choice(['A', 'B', 'C', 'D'], num_rows)}), 
    'fcon': pd.DataFrame({'col': np.random.normal(loc=20, scale=2, size=num_rows)}), 
    'icon': pd.DataFrame({'col': np.random.randint(50, 101, size=num_rows)}), 
}

# sort each of the dataframes in basic_cols
for col_type in basic_cols:
    basic_cols[col_type] = pd.DataFrame({'col': sorted(basic_cols[col_type]['col'])})

coltypes = [
    ('cat', 'cat'),
    ('cat', 'fcon'),
    ('cat', 'icon'),
    ('fcon', 'cat'),
    ('fcon', 'fcon'),
    ('fcon', 'icon'),
    ('icon', 'cat'),
    ('icon', 'fcon'),
    ('icon', 'icon'),
]

swap_fractions = [0, 0.05, 0.1, 0.2, 0.4, 0.8]

def test_make_df():
    for col1_type, col2_type in coltypes:
        num_differing_rows = []
        for swap_frac in swap_fractions:
            df = make_df(col1_type, col2_type, swap_frac)
            # make sure df has the right number of rows
            assert df.shape[0] == num_rows
            # count the number of values in df[col2] that differ from the corresponding row in basic_cols[col2_type]
            num_differing_rows.append(int((df['col2'] != basic_cols[col2_type]['col']).sum()))
        # assert that each value in num_differing_rows increases
        assert all(num_differing_rows[i] < num_differing_rows[i+1] for i in range(len(num_differing_rows)-1))
        # assert that num_different_rows is 0 for the first element
        assert num_differing_rows[0] == 0

def make_df(col1_type, col2_type, swap_frac):
    ''' Make a df with two columns 'col1' and 'col2', where each column is taken from basic_cols according to col1_type and col2_type.
    Compute the number of rows `nrows` that correspond to swap_frac (round down to integer)
    Select nrow rows randomly.
    For each selected row, select one other random row, and swap the values of col2 between the two rows.
    Return the resulting df.
    '''
    # Create the initial DataFrame
    df = pd.DataFrame({
        'col1': sorted(basic_cols[col1_type]['col']),
        'col2': sorted(basic_cols[col2_type]['col'])
    })
    
    # Compute the number of rows to swap
    nrows = int(num_rows * swap_frac)
    
    # Select nrows rows randomly
    swap_indices = np.random.choice(num_rows, nrows, replace=False)
    
    # Swap the values of col2 between the selected rows
    for idx in swap_indices:
        swap_with = np.random.choice(num_rows)
        df.at[idx, 'col2'], df.at[swap_with, 'col2'] = df.at[swap_with, 'col2'], df.at[idx, 'col2']
    return df


def run_tests():
    results = []
    for col1_type, col2_type in coltypes:
        for swap_frac in swap_fractions:
            df = make_df(col1_type, col2_type, swap_frac)
            columns = df.columns
            stat_tests = StatTests(df, columns[0], columns[1])
            for test_name in test_types:
                print(f"Run {test_name} for {col1_type}-{col2_type} with swap fraction {swap_frac}")
                result = stat_tests.run_stat_test(test_name)
                if result is not None:
                    results.append({
                        'col1_type': col1_type,
                        'col2_type': col2_type,
                        'swap_frac': swap_frac,
                        'test_name': test_name,
                        'score': result['score'],
                        'elapsed_time': result['elapsed_time']
                    })
            # write results to file results.parquet 
            df_results = pd.DataFrame(results)
            df_results.to_parquet('run_stat_test_results.parquet')

def plot_median_diff(df):
    ''' Make a plot with six subplots, organized in 3 rows by 2 columns.
        In all subplots, the y axis is the set of test_names.
        For each test_name, there is a boxplot, where the x axis is diff_from_median.
        cat-cat, fcon-fcon, icon-icon, everything else with cat, everythings else without cat
        The subplot in the upper left is derived from all rows.
        The subplot in the upper right is derived from rows where col1_type and col2_type are 'cat'.
        The subplot in the middle left is derived from rows where col1_type and col2_type are 'fcon'.
        The subplot in the middle right is derived from rows where col1_type and col2_type are 'icon'.
        The subplot in the lower left is derived from rows where either col1_type or col2_type is 'cat', but not both.
        The subplot in the lower right is derived from rows where neither col1_type nor col2_type is 'cat', and col1_type != col2_type.
    '''
    fig, ax = plt.subplots(3, 2, figsize=(8, 13))
    
    # Function to ensure all test_names are present in the data
    def ensure_all_test_names(data):
        missing_test_names = [test_name for test_name in test_types if test_name not in data['test_name'].values]
        if missing_test_names:
            missing_data = pd.DataFrame({'test_name': missing_test_names, 'diff_from_median': [np.nan] * len(missing_test_names)})
            data = pd.concat([data, missing_data], ignore_index=True)
        return data
    
    # Create a palette based on test_colors
    palette = {test_name: test_colors[test_name] for test_name in test_types}
    
    # Plot all rows
    data = ensure_all_test_names(df)
    sns.boxplot(x='diff_from_median', y='test_name', data=data, ax=ax[0, 0], palette=palette, order=test_types)
    x_limits = ax[0, 0].get_xlim()
    ax[0, 0].set_xlabel('Difference from Median Score\n(All datasets)')
    
    # Plot rows where col1_type and col2_type are 'cat'
    data = ensure_all_test_names(df[(df['col1_type'] == 'cat') & (df['col2_type'] == 'cat')])
    sns.boxplot(x='diff_from_median', y='test_name', data=data, ax=ax[0, 1], palette=palette, order=test_types)
    ax[0, 1].set_xlabel('Difference from Median Score\n(Cat-Cat datasets)')
    
    # Plot rows where col1_type and col2_type are 'fcon'
    data = ensure_all_test_names(df[(df['col1_type'] == 'fcon') & (df['col2_type'] == 'fcon')])
    sns.boxplot(x='diff_from_median', y='test_name', data=data, ax=ax[1, 0], palette=palette, order=test_types)
    ax[1, 0].set_xlabel('Difference from Median Score\n(Float-Float datasets)')
    
    # Plot rows where col1_type and col2_type are 'icon'
    data = ensure_all_test_names(df[(df['col1_type'] == 'icon') & (df['col2_type'] == 'icon')])
    sns.boxplot(x='diff_from_median', y='test_name', data=data, ax=ax[1, 1], palette=palette, order=test_types)
    ax[1, 1].set_xlabel('Difference from Median Score\n(Int-Int datasets)')
    
    # Plot rows where either col1_type or col2_type is 'cat', but not both
    data = ensure_all_test_names(df[(df['col1_type'] == 'cat') ^ (df['col2_type'] == 'cat')])
    sns.boxplot(x='diff_from_median', y='test_name', data=data, ax=ax[2, 0], palette=palette, order=test_types)
    ax[2, 0].set_xlabel('Difference from Median Score\n(Cat-NonCat datasets)')
    
    # Plot rows where neither col1_type nor col2_type is 'cat', and col1_type != col2_type
    data = ensure_all_test_names(df[(df['col1_type'] != 'cat') & (df['col2_type'] != 'cat') & (df['col1_type'] != df['col2_type'])])
    sns.boxplot(x='diff_from_median', y='test_name', data=data, ax=ax[2, 1], palette=palette, order=test_types)
    ax[2, 1].set_xlabel('Difference from Median Score\n(NonCat-NonCat datasets)')
    
    # Customize the plots
    for i in range(3):
        for j in range(2):
            ax[i, j].set_ylabel('')
            ax[i, j].axvline(0, ls=':', color='k')
            ax[i, j].set_xlim(x_limits)  # Set the x-axis limits to match the subplot at [0, 0]
            ax[i, j].set_yticks(range(len(test_types)))
    for i in range(3):
        ax[i, 0].set_yticklabels(test_types)
        ax[i, 1].set_yticklabels([])
    
    plt.tight_layout()
    plt.savefig(os.path.join('simple_results', 'median_diff.png'))
    plt.savefig(os.path.join('simple_results', 'median_diff.pdf'))
    plt.close()

def plot_basic_per_test(df):
    ''' Make a seaborn plot with two subplots, arranged horizontally.
        In both subplots, the y axis is the set of test_names
        For each test_name, there is a boxplot
        In the first subplot, the x axis for the boxplots is score
        In the second subplot, the x axis for the boxplots is elapsed_time
    '''
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    
    # Create a palette based on test_colors
    palette = {test_name: test_colors[test_name] for test_name in test_types}
    
    sns.boxplot(x='diff_from_median', y='test_name', data=df, ax=ax[0], palette=palette, order=test_types)
    sns.boxplot(x='elapsed_time', y='test_name', data=df, ax=ax[1], palette=palette, order=test_types)
    
    ax[1].set_yticks([])
    ax[1].set_ylabel('')
    ax[0].set_ylabel('')
    ax[1].set_xscale('log')
    ax[0].axvline(0, ls=':', color='k')
    # Set the x-axis label for the first subplot
    ax[0].set_xlabel('Difference from Median Score')
    ax[1].set_xlabel('Elapsed Time (log scale)')
    plt.tight_layout()
    plt.savefig(os.path.join('simple_results', 'basic_per_test.png'))
    plt.savefig(os.path.join('simple_results', 'basic_per_test.pdf'))
    plt.close()

def plot_swap_frac_per_test(df):
    ''' Make a plot with 13 subplots laid out in 5 rows and 3 columns.
        Each subplot has swap_frac on the x axis, interpreted as categorical values, and score on the y axis.
        Each subplot has a boxplot per swap_frac value, oriented horizontally.
        Each subplot is limited to a single test_name.
    '''
    # Create the subplots
    fig, axes = plt.subplots(3, 5, figsize=(16, 10))
    axes = axes.flatten()

    # Plot each subplot
    # First make the plot for everything
    sns.boxplot(x='score', y='swap_frac', data=df, ax=axes[0], orient='h', color='grey')
    axes[0].set_ylabel('Swap Fraction')
    axes[0].set_xlabel('Score')
    axes[0].set_title('All Test Types')
    for i, test_name in enumerate(test_types):
        ax = axes[i+2]
        sns.boxplot(x='score', y='swap_frac', data=df[df['test_name'] == test_name], ax=ax, orient='h', color=test_colors[test_name])
        ax.set_title(test_name)
        ax.set_ylabel('Swap Fraction')
        ax.set_xlabel(f'Score')

    # Remove any empty subplots
    fig.delaxes(axes[1])

    plt.tight_layout()
    plt.savefig(os.path.join('simple_results', 'swap_by_test.png'))
    plt.savefig(os.path.join('simple_results', 'swap_by_test.pdf'))

def plot_basic_per_swap_frac(df):
    ''' Make a seaborn plot where the y axis is swap_frac.
        Each swap_frac should be treated as a category.
        Each swap_frac has a boxplot.
        The x axis for the boxplots is score.
    '''
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='score', y='swap_frac', data=df, orient='h')
    plt.xlabel('Score')
    plt.ylabel('Swap Fraction')
    plt.tight_layout()
    plt.savefig(os.path.join('simple_results', 'basic_per_swap_frac.png'))
    plt.savefig(os.path.join('simple_results', 'basic_per_swap_frac.pdf'))
    plt.close()

def plot_score_per_test_type(df):
    plt.figure(figsize=(6, 4))
    
    # Create a palette based on test_colors
    palette = {test_name: test_colors[test_name] for test_name in test_types}
    
    sns.boxplot(x='score', y='test_name', data=df, orient='h', palette=palette, order=test_types)
    plt.tight_layout()
    plt.savefig(os.path.join('simple_results', 'score_per_test_type.png'))
    plt.savefig(os.path.join('simple_results', 'score_per_test_type.pdf'))
    plt.close()

def set_diff_from_best(df):
    ''' Add a column 'diff_from_best' to the DataFrame.
    For each combination of col1_type, col2_type, and swap_frac, find the highest score among the different test_type values.
    Set diff_from_best to be the difference between the highest score and the score of the row's test_type.
    '''
    # Group by col1_type, col2_type, and swap_frac and find the highest score in each group
    max_scores = df.groupby(['col1_type', 'col2_type', 'swap_frac'])['score'].transform('max')
    # Calculate the difference from the best score
    df['diff_from_best'] = max_scores - df['score']
    print(df.head(50))
    return df

def set_diff_from_median(df):
    ''' Add a column 'diff_from_median' to the DataFrame. 
        For each combination of col1_type, col2_type, and swap_frac, compute the median score among the different test_type values.
        Set diff_from_median to be the difference between the median score and the score of the row's test_type.
    '''
    # Group by col1_type, col2_type, and swap_frac and find the median score in each group
    median_scores = df.groupby(['col1_type', 'col2_type', 'swap_frac'])['score'].transform('median')
    # Calculate the difference from the median score
    df['diff_from_median'] = median_scores - df['score']
    return df

if __name__ == "__main__":
    # make directory simple_results if it doesn't exist
    os.makedirs('simple_results', exist_ok=True)
    if False:
        test_make_df()
    # check to see if file run_stat_test_results.parquet exists
    if not os.path.exists('run_stat_test_results.parquet'):
        run_tests()
    df = pd.read_parquet('run_stat_test_results.parquet')
    df = df[df['test_name'] != 'logistic_regression']
    df = set_diff_from_best(df.copy())
    df = set_diff_from_median(df.copy())
    plot_swap_frac_per_test(df.copy())
    plot_median_diff(df.copy())
    plot_score_per_test_type(df.copy())
    plot_basic_per_test(df.copy())
    plot_basic_per_swap_frac(df.copy())