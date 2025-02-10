import pandas as pd
import numpy as np
import pprint
from stat_tests import StatTests

pp = pprint.PrettyPrinter(indent=4)
num_rows = 1000

def run_tests():
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
        'logistic_regression',
        'gradient_boosting_classifier',
        'random_forest_classifier',
    ]

    # Define dataframes for each test type
    dataframes = {
        'cat-cat-weak': pd.DataFrame({'col1': np.random.choice(['A', 'B', 'C'], num_rows), 'col2': np.random.choice(['A', 'B', 'C'], num_rows)}),
        'cat-fcon-weak': pd.DataFrame({'col1': np.random.choice(['A', 'B', 'C'], num_rows), 'col2': np.random.normal(loc=20, scale=2, size=num_rows)}),
        'fcon-fcon-weak': pd.DataFrame({'col1': np.random.normal(loc=20, scale=2, size=num_rows), 'col2': np.random.normal(loc=20, scale=2, size=num_rows)}),
        'fcon-cat-weak': pd.DataFrame({ 'col1': np.random.normal(loc=20, scale=2, size=num_rows), 'col2': np.random.choice(['A', 'B', 'C'], num_rows), }),
        'cat-icon-weak': pd.DataFrame({'col1': np.random.choice(['A', 'B', 'C'], num_rows), 'col2': np.random.randint(50, 101, size=num_rows)}),
        'icon-icon-weak': pd.DataFrame({'col1': np.random.randint(50, 101, size=num_rows), 'col2': np.random.randint(50, 101, size=num_rows)}),
        'icon-cat-weak': pd.DataFrame({ 'col1': np.random.randint(50, 101, size=num_rows), 'col2': np.random.choice(['A', 'B', 'C'], num_rows), }),
    }
    df = dataframes['cat-cat-weak'].copy()
    df['col2'] = df['col1']
    dataframes['cat-cat-strong'] = df

    df = dataframes['fcon-fcon-weak'].copy()
    df['col2'] = df['col1']
    dataframes['fcon-fcon-strong'] = df

    df = dataframes['cat-fcon-weak']
    df1 = pd.DataFrame({
        'col1': sorted(df['col1']),
        'col2': sorted(df['col2'])
    })
    dataframes['cat-fcon-strong'] = df1

    df = dataframes['fcon-cat-weak']
    df1 = pd.DataFrame({
        'col1': sorted(df['col1']),
        'col2': sorted(df['col2'])
    })
    dataframes['fcon-cat-strong'] = df1

    df = dataframes['icon-icon-weak'].copy()
    df['col2'] = df['col1']
    dataframes['icon-icon-strong'] = df

    df = dataframes['cat-icon-weak']
    df1 = pd.DataFrame({
        'col1': sorted(df['col1']),
        'col2': sorted(df['col2'])
    })
    dataframes['cat-icon-strong'] = df1

    df = dataframes['icon-cat-weak']
    df1 = pd.DataFrame({
        'col1': sorted(df['col1']),
        'col2': sorted(df['col2'])
    })
    dataframes['icon-cat-strong'] = df1

    all_scores = {}
    for test_name in test_types:
        all_scores[test_name] = {}
    for dataframe_name in dataframes.keys():
    #for dataframe_name in ['fcon-fcon-weak', 'fcon-fcon-strong', 'cat-cat-weak', 'cat-cat-strong']:
        df = dataframes[dataframe_name].copy()
        print(f"Run scores for {dataframe_name}:")
        col1, col2 = df.columns
        stat_tests = StatTests(df, col1, col2)
        for test_name in test_types:
            print(f"    Running {test_name} test")
            result = stat_tests.run_stat_test(test_name)
            all_scores[test_name][dataframe_name] = result['score'] if result is not None else None
    pp.pprint(all_scores)

if __name__ == "__main__":
    run_tests()