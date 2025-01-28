import pandas as pd
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer

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

def compute_mutual_information(df, col1, col2):
    col1_processed = preprocess_column(df[col1])
    col2_processed = preprocess_column(df[col2])
    
    # Discretize continuous columns
    if pd.api.types.is_numeric_dtype(col1_processed):
        col1_processed = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform').fit_transform(col1_processed.values.reshape(-1, 1)).flatten()
    if pd.api.types.is_numeric_dtype(col2_processed):
        col2_processed = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform').fit_transform(col2_processed.values.reshape(-1, 1)).flatten()
    
    # Compute mutual information
    mi = mutual_info_score(col1_processed, col2_processed)
    
    # Normalize the mutual information score
    h_col1 = mutual_info_score(col1_processed, col1_processed)
    h_col2 = mutual_info_score(col2_processed, col2_processed)
    normalized_mi = mi / min(h_col1, h_col2)
    
    return normalized_mi

# Example usage
df = pd.DataFrame({
    'col1': ['a', 'b', 'a', 'b', 'a'],
    'col2': ['x', 'x', 'y', 'y', 'x'],
    'col3': [1, 2, 1, 2, 1],
    'col4': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])
})

mi_score = compute_mutual_information(df, 'col1', 'col4')
print(f"Mutual Information between 'col1' and 'col4': {mi_score}")