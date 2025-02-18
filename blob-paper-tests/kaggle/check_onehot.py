import pandas as pd

def find_onehot_encoded_sets(df):
    onehot_sets = []
    visited_columns = set()
    
    # Filter columns that contain only 0 and 1 and have at least one 1
    binary_columns = [col for col in df.columns if df[col].isin([0, 1]).all() and df[col].sum() > 0]
    
    for column in binary_columns:
        if column in visited_columns:
            continue
        
        # Find potential one-hot encoded set
        potential_set = [column]
        for other_column in binary_columns:
            if other_column != column and other_column not in visited_columns:
                # Check if the columns are mutually exclusive
                if (df[[column, other_column]].sum(axis=1) <= 1).all():
                    potential_set.append(other_column)
        
        # Check if the potential set is a valid one-hot encoded set
        if len(potential_set) > 1 and (df[potential_set].sum(axis=1) == 1).all():
            onehot_sets.append(potential_set)
            visited_columns.update(potential_set)
    
    return onehot_sets