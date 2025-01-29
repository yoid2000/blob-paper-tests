import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_mutual_info(df):
    # Sort the DataFrame by 'mi_orig' column
    df = df.sort_values(by='mi_orig').reset_index(drop=True)
    
    # Create the plots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # First subplot: plot mi_orig and mi_syn
    sns.lineplot(ax=axes[0], data=df[['mi_syn', 'mi_orig']])
    axes[0].set_title('Mutual Information: Original vs Synthetic')
    axes[0].set_xlabel('Index')
    axes[0].set_ylabel('Mutual Information')
    axes[0].legend(['mi_syn', 'mi_orig'])
    
    # Second subplot: plot mi_orig - mi_syn
    sns.lineplot(ax=axes[1], data=df['mi_err'])
    axes[1].set_title('Difference in Mutual Information (mi_orig - mi_syn)')
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('Difference in Mutual Information')
    
    plt.tight_layout()
    return(plt)

blob_code_path = '.'
# check if there is an environment variable called BLOB_TEST_DIR
if 'BLOB_TEST_DIR' in os.environ:
    blob_path = os.environ['BLOB_TEST_DIR']
    blob_code_path = os.environ['BLOB_TEST_CODE']
else:
    blob_path = os.path.join(os.path.dirname(os.getcwd()), 'blob_tests')

print(f"Blob path: {blob_path}")

os.makedirs(blob_path, exist_ok=True)
results_path = os.path.join(blob_path, 'results')
mi_results_path = os.path.join(results_path, 'mutual_info')
mi_results_complete_path = os.path.join(results_path, 'mi_results.parquet')

# check if mi_results_complete_path exists
if not os.path.exists(mi_results_complete_path):
    # Read all parquet files in the directory and concatenate them into a single DataFrame
    print("Reading parquet files")
    df_list = []
    for file_name in os.listdir(mi_results_path):
        if file_name.endswith('.parquet'):
            file_path = os.path.join(mi_results_path, file_name)
            df_list.append(pd.read_parquet(file_path))
    df = pd.concat(df_list, ignore_index=True)
    print(f"Saving DataFrame to {mi_results_complete_path}")
    df.to_parquet(mi_results_complete_path)
else:
    df = pd.read_parquet(mi_results_complete_path)

def get_various_stats(df):
    pass
    

def plot_err_nunique(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # First subplot: num_unique_syn_col1 vs mi_err
    sns.scatterplot(ax=axes[0, 0], x='num_unique_syn_col1', y='mi_err', data=df)
    axes[0, 0].set_title('num_unique_syn_col1 vs mi_err')
    axes[0, 0].set_xlabel('num_unique_syn_col1')
    axes[0, 0].set_ylabel('mi_err')
    
    # Second subplot: num_unique_syn_col2 vs mi_err
    sns.scatterplot(ax=axes[0, 1], x='num_unique_syn_col2', y='mi_err', data=df)
    axes[0, 1].set_title('num_unique_syn_col2 vs mi_err')
    axes[0, 1].set_xlabel('num_unique_syn_col2')
    axes[0, 1].set_ylabel('mi_err')
    
    # Third subplot: num_unique_orig_col1 vs mi_err
    sns.scatterplot(ax=axes[1, 0], x='num_unique_orig_col1', y='mi_err', data=df)
    axes[1, 0].set_title('num_unique_orig_col1 vs mi_err')
    axes[1, 0].set_xlabel('num_unique_orig_col1')
    axes[1, 0].set_ylabel('mi_err')
    
    # Fourth subplot: num_unique_orig_col2 vs mi_err
    sns.scatterplot(ax=axes[1, 1], x='num_unique_orig_col2', y='mi_err', data=df)
    axes[1, 1].set_title('num_unique_orig_col2 vs mi_err')
    axes[1, 1].set_xlabel('num_unique_orig_col2')
    axes[1, 1].set_ylabel('mi_err')
    
    plt.tight_layout()
    return plt

df['mi_err'] = df['mi_orig'] - df['mi_syn']
df['type_match'] = (df['col1_type_syn'] == df['col1_type_orig']) & (df['col2_type_syn'] == df['col2_type_orig'])
get_various_stats(df)
plt = plot_mutual_info(df)
plot_path = os.path.join(results_path, 'mutual_info_plot.png')
plt.savefig(plot_path)

plt = plot_err_nunique(df)
plot_path = os.path.join(results_path, 'err_nunique.png')
plt.savefig(plot_path)