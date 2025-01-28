import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_mutual_info(path_to_dir):
    # Read all parquet files in the directory and concatenate them into a single DataFrame
    df_list = []
    for file_name in os.listdir(path_to_dir):
        if file_name.endswith('.parquet'):
            file_path = os.path.join(path_to_dir, file_name)
            df_list.append(pd.read_parquet(file_path))
    df = pd.concat(df_list, ignore_index=True)
    
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
    df['mi_diff'] = df['mi_orig'] - df['mi_syn']
    sns.lineplot(ax=axes[1], data=df['mi_diff'])
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

plt = plot_mutual_info(mi_results_path)
plot_path = os.path.join(results_path, 'mutual_info_plot.png')
plt.savefig(plot_path)