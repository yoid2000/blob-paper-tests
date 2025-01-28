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
    sns.lineplot(ax=axes[0], data=df[['mi_orig', 'mi_syn']])
    axes[0].set_title('Mutual Information: Original vs Synthetic')
    axes[0].set_xlabel('Index')
    axes[0].set_ylabel('Mutual Information')
    axes[0].legend(['mi_orig', 'mi_syn'])
    
    # Second subplot: plot mi_orig - mi_syn
    df['mi_diff'] = df['mi_orig'] - df['mi_syn']
    sns.lineplot(ax=axes[1], data=df['mi_diff'])
    axes[1].set_title('Difference in Mutual Information (mi_orig - mi_syn)')
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('Difference in Mutual Information')
    
    plt.tight_layout()
    plt.show()

# Example usage
# plot_mutual_info('/path/to/your/directory')