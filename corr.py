import pandas as pd 
import numpy as np
from src.utils import make_latex_table, color_table_by_method, color_table_by_method_type
path = 'output/results_temp.csv'
df = pd.read_csv(path)


## add size rank
df['size_rank'] = np.ones(len(df))
df['size_rank'].mask(df["Dataset"] == 'Kolodziejczyk', 2, inplace=True)
df['size_rank'].mask(df["Dataset"] == 'Zeisel', 4, inplace=True)
df['size_rank'].mask(df["Dataset"] == 'Klein', 3, inplace=True)
## add size
df['size'] = np.ones(len(df)) * 318
df['size'].mask(df["Dataset"] == 'Kolodziejczyk', 705, inplace=True)
df['size'].mask(df["Dataset"] == 'Zeisel', 2718, inplace=True)
df['size'].mask(df["Dataset"] == 'Klein', 3006, inplace=True)
## add number classes
df['num_classes'] = np.ones(len(df)) * 4
df['num_classes'].mask(df["Dataset"] == 'Kolodziejczyk', 4, inplace=True)
df['num_classes'].mask(df["Dataset"] == 'Zeisel', 6, inplace=True)
## average cells per class
df['avg_cells_per_class'] = df['size'] / df['num_classes']
unique_methods = df[' Method'].unique()
unique_metrics = [' Silhouette Score', ' Adjusted Rand Index',
       ' Normalized Mutual Info Score']
unique_cor = ['size_rank', 'num_classes', 'avg_cells_per_class']
results_df = pd.DataFrame(columns=['method', 'metric', 'corr', 'value'])
results_array = []
for method in unique_methods:
    for metric in unique_metrics:
        working_frame = df[(df[' Method'] == method)]
        for cor in unique_cor:
            print(f"Correlation between {metric} and {cor} for {method}")
            # print(working_frame[metric].corr(working_frame[cor]))
            results_array.append([method, metric, cor, working_frame[metric].corr(working_frame[cor])])
        # print('-'*100)
results_df = pd.DataFrame(results_array, columns=['method', 'metric', 'corr', 'value'])
results_df[(results_df["metric"] == ' Adjusted Rand Index') & (results_df["corr"] == 'size_rank')][['method', 'value']] 
results_df[(results_df["metric"] == ' Normalized Mutual Info Score') & (results_df["corr"] == 'size_rank')][['method', 'value']]
results_df[(results_df["metric"] == ' Silhouette Score') & (results_df["corr"] == 'size_rank')][['method', 'value']]

results_df[(results_df["metric"] == ' Adjusted Rand Index') & (results_df["corr"] == 'num_classes')][['method', 'value']]
results_df[(results_df["metric"] == ' Normalized Mutual Info Score') & (results_df["corr"] == 'num_classes')][['method', 'value']]
results_df[(results_df["metric"] == ' Silhouette Score') & (results_df["corr"] == 'num_classes')][['method', 'value']]

results_df[(results_df["metric"] == ' Adjusted Rand Index') & (results_df["corr"] == 'avg_cells_per_class')][['method', 'value']]
results_df[(results_df["metric"] == ' Normalized Mutual Info Score') & (results_df["corr"] == 'avg_cells_per_class')][['method', 'value']]

nmi_df = results_df[results_df["metric"] == ' Normalized Mutual Info Score'][['method','corr' ,'value']]
## rename the columns of nmi_df
nmi_df.columns = ['Method', 'Comparison', 'Pearson Correlation']
nmi_df = nmi_df.groupby(['Comparison', 'Method']).mean()

dataset_order = ["size_rank" , 'num_classes', 'avg_cells_per_class']
new_dataset_order = ["Number of Cells", "Number of Classes", "Average Cells per Class"]
old_method_order = [' raw', ' scvi', ' scvi_ld', ' clear', ' contrastive_sc', ' scmae', ' scgnn']
new_method_order = ['PCA + K-Means', 'scVI', 'scVI-LD', 'CLEAR', 'Contrastive-SC', 'scMAE', 'scGNN']
idx_1 = pd.MultiIndex.from_product([dataset_order, old_method_order], names=['Comparison', ' Method'])
idx_2 = pd.MultiIndex.from_product([dataset_order, new_method_order], names=['Comparison', ' Method'])
idx_3 = pd.MultiIndex.from_product([new_dataset_order, new_method_order], names=['Comparison', ' Method'])
results = nmi_df.reindex(idx_1)
df = results.copy()
temp = results.index.set_levels([dataset_order, new_method_order])
temp = temp.reindex(idx_2)
results.index = temp[0]
results.to_latex(index = True, buf='output/corr_results.tex', escape=True, float_format="%.2f")
color_table_by_method_type(latex_path='output/corr_results.tex')