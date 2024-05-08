import scanpy as sc
import pandas as pd 
from src.utils import *
import argparse
""""
TODO: 
[x] try to run methods on other datasets
[x] run scgnn and clear on the other daatasets  
[] consier adding scmae
[] want to  have a good way to plot results 
[]
"""


def main():
    # methods = ['raw', 'scvi', 'scvi_ld', 'clear', 'contrastive_sc', 'scmae', 'scgnn']
    # methods = ['raw',  'scvi', 'scvi_ld', 'clear', 'contrastive_sc', 'scmae', 'scgnn']
    methods = ['scvi']
    # methods = ['scmae', 'scgnn']
    # datasets = ['Chung', 'Klein', 'Kolodziejczyk','Zeisel' ]
    datasets = ['Chung','Klein','Kolodziejczyk','Zeisel' ]
    # datasets = ['Chung']
    use_manual_args = True
    if not use_manual_args:
        print("reading in arguments:")
        parser = argparse.AsrgumentParser(description='Run the evaluation pipeline')
        parser.add_argument('--method', type=str, default='scgnn', help='The method to use for embeddings')
        parser.add_argument('--dataset', type=str, default='Chung', help='The dataset to use for embeddings')
        args = parser.parse_args()
        method_name = args.method
        dataset_name = args.dataset
    else:
        method_name = 'scvi'
        # dataset_name = 'Chung'
        dataset_name = 'Klein'
    for dataset_name in datasets:
        for method_name in methods:
            print("method_name: ", method_name)
            print("dataset_name: ", dataset_name)
            raw_path = 'data/benchmark_data/{0}/T2000_LTMG.txt'.format(dataset_name)
            anndata= sc.read_text(raw_path).T
            anndata.obs['cell_type'] = pd.read_csv("./data/benchmark_data/{0}/{0}_cell_label.csv".format(dataset_name))['cell_type'].values
            ## preprocess 
            if method_name == "raw":
                manual_preprocess(anndata, method_name=method_name, dataset_name=dataset_name)
                anndata.obsm["raw_embedings"] = anndata.X ## in this case will just use the raws
            elif method_name == 'scgnn':
                anndata = manual_preprocess(anndata, method_name=method_name, dataset_name=dataset_name)
                run_scGNN(dataset_name)
                anndata.obsm["raw_embedings"] = pd.read_csv('output/scgnn/{0}/{0}_LTMG_10-0.1-0.9-0.0-0.3-0.1_1.0_0.0_embedding.csv'.format(dataset_name), header=None).values  
            elif method_name == 'scvi':
                manual_preprocess(anndata, method_name=method_name, dataset_name=dataset_name)
                model = fit_scvi(anndata)
                anndata.obsm["raw_embedings"] = model.get_latent_representation()
            elif method_name == "scvi_ld":
                manual_preprocess(anndata, method_name=method_name, dataset_name=dataset_name)
                model = fit_scvi_ld(anndata)
                anndata.obsm["raw_embedings"] = model.get_latent_representation()
            elif method_name == "clear":
                manual_preprocess(anndata=anndata, method_name=method_name, dataset_name=dataset_name)
                run_clear(dataset_name  = dataset_name)
                anndata.obsm["raw_embedings"] = get_clear_embeds(anndata, dataset_name)
            elif method_name == "contrastive_sc":
                manual_preprocess(anndata=anndata, method_name=method_name, dataset_name=dataset_name)
                anndata.obsm["raw_embedings"] = run_contrastive_sc(anndata, dataset_name)
            elif method_name == 'scmae':
                manual_preprocess(anndata=anndata, method_name=method_name, dataset_name=dataset_name)
                anndata.obsm["raw_embedings"] = run_scmae(anndata, dataset_name=dataset_name)
            
            else:
                raise NotImplementedError
            anndata.obs["predicted_labels"] = get_predicted_labels(anndata.obs['cell_type'], anndata.obsm["raw_embedings"], clustering_method="KMeans", method_name=method_name, dataset_name=dataset_name)
            print("model: ", method_name)
            metrics = get_metrics(anndata.obs['cell_type'], anndata.obs["predicted_labels"], anndata.obsm["raw_embedings"])
            print("metrics: ", metrics)
            write_results_to_file(dataset_name=dataset_name, method_name=method_name, scores=metrics)
            plot_clusters(anndata, dataset_name=dataset_name, method_name=method_name)
if __name__ == "__main__":
    main()