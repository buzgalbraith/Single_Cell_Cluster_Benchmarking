from sklearn import metrics
from munkres import Munkres
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np 
import scvi
import scanpy as sc
import h5py
import scMAE.main as SCMAEA
from CLEAR.preprocess.generate_h5ad import *
import contrastive_sc.train as train
import subprocess
import anndata as ad
import re
import matplotlib.pyplot as plt

def reduce_dimensionality(embedings, method="PCA", n_components=2):
    """reduces the dimensionality of the data"""
    if method == "PCA":
        pca = PCA(n_components=n_components)
        reduced_dim_emebdigs = pca.fit_transform(embedings)
    else:
        print('Method not implemented')
        raise NotImplementedError
    return reduced_dim_emebdigs
def run_scmae(anndata, dataset_name="Chung",):
    
    num_classes = anndata.obs['cell_type'].nunique() 
    data_dim = 1000 ## number of unique genes after filtering.
    data_path = 'data/benchmark_data/{0}/'.format(dataset_name)
    result_path = 'output/scmae/{0}/'.format(dataset_name)
    ## if file does not exist, then we need to create it
    os.makedirs(result_path, exist_ok=True)
    paths = {"data":data_path, "results":result_path}
    train_args = {
        'dataset' : dataset_name,
        'n_classes': num_classes,
        'paths': paths,
        'batch_size': 256,
        'data_dim': data_dim,
        'epochs': 100,
        'num_workers': 4,
        'learning_rate': 0.001,
        'latent_dim': 32,
        'save_path': result_path
    }
    # import ipdb; ipdb.set_trace()
    SCMAEA.train(train_args)
    # embeds = pd.read_csv(result_path + 'embedding_80.npy', header=None).values
    embeds = np.load(result_path + 'embedding_80.npy')
    return embeds
def run_scGNN(dataset_name="Chung"):
    """running this, requires executing a few python files, so we will use subprocess to do this"""
    os.makedirs('output/scgnn/{0}'.format(dataset_name), exist_ok=True)
    pre_processing_comand = f"python scGNN/Preprocessing_main.py \
    --expression-name {dataset_name} \
    --featureDir data/benchmark_data/{dataset_name}/"
    clean_comand_data = f"rm -r data/sc/"
    clean_comand_scgnn = f"rm -r scGNN/data/sc"
    copy_comand = f"cp -r data/sc/ scGNN/data/sc/"
    scgnn_run_comand = f'python -W ignore scGNN/main_benchmark.py \
    --datasetName {dataset_name} \
    --benchmark data/benchmark_data/{dataset_name}/{dataset_name}_cell_label.csv \
    --LTMGDir data/benchmark_data/ \
    --regulized-type LTMG \
    --EMtype celltypeEM \
    --clustering-method LouvainK \
    --useGAEembedding \
    --npyDir output/scgnn/{dataset_name}/ \
    --debuginfo'.format(dataset_name)

### here 
    # scgnn_run_comand = f'python -W ignore scGNN/scGNN.py \
    #         --datasetName {dataset_name} \
    #         --datasetDir data/benchmark_data/ \
    #         --LTMGDir data/benchmark_data/{dataset_name} \
    #         --outputDir output/scgnn/{dataset_name} \
    #         --quickmode \
    #         --nonsparseMode \
    #         --regulized-type LTMG'
    # df = pd.read_csv('data/benchmark_data/{0}/{0}.csv'.format(dataset_name))
    # df.T.to_csv('data/benchmark_data/{0}/{0}_transpose.csv'.format(dataset_name), header=True, index=True)
    # # import ipdb; ipdb.set_trace()
    # pre_processing_comand = f'python -W ignore scGNN/PreprocessingscGNN.py \
    #     --datasetName {dataset_name}_transpose.csv \
    #     --datasetDir data/benchmark_data/{dataset_name}/ \
    #     --LTMGDir data/benchmark_data/{dataset_name}/  \
    #     --filetype CSV \
    #     --geneSelectnum 2000'
    # # data/benchmark_data/Chung/Use_expression.csv
    # scgnn_run_comand = f'python -W ignore scGNN/scGNN.py \
    #         --datasetName {dataset_name} \
    #         --datasetDir data/benchmark_data/ \
    #         --outputDir output/scgnn/ \
    #         --EM-iteration 50 \
    #         --Regu-epochs 250 \
    #         --EM-epochs 40 \
    #         --quickmode \
    #         --nonsparseMode'



## here 


    clean_comand_data_list = clean_comand_data.split()
    clean_comand_scgnn_list = clean_comand_scgnn.split()
    copy_comand_list = copy_comand.split()
    pre_processing_comand_list = pre_processing_comand.split()
    scgnn_run_comand_list = scgnn_run_comand.split()
    print("Preprocessing....")
    subprocess.run('pwd')
    subprocess.run(clean_comand_data_list)
    subprocess.run(clean_comand_scgnn_list)
    subprocess.run(pre_processing_comand_list)
    print("Copying files....")
    subprocess.run(copy_comand_list)
    print("Running scGNN....")
    subprocess.run(scgnn_run_comand_list)
    print("finshed running")
def run_clear(dataset_name="Chung"):
    clear_path='data/benchmark_data/{0}/{0}_preprocessed.h5ad'.format(dataset_name)
    output_path = 'output/'.format(dataset_name)
    command = f"python CLEAR/CLEAR.py \
    --input_h5ad_path {clear_path} \
    --epochs 100 \
    --lr 0.01 \
    --batch_size 512 \
    --pcl_r 1024 \
    --cos \
    --save_dir {output_path} \
    --gpu 0"
    command_list = command.split()
    subprocess.run(command_list)
    print("finshed running")

def get_predicted_labels(true_labels, embedings, clustering_method="KMeans",method_name="raw", dataset_name="Chung"):
    """gets the predicted labels for the given method"""
    if method_name == "scmae":
        new_labels = pd.read_csv('output/scmae/{0}/types_80.txt'.format(dataset_name))['Pred'].values    
        new_labels = allign_cluster_labels(true_labels, new_labels)
    elif method_name == "scgnn":
        # import ipdb; ipdb.set_trace()
        new_labels = pd.read_csv('output/scgnn/{0}/{0}_LTMG_10-0.1-0.9-0.0-0.3-0.1_1.0_0.0_results.txt'.format(dataset_name), header=None).values.flatten()
        new_labels = allign_cluster_labels(true_labels, new_labels)
    else:
        if clustering_method == "KMeans":
            kmeans = KMeans(n_clusters=np.unique(true_labels).shape[0])##.fit(anndata.obsm['X_pca'])
            original_labels =  kmeans.fit_predict(embedings)
            new_labels = allign_cluster_labels(true_labels, original_labels)
        else:
            print('Method not implemented ~_~')
            raise NotImplementedError
    ## as these are unsupervised methods, we need to allign the labels (here we are using  method lifted from SGCNN)
    return new_labels 
def get_metrics(true_label, pred_label, emebdings, dimension_reduction_method="PCA"):
    scores = {}
    ## get silhouette score
    scores["silhouette_score"] = metrics.silhouette_score(emebdings, true_label)
    ## get adjust random index 
    scores["adjusted_rand_index"] = metrics.adjusted_rand_score(true_label, pred_label)                 
    ## get normalized mutual info score
    scores["normalized_mutual_info_score"] = metrics.normalized_mutual_info_score(true_label, pred_label)
    scores["accuracy"] = metrics.accuracy_score(true_label, pred_label)
    scores['f1_macro'] = metrics.f1_score(true_label, pred_label, average='macro')
    scores['precision_macro'] = metrics.precision_score(true_label, pred_label, average='macro')
    scores['recall_macro'] = metrics.recall_score(true_label, pred_label, average='macro')
    scores['f1_micro'] = metrics.f1_score(true_label, pred_label, average='micro')
    scores['precision_micro'] = metrics.precision_score(true_label, pred_label, average='micro')
    scores['recall_micro'] = metrics.recall_score(true_label, pred_label, average='micro')
    return scores

def make_file_header(path="output/results.csv"):
    with open (path, 'w') as f:
        f.write("Dataset, Method, Silhouette Score, Adjusted Rand Index, Normalized Mutual Info Score, Accuracy, F1 Macro, Precision Macro, Recall Macro, F1 Micro, Precision Micro, Recall Micro\n")
def write_results_to_file(path="output/results.csv", dataset_name="Chung", method_name="raw", scores={}):
    with open (path, 'a') as f:
        f.write("{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}\n".format(dataset_name, method_name, scores["silhouette_score"], scores["adjusted_rand_index"], scores["normalized_mutual_info_score"], scores["accuracy"], scores['f1_macro'], scores['precision_macro'], scores['recall_macro'], scores['f1_micro'], scores['precision_micro'], scores['recall_micro'])
        )

def allign_cluster_labels(true_label, pred_label):
    l1 = list(set(true_label))
    numclass1 = len(l1)

    l2 = list(set(pred_label))
    numclass2 = len(l2)
    if numclass1 != numclass2:
        print('Class Not equal, Error!!!!')
        return 0

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(true_label) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if pred_label[i1] == c2]

            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()

    indexes = m.compute(cost) ##

    # get the match results
    predicted_label = np.zeros(len(pred_label))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(pred_label) if elm == c2]
        predicted_label[ai] = c
    return predicted_label

def fit_scvi(anndata, save_path="output/scvi/model.pt"):
    model = scvi.model.SCVI(anndata)
    model.train()
    model.save(save_path, overwrite=True)
    return model
def fit_scvi_ld(anndata, save_path="output/scvi_ld/model.pt"):
    model = scvi.model.LinearSCVI(anndata)
    model.train()
    model.save(save_path, overwrite=True)
    return model
def manual_preprocess(anndata ,method_name="scvi", dataset_name="Chung"):
    """
    SCGNN defines there pre-processing step as:
    'Data filtering and quality control are the first steps of data preprocessing.
    Due to the high dropout rate of scRNA-seq expression data, only genes expressed
    as non-zero in more than 1% of cells, and cells expressed as non-zero in more than
    1% of genes are kept. Then, genes are ranked by standard deviation, i.e., the top
    2000 genes in variances are used for the study. All the data are log-transformed.'
    there is another section on Left-truncated mixed Gaussian (LTMG) modeling that i 
    think is more of a complex modeling assumption so I am just going to ignore it for other 
    models 
    """
    min_cells = round(anndata.X.shape[0]*0.01)
    min_genes = round(anndata.X.shape[1]*0.01)
    if method_name == "raw":
        sc.pp.filter_cells(data = anndata,min_genes=min_genes, inplace= True)
        sc.pp.filter_genes(data = anndata,min_cells=min_cells, inplace= True)
        sc.pp.log1p(anndata, copy=False)
    elif method_name == 'scgnn':
        anndata = sc.read_csv('./data/benchmark_data/{0}/{0}.csv'.format(dataset_name))
        anndata.obs['cell_type'] = pd.read_csv("./data/benchmark_data/{0}/{0}_cell_label.csv".format(dataset_name))['cell_type'].values
        return anndata
    elif method_name == "scvi":
        sc.pp.filter_cells(data = anndata,min_genes=min_genes, inplace= True)
        sc.pp.filter_genes(data = anndata,min_cells=min_cells, inplace= True)
        scvi.model.SCVI.setup_anndata(anndata, labels_key='cell_type')
        sc.pp.log1p(anndata, copy=False)
    elif method_name == "scvi_ld":
        sc.pp.filter_cells(data = anndata,min_genes=min_genes, inplace= True)
        sc.pp.filter_genes(data = anndata,min_cells=min_cells, inplace= True)
        scvi.model.LinearSCVI.setup_anndata(anndata, labels_key='cell_type')
        sc.pp.log1p(anndata, copy=False)
    elif method_name == "clear":
        sc.pp.filter_cells(data = anndata,min_genes=min_genes, inplace= True)
        sc.pp.filter_genes(data = anndata,min_cells=min_cells, inplace= True)
        sc.pp.log1p(anndata)
        save_path = "data/benchmark_data/{0}/{0}_preprocessed.h5ad".format(dataset_name)
        anndata.write(save_path)
    elif method_name == "contrastive_sc":
        sc.pp.filter_cells(data = anndata,min_genes=min_genes, inplace= True)
        sc.pp.filter_genes(data = anndata,min_cells=min_cells, inplace= True)
        sc.pp.log1p(anndata)
        save_path = "data/benchmark_data/{0}/{0}_preprocessed.h5".format(dataset_name)
        anndata.write(save_path)
    elif method_name == 'scmae':
        sc.pp.filter_cells(data = anndata,min_genes=min_genes, inplace= True)
        sc.pp.filter_genes(data = anndata,min_cells=min_cells, inplace= True)
        sc.pp.log1p(anndata)
        anndata.Y = anndata.obs['cell_type']
        save_path = "data/benchmark_data/{0}/{0}.h5".format(dataset_name)
        anndata.write(save_path)        
    else:
        raise NotImplementedError
def get_clear_embeds(anndata, dataset_name):
    embed_path = 'output/CLEAR/feature_CLEAR_{0}_preprocessed.csv'.format(dataset_name)
    return pd.read_csv(embed_path, header=None).values
def run_contrastive_sc(anndata, dataset_name):
    X = anndata.X
    y = anndata.obs['cell_type']
    cluster_number = np.unique(y).shape[0]
    results = train.run(X,
                        cluster_number,
                        dataset=dataset_name,
                        Y=y,
                        nb_epochs=30,
                        layers=[200, 40, 60],
                        dropout = 0.9,
                        save_pred = True,
                        cluster_methods =["KMeans", "Leiden"])
    return results['features']
def plot_PCA(anndata, method_name="raw", dataset_name="Chung"):
    plt.clf()
    sc.pp.neighbors(anndata, n_neighbors=10, use_rep="raw_embedings")
    # import ipdb; ipdb.set_trace()
    pca_returns = sc.tl.pca(anndata.obsm["raw_embedings"], return_info= True)
    anndata.obsm["X_pca"] = pca_returns[0]
    # import ipdb; ipdb.set_trace()
    # anndata.varm["PCs"] = pca_returns[1]
    anndata.uns['pca'] = {'variance': pca_returns[2], 'variance_ratio': pca_returns[3]}
    fig_title = "PCA of {0} on {1}".format(dataset_name, method_name)
    pca_plot = sc.pl.pca(anndata, color='cell_type',title=fig_title, show=False)#, save = 'figs/{0}/{1}/pca_plot'.format(dataset_name, method_name))
    os.makedirs('figs/{0}/{1}'.format(dataset_name, method_name), exist_ok=True)
    plt.savefig('figs/{0}/{1}/pca_plot'.format(dataset_name, method_name))
    pca_var  = sc.pl.pca_variance_ratio(anndata, n_pcs=50, log=True)
    plt.savefig('figs/{0}/{1}/pca_var'.format(dataset_name, method_name))
def plot_umap(anndata, method_name="raw", dataset_name="Chung"):
    plt.clf()
    sc.pp.neighbors(anndata, n_neighbors=10, use_rep="raw_embedings")
    sc.tl.umap(anndata)
    fig_title = "Umap of {0} on {1}".format(dataset_name, method_name)
    umap_plot = sc.pl.umap(anndata, color='cell_type',title=fig_title, show=False)#, save = 'figs/{0}/{1}/umap_plot'.format(dataset_name, method_name))
    os.makedirs('figs/{0}/{1}'.format(dataset_name, method_name), exist_ok=True)
    plt.savefig('figs/{0}/{1}/umap_plot'.format(dataset_name, method_name))
def plot_raw(anndata, method_name="raw", dataset_name="Chung"):
    ## clear figure 
    plt.clf()
    plt.scatter(anndata.obsm["raw_embedings"][:,0], anndata.obsm["raw_embedings"][:,1], c=anndata.obs['cell_type'])
    plt.savefig('figs/{0}/{1}/raw_learned_emebdings_plot'.format(dataset_name, method_name))
 
def plot_tsne(anndata, method_name="raw", dataset_name="Chung"):
    plt.clf()
    sc.pp.neighbors(anndata, n_neighbors=10, use_rep="raw_embedings")
    sc.tl.tsne(anndata, n_pcs=2, perplexity=30, learning_rate=1000, n_jobs=4)
    sc.pl.tsne(anndata, color='cell_type', show=False)
    plt.savefig('figs/{0}/{1}/tsne_plot'.format(dataset_name, method_name))


def plot_clusters(anndata, method_name="raw", dataset_name="Chung"):
    plot_PCA(anndata, method_name, dataset_name)
    plot_umap(anndata, method_name, dataset_name)
    plot_tsne(anndata, method_name, dataset_name)
    plot_raw(anndata, method_name, dataset_name)
def make_latex_table(results_path="output/results_temp.csv", ML_metrics=False, micro=False):
    results = pd.read_csv(results_path)
    if not micro:
        to_drop = [' F1 Micro', ' Precision Micro',
       ' Recall Micro']
        to_keep_rename = {' F1 Macro': 'F1', ' Precision Macro': 'Precision', ' Recall Macro': 'Recall'}
    else:
        to_drop=[ ' F1 Macro', ' Precision Macro',
       ' Recall Macro']
        to_keep_rename = {' F1 Micro': 'F1', ' Precision Micro': 'Precision', ' Recall Micro': 'Recall'}
    if ML_metrics:
        results = results.drop(to_drop, axis=1)
        results = results.rename(columns = {' Silhouette Score': 'Silhouette', ' Adjusted Rand Index': 'ARI', ' Normalized Mutual Info Score': 'NMI'})
        results = results.rename(columns = to_keep_rename)
    else:
        results = results.drop([' F1 Macro', ' Precision Macro',
       ' Recall Macro', ' F1 Micro', ' Precision Micro',
       ' Recall Micro', ' Accuracy'], axis=1)
        results = results.rename(columns = {' Silhouette Score': 'Silhouette', ' Adjusted Rand Index': 'ARI', ' Normalized Mutual Info Score': 'NMI'})
    results = results.groupby(['Dataset', ' Method']).mean()
    dataset_order = ["Chung" , 'Kolodziejczyk', 'Klein', 'Zeisel']
    old_method_order = [' raw', ' scvi', ' scvi_ld', ' clear', ' contrastive_sc', ' scmae', ' scgnn']
    new_method_order = ['PCA + K-Means', 'scVI', 'scVI-LD', 'CLEAR', 'Contrastive-SC', 'scMAE', 'scGNN']
    idx_1 = pd.MultiIndex.from_product([dataset_order, old_method_order], names=['Dataset', ' Method'])
    idx_2 = pd.MultiIndex.from_product([dataset_order, new_method_order], names=['Dataset', ' Method'])
    results = results.reindex(idx_1)
    df = results.copy()
    temp = results.index.set_levels([dataset_order, new_method_order])
    temp = temp.reindex(idx_2)
    results.index = temp[0]
    # output_vals = results.values[temp[1]]
    # output_df = pd.DataFrame(output_vals, index = temp[0], columns = results.columns)
    results.to_latex(index = True, buf='output/results.tex', escape=True, float_format="%.2f")

def color_table_by_method(colors=['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'brown'], latex_path="output/results.tex"):
    # base_str = '\cellcolor{red!10}'
    digit_pattern = r'\d{2}'
    color_pointer = 0 
    with open(latex_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            match = re.findall(digit_pattern, lines[i])
            if len(match)>0:
                cell_color = '\cellcolor{' + colors[color_pointer] + '!10}'
                lines[i] = lines[i].replace('&',  '&'+cell_color)
                if color_pointer < len(colors)-1:
                    color_pointer += 1
                else:
                    color_pointer = 0
    with open(latex_path, 'w') as f:
        f.writelines(lines)
def color_table_by_method_type(colors=['red','blue', 'green', 'yellow', 'purple', 'orange', 'brown'], latex_path="output/results.tex"):
    method_type_map = {'PCA + K-Means':'dim_reduction', 'scVI':'dim_reduction', 'scVI-LD':'dim_reduction', 'CLEAR':'contrastive', 'Contrastive-SC':'contrastive', 'scMAE':'contrastive', 'scGNN':'graphical'}
    type_color_map = {'dim_reduction':'red', 'contrastive':'blue', 'graphical':'green'}
    digit_pattern = r'\d{2}'
    with open(latex_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            match = re.findall(digit_pattern, lines[i])
            if len(match)>0:
                method = lines[i].split('&')[1].strip()
                method_type = method_type_map[method]
                cell_color = '\cellcolor{' + type_color_map[method_type] + '!10}'
                lines[i] = lines[i].replace('&',  '&'+cell_color)
    with open(latex_path, 'w') as f:
        f.writelines(lines)