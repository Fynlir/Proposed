### Import required packages
import scipy.sparse as sp
import numpy as np
import pandas as pd
import igraph as ig
import matplotlib.pyplot as plt
import warnings
import time
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from typing import Dict, List, Set
from sklearn.decomposition import TruncatedSVD
from datetime import datetime
from itertools import combinations
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.stats import ttest_ind, stats
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from scipy.sparse import csr_matrix, diags
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import KNeighborsClassifier, KNeighborsTransformer
import networkx as nx
import community as community_louvain
import seaborn as sns

### Data importing 
log_scdata = pd.read_csv('deng_test1.csv', index_col=0) ## This one is normalized data
log_scdata = pd.DataFrame(log_scdata)
data_label = pd.read_csv('deng_label.csv',index_col=0)
print(log_scdata.shape[1])
data_label = pd.DataFrame(data_label)

log_scdata.index = log_scdata.index.str.upper()
log_scdata.columns = log_scdata.columns.str.upper()
unique_log_scdata_genes = set(log_scdata.index)


pathways_df = pd.read_csv('exported_pathways.csv', 
                           index_col=0,  # Use first column as index
                           low_memory=False,  # Handle mixed type columns
                           dtype=object)  # Read all columns as strings initially


def analyze_gene_features_and_pathways(log_scdata, pathway_input, p_feat=0.2, expression_threshold=90, verbose=True):
    """
    Filters genes based on pathways first, removing genes with zero expression. Then, adds genes with top 10% highest 
    expression and variance across cells (calculated from the original dataset). Also returns topK and variable genes.
    """
    if verbose:
        print("=== Pathway Analysis ===")

    
    # Load pathway data
    if isinstance(pathway_input, pd.DataFrame):
        pathways_df = pathway_input
    elif isinstance(pathway_input, str):
        pathways_df = pd.read_csv(pathway_input, index_col=0, low_memory=False, dtype=object)
    else:
        raise ValueError("pathway_input must be either a pandas DataFrame or a path to a CSV file")

    # Collect all pathway genes
    all_pathway_genes = set(gene for pathway in pathways_df.index 
                            for gene in pathways_df.loc[pathway] if pd.notna(gene))
    
    # Filter genes to keep only those found in pathways and remove zero-expression genes
    pathway_genes = list(set(log_scdata.index).intersection(all_pathway_genes))
    filtered_scdata = log_scdata.loc[pathway_genes]
    nonzero_mask = (filtered_scdata > 0).sum(axis=1) > 0
    filtered_scdata = filtered_scdata.loc[nonzero_mask]
    print("filter_genes:  ", len(filtered_scdata))
    
    if verbose:
        print(f"Genes in pathways: {len(pathway_genes)}")
        print(f"Genes retained after removing zero-expression genes: {len(filtered_scdata)}")
    
    # Compute top 10% highest expression & variance genes (from the original dataset)
    rm = np.mean(log_scdata.values, axis=1)
    gene_means = log_scdata.mean(axis=1)
    gene_variances = log_scdata.var(axis=1)
    top_expr_threshold = np.percentile(rm, expression_threshold)
    top_var_threshold = np.percentile(gene_variances, expression_threshold)
    
    top_expr_genes = log_scdata.index[rm >= top_expr_threshold]
    top_var_genes = log_scdata.index[gene_variances >= top_var_threshold]
    
    # Compute variable genes using MAD-based approach
    rm = np.median(filtered_scdata.values, axis=1)
    mad = np.median(np.abs(filtered_scdata.values - rm[:, np.newaxis]), axis=1) * 1.4826
    normalized_mad = (mad - mad.min()) / (mad.max() - mad.min() + 1e-10)
    composite_score = normalized_mad

    
    sorted_indices = np.argsort(composite_score)
    n_vargenes = max(1, int(np.ceil(p_feat * filtered_scdata.shape[0])))
    vargenes = filtered_scdata.index[sorted_indices[-n_vargenes:]]

    # Compute topK genes
    n_topK = max(1, int(0.05 * log_scdata.shape[0]))
    topK = log_scdata.index[sorted_indices[-n_topK:]]
    final_genes = set(vargenes).union((set(top_expr_genes)).intersection(set(top_var_genes)))
   
    
    if verbose:
        print(f"Top 10% highest expression genes: {len(top_expr_genes)}")
        # print(f"Top 10% highest variance genes: {len(top_var_genes)}")
        print(f"Total genes retained after all filtering: {len(final_genes)}")
        print(f"Variable genes (VarGenes): {len(vargenes)}")
        print(f"Top K genes: {len(topK)}")
    
    return {
        "filtered_genes": list(final_genes),
        "VarGenes": list(vargenes),
        "topK": list(topK)
    }



def filter_missing_genes_from_pathways(log_scdata, pathway_input, min_retention_rate, verbose=True, save_retained=None):
    """
    Removes genes not present in expression data and keeps only pathways with specified minimum retention rate.
    
    Parameters:
    -----------
    log_scdata : pandas.DataFrame
        Expression data with genes as index
    pathway_input : str or pandas.DataFrame
        Either a path to CSV file or DataFrame containing pathway information
    min_retention_rate : float
        Minimum fraction of genes that must be retained for a pathway (default: 0.8)
    verbose : bool
        Whether to print summary statistics (default: True)
    save_retained : str, optional
        Path to save the list of retained pathways (default: None)
        
    Returns:
    --------
    tuple
        (filtered_pathway_df, retained_pathway_df) where:
        - filtered_pathway_df: DataFrame with filtered genes
        - retained_pathway_df: DataFrame in original format but only with retained pathways
    """
    if isinstance(pathway_input, pd.DataFrame):
        pathways_df = pathway_input.copy()
    elif isinstance(pathway_input, str):
        pathways_df = pd.read_csv(pathway_input, index_col=0, low_memory=False, dtype=object)
    else:
        raise ValueError("pathway_input must be either a pandas DataFrame or a path to a CSV file")

    expression_genes = set(log_scdata.index)
    
    # Count before filtering
    genes_before = sum(pathways_df.notna().sum())
    unique_genes_before = len(set([gene for genes in pathways_df.values for gene in genes if pd.notna(gene)]))
    
    # Filter genes and calculate retention rates
    filtered_df = pathways_df.copy()
    retention_rates = {}
    
    for pathway in filtered_df.index:
        pathway_genes = filtered_df.loc[pathway]
        original_gene_count = sum(pd.notna(pathway_genes))
        
        filtered_genes = [gene if (pd.notna(gene) and gene in expression_genes) else np.nan
                        for gene in pathway_genes]
        filtered_df.loc[pathway] = filtered_genes
        
        retained_gene_count = sum(pd.notna(filtered_genes))
        retention_rate = retained_gene_count / original_gene_count if original_gene_count > 0 else 0
        retention_rates[pathway] = retention_rate
    
    # Keep only pathways meeting the retention rate threshold
    pathways_to_keep = [pathway for pathway, rate in retention_rates.items() 
                       if rate >= min_retention_rate]
    
    # Create two versions of filtered data:
    # 1. Filtered genes version (as before)
    filtered_df = filtered_df.loc[pathways_to_keep]
    filtered_df = filtered_df.dropna(axis=1, how='all')
    
    # 2. Original format version with only retained pathways
    retained_pathway_df = pathways_df.loc[pathways_to_keep]
    
    # Count after filtering
    genes_after = sum(filtered_df.notna().sum())
    unique_genes_after = len(set([gene for genes in filtered_df.values for gene in genes if pd.notna(gene)]))
    
    if verbose:
        print(f"Before filtering:")
        print(f"Shape: {pathways_df.shape}")
        print(f"Total genes: {genes_before}")
        print(f"Unique genes: {unique_genes_before}")
        print(f"\nAfter filtering:")
        print(f"Shape: {filtered_df.shape}")
        print(f"Total genes: {genes_after}")
        print(f"Unique genes: {unique_genes_after}")
        print(f"\nPathways retained: {len(pathways_to_keep)} out of {len(pathways_df)} " 
              f"({len(pathways_to_keep)/len(pathways_df)*100:.1f}%)")
       
    # Save retained pathways to file if path is provided
    if save_retained:
        with open(save_retained, 'w') as f:
            for pathway in pathways_to_keep:
                f.write(f"{pathway}\t{retention_rates[pathway]:.3f}\n")
    
    return filtered_df, retained_pathway_df





#### Make KNN function
def make_knn(data, k_num):
    # Calculate pairwise Euclidean distances
    edist = pdist(data, metric='euclidean')
    edist = squareform(edist)
    # Create the k-nearest neighbors graph
    knn_graph = kneighbors_graph(data, n_neighbors=k_num, mode='connectivity', include_self=False)
    # Convert to sparse matrix
    knn_sparse = csr_matrix(knn_graph)
    return knn_sparse
#### Compute separation socres for merging clusters 

def compute_sep_scores(log_scdata, membership, topK):
    """
    Computes separation scores between clusters using t-test on topK genes.
    FIX: Added axis=1 to ttest_ind to handle (n_genes, n_cells) shape correctly.
    """
    import numpy as np
    
    # Ensure membership is an array
    membership = np.array(membership)
    unique_labels = np.sort(np.unique(membership))
    n_clusters = len(unique_labels)
    
    # Initialize score matrix
    sep_scores = np.zeros((n_clusters, n_clusters))
    
    # Pre-select data for topK genes to speed up
    # Assuming log_scdata is (Genes x Cells)
    data_topK = log_scdata.loc[topK]
    
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            label_i = unique_labels[i]
            label_j = unique_labels[j]
            
            # Get cells for each cluster
            # Note: Assuming columns are cells. If using indices, use .iloc or ensure alignment
            cells_i = (membership == label_i)
            cells_j = (membership == label_j)
            
            # Check if clusters are empty (safety check)
            if np.sum(cells_i) == 0 or np.sum(cells_j) == 0:
                sep_scores[i, j] = sep_scores[j, i] = 0
                continue
            
            # Extract expression data: Shape (n_topK_genes, n_cells_in_cluster)
            data1 = data_topK.loc[:, cells_i].values
            data2 = data_topK.loc[:, cells_j].values
            
            # --- CRITICAL FIX: axis=1 ---
            # Compute t-test across cells (axis=1) for each gene
            # nan_policy='omit' helps if there are NaNs
            t_stat, p_val = ttest_ind(data1, data2, axis=1, equal_var=False, nan_policy='omit')
            
            # Compute separation score (e.g., mean absolute t-statistic)
            # You can adjust this metric based on your original logic
            # Usually we want the minimum separation or average separation
            score = np.nanmean(np.abs(t_stat))
            
            sep_scores[i, j] = score
            sep_scores[j, i] = score
            
    return sep_scores


### Merging clusters
def merge_clusters(membership, sep_scores):
    """
    Merge clusters with improved merge strategy and robust handling of cluster labels.
    """
    membership = np.array(membership).copy()
    ncl = len(np.unique(membership))
    my_membership = {}
    my_membership[str(ncl)] = membership.copy()
    
    # Track current cluster labels
    current_labels = np.unique(membership)
    
    for target_clusters in range(ncl - 1, 0, -1):
        cur_mem = my_membership[str(target_clusters + 1)].copy()
        current_labels = np.unique(cur_mem)
        
        if len(current_labels) <= target_clusters:
            my_membership[str(target_clusters)] = cur_mem.copy()
            continue
            
        # Find best pair to merge based on separation scores
        min_score = np.inf
        merge_pair = None
        
        for i, label1 in enumerate(current_labels):
            for j, label2 in enumerate(current_labels[i+1:], i+1):
                if i < sep_scores.shape[0] and j < sep_scores.shape[0]:
                    score = sep_scores[i, j]
                    if score < min_score:
                        min_score = score
                        merge_pair = (label1, label2)
        
        if merge_pair is None:
            my_membership[str(target_clusters)] = cur_mem.copy()
            continue
            
        # Perform merge
        mask = cur_mem == merge_pair[0]
        cur_mem[mask] = merge_pair[1]
        my_membership[str(target_clusters)] = cur_mem.copy()
    
    # Relabel clusters to ensure consecutive integers starting from 1
    new_membership = {}
    for n_clusters, mem in my_membership.items():
        mem = mem.copy()
        unique_labels = np.unique(mem)
        label_map = {old: new for new, old in enumerate(unique_labels, 1)}
        new_mem = np.array([label_map[x] for x in mem])
        new_membership[str(len(unique_labels))] = new_mem
    
    return new_membership
 
#### Make doubly stochastic matrix function
def make_dsm(my_mat):
    # Ensure input is a sparse matrix
    my_mat = csr_matrix(my_mat)
    # Compute D1 matrix
    row_sums = np.array(my_mat.sum(axis=1)).flatten()
    D1 = diags(1 / row_sums)
    # Compute intermediate TT matrix
    TT = D1.dot(my_mat)
    # Compute D2 matrix
    col_sums = np.array(TT.sum(axis=0)).flatten()
    D2 = diags(1 / np.sqrt(col_sums))
    # Compute final TT matrix
    TT = TT.dot(D2)
    # Compute DSM matrix
    my_dsm = TT.dot(TT.T)
    return my_dsm


    

def calculate_clustering_metrics(true_labels, cluster_labels):

    
    
    # Ensure inputs are numpy arrays
    true_labels = np.array(true_labels)
    cluster_labels = np.array(cluster_labels)
    
    # Convert labels to a uniform integer encoding to avoid mixed types
    le = LabelEncoder()
    true_labels = le.fit_transform(true_labels)
    cluster_labels = le.fit_transform(cluster_labels)
    
    # Calculate metrics
    ari = adjusted_rand_score(true_labels, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)
    
    # Compute Clustering Purity
    def clustering_purity(true_labels, cluster_labels):
        clusters = np.unique(cluster_labels)
        total_samples = len(true_labels)
        purity_sum = 0
        
        for cluster in clusters:
            cluster_indices = np.where(cluster_labels == cluster)[0]
            true_labels_in_cluster = true_labels[cluster_indices]
            most_common_label_count = Counter(true_labels_in_cluster).most_common(1)[0][1]
            purity_sum += most_common_label_count
        
        return purity_sum / total_samples
    
    purity = clustering_purity(true_labels, cluster_labels)
    
    # Print detailed information
    print("=== Clustering Evaluation ===")
    print(f"Number of true classes: {len(np.unique(true_labels))}")
    print("True label distribution:")
    unique_true, counts_true = np.unique(true_labels, return_counts=True)
    for label, count in zip(unique_true, counts_true):
        print(f"  Label {label}: {count}")
    
    print(f"\nNumber of clusters: {len(np.unique(cluster_labels))}")
    print("Cluster distribution:")
    unique_cluster, counts_cluster = np.unique(cluster_labels, return_counts=True)
    for cluster, count in zip(unique_cluster, counts_cluster):
        print(f"  Cluster {cluster}: {count}")
    
    print(f"\nAdjusted Rand Index: {ari:.4f}")
    print(f"Normalized Mutual Information: {nmi:.4f}")
    print(f"Clustering Purity: {purity:.4f}")
    
    return ari, nmi, purity





def scCLUE(log_scdata, nEns=15, K=[5,10], smin=0.6, smax=1, nPCs=15, nCls=3, resolution=1.0):
    start_time = time.time()
    component_times = {}
    
    print("Running Identifying features")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    

    features1 = analyze_gene_features_and_pathways(log_scdata, pathways_df)
    topK = features1["topK"]
    common_genes = features1["filtered_genes"]
    print(len(common_genes))
    #varGenes = features1["VarGenes"]
    nCells = log_scdata.shape[1]
    n = 30
    ensNet = csr_matrix((nCells, nCells), dtype=np.float64)
    
    # Time the ensemble creation
    ensemble_start = time.time()
    print("Running scCLUE")
    for ix in range(nEns):
        for ik in K:
            # Sample genes
            sample_rng = round(np.random.uniform(smin, smax), 2)
            min_gene_threshold = nCells  
            sample_size = max(min_gene_threshold, int(np.ceil(sample_rng * len(common_genes))))
            sampled_genes = np.random.choice(common_genes, size=sample_size, replace=False)
            
            reduction_data = log_scdata.loc[sampled_genes].T

            
            # UMAP reduction
            umap_res = UMAP(n_components=nPCs).fit_transform(reduction_data)
            umap_knn = make_knn(umap_res, k_num=ik)
            umap_knn = umap_knn + umap_knn.T
            
            # PCA reduction

            try:
                pca_res = PCA(n_components=nCells).fit_transform(reduction_data)
            except Exception as e:
                print(f"[Warning] PCA with n_components={nCells} failed: {e}")
                max_components = min(reduction_data.shape[0], reduction_data.shape[1])
                pca_res = PCA(n_components=min(nCells, max_components)).fit_transform(reduction_data)

    
            
            # Time KNN creation
            cor_dist = np.corrcoef(pca_res.T)
            cor_knn = KNeighborsTransformer(n_neighbors=ik, metric='precomputed').fit_transform(1 - cor_dist)
            cor_knn = cor_knn + cor_knn.T
            
            
            
            temp = umap_knn + cor_knn
            ensNet += make_dsm(temp)
            ensNet_array = ensNet.toarray()
   
    
    # Time community detection
    A = np.array(ensNet_array)
    G = nx.Graph(A)
    gg = nx.from_numpy_array(ensNet)
    print(gg)
    partition = community_louvain.best_partition(gg, resolution=1.0)
    membership = np.array(list(partition.values())) + 1
    
    
    # Time single cluster handling
    
    sind = np.where(np.bincount(membership) == 1)[0]
    if len(sind) != 0:
        clid = np.setdiff1d(np.unique(membership), sind)
        for ss in sind:
            single = np.where(membership == ss)[0]
            mscore = np.zeros((1, len(clid)))
            for ii, cc in enumerate(clid):
                cid = np.where(membership == cc)[0]
                mscore[0, ii] = np.mean(ensNet[single[:, None], cid])
            membership[single] = clid[np.argmax(mscore)]

        clid = np.unique(membership)
        new_mem = membership.copy()
        for cid, mk in enumerate(clid, start=1):
            idx = np.where(membership == mk)[0]
            new_mem[idx] = cid
        membership = new_mem
    print("Louvain membership is",len(set(membership)))
    
    # Time cluster merging
  
    unique_clusters = np.unique(membership)
    if len(unique_clusters) > nCls:
        sep_scores = compute_sep_scores(log_scdata=log_scdata, membership=membership, topK=features1["topK"])
        membership = merge_clusters(membership, sep_scores)
    print('membership',len(membership))

    # Calculate total time
    total_time = time.time() - start_time
    
    # Print timing results
    print("\nTiming Results:")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return {
        'membership': membership,
        'execution_time': total_time,
        
    }

##Run the algorithm with timing
final_cls = scCLUE(log_scdata, nCls=5)
cluster = final_cls['membership']

try:
    Eva = calculate_clustering_metrics(data_label['x'].values, cluster)
except Exception:
    Eva = calculate_clustering_metrics(data_label['x'].values, cluster['5'])




