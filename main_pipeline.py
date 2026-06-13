### ============================================================
### Pipeline: ccImpute → scCLUE
### Stage 1: Dropout imputation using ccImpute
### Stage 2: Cell type clustering using scCLUE
### ============================================================

### ---------- Imports ----------
import numpy as np
import pandas as pd
import multiprocessing
import scipy.sparse as sp
import igraph as ig
import matplotlib.pyplot as plt
import warnings
import time
import networkx as nx
import community as community_louvain
import seaborn as sns

from scipy.sparse.linalg import svds
from scipy.sparse import issparse, csr_matrix, diags
from scipy.stats import ttest_ind
from scipy.spatial.distance import pdist, squareform
from joblib import Parallel, delayed
from sklearn.utils.extmath import randomized_svd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import kneighbors_graph, KNeighborsTransformer
from sklearn.manifold import TSNE
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Union, Dict, List, Set
from collections import Counter
from datetime import datetime
from itertools import combinations
from multiprocessing import Pool
from umap import UMAP

import anndata as ad


### ============================================================
### SECTION 1 — Load Input Data
### ============================================================

logX = pd.read_csv(
    "log_usoskin.csv",      ### <-- Change to your normalized gene expression dataset
    index_col=0,
    engine="python"
)

data_label  = pd.read_csv("usoskin_label.csv", index_col=0)
pathways_df = pd.read_csv(
    "exported_pathways.csv",
    index_col=0,
    low_memory=False,
    dtype=object
)

print("Input data loaded successfully.")
print(f"Expression matrix shape: {logX.shape}")


### ============================================================
### SECTION 2 — ccImpute Functions
### ============================================================

def calculate_weights_by_variance(x):
    return np.var(x, axis=1)

def column_wise_spearman_corr(x):
    ranked_x = np.apply_along_axis(lambda a: np.argsort(np.argsort(a)), 0, x)
    return np.corrcoef(ranked_x, rowvar=False)

def weighted_corr(x, w, method='pearson'):
    if method == 'spearman':
        x = np.apply_along_axis(lambda a: np.argsort(np.argsort(a)), 0, x)
    mean_w  = np.average(x, axis=0, weights=w)
    cov_w   = np.cov(x, rowvar=False, aweights=w)
    std_w   = np.sqrt(np.diag(cov_w))
    return cov_w / np.outer(std_w, std_w)

def calculate_correlation(method, x, w=None, n_cores=1):
    if issparse(x):
        x = x.toarray()
    if method == 'spearman':
        return weighted_corr(x, w, method='spearman') if w is not None else column_wise_spearman_corr(x)
    elif method == 'pearson':
        return weighted_corr(x, w, method='pearson') if w is not None else np.corrcoef(x, rowvar=False)
    else:
        raise ValueError("Selected correlation method not available.")

def get_scale(x, n_cores=1):
    return {'means': np.mean(x, axis=0), 'sds': np.std(x, axis=0, ddof=1)}

def doSVD(x, svdMaxRatio=0.08, nCeil=2000, nCores=1):
    nv    = int(np.ceil(svdMaxRatio * min(nCeil, x.shape[1])))
    scale = get_scale(x, nCores)
    centered_scaled_x = (x - scale['means']) / scale['sds']
    u, s, v = randomized_svd(centered_scaled_x, n_components=nv, random_state=None)
    return v

def estkTW(x):
    n, p = x.shape
    sqrt_term = np.sqrt(n - 1) + np.sqrt(p)
    muTW      = sqrt_term ** 2
    sigmaTW   = sqrt_term * ((1 / np.sqrt(n - 1)) + (1 / np.sqrt(p))) ** (1/3)
    bd        = 3.273 * sigmaTW + muTW
    x_scaled  = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    cov_matrix = np.dot(x_scaled.T, x_scaled)
    evals = np.linalg.eigvalsh(cov_matrix)
    return np.sum(evals > bd)

def getConsMtx(cluster_results, consMin):
    consMin   = 0.75
    consensus = np.corrcoef(cluster_results)
    consensus[consensus < consMin] = 0
    return consensus

def getPConsMtx(kmResults, consMin):
    consMin  = 0.75
    d        = np.array(kmResults).reshape(len(kmResults[0]), -1)
    consMtx  = getConsMtx(d, consMin=0.75)
    np.fill_diagonal(consMtx, 0)
    consMtx[consMtx < consMin] = 0
    consMtx  = np.apply_along_axis(
        lambda i: i / np.sum(i) if np.sum(i) != 0 else i, axis=1, arr=consMtx
    )
    consMtx  = np.nan_to_num(consMtx)
    return consMtx

def kmAux(i, input, k, kmNStart, kmMax):
    x = input[:, :i]
    kmeans = KMeans(n_clusters=k, max_iter=kmMax, n_init=kmNStart,
                    init='k-means++', random_state=0)
    kmeans.fit(x)
    return kmeans.labels_

def runKM(logX, v, maxSets=8, k=None, consMin=0.75, km_n_start=None, km_max=1000, n_cores=4):
    lv   = int(0.5 * v.shape[1])
    rv   = v.shape[1]
    nDim = np.linspace(lv, rv, maxSets).astype(int)
    if k is None:
        k = estkTW(logX)
    if km_n_start is None:
        km_n_start = 50 if logX.shape[1] >= 2000 else 1000
    km_results = Parallel(n_jobs=n_cores)(
        delayed(kmAux)(i, v, k, km_n_start, km_max) for i in nDim
    )
    cons_mtx = getConsMtx(np.array(km_results).T, consMin)
    return cons_mtx

def findDropouts(logX, consMtx):
    if issparse(logX):
        logX = logX.toarray()
    if isinstance(logX, pd.DataFrame):
        logX = logX.values
    if isinstance(consMtx, pd.DataFrame):
        consMtx = consMtx.values
    logX         = logX.T
    cells, genes = logX.shape
    zero_indices  = (logX == 0)
    vote_m        = zero_indices * 2.0 - 1.0
    result        = (vote_m.T @ consMtx)
    result_reshaped       = result.reshape(genes, -1)
    zero_indices_reshaped = zero_indices.T.reshape(genes, -1)
    votes         = result_reshaped * zero_indices_reshaped
    dropout_indices = np.argwhere(votes < 0)
    return dropout_indices

def calculate_single_dropout(args):
    row, col, em_t, cm = args
    if isinstance(em_t, (pd.DataFrame, pd.Series)):
        em_t = em_t.to_numpy()
    if isinstance(cm, (pd.DataFrame, pd.Series)):
        cm = cm.to_numpy()
    em_values = em_t[:, row] if em_t.ndim == 2 else em_t
    cm_values = cm[:, col]   if cm.ndim == 2   else cm
    em_mask   = (em_values > 0).astype(float)
    div2      = np.dot(em_mask, cm_values)
    numerator = np.dot(em_values, cm_values)
    return numerator / div2 if div2 != 0 else 0

def solver2(cm, em, ids, n_cores=None):
    if isinstance(cm,  pd.DataFrame): cm  = cm.to_numpy()
    if isinstance(em,  pd.DataFrame): em  = em.to_numpy()
    if isinstance(ids, pd.DataFrame): ids = ids.to_numpy()
    ids = ids - 1 if np.min(ids) > 0 else ids
    ids = ids.astype(int)
    em_t = em.T
    if n_cores is None:
        n_cores = max(1, multiprocessing.cpu_count() - 1)
    args = [(int(row), int(col), em_t, cm) for row, col in ids]
    with ThreadPoolExecutor(max_workers=n_cores) as executor:
        imp = np.array(list(executor.map(calculate_single_dropout, args)))
    return imp

def computeDropouts(consMtx, logX, dropIds, fastSolver=True, nCores=1):
    is_x_sparse = issparse(logX)
    if fastSolver:
        imputed_values   = solver2(consMtx, logX, dropIds, nCores)
        imputed_logX     = logX.copy()
        impute_logX_values = imputed_logX.values
        impute_logX_values[tuple(dropIds.T)] = imputed_values
        return imputed_logX
    else:
        if is_x_sparse:
            raise ValueError("Slow solver not supported for sparse matrices.")
        print("Warning: Slow solver selected.")
        imputed_logX = logX.copy()
        imputed_logX.flat[dropIds] = np.random.rand(len(dropIds))
        return imputed_logX

def ccImpute(log_scdata, dist=None, nCeil=2000, svdMaxRatio=0.08, maxSets=8, k=None,
             consMin=0.80, kmNStart=None, kmMax=1000, fastSolver=True, nCores=4, verbose=True):
    start_time = time.time()
    isXSparse  = issparse(log_scdata)
    if verbose:
        print(f"Running ccImpute on dataset ({log_scdata.shape[1]} cells) with {nCores} cores.")
    if dist is None:
        dist = 1 - pairwise_distances(log_scdata.T, metric='correlation')
        if verbose:
            print(f"Distance matrix computed. Time elapsed: {time.time() - start_time:.2f}s")
    v = doSVD(dist, svdMaxRatio=svdMaxRatio, nCeil=nCeil, nCores=nCores)
    if verbose:
        print(f"Dimensional reduction completed. Time elapsed: {time.time() - start_time:.2f}s")
    consMin   = 0.75
    kmResults = runKM(log_scdata, v.T, maxSets=maxSets, k=k, consMin=consMin)
    consMtx   = getPConsMtx(kmResults, consMin=consMin)
    if verbose:
        print(f"Clustering completed. Time elapsed: {time.time() - start_time:.2f}s")
    dropIds = findDropouts(log_scdata, consMtx)
    if verbose:
        print(f"Dropouts identified. Time elapsed: {time.time() - start_time:.2f}s")
    impLogX = computeDropouts(consMtx, log_scdata, dropIds, fastSolver=fastSolver, nCores=nCores)
    if verbose:
        print(f"Dropouts imputed. Time elapsed: {time.time() - start_time:.2f}s")
    return impLogX


### ============================================================
### SECTION 3 — scCLUE Functions
### ============================================================

def analyze_gene_features_and_pathways(log_scdata, pathway_input, p_feat=0.2,
                                        expression_threshold=90, verbose=True):
    if verbose:
        print("=== Pathway Analysis ===")
    if isinstance(pathway_input, pd.DataFrame):
        pathways_df = pathway_input
    elif isinstance(pathway_input, str):
        pathways_df = pd.read_csv(pathway_input, index_col=0, low_memory=False, dtype=object)
    else:
        raise ValueError("pathway_input must be a DataFrame or file path.")

    all_pathway_genes = set(gene for pathway in pathways_df.index
                            for gene in pathways_df.loc[pathway] if pd.notna(gene))
    pathway_genes     = list(set(log_scdata.index).intersection(all_pathway_genes))
    filtered_scdata   = log_scdata.loc[pathway_genes]
    nonzero_mask      = (filtered_scdata > 0).sum(axis=1) > 0
    filtered_scdata   = filtered_scdata.loc[nonzero_mask]

    rm              = np.mean(log_scdata.values, axis=1)
    gene_variances  = log_scdata.var(axis=1)
    top_expr_genes  = log_scdata.index[rm >= np.percentile(rm, expression_threshold)]
    top_var_genes   = log_scdata.index[gene_variances >= np.percentile(gene_variances, expression_threshold)]

    rm_filt = np.median(filtered_scdata.values, axis=1)
    mad     = np.median(np.abs(filtered_scdata.values - rm_filt[:, np.newaxis]), axis=1) * 1.4826
    normalized_mad  = (mad - mad.min()) / (mad.max() - mad.min() + 1e-10)
    sorted_indices  = np.argsort(normalized_mad)

    n_vargenes = max(1, int(np.ceil(p_feat * filtered_scdata.shape[0])))
    vargenes   = filtered_scdata.index[sorted_indices[-n_vargenes:]]
    n_topK     = max(1, int(0.05 * log_scdata.shape[0]))
    topK       = log_scdata.index[sorted_indices[-n_topK:]]
    final_genes = set(vargenes).union(set(top_expr_genes).intersection(set(top_var_genes)))

    if verbose:
        print(f"Genes in pathways: {len(pathway_genes)}")
        print(f"Genes retained after filtering: {len(final_genes)}")

    return {"filtered_genes": list(final_genes), "VarGenes": list(vargenes), "topK": list(topK)}

def make_knn(data, k_num):
    edist     = squareform(pdist(data, metric='euclidean'))
    knn_graph = kneighbors_graph(data, n_neighbors=k_num, mode='connectivity', include_self=False)
    return csr_matrix(knn_graph)

def compute_sep_scores(log_scdata, membership, topK):
    membership    = np.array(membership)
    unique_labels = np.sort(np.unique(membership))
    n_clusters    = len(unique_labels)
    sep_scores    = np.zeros((n_clusters, n_clusters))
    data_topK     = log_scdata.loc[topK]
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            cells_i = (membership == unique_labels[i])
            cells_j = (membership == unique_labels[j])
            if np.sum(cells_i) == 0 or np.sum(cells_j) == 0:
                continue
            data1 = data_topK.loc[:, cells_i].values
            data2 = data_topK.loc[:, cells_j].values
            t_stat, _ = ttest_ind(data1, data2, axis=1, equal_var=False, nan_policy='omit')
            score = np.nanmean(np.abs(t_stat))
            sep_scores[i, j] = sep_scores[j, i] = score
    return sep_scores

def merge_clusters(membership, sep_scores):
    membership = np.array(membership).copy()
    ncl        = len(np.unique(membership))
    my_membership = {str(ncl): membership.copy()}
    for target_clusters in range(ncl - 1, 0, -1):
        cur_mem       = my_membership[str(target_clusters + 1)].copy()
        current_labels = np.unique(cur_mem)
        if len(current_labels) <= target_clusters:
            my_membership[str(target_clusters)] = cur_mem.copy()
            continue
        min_score, merge_pair = np.inf, None
        for i, label1 in enumerate(current_labels):
            for j, label2 in enumerate(current_labels[i+1:], i+1):
                if i < sep_scores.shape[0] and j < sep_scores.shape[0]:
                    score = sep_scores[i, j]
                    if score < min_score:
                        min_score, merge_pair = score, (label1, label2)
        if merge_pair is None:
            my_membership[str(target_clusters)] = cur_mem.copy()
            continue
        cur_mem[cur_mem == merge_pair[0]] = merge_pair[1]
        my_membership[str(target_clusters)] = cur_mem.copy()
    new_membership = {}
    for n_clusters, mem in my_membership.items():
        mem         = mem.copy()
        unique_labels = np.unique(mem)
        label_map   = {old: new for new, old in enumerate(unique_labels, 1)}
        new_mem     = np.array([label_map[x] for x in mem])
        new_membership[str(len(unique_labels))] = new_mem
    return new_membership

def make_dsm(my_mat):
    my_mat   = csr_matrix(my_mat)
    row_sums = np.array(my_mat.sum(axis=1)).flatten()
    D1       = diags(1 / row_sums)
    TT       = D1.dot(my_mat)
    col_sums = np.array(TT.sum(axis=0)).flatten()
    D2       = diags(1 / np.sqrt(col_sums))
    TT       = TT.dot(D2)
    return TT.dot(TT.T)

def calculate_clustering_metrics(true_labels, cluster_labels):
    true_labels    = np.array(true_labels)
    cluster_labels = np.array(cluster_labels)
    le             = LabelEncoder()
    true_labels    = le.fit_transform(true_labels)
    cluster_labels = le.fit_transform(cluster_labels)
    ari = adjusted_rand_score(true_labels, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)
    def clustering_purity(true_labels, cluster_labels):
        purity_sum = sum(
            Counter(true_labels[np.where(cluster_labels == c)[0]]).most_common(1)[0][1]
            for c in np.unique(cluster_labels)
        )
        return purity_sum / len(true_labels)
    purity = clustering_purity(true_labels, cluster_labels)
    print("=== Clustering Evaluation ===")
    print(f"ARI: {ari:.4f} | NMI: {nmi:.4f} | Purity: {purity:.4f}")
    return ari, nmi, purity

def scCLUE(log_scdata, nEns=15, K=[5, 10], smin=0.6, smax=1, nPCs=15, nCls=3, resolution=1.0):
    start_time = time.time()
    print("Running scCLUE — Identifying features")
    features1   = analyze_gene_features_and_pathways(log_scdata, pathways_df)
    topK        = features1["topK"]
    common_genes = features1["filtered_genes"]
    nCells      = log_scdata.shape[1]
    ensNet      = csr_matrix((nCells, nCells), dtype=np.float64)

    print("Building ensemble network...")
    for ix in range(nEns):
        for ik in K:
            sample_rng  = round(np.random.uniform(smin, smax), 2)
            sample_size = max(nCells, int(np.ceil(sample_rng * len(common_genes))))
            sampled_genes = np.random.choice(common_genes, size=sample_size, replace=False)
            reduction_data = log_scdata.loc[sampled_genes].T

            umap_res = UMAP(n_components=nPCs).fit_transform(reduction_data)
            umap_knn = make_knn(umap_res, k_num=ik)
            umap_knn = umap_knn + umap_knn.T

            try:
                pca_res = PCA(n_components=nCells).fit_transform(reduction_data)
            except Exception as e:
                max_comp = min(reduction_data.shape[0], reduction_data.shape[1])
                pca_res  = PCA(n_components=min(nCells, max_comp)).fit_transform(reduction_data)

            cor_dist = np.corrcoef(pca_res.T)
            cor_knn  = KNeighborsTransformer(n_neighbors=ik, metric='precomputed').fit_transform(1 - cor_dist)
            cor_knn  = cor_knn + cor_knn.T
            ensNet  += make_dsm(umap_knn + cor_knn)

    ensNet_array = ensNet.toarray()
    gg        = nx.from_numpy_array(ensNet)
    partition = community_louvain.best_partition(gg, resolution=resolution)
    membership = np.array(list(partition.values())) + 1

    sind = np.where(np.bincount(membership) == 1)[0]
    if len(sind) != 0:
        clid = np.setdiff1d(np.unique(membership), sind)
        for ss in sind:
            single = np.where(membership == ss)[0]
            mscore = np.array([np.mean(ensNet[single[:, None], np.where(membership == cc)[0]])
                               for cc in clid])
            membership[single] = clid[np.argmax(mscore)]
        clid    = np.unique(membership)
        new_mem = membership.copy()
        for cid, mk in enumerate(clid, start=1):
            new_mem[np.where(membership == mk)[0]] = cid
        membership = new_mem

    print(f"Louvain clusters: {len(set(membership))}")
    if len(np.unique(membership)) > nCls:
        sep_scores = compute_sep_scores(log_scdata, membership, topK)
        membership = merge_clusters(membership, sep_scores)

    total_time = time.time() - start_time
    print(f"scCLUE completed in {total_time:.2f} seconds.")
    return {'membership': membership, 'execution_time': total_time}


### ============================================================
### SECTION 4 — Run Pipeline
### ============================================================

print("\n" + "="*60)
print("STAGE 1: ccImpute — Dropout Imputation")
print("="*60)
imputed_data = ccImpute(logX, nCeil=2000, svdMaxRatio=0.08, nCores=6, k=4)

# Save imputed output (optional)
imputed_data.to_csv("dataset_imputed.csv", index=True)
print("Imputed data saved to dataset_imputed.csv")

print("\n" + "="*60)
print("STAGE 2: scCLUE — Cell Type Clustering")
print("="*60)

# Prepare imputed data for scCLUE
log_scdata = pd.DataFrame(imputed_data)
log_scdata.index   = log_scdata.index.str.upper()
log_scdata.columns = log_scdata.columns.str.upper()

final_cls  = scCLUE(log_scdata, nCls=4)
cluster    = final_cls['membership']

print("\n" + "="*60)
print("EVALUATION")
print("="*60)
try:
    Eva = calculate_clustering_metrics(data_label['x'].values, cluster)
except Exception:
    Eva = calculate_clustering_metrics(data_label['x'].values, cluster['4'])
