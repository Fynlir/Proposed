import numpy as np
import pandas as pd
import multiprocessing
from scipy.sparse.linalg import svds
from joblib import Parallel, delayed
from scipy.stats import spearmanr, pearsonr
from scipy.sparse import issparse, csr_matrix
from sklearn.utils.extmath import randomized_svd
from multiprocessing import Pool
from sklearn.cluster import KMeans
import time
#import scanpy as sc
import anndata as ad
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import pairwise_distances
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Union

logX = pd.read_csv(
    "log_usoskin.csv", #### Use your normalized gene expression dataset 
    index_col=0,
    engine="python"
)
def cpm(counts):
    return (counts / counts.sum(axis=0)) * 1e6

# logX = cpm(logX) 
#logX = np.log10(1 + (logX))
print(logX)


def calculate_weights_by_variance(x):
    """Calculate feature weights using variance of each row."""
    variances = np.var(x, axis=1)  # Variance along each row (axis=1)
    return variances
w = calculate_weights_by_variance(logX)
def column_wise_spearman_corr(x):
    """Compute column-wise Spearman correlation."""
    ranked_x = np.apply_along_axis(lambda a: np.argsort(np.argsort(a)), 0, x)
    return np.corrcoef(ranked_x, rowvar=False)

def weighted_corr(x, w, method='pearson'):
    """Compute weighted Pearson or Spearman correlation."""
    if method == 'spearman':
        x = np.apply_along_axis(lambda a: np.argsort(np.argsort(a)), 0, x)
    
    mean_w = np.average(x, axis=0, weights=w)
    cov_w = np.cov(x, rowvar=False, aweights=w)
    std_w = np.sqrt(np.diag(cov_w))
    
    return cov_w / np.outer(std_w, std_w)

def calculate_correlation(method, x, w=None, n_cores=1):
    """Calculate column-wise correlation matrix with optional weighting."""
    if issparse(x):
        x = x.toarray()
    
    if method == 'spearman':
        if w is None:
            # Spearman without weights
            with Pool(n_cores) as pool:
                return column_wise_spearman_corr(x)
        else:
            # Spearman with weights
            return weighted_corr(x, w, method='spearman')
    
    elif method == 'pearson':
        if w is None:
            # Pearson without weights
            return np.corrcoef(x, rowvar=False)
        else:
            # Pearson with weights
            return weighted_corr(x, w, method='pearson')
    
    else:
        raise ValueError("Selected correlation method not available.")





def get_scale(x, n_cores=1):
    """
    Computes the mean and standard deviation of the matrix x for scaling.
    
    Parameters:
    x (np.ndarray): Input matrix.
    n_cores (int): Number of cores to use for parallel processing (if applicable).
    
    Returns:
    dict: A dictionary containing means and standard deviations.
    """
    means = np.mean(x, axis=0)
    sds = np.std(x, axis=0, ddof=1)
    return {'means': means, 'sds': sds}

def doSVD(x, svdMaxRatio=0.08, nCeil=2000, nCores=1):
    """
    Perform Truncated Singular Value Decomposition (SVD) on the input matrix.

    Parameters:
    x (np.ndarray): The input matrix on which to perform SVD.
    svd_max_ratio (float): The ratio to determine the number of singular vectors to retain.
    n_ceil (int): Maximum number of components to retain.
    n_cores (int): Number of cores to use for parallel processing.
    
    Returns:
    np.ndarray: The right singular vectors (V matrix) from the SVD.
    """
    
    # Determine the number of singular vectors to retain
    nv = int(np.ceil(svdMaxRatio * min(nCeil, x.shape[1])))
    
    # Scale the matrix
    scale = get_scale(x, nCores)
    centered_scaled_x = (x - scale['means']) / scale['sds']
    
    # Perform randomized truncated SVD
    u, s, v = randomized_svd(centered_scaled_x, n_components=nv, random_state=None)
    print('v', v.T)
    return v




def estkTW(x): ### Estimate the number of cluster from the gene expression dataset.
    n, p = x.shape

    # Compute Tracy-Widom bound parameters
    sqrt_term = np.sqrt(n - 1) + np.sqrt(p)
    muTW = sqrt_term ** 2
    sigmaTW = sqrt_term * ((1 / np.sqrt(n - 1)) + (1 / np.sqrt(p))) ** (1/3)
    bd = 3.273 * sigmaTW + muTW

    # Scale the data
    x_scaled = (x - np.mean(x, axis=0)) / np.std(x, axis=0)

    # Compute the eigenvalues of the covariance matrix
    cov_matrix = np.dot(x_scaled.T, x_scaled)
    evals = np.linalg.eigvalsh(cov_matrix)  # Use eigvalsh for symmetric matrix (Hermitian)
    
    # Return the number of eigenvalues greater than the Tracy-Widom bound
    return np.sum(evals > bd)



def getConsMtx(cluster_results, consMin):
    # This is a placeholder, replace with actual logic to generate consensus matrix
    # Cluster results shape will be (n_samples, n_runs)
    consMin = 0.75
    consensus = np.corrcoef(cluster_results)
    consensus[consensus < consMin] = 0
    return consensus

def getPConsMtx(kmResults, consMin):
    consMin = 0.75
    d = np.array(kmResults).reshape(len(kmResults[0]), -1)
    consMtx = getConsMtx(d, consMin=0.75)
    np.fill_diagonal(consMtx, 0)
    consMtx[consMtx < consMin] = 0
    consMtx = np.apply_along_axis(lambda i: i / np.sum(i) if np.sum(i) != 0 else i, axis=1, arr=consMtx)
    consMtx = np.nan_to_num(consMtx)
    return consMtx

# Function to perform K-means clustering on a subset of SVD vectors

def kmAux(i, input, k, kmNStart, kmMax):
    x = input[:, :i]
    kmeans = KMeans(n_clusters=k, max_iter=kmMax, n_init=kmNStart, init='k-means++', random_state=0)
    kmeans.fit(x)
    return kmeans.labels_

# Function to run K-means in parallel on multiple subsets
def runKM(logX, v, maxSets=8, k=None, consMin=0.75, km_n_start=None, km_max=1000, n_cores=4):
    
    lv = int(0.5 * v.shape[1])  # Lower bound for SVD vector selection
    rv = v.shape[1]             # Upper bound for SVD vector selection

    # Generate a sequence of SVD vector indices for defining subsets
    nDim = np.linspace(lv, rv, maxSets).astype(int)
    
    if k is None:
        k = estkTW(logX)  # Replace with actual function for estimating clusters
        
    # Determine km_n_start based on the number of cells (columns of logX)
    if km_n_start is None:
        km_n_start = 50 if logX.shape[1] >= 2000 else 1000

    # Run K-means in parallel using multiprocessing
    # with Pool(n_cores) as pool:
    km_results = Parallel(n_jobs=n_cores)(
    delayed(kmAux)(i, v, k, km_n_start, km_max) for i in nDim
)

    # Generate the consensus matrix (matrix of clustering results)
    cons_mtx = getConsMtx(np.array(km_results).T, consMin)
    
    return cons_mtx


def findDropouts(logX, consMtx):
    # Convert sparse matrix to dense if needed
    if issparse(logX):
        logX = logX.toarray()
    
    if isinstance(logX, pd.DataFrame):
        logX = logX.values  # Convert DataFrame to NumPy array if necessary

    if isinstance(consMtx, pd.DataFrame):
        consMtx = consMtx.values  # Convert DataFrame to NumPy array if necessary

    # Transpose logX so shape becomes (cells, genes)
    logX = logX.T  
    cells, genes = logX.shape  # Automatically determine dimensions

    print('consMtx shape:', consMtx.shape)
    
    # Identify zero indices
    zero_indices = (logX == 0)
    zero_indices_int = zero_indices.astype(int)

    print('zero shape:', zero_indices.shape)  # Should be (cells, genes)
    
    # Create vote matrix (zero_indices * 2 - 1)
    vote_m = zero_indices * 2.0 - 1.0
    
    # Perform matrix multiplication
    result = (vote_m.T @ consMtx)  # Shape will be (genes, consMtx.shape[1])

    print('result shape:', result.shape)

    # Automatically reshape based on computed dimensions
    result_reshaped = result.reshape(genes, -1)  # Auto reshape

    # Ensure zero_indices is properly reshaped
    zero_indices_reshaped = zero_indices.T.reshape(genes, -1)

    votes = result_reshaped * zero_indices_reshaped

    print('votes shape:', votes.shape)

    # Find indices where votes < 0, indicating a dropout event
    dropout_indices = np.argwhere(votes < 0)
    print('dropout_indices shape:', dropout_indices.shape)
    
    return dropout_indices


def calculate_single_dropout(args):
    """
    Helper function to calculate dropout value for a single gene-sample pair
    """
    row, col, em_t, cm = args
    
    # Convert to numpy array if pandas DataFrame/Series
    if isinstance(em_t, (pd.DataFrame, pd.Series)):
        em_t = em_t.to_numpy()
    if isinstance(cm, (pd.DataFrame, pd.Series)):
        cm = cm.to_numpy()
    
    # Handle both 1D and 2D arrays
    if em_t.ndim == 2:
        em_values = em_t[:, row]
    else:
        em_values = em_t
        
    if cm.ndim == 2:
        cm_values = cm[:, col]
    else:
        cm_values = cm
    
    # Create binary mask where expression values are > 0
    em_mask = (em_values > 0).astype(float)
    
    # Calculate denominator (sum of consensus values where expression > 0)
    div2 = np.dot(em_mask, cm_values)
    
    # Calculate numerator (dot product of expression and consensus values)
    numerator = np.dot(em_values, cm_values)
    
    return numerator / div2 if div2 != 0 else 0

def solver2(cm: Union[np.ndarray, pd.DataFrame], 
           em: Union[np.ndarray, pd.DataFrame], 
           ids: Union[np.ndarray, pd.DataFrame], 
           n_cores: Optional[int] = None) -> np.ndarray:
    """
    Fast calculation of dropout values for gene expression data.
    
    Parameters:
    -----------
    cm : np.ndarray or pd.DataFrame
        Consensus matrix
    em : np.ndarray or pd.DataFrame
        Gene expression matrix where rows are genes and columns are samples
    ids : np.ndarray or pd.DataFrame
        Matrix of row and column indices for which to calculate importance scores.
        Each row should contain [gene_index, sample_index]
    n_cores : int, optional
        Number of cores to use for parallel processing. 
        If None, uses number of CPU cores - 1
        
    Returns:
    --------
    np.ndarray
        Vector of imputed dropout values corresponding to the entries specified in ids
    """
    # Convert inputs to numpy arrays if they're pandas DataFrames
    if isinstance(cm, pd.DataFrame):
        cm = cm.to_numpy()
    if isinstance(em, pd.DataFrame):
        em = em.to_numpy()
    if isinstance(ids, pd.DataFrame):
        ids = ids.to_numpy()
    
    # Convert to 0-based indexing if input uses 1-based indexing
    ids = ids - 1 if np.min(ids) > 0 else ids
    
    # Ensure ids are integers
    ids = ids.astype(int)
    
    # Transpose expression matrix for efficient column access
    em_t = em.T
    
    # Prepare output array
    imp = np.zeros(len(ids))
    
    # If no cores specified, use CPU count - 1
    if n_cores is None:
        n_cores = max(1, multiprocessing.cpu_count() - 1)
    
    # Prepare arguments for parallel processing
    args = [(int(row), int(col), em_t, cm) for row, col in ids]
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=n_cores) as executor:
        imp = np.array(list(executor.map(calculate_single_dropout, args)))
    
    return imp

def solver(consMtx, logX, dropIds, n_cores):
    """
    Placeholder for the slow solver (linear equations).
    
    Parameters:
    consMtx (numpy.ndarray): Consensus matrix from clustering.
    logX (numpy.ndarray): Log-normalized expression matrix (dense).
    dropIds (numpy.ndarray): Indices of the dropouts to be imputed.
    n_cores (int): Number of cores for parallel processing.
    
    Returns:
    numpy.ndarray: Imputed values for dropouts.
    """
    # Implement the slow solver here
    # This is a placeholder function
    imputed_values = np.random.rand(len(dropIds))  # Dummy implementation
    return imputed_values

def computeDropouts(consMtx, logX, dropIds, fastSolver=True, nCores=1): #### Compute values that should be replace at the dropout positions
   
    # Check if logX is sparse
    is_x_sparse = issparse(logX)
    
    if fastSolver:
        # Use the fast solver
        if is_x_sparse:
            imputed_values = solver2(consMtx, logX, dropIds, nCores)
        else:
            imputed_values = solver2(consMtx, logX, dropIds, nCores)

        # Create a copy of logX to impute values
        print('impute value',len(imputed_values))
    
        imputed_logX = logX.copy()
        print('impute_logX0', imputed_logX.keys)
        impute_logX_values = imputed_logX.values
        print('impute_logX1',imputed_logX)
        impute_logX_values[tuple(dropIds.T)] = imputed_values
        print('imputed_logX', imputed_logX)
        

        return imputed_logX

    else:
        if is_x_sparse:
            raise ValueError("Slow solver not supported for sparse matrices.")
        else:
            # Warning for the slow solver
            print("Warning: Slow solver selected. This might significantly increase processing time.")
        
        # Use the slow solver
        imputed_values = solver(consMtx, logX, dropIds, nCores)

        # Create a copy of logX to impute values
        imputed_logX = logX.copy()
        imputed_logX.flat[dropIds] = imputed_values  # Fill dropouts with imputed values

        return imputed_logX


def ccImpute(log_scdata, dist=None, nCeil=2000, svdMaxRatio=0.08, maxSets=8, k=None,
             consMin=0.80, kmNStart=None, kmMax=1000, fastSolver=True, nCores=4, verbose=True):
    
    start_time = time.time()  # Start time for benchmarking
    

    isXSparse = issparse(logX)
    
    if verbose:
        n = logX.shape[1]
        print(f"Running ccImpute on dataset ({n} cells) with {nCores} cores.")
    
    # Step 2: Compute the distance matrix if not provided
    if dist is None:
        if isXSparse:
            w = logX.power(2).mean(axis=0) - np.square(logX.mean(axis=0))
        else:
            w = logX.var(axis=0)
        dist = 1 - pairwise_distances(logX.T, metric='correlation')  # Spearman-like distance
        
        if verbose:
            print(f"Distance matrix computed. Time elapsed: {time.time() - start_time:.2f} seconds.")
    print('dist', dist)
    # Step 3: Perform truncated SVD on the distance matrix
    v = doSVD(dist, svdMaxRatio=svdMaxRatio, nCeil=nCeil, nCores=nCores)
    
    if verbose:
        print(f"Dimensional reduction completed. Time elapsed: {time.time() - start_time:.2f} seconds.")
    consMin = 0.75
    # Step 4: Run k-means clustering on the reduced dimensions
    kmResults = runKM(logX, v.T, maxSets=maxSets, k=k, consMin=consMin)
    consMtx = getPConsMtx(kmResults, consMin=consMin)
    print("kmResult", kmResults.shape)
    print('consMtx', consMtx)
    print("consMtx Shape", consMtx.shape) 
    
    if verbose:
        print(f"Clustering completed. Time elapsed: {time.time() - start_time:.2f} seconds.")
    
    # Step 5: Identify dropouts (zero values in the log-normalized matrix)
    dropIds = findDropouts(logX, consMtx)
    print('dropIds', dropIds)
    
    if verbose:
        print(f"Dropouts identified. Time elapsed: {time.time() - start_time:.2f} seconds.")
    
    # Step 6: Compute the dropout imputation
    impLogX = computeDropouts(consMtx, logX, dropIds, fastSolver=fastSolver, nCores=4)
    
    if verbose:
        print(f"Dropouts imputed. Time elapsed: {time.time() - start_time:.2f} seconds.")
    
    
    return impLogX

#Load or create an AnnData object
#Perform ccImpute on the dataset
adata_imputed = ccImpute(logX, nCeil=2000, svdMaxRatio=0.08, nCores=6, k=4)

# View the result
print('adata_impute result ____',adata_imputed)
iLogX_df = pd.DataFrame(adata_imputed)
# Define the file path where you want to save the CSV file
output_file_path = '/Users/synestheisa/Documents/Python/dataset_imputed.csv'

# Save the DataFrame to a CSV file with row names included
iLogX_df.to_csv(output_file_path, index=True)




