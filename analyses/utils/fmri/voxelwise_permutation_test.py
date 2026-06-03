import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm



def voxelwise_permutation_test(pred, test_data, chunk_length=10, n_permutations=1000,
                                        pval=0.05, seed=None, n_jobs=1, return_null=True):
    """
    Optimized permutation test for voxelwise encoding model performance.

    This function tests whether predicted voxel time series (`pred`) are significantly correlated 
    with the true brain data (`test_data`) by performing a chunked permutation test. The permutations 
    are applied only to the predicted data to preserve temporal autocorrelation structure.

    # Note: scores that we get from banded ridge are R^2, while here we are calculating correlation. This
    # is still fine because correlation is a monotonic function of R^2 for mean-centered data, so the 
    # permutation null distribution based on correlation preserves the ranking of model performance. 
    # Using correlation for the permutation test is therefore valid for assessing significance, 
    # and faster to compute than repeatedly calling banded_ridge.score.

    Args:
        pred (ndarray): Predicted time series of shape (T, V), where T is time and V is voxels.
        test_data (ndarray): True brain data of shape (T, V).
        chunk_length (int): Length of temporal chunks used for shuffling. Default is 10.
        n_permutations (int): Number of permutations. Default is 1000.
        pval (float): Significance threshold for p-values. Default is 0.05.
        seed (int or None): Random seed for reproducibility. Default is None.
        n_jobs (int): Number of parallel jobs for permutation. Default is 1.
        return_null (bool): If True, return full null distribution. Default is True.

    Returns:
        voxcorrs_true (ndarray): (V,) array of true voxelwise correlations.
        sig_voxels_mask (ndarray): (V,) boolean array where True indicates significance at given p-value.
        pval_map (ndarray): (V,) array of empirical p-values.
        null_distrib (ndarray): (n_permutations, V) array of null correlations (if return_null=True).

    
    Example usage:
        pred = test_feats @ wt  # predicted response
        voxcorrs_true, sig_mask, pval_map, null_distrib = voxelwise_permutation_test(
                    pred, test_data, chunk_length=10, n_permutations=1000, pval=0.05, seed=42, n_jobs=4, return_null=True)
"""


    if seed is not None:
        np.random.seed(seed)
    
    T, V = test_data.shape # get time and voxel shapes
    
    # Prepare chunked data
    n_chunks = T // chunk_length
    trunc_T = n_chunks * chunk_length
    pred_trunc = pred[:trunc_T]
    test_data_trunc = test_data[:trunc_T]
    
    # Vectorized correlation computation
    def fast_corrcoef_columns(x, y):
        """Fast vectorized correlation between columns of x and y"""
        # Center the data
        x_centered = x - np.mean(x, axis=0, keepdims=True)
        y_centered = y - np.mean(y, axis=0, keepdims=True)
        
        # Compute correlation using einsum (faster than loops)
        numerator = np.einsum('tv,tv->v', x_centered, y_centered)
        x_norm = np.sqrt(np.einsum('tv,tv->v', x_centered, x_centered))
        y_norm = np.sqrt(np.einsum('tv,tv->v', y_centered, y_centered))
        
        # Avoid division by zero
        denominator = x_norm * y_norm
        corr = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
        return corr
    
    # True correlations (vectorized)
    voxcorrs_true = fast_corrcoef_columns(test_data_trunc, pred_trunc)
    
    # Reshape for chunked permutation
    pred_chunks = pred_trunc.reshape(n_chunks, chunk_length, V)
    
    # Batch permutation processing to reduce joblib overhead
    batch_size = max(1, min(50, n_permutations // n_jobs))  # Adaptive batch size
    
    def permute_batch(batch_indices):
        """Process a batch of permutations"""
        batch_results = []
        for _ in batch_indices:
            perm_indices = np.random.permutation(n_chunks)
            shuffled_pred = pred_chunks[perm_indices].reshape(trunc_T, V)
            batch_corrs = fast_corrcoef_columns(test_data_trunc, shuffled_pred)
            batch_results.append(batch_corrs)
        return np.array(batch_results)
    
    # Create batches
    batches = []
    for i in range(0, n_permutations, batch_size):
        batch_end = min(i + batch_size, n_permutations)
        batches.append(range(batch_end - i))
    
    # Run batched permutations
    batch_results = Parallel(n_jobs=n_jobs)(
        delayed(permute_batch)(batch) 
        for batch in tqdm(batches, desc="Permutation batches")
    )
    
    # Concatenate results
    null_distrib = np.concatenate(batch_results, axis=0)
    
    # Vectorized p-value computation
    pval_map = (np.sum(null_distrib >= voxcorrs_true[np.newaxis, :], axis=0) + 1) / (n_permutations + 1)
    
    # Binary significance mask
    sig_voxels_mask = pval_map < pval
    
    if return_null:
        return voxcorrs_true, sig_voxels_mask, pval_map, null_distrib
    else:
        return voxcorrs_true, sig_voxels_mask, pval_map