import numpy as np
from nilearn import image
from scipy import ndimage
import warnings

def get_topN_cluster_coordinates(
    img,
    threshold,
    top_n=5,
    direction="positive",
    method="mean",
    verbose=True
):
    """
    Identify coordinates of the largest clusters above a given threshold.

    Parameters
    ----------
    img : nibabel.Nifti1Image
        Statistical or activation map.
    threshold : float
        Threshold for detecting clusters (applied to absolute or signed values
        depending on `direction`).
    top_n : int, optional
        Number of top clusters to return (default=5).
    direction : {'positive', 'negative', 'bidirectional'}, optional
        Which clusters to extract:
            - 'positive'     : only values > threshold
            - 'negative'     : only values < -threshold
            - 'bidirectional': both positive and negative clusters
    method : {'extent', 'mean'}, optional
        Criterion for ranking clusters:
            - 'mean'   : rank by mean intensity within each cluster [default]
            - 'extent' : rank by cluster size (number of voxels)
    verbose : bool, optional
        Whether to print cluster info (default=True).

    Returns
    -------
    clusters_dict : dict
        Dictionary with keys corresponding to directions ('positive', 'negative'),
        and values being lists of (x, y, z) MNI coordinates.
        For example:
            {'positive': [...]} or {'negative': [...]} or
            {'positive': [...], 'negative': [...]}
    """

    data = img.get_fdata()
    affine = img.affine

    if np.any(np.isnan(data)):
        warnings.warn("Input image contains NaN values")

    def _find_clusters(mask, data, sign_label):
        """Internal helper to find clusters given a mask."""
        labeled, n_clusters = ndimage.label(mask)
        if n_clusters == 0:
            if verbose:
                print(f"No {sign_label} clusters found.")
            return []

        # Compute cluster scores according to method
        cluster_ids = range(1, n_clusters + 1)
        if method == "mean":
            cluster_scores = np.array([np.mean(data[labeled == i]) for i in cluster_ids])
        else:  # extent
            cluster_scores = ndimage.sum(mask, labeled, cluster_ids)
        sorted_idx = np.argsort(np.abs(cluster_scores))[::-1]


        cluster_coordinates = []
        for rank, cluster_id in enumerate(sorted_idx[:top_n], start=1):
            cluster_mask = (labeled == (cluster_id + 1))
            coords = np.argwhere(cluster_mask)

            # Find voxel of max (or min) intensity depending on sign
            if sign_label == "positive":
                peak_voxel = coords[np.argmax(data[cluster_mask])]
            else:
                peak_voxel = coords[np.argmin(data[cluster_mask])]

            peak_mni = image.coord_transform(*peak_voxel, affine)
            cluster_coordinates.append(peak_mni)

            if verbose:
                score = cluster_scores[cluster_id]
                score_label = "mean" if method == "mean" else "size"
                print(f"{sign_label.capitalize()} cluster {rank}: "
                      f"{score_label}={score:.4f}, peak @ {np.round(peak_mni, 2)}")

        return cluster_coordinates

    # --- Handle direction modes ---
    clusters_dict = {}

    if direction in ("positive", "bidirectional"):
        clusters_dict["positive"] = _find_clusters(data > threshold, data, "positive")

    if direction in ("negative", "bidirectional"):
        clusters_dict["negative"] = _find_clusters(data < -threshold, data, "negative")

    # Safety: ensure both keys exist (for consistent structure)
    for key in ["positive", "negative"]:
        clusters_dict.setdefault(key, [])

    return clusters_dict