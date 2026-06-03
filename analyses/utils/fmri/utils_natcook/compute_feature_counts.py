import numpy as np

def compute_feature_counts(runs_feats, train_run_numbers, feats_legend, presence_threshold=0.0):
    """
    Compute how often each feature is present (non-zero) across all frames in training runs.

    Parameters
    ----------
    runs_feats : dict[int, np.ndarray]
        Dictionary mapping run number to features (shape: n_times x n_feats).
    train_run_numbers : list[int]
        List of run numbers to include in the count.
    feats_legend : list[str]
        List of feature names corresponding to columns in the feature arrays.
    presence_threshold : float
        Minimum value to consider a feature as 'present'. Defaults to 0.0.

    Returns
    -------
    features_counts : np.ndarray
        Array of shape (n_feats,) with counts of how many times each feature was present.
    features_freqs_sorted : list[tuple[str, int]]
        List of (feature_name, count) pairs sorted by count (ascending).
    """
    n_features = len(feats_legend)
    features_counts = np.zeros(n_features, dtype=int)

    for run_number in train_run_numbers:
        feats = runs_feats[run_number]
        present = feats > presence_threshold
        features_counts += present.sum(axis=0)

    features_freqs = list(zip(feats_legend, features_counts))
    features_freqs_sorted = sorted(features_freqs, key=lambda x: x[1])

    return features_counts, features_freqs_sorted
