from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
import numpy as np
import pandas as pd


def fit_binaryencoder_across_runs(dfs, colname):
    """
    Fit a LabelBinarizer on all unique, non-NaN labels from a specified column
    across multiple DataFrames.

    Parameters:
    -----------
    dfs : iterable or dict of pd.DataFrame
        Collection of DataFrames (e.g., list or dict.values()) to extract labels from.
    colname : str
        Name of the column containing labels to fit the binarizer on.

    Returns:
    --------
    lb : LabelBinarizer
        A fitted LabelBinarizer on all unique labels across all DataFrames.
    """
    all_labels = set()
    
    # If input is dict, iterate over its values (DataFrames)
    if isinstance(dfs, dict):
        dfs_iter = dfs.values()
    else:
        dfs_iter = dfs

    # Collect all unique non-NaN labels from each DataFrame's specified column
    for df in dfs_iter:
        labels = df[colname].dropna().unique()
        all_labels.update(labels)

    all_labels = sorted(all_labels)  # sort for consistency

    # Fit the LabelBinarizer
    lb = LabelBinarizer()
    lb.fit(all_labels)
    
    return lb



def one_hot_encode_column(df_annotations, lb, colname):
    """
    One-hot encodes the specified column in a DataFrame using a pre-fitted LabelBinarizer.
    
    Frames with missing labels (NaN) will be encoded as all-zero vectors.

    Parameters:
    -----------
    df_annotations : pd.DataFrame
        DataFrame with one row per frame and a column containing class labels.
    lb : LabelBinarizer
        A fitted LabelBinarizer instance for transforming the column values.
        This should be pre-fitted to ensure consistency across multiple datasets/dfs/runs.
    colname : str
        Name of the column to one-hot encode (e.g., 'noun_class_name').

    Returns:
    --------
    np.ndarray
        A 2D numpy array of shape (n_frames, n_classes) with one-hot vectors per frame.
    """
    
    # Extract column and identify valid (non-NaN) values
    labels = df_annotations[colname]
    is_valid = labels.notna()

    # Check if any valid labels are not in the fitted LabelBinarizer classes
    unknown_labels = set(labels[is_valid]) - set(lb.classes_)
    if unknown_labels:
        raise ValueError(f"Found unknown labels not in LabelBinarizer classes: {unknown_labels}")

    # Initialize output matrix with all zeros
    one_hot = np.zeros((len(labels), len(lb.classes_)), dtype=int)

    # Fill only rows with valid labels; NaNs remain as zero rows
    one_hot[is_valid.values] = lb.transform(labels[is_valid])

    # At this point, any row corresponding to a NaN label is left as an all-zero vector
    return one_hot



def fit_multilabelbinarizer_across_runs(dfs, colname):
    """
    Fit a MultiLabelBinarizer on all unique items in a column that contains lists of strings,
    across multiple DataFrames.

    Parameters:
    -----------
    dfs : iterable or dict of pd.DataFrame
        Collection of DataFrames (e.g., list or dict.values()) to extract labels from.
    colname : str
        Name of the column containing lists of string labels.

    Returns:
    --------
    mlb : MultiLabelBinarizer
        A fitted MultiLabelBinarizer.
    """
    all_items = []

    if isinstance(dfs, dict):
        dfs_iter = dfs.values()
    else:
        dfs_iter = dfs

    for df in dfs_iter:
        # Drop NaNs and extend the list with all items
        items = df[colname].dropna().tolist()
        all_items.extend(items)

    mlb = MultiLabelBinarizer()
    mlb.fit(all_items)

    return mlb


def n_hot_encode_column(df_annotations, mlb, colname):
    """
    Apply a MultiLabelBinarizer to a DataFrame column containing lists of strings.
    Missing values (NaNs) will be treated as empty lists (i.e., all-zero n-hot vector).

    Parameters:
    -----------
    df_annotations : pd.DataFrame
        DataFrame with one row per frame and a column containing lists of items.
    mlb : MultiLabelBinarizer
        A pre-fitted MultiLabelBinarizer instance.
    colname : str
        Name of the column to n-hot encode.

    Returns:
    --------
    np.ndarray
        A 2D numpy array of shape (n_frames, n_classes) with n-hot encoded vectors.
    """
    items_list = df_annotations[colname].apply(lambda x: x if isinstance(x, list) else [])

    # Check for unknown items (optional, but good for consistency)
    all_items_in_data = set()
    for item_list in items_list:
        all_items_in_data.update(item_list)
    
    unknown_items = all_items_in_data - set(mlb.classes_)
    if unknown_items:
        raise ValueError(f"Found unknown items not in MultiLabelBinarizer classes: {unknown_items}")


    return mlb.transform(items_list)




def create_feature_spaces_dict(feats_legend, categories):

    '''
    Create a dictionary mapping feature space names to indices of features belonging to each space.
    This is then used as input for himalaya.kernel_ridge.ColumnKernelizer

    This function parses a list of feature names (`feats_legend`) using a list of prefix-category 
    tuples (`categories`), and returns a dictionary that maps each feature group (e.g., 'hand_features') 
    to the indices of features that begin with the corresponding prefix (e.g., 'hands_').

    It ensures that all features are uniquely assigned to a group with no overlaps or omissions.

    Parameters
    ----------
    feats_legend : list or np.ndarray of str
        List of feature names. Ususally each name should start with one of the prefixes in "categories"

    categories : list of (str, str) tuples
        Each tuple contains a prefix and the corresponding group name.
        For example: [('hands_', 'hand_features'), ('action_', 'action_features')]

    Returns
    -------
    feature_spaces : dict
        Dictionary mapping group names to arrays of feature indices.
        For example: {'hand_features': array([0, 1, 2]), 'action_features': array([3, 4])}

    Raises
    ------
    AssertionError
        If any features are missing (not matched to any group) or assigned to multiple groups.

    Example
    -------
    categories = [
        ('hands_', 'hand_features'),
        ('action_', 'action_features'),
        ('target_', 'target_features'),
        ('interaction_', 'interaction_features'),
        ('tool_', 'tool_features'),
        ('object_', 'object_features')
    ]
    
    feature_spaces = create_feature_spaces_dict(feats_legend, categories)
    '''


    # Ensure feats_legend is a numpy array
    feats_legend = np.array(feats_legend)

    # Initialize feature space index mapping
    feature_spaces = {}

    # Populate index arrays for each group
    for prefix, group_name in categories:
        inds = np.where(np.char.startswith(feats_legend, prefix))[0]
        if len(inds) > 0:
            feature_spaces[group_name] = inds

    # Report
    print(f"Feature spaces defined: {list(feature_spaces.keys())}")
    for k, v in feature_spaces.items():
        print(f"  {k}: {len(v)} features")

    # Optional: check for full coverage (no overlap or missing)
    all_indices = np.concatenate(list(feature_spaces.values()))
    unique_indices = np.unique(all_indices)

    assert len(all_indices) == len(unique_indices), \
        f"Overlapping features detected! {len(all_indices)} total, {len(unique_indices)} unique"

    assert len(unique_indices) == len(feats_legend), \
        f"Some features are missing or misclassified! {len(unique_indices)} vs {len(feats_legend)}"
    
    
    return feature_spaces
