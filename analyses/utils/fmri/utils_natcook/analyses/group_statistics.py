"""
Group-level statistical testing utilities for neuroimaging analysis.
"""

import numpy as np
from scipy.stats import ttest_1samp
from nilearn.image import smooth_img, math_img, new_img_like
from nilearn.maskers import NiftiMasker
from nilearn.glm import threshold_stats_img


def group_level_1sample_ttest(
    stat_imgs,
    popmean=0,
    mask_img=None,
    fwhm=6,
    alpha=0.05,
    height_control='fdr',
    cluster_threshold=10,
    two_sided=False,
    verbose=True
):
    """
    Perform group-level one-sample t-test on neuroimaging data.
    
    This function smooths individual statistical maps, performs voxelwise one-sample
    t-tests against a population mean, applies multiple comparisons correction, and
    returns a binary significance mask for thresholding group average maps.
    
    Parameters
    ----------
    stat_imgs : list of Niimg-like
        List of individual statistical maps (one per subject), e.g., R² maps or
        correlation maps from encoding models.
    popmean : float, default=0
        Value against which to perform the one-sample t-test. Use 0 to test whether
        effects are greater than zero, or another value (e.g., 0.5 for accuracy maps)
        to test against chance level.
    mask_img : Niimg-like, optional
        Brain mask image defining voxels to analyze. If None, uses the non-zero 
        voxels from the first image.
    fwhm : float, default=6
        Full-width at half-maximum for Gaussian smoothing kernel (in mm).
        Set to None or 0 to skip smoothing.
    alpha : float, default=0.05
        Significance threshold for multiple comparisons correction.
    height_control : str, default='fdr'
        Method for multiple comparisons correction. Options: 'fdr' (false discovery 
        rate), 'bonferroni', or 'fpr' (false positive rate).
    cluster_threshold : int, default=10
        Minimum cluster size in voxels. Clusters smaller than this are removed.
    two_sided : bool, default=False
        Whether to perform two-sided test. If False, performs one-sided test
        (tests whether values are greater than popmean).
    verbose : bool, default=True
        Whether to print progress and results.
    
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'significance_mask': Binary mask (Nifti1Image) indicating significant voxels.
          Use this to mask group average maps for visualization.
        - 'thresholded_t_map': Thresholded t-statistic map (Nifti1Image) showing 
          t-values only for significant voxels (zeros elsewhere).
        - tthreshold : The voxel-level threshold used actually (returned by nilearn.glm.threshold_stats_img)

          
    """
    
    # Validate inputs
    if len(stat_imgs) < 2:
        raise ValueError(f"Need at least 2 images for group analysis, got {len(stat_imgs)}")
    
    # Step 1: Smooth individual maps
    if fwhm and fwhm > 0:
        if verbose:
            print(f"Smoothing {len(stat_imgs)} images with FWHM={fwhm}mm...")
        smoothed_imgs = [smooth_img(img, fwhm=fwhm) for img in stat_imgs]
    else:
        if verbose:
            print(f"Skipping smoothing (fwhm={fwhm})...")
        smoothed_imgs = stat_imgs
    
    # Step 2: Mask and vectorize
    if verbose:
        print("Masking and vectorizing data...")
    
    masker = NiftiMasker(mask_img=mask_img).fit()
    group_data = np.vstack([masker.transform(img) for img in smoothed_imgs])
    group_data = np.nan_to_num(group_data, nan=0.0)
    
    if verbose:
        print(f"Data shape: {group_data.shape} (subjects x voxels)")
        print(f"Testing against population mean: {popmean}")
        print(f"Sample mean: {np.mean(group_data):.4f}")
        print(f"Sample std: {np.std(group_data):.4f}")
    
    # Step 3: Voxelwise one-sample t-test
    if verbose:
        print("Performing voxelwise one-sample t-test...")
    
    t_vals, p_vals = ttest_1samp(group_data, popmean=popmean, axis=0)
    t_map_img = masker.inverse_transform(t_vals)

    # Step 4: Multiple comparisons correction
    if verbose:
        print(f"Applying {height_control.upper()} correction (alpha={alpha})...")
    
    thresholded_map, t_threshold = threshold_stats_img(
        stat_img=t_map_img,
        mask_img=mask_img,
        alpha=alpha,
        height_control=height_control,
        cluster_threshold=cluster_threshold,
        two_sided=two_sided
    )
    
    n_sig_voxels = np.sum(thresholded_map.get_fdata() != 0)

    # Step 5: Create binary significance mask
    significance_mask = math_img("img != 0", img=thresholded_map)

    
    if verbose:
        print(f"Voxel-level threshold (t-value): {t_threshold:.2f}")
        print(f"Number of significant voxels: {n_sig_voxels}")
        if n_sig_voxels == 0:
            print("Warning: No significant voxels found. Consider:")
            print("  - Using a less conservative correction method")
            print("  - Increasing alpha threshold")
            print("  - Checking if input maps have sufficient signal")
    
    # Prepare results dictionary
    results = {
        'significance_mask': significance_mask,
        'thresholded_t_map': thresholded_map,
        'tthreshold': t_threshold
    }

    return results


def calculate_contrast_coeffs(group_coeffs, feature_spaces_to_compare):
    """
    Compute contrast of coefficients between two feature spaces across subjects.
    No masking applied - all voxels are used. Masking should be done outside the function if needed.

    Parameters
    ----------
    group_coeffs : list of dict
        List of length n_subjects. Each element is a dictionary where:
        - keys are feature names 
        - values are NiftiImages of the corresponding coefficients
    feature_spaces_to_compare : list of str
        List of exactly two feature space names to compare (e.g., ['object_features', 'target_features']).
        The contrast will be computed as: first space - second space.

    Returns
    -------
    average_contrast_coeffs_img : NiftiImage
        Group average contrast map
    group_contrast_coeffs : list of NiftiImage
        Individual subject contrast maps
        
    Raises
    ------
    ValueError
        If feature_spaces_to_compare doesn't contain exactly 2 feature spaces
        If any specified feature space has no corresponding features
    """
    # Import here to avoid circular dependency
    from .nilearn_utils import average_list_of_imgs
    
    # Input validation
    if len(feature_spaces_to_compare) != 2:
        raise ValueError("Must specify exactly 2 feature spaces to compare")
        
    if len(group_coeffs) == 0:
        raise ValueError("group_coeffs is empty")

    # Extract prefixes for each feature space (e.g., 'object_', 'target_')
    feature_spaces_prefixes = [f"{i.split('_')[0]}_" for i in feature_spaces_to_compare]
    print('Computing contrast between feature spaces with prefixes:', feature_spaces_prefixes)

    group_contrast_coeffs = []  # List to store each subject's contrast map
    
    for subj_coeffs in group_coeffs:
        # Get all feature names for this subject
        feats_legend = list(subj_coeffs.keys())

        # Get feature names for each space
        feature_spaces_allfeatnames = [
            [i for i in feats_legend if i.startswith(prefix)] 
            for prefix in feature_spaces_prefixes
        ]

        # Check we found features for both spaces
        if not all(len(features) > 0 for features in feature_spaces_allfeatnames):
            raise ValueError(
                f"No features found for one or more spaces. "
                f"Features found: {[len(f) for f in feature_spaces_allfeatnames]}"
            )

        # Get average coefficient maps for each feature space
        coeffs_imgs = []
        for featnames in feature_spaces_allfeatnames:
            space_coeffs_imgs = [subj_coeffs[feat_name] for feat_name in featnames]
            space_avg_img = average_list_of_imgs(space_coeffs_imgs)
            coeffs_imgs.append(space_avg_img)

        # Calculate contrast
        contrast_coeffs_img = math_img(
            'img1 - img2',
            img1=coeffs_imgs[0],
            img2=coeffs_imgs[1]
        )
        group_contrast_coeffs.append(contrast_coeffs_img)

    # Calculate group average
    average_contrast_coeffs_img = average_list_of_imgs(group_contrast_coeffs)

    return average_contrast_coeffs_img, group_contrast_coeffs


def calculate_contrast_coeffs_from_featurelists(group_coeffs, features_to_compare):
    """
    Compute contrast of coefficients between two sets of features across subjects.
    No masking applied - all voxels are used. Masking should be done outside the function if needed.

    Parameters
    ----------
    group_coeffs : list of dict
        List of length n_subjects. Each element is a dictionary where:
        - keys are feature names 
        - values are NiftiImages of the corresponding coefficients
    features_to_compare : list of list of str
        List of exactly two lists, each containing feature names (from feats_legend).
        The contrast will be computed as: mean(features_to_compare[0]) - mean(features_to_compare[1])

    Returns
    -------
    average_contrast_coeffs_img : NiftiImage
        Group average contrast map
    group_contrast_coeffs : list of NiftiImage
        Individual subject contrast maps

    Raises
    ------
    ValueError
        If features_to_compare doesn't contain exactly 2 lists
        If any specified feature list has no corresponding features
    """
    # Import here to avoid circular dependency
    from .nilearn_utils import average_list_of_imgs
    
    if len(features_to_compare) != 2:
        raise ValueError("Must specify exactly 2 feature lists to compare")
    if len(group_coeffs) == 0:
        raise ValueError("group_coeffs is empty")

    group_contrast_coeffs = []
    for subj_coeffs in group_coeffs:
        feats_legend = list(subj_coeffs.keys())

        # For each feature list, get the images for those features
        feature_imgs = []
        for feature_list in features_to_compare:
            # Check all features exist
            missing = [f for f in feature_list if f not in feats_legend]
            if missing:
                raise ValueError(f"Features not found in subject: {missing}")
            imgs = [subj_coeffs[f] for f in feature_list]
            avg_img = average_list_of_imgs(imgs)
            feature_imgs.append(avg_img)

        # Contrast: mean of first list minus mean of second list
        contrast_coeffs_img = math_img(
            'img1 - img2',
            img1=feature_imgs[0],
            img2=feature_imgs[1]
        )
        group_contrast_coeffs.append(contrast_coeffs_img)

    average_contrast_coeffs_img = average_list_of_imgs(group_contrast_coeffs)
    return average_contrast_coeffs_img, group_contrast_coeffs
