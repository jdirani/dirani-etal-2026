import numpy as np
from nilearn.image import new_img_like, concat_imgs


def replace_ns_with_nan(img, mask):
    """
    Replace non-significant voxels with NaN.
    
    Parameters
    ----------
    img : nibabel.Nifti1Image
        Input image.
    mask : nibabel.Nifti1Image
        Binary mask indicating significant voxels.
        
    Returns
    -------
    out : nibabel.Nifti1Image
        Image with non-significant voxels set to NaN.
    """
    data = img.get_fdata().copy()
    data[~mask.get_fdata().astype(bool)] = np.nan  # set non-significant voxels to NaN
    out = new_img_like(img, data)
    return out


def replace_zeros_with_nan(img, tol=1e-8):
    """
    Replace near-zero voxels with NaN.
    
    Parameters
    ----------
    img : nibabel.Nifti1Image
        Input image.
    tol : float
        Absolute threshold below which values are considered zero.
        
    Returns
    -------
    out : nibabel.Nifti1Image
        Image with near-zero values set to NaN.
    """
    data = img.get_fdata().copy()
    mask_small = np.abs(data) < tol
    data[mask_small] = np.nan
    out = new_img_like(img, data)
    return out


def average_list_of_imgs(imgs_list):
    """
    Average a list of NIfTI images across subjects/sessions.
    
    Parameters
    ----------
    imgs_list : list of nibabel.Nifti1Image
        List of images to average.
        
    Returns
    -------
    averaged_img : nibabel.Nifti1Image
        Averaged image.
    """
    concatenated_imgs = concat_imgs(imgs_list)  # concatenate as a single nilearn object
    averaged_data = np.nanmean(concatenated_imgs.get_fdata(), axis=3)  # get the data and average over subjects
    averaged_img = new_img_like(imgs_list[0], averaged_data)
    
    return averaged_img


def nan_argmax(arr, axis=None):
    """
    Argmax that ignores NaNs but returns NaN where all values are NaN.
    
    Parameters
    ----------
    arr : np.ndarray
        Input array.
    axis : int or None
        Axis along which to perform argmax. If None, the array is flattened.
    
    Returns
    -------
    out : np.ndarray or float
        Indices of the maximum values along the specified axis.
        NaN where all values are NaN.
    """
    # Check where NaNs are
    nan_mask = np.isnan(arr)
    
    # Replace NaNs with -inf for argmax computation
    arr_filled = np.where(nan_mask, -np.inf, arr)
    
    # Compute argmax
    argmax_indices = np.argmax(arr_filled, axis=axis)
    
    # Identify slices that are all NaN
    all_nan = np.all(nan_mask, axis=axis)
    
    # Convert to float array to allow NaNs in output
    argmax_indices = argmax_indices.astype(float)
    
    # Assign NaN where all values were NaN
    argmax_indices[all_nan] = np.nan
    
    return argmax_indices
