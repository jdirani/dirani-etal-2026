"""
Brain mask computation utilities for group-level neuroimaging analysis.
"""

import os
import numpy as np
import nibabel as nib
from nilearn import image


def compute_group_mask_nMinSubjects(modeled_masks, n_min_subjects):
    """
    Compute a group-level brain mask where at least n_min_subjects have modeled voxels.

    Parameters
    ----------
    modeled_masks : list of Nifti1Image
        The subject-level modeled voxel masks (same shape and affine).
    n_min_subjects : int
        Minimum number of subjects that must have modeled voxels at a voxel for it to be included in the group mask.

    Returns
    -------
    group_mask_img : nibabel.Nifti1Image
        The computed group mask image.
    """
    # check they have the same shape and affine
    mask_shapes = [mask.shape for mask in modeled_masks]
    assert all(np.array_equal(mask_shapes[0], m) for m in mask_shapes), "Masks shapes differ!"
    mask_affines = [mask.affine for mask in modeled_masks]
    assert all(np.array_equal(mask_affines[0], m) for m in mask_affines), "Masks affines differ!"

    # Stack all masks into a 4D array
    modeled_masks_4d = image.concat_imgs(modeled_masks)
    modeled_masks_data = modeled_masks_4d.get_fdata()  # Shape: (x, y, z, n_subjects)

    # Count how many subjects have valid data at each voxel
    n_subjects_per_voxel = np.sum(modeled_masks_data > 0, axis=3)

    # Create group mask (1 where >= n_min_subjects have data, 0 elsewhere)
    group_mask_data = (n_subjects_per_voxel >= n_min_subjects).astype(int)

    # Create a Nifti image from the group mask
    group_mask_img = image.new_img_like(modeled_masks[0], group_mask_data)

    return group_mask_img


def compute_group_mask(mask_imgs, mode='union', save_path=None):
    """
    Compute a group-level brain mask (union or intersection).

    Parameters
    ----------
    mask_imgs : list of Nifti1Image or file paths
        The subject-level mask images (same shape and affine).
    mode : str, 'union' or 'intersection'
        Whether to compute the union (any voxel in any mask)
        or intersection (only voxels present in all masks).
    save_path : str or None
        If provided, saves the resulting mask to this path.

    Returns
    -------
    group_mask_img : nibabel.Nifti1Image
        The computed group mask image.
    """
    mask_data = []
    for m in mask_imgs:
        if isinstance(m, str):
            m = nib.load(m)
        mask_data.append((m.get_fdata() > 0).astype(np.uint8))

    mask_data = np.stack(mask_data, axis=0)

    if mode == 'union':
        group_mask = (np.any(mask_data, axis=0)).astype(np.uint8)
    elif mode == 'intersection':
        group_mask = (np.all(mask_data, axis=0)).astype(np.uint8)
    else:
        raise ValueError("mode must be 'union' or 'intersection'")

    group_mask_img = nib.Nifti1Image(group_mask, mask_imgs[0].affine, mask_imgs[0].header)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        nib.save(group_mask_img, save_path)
        print(f"Saved group {mode} mask to: {save_path}")

    return group_mask_img
