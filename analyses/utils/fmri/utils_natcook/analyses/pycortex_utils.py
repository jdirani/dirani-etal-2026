from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba_array
from matplotlib.image import imread

import cortex
from cortex import quickflat


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _nilearn_to_pycortex_array(img):
    """Convert a nilearn image to a (Z, Y, X) float array for pycortex."""
    return img.get_fdata().T.astype(float)


def _resolve_xfm(subj, xfmname):
    return xfmname if xfmname is not None else f'{subj}_transform'


def _to_u8(arr):
    """Clip to [0, 1] and convert to uint8 for VolumeRGB channels."""
    return np.clip(np.asarray(arr) * 255.0, 0, 255).astype(np.uint8)


def _build_alpha(
    data_for_mask,
    alpha_img=None,
    hide_nan=True,
    hide_zero=False,
    hide_below=None,
):
    """
    Build an alpha volume (floats in [0, 1]) in the same (Z, Y, X) shape
    as the data. If `alpha_img` is given, it takes precedence over the
    hide_* rules.
    """
    if alpha_img is not None:
        return np.clip(_nilearn_to_pycortex_array(alpha_img), 0, 1)

    alpha = np.ones_like(data_for_mask, dtype=float)
    if hide_nan:
        alpha *= (~np.isnan(data_for_mask)).astype(float)
    if hide_zero:
        alpha *= (data_for_mask != 0).astype(float)
    if hide_below is not None:
        alpha *= (data_for_mask >= hide_below).astype(float)
    return np.clip(alpha, 0, 1)


def _as_volume_rgb(rgb_float, alpha_float, subj, xfmname):
    """Package RGB + alpha float arrays into a cortex.VolumeRGB."""
    return cortex.VolumeRGB(
        _to_u8(rgb_float[..., 0]),
        _to_u8(rgb_float[..., 1]),
        _to_u8(rgb_float[..., 2]),
        subj, _resolve_xfm(subj, xfmname),
        alpha=_to_u8(alpha_float),
    )


# ---------------------------------------------------------------------------
# Legacy scalar plotters (no masking)
# ---------------------------------------------------------------------------

def pycortex_plot_img(nilearn_img, subj, title='', cortex_Volume_kwargs={}):
    """
    Display a NIfTI image using the pycortex webviewer.

    Parameters
    ----------
    nilearn_img : nibabel.Nifti1Image
        NIfTI image to display.
    subj : str
        Pycortex subject name.
    title : str, optional
        Title for the visualization.
    cortex_Volume_kwargs : dict, optional
        Additional keyword arguments for cortex.Volume.
    """
    pycx_voldata = cortex.Volume(
        nilearn_img.get_fdata().T, subj, f'{subj}_transform',
        **cortex_Volume_kwargs,
    )
    cortex.webshow(pycx_voldata, title=title)


def pycortex_plot_flatmap(
    nilearn_img,
    subj,
    title='',
    cbar_label=None,
    figsize=(16, 4),
    cortex_Volume_kwargs={},
    quickflat_kwargs={},
):
    """
    Create a flatmap visualization from a scalar nilearn image.

    Under the hood this builds a `cortex.VolumeRGB` via
    `pycortex_make_volume_rgba` so NaN voxels render as truly transparent
    — no interpolation-bleed artifact, no mid-ramp halos at mask edges.
    The public API is unchanged from the legacy scalar-Volume version,
    so existing call sites don't need to be touched.

    The colorbar is drawn manually from the (cmap, vmin, vmax) in
    `cortex_Volume_kwargs` because VolumeRGB has no scalar→color mapping
    for quickflat to auto-generate one.

    Parameters
    ----------
    nilearn_img : nibabel.Nifti1Image
        NIfTI image to display. NaN voxels render as transparent.
    subj : str
        Pycortex subject name.
    title : str, optional
    cbar_label : str, optional
        Label for the colorbar.
    figsize : tuple, optional
        Figure size (width, height) in inches.
    cortex_Volume_kwargs : dict, optional
        Recognized keys: 'vmin', 'vmax', 'cmap'. Other keys are ignored
        (VolumeRGB doesn't accept the same options cortex.Volume does).
    quickflat_kwargs : dict, optional
        Forwarded to `quickflat.make_figure`. `with_colorbar` is honored
        (default True) but handled by this function rather than quickflat.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    # Extract the scalar-colormap params that drive both the RGBA
    # construction and the colorbar.
    cmap = cortex_Volume_kwargs.get('cmap', 'RdBu_r')
    vmin = cortex_Volume_kwargs.get('vmin', None)
    vmax = cortex_Volume_kwargs.get('vmax', None)

    # Build a VolumeRGB with alpha=0 at NaN voxels (fixes the bleed).
    vol = pycortex_make_volume_rgba(
        nilearn_img, subj,
        cmap=cmap, vmin=vmin, vmax=vmax,
        hide_nan=True,
    )

    # We render the colorbar ourselves. Pop the user's request and then
    # force quickflat's own colorbar off — otherwise quickflat falls back
    # to its own default (True) and draws a (useless) VolumeRGB bar even
    # when the caller asked for `with_colorbar=False`.
    quickflat_kwargs = dict(quickflat_kwargs)
    quickflat_kwargs.setdefault('with_curvature', True)
    with_colorbar = quickflat_kwargs.pop('with_colorbar', True)
    quickflat_kwargs['with_colorbar'] = False

    fig, ax = plt.subplots(figsize=figsize)
    quickflat.make_figure(vol, fig=fig, **quickflat_kwargs)
    ax.set_title(title)
    ax.set_axis_off()

    if with_colorbar:
        # Resolve the effective vmin/vmax that the RGBA mapping actually
        # used, so the colorbar matches what's on the flatmap.
        data = nilearn_img.get_fdata().astype(float)
        effective_vmin = 0.0 if vmin is None else vmin
        effective_vmax = (float(np.nanpercentile(data, 95))
                         if vmax is None else vmax)

        cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
        sm = ScalarMappable(
            cmap=cmap_obj,
            norm=Normalize(vmin=effective_vmin, vmax=effective_vmax),
        )
        sm.set_array([])

        # Horizontal bar near the bottom of the figure — matches the
        # layout pycortex's own colorbar used to occupy.
        cbar_ax = fig.add_axes([0.4, 0.07, 0.2, 0.03])
        fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')

        if cbar_label:
            # Same label placement the legacy function used, so downstream
            # `cbar_label='R²'` style calls look identical.
            cbar_ax.set_ylabel(cbar_label, fontsize=12, rotation=0)
            cbar_ax.yaxis.set_label_coords(-0.05, -0.2)

    return fig, ax


# ---------------------------------------------------------------------------
# Scalar (1D) volume → RGBA
# ---------------------------------------------------------------------------

def pycortex_make_volume_rgba(
    nilearn_img,
    subj,
    cmap='viridis',
    vmin=None,
    vmax=None,
    alpha_img=None,
    hide_nan=True,
    hide_zero=False,
    hide_below=None,
    vmax_percentile=95,
    nan_fill_value=0.0,
    xfmname=None,
):
    """
    Build a VolumeRGB from a single scalar volume, with a true alpha
    channel so transparent voxels stay transparent at edges (no mid-ramp
    halo, no NaN propagation).

    Parameters
    ----------
    nilearn_img : nilearn image
        Scalar data volume (e.g. R^2 scores, contrasts, RSA correlations).
    subj : str
        Pycortex subject id.
    cmap : str or matplotlib.colors.Colormap
        Colormap applied to the scalar values.
    vmin, vmax : float, optional
        Display range. Defaults: vmin=0, vmax=`vmax_percentile` of finite
        data.
    alpha_img : nilearn image, optional
        Explicit visibility mask (0 = transparent, 1 = opaque, values in
        between are allowed). If provided, overrides the `hide_*` flags.
    hide_nan, hide_zero : bool
        Default rules when `alpha_img` is not provided.
    hide_below : float, optional
        Also hide voxels strictly below this value.
    vmax_percentile : float
        Percentile used to compute `vmax` when not given.
    nan_fill_value : float
        Data-space value whose color is assigned to NaN voxels' RGB
        channel (their alpha is 0, so they're invisible — but their RGB
        still gets interpolated with neighboring valid voxels during
        flatmap resampling, which creates a colored halo at mask edges).
        Default 0.0, which lands:
          * at the midpoint of a diverging cmap (vmin<0<vmax) → neutral
            color (e.g. white for 'bwr'), so red/blue clusters fade
            cleanly to the curvature instead of producing blue/red halos,
          * at the bottom of a sequential cmap (vmin=0) → same as the
            pre-fix behavior.
        Override only if you want non-default halo behaviour.
    xfmname : str, optional
        Pycortex transform name. Defaults to f'{subj}_transform'.

    Returns
    -------
    cortex.VolumeRGB
    """
    data = _nilearn_to_pycortex_array(nilearn_img)

    if vmin is None:
        vmin = 0.0
    if vmax is None:
        vmax = float(np.nanpercentile(data, vmax_percentile))

    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap

    denom = (vmax - vmin) if (vmax - vmin) != 0 else 1.0
    norm = np.clip((data - vmin) / denom, 0, 1)
    # Colour of NaN voxels' RGB channel (alpha is 0 for them, but their
    # RGB bleeds into neighbours during flatmap interpolation — so we
    # pick a neutral colour rather than the cmap's vmin colour).
    fill_norm = float(np.clip((nan_fill_value - vmin) / denom, 0.0, 1.0))
    norm_filled = np.nan_to_num(norm, nan=fill_norm)
    rgba = cmap_obj(norm_filled)  # (..., 4) floats in [0, 1]

    alpha = _build_alpha(
        data_for_mask=data,
        alpha_img=alpha_img,
        hide_nan=hide_nan,
        hide_zero=hide_zero,
        hide_below=hide_below,
    )

    return _as_volume_rgb(rgba[..., :3], alpha, subj, xfmname)


# ---------------------------------------------------------------------------
# 2D volume (two scalars + 2D colormap) → RGBA
# ---------------------------------------------------------------------------

def pycortex_make_vol2D_rgba(
    nilearn_img1,
    nilearn_img2,
    subj,
    cmap=None,
    vmin=0,
    vmax=None,
    vmin2=0,
    vmax2=None,
    alpha_img=None,
    hide_nan=True,
    hide_both_zero=False,
    vmax_percentile=95,
    flip_y=True,
    xfmname=None,
):
    """
    Build a VolumeRGB from two scalar volumes mapped through a pycortex
    2D colormap. Drop-in replacement for `cortex.Volume2D` that supports
    a true alpha channel.

    `nilearn_img1` goes on the X axis of the 2D colormap,
    `nilearn_img2` on the Y axis.

    The returned VolumeRGB has the cmap name and axis ranges stashed on
    it as attributes (`_cmap2d`, `_vmin`, `_vmax`, `_vmin2`, `_vmax2`,
    `_flip_y`) so `add_2d_colorbar` can build a matching legend without
    the caller re-specifying everything.

    Parameters
    ----------
    nilearn_img1, nilearn_img2 : nilearn images
        Scalar volumes for the X and Y axes.
    subj : str
        Pycortex subject id.
    cmap : str, optional
        Name of a pycortex 2D colormap (PNG in pycortex's colormap dir).
        Defaults to pycortex's `basic.default_cmap2D` if available, else
        'RdBu_covar'.
    vmin, vmax, vmin2, vmax2 : float
        Display ranges. If `vmax` or `vmax2` is None, both default to the
        combined `vmax_percentile` of the two inputs so the axes match.
    alpha_img : nilearn image, optional
        Explicit visibility mask. Overrides the `hide_*` flags.
    hide_nan : bool
        Hide voxels that are NaN in either input.
    hide_both_zero : bool
        Hide voxels where both inputs are exactly zero.
    vmax_percentile : float
        Percentile used for default `vmax` / `vmax2`.
    flip_y : bool
        Flip the Y axis when indexing into the 2D colormap PNG. Toggle if
        the rendered image looks vertically mirrored relative to
        `cortex.webshow`.
    xfmname : str, optional
        Pycortex transform name.

    Returns
    -------
    cortex.VolumeRGB
    """
    data1 = _nilearn_to_pycortex_array(nilearn_img1)
    data2 = _nilearn_to_pycortex_array(nilearn_img2)

    shared_vmax = float(np.nanmax([
        np.nanpercentile(data1, vmax_percentile),
        np.nanpercentile(data2, vmax_percentile),
    ]))
    if vmax is None:
        vmax = shared_vmax
    if vmax2 is None:
        vmax2 = shared_vmax

    if cmap is None:
        try:
            cmap = cortex.options.config.get('basic', 'default_cmap2D')
        except Exception:
            cmap = 'RdBu_covar'
    cmap_dir = cortex.options.config.get('webgl', 'colormaps')
    png_path = os.path.join(cmap_dir, cmap + '.png')
    if not os.path.isfile(png_path):
        raise FileNotFoundError(
            f"2D colormap PNG not found at {png_path!r}. "
            f"Check the `cmap` name or pycortex's colormap directory."
        )
    cmap_img = imread(png_path)  # (H, W, 3 or 4), floats in [0, 1]
    H, W = cmap_img.shape[:2]

    denom1 = (vmax - vmin)   if (vmax - vmin)   != 0 else 1.0
    denom2 = (vmax2 - vmin2) if (vmax2 - vmin2) != 0 else 1.0
    n1 = np.nan_to_num(np.clip((data1 - vmin)  / denom1, 0, 1), nan=0.0)
    n2 = np.nan_to_num(np.clip((data2 - vmin2) / denom2, 0, 1), nan=0.0)
    i = (n1 * (W - 1)).astype(int)
    j = ((1 - n2) * (H - 1)).astype(int) if flip_y else (n2 * (H - 1)).astype(int)
    rgb = cmap_img[j, i, :3]

    if alpha_img is not None:
        alpha = _nilearn_to_pycortex_array(alpha_img)
    else:
        alpha = np.ones_like(data1, dtype=float)
        if hide_nan:
            alpha *= (~(np.isnan(data1) | np.isnan(data2))).astype(float)
        if hide_both_zero:
            alpha *= ((data1 != 0) | (data2 != 0)).astype(float)
    alpha = np.clip(alpha, 0, 1)

    vol = _as_volume_rgb(rgb, alpha, subj, xfmname)
    # Stash metadata so `add_2d_colorbar(fig, vol, ...)` can build a
    # matching legend without re-specifying cmap/ranges.
    vol._cmap2d = cmap
    vol._vmin, vol._vmax = vmin, vmax
    vol._vmin2, vol._vmax2 = vmin2, vmax2
    vol._flip_y = flip_y
    return vol


# ---------------------------------------------------------------------------
# Categorical / WTA volume → RGBA
# ---------------------------------------------------------------------------

def pycortex_make_categorical_rgba(
    label_img,
    subj,
    colors,
    alpha_img=None,
    bg_label=None,
    xfmname=None,
):
    """
    Build a VolumeRGB from an integer-label volume (e.g. a winner-takes-all
    map), with one discrete color per category.

    Category boundaries fade smoothly into the curvature instead of
    producing false-color halos, because RGB and alpha are interpolated
    independently.

    Parameters
    ----------
    label_img : nilearn image
        Integer-label volume. Non-integer dtype is allowed; values are
        rounded to int. NaN voxels are treated as background
        (transparent) regardless of `bg_label`.
    subj : str
        Pycortex subject id.
    colors : sequence of matplotlib color specs
        One color per category, indexed by label value. Accepts anything
        matplotlib understands ('red', '#ff0000', (1, 0, 0), etc.).
    alpha_img : nilearn image, optional
        Explicit visibility mask. Overrides `bg_label` behaviour.
    bg_label : int, optional
        Label value to treat as background (alpha = 0). Pass 0 to hide a
        zero-valued background. If None, only NaN voxels are transparent.
    xfmname : str, optional
        Pycortex transform name.

    Returns
    -------
    cortex.VolumeRGB
    """
    labels = _nilearn_to_pycortex_array(label_img)
    rgba_lookup = to_rgba_array(list(colors))  # (n_cat, 4) floats in [0, 1]

    nan_mask = np.isnan(labels)
    labels_int = np.where(nan_mask, 0, labels).astype(int)

    if labels_int.max() >= len(rgba_lookup) or labels_int.min() < 0:
        out_of_range = (labels_int >= len(rgba_lookup)) | (labels_int < 0)
        labels_int = np.clip(labels_int, 0, len(rgba_lookup) - 1)
    else:
        out_of_range = np.zeros_like(labels_int, dtype=bool)

    rgb = rgba_lookup[labels_int, :3]

    if alpha_img is not None:
        alpha = _nilearn_to_pycortex_array(alpha_img)
    else:
        alpha = (~nan_mask).astype(float)
        if bg_label is not None:
            alpha *= (labels != bg_label).astype(float)
        alpha *= (~out_of_range).astype(float)
    alpha = np.clip(alpha, 0, 1)

    return _as_volume_rgb(rgb, alpha, subj, xfmname)


# ---------------------------------------------------------------------------
# 2D colorbar inset
# ---------------------------------------------------------------------------

def add_2d_colorbar(
    fig,
    vol=None,
    cmap=None,
    vmin=None, vmax=None,
    vmin2=None, vmax2=None,
    xlabel='', ylabel='',
    position=(0.86, 0.15, 0.10, 0.25),
    flip_y=None,
    labelsize=9,
    ticksize=8,
):
    """
    Add a 2D colormap legend as an inset on an existing figure, for
    flatmaps built with `pycortex_make_vol2D_rgba`.

    Pass either the `vol` returned by `pycortex_make_vol2D_rgba` (which
    carries its cmap/ranges as attributes), or the cmap name and ranges
    explicitly. Explicit args win when both are given.

    Usage
    -----
    >>> vol = pycortex_make_vol2D_rgba(img1, img2, subj,
    ...                                cmap='RdBu_covar', vmax_percentile=99)
    >>> fig, _ = pycortex_plot_flatmap_from_vol(vol)
    >>> add_2d_colorbar(fig, vol, xlabel='Targets R²', ylabel='Objects R²')

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to attach the inset to (usually the one returned by
        `pycortex_plot_flatmap_from_vol`).
    vol : cortex.VolumeRGB, optional
        A volume from `pycortex_make_vol2D_rgba`. Its stashed metadata
        is used for any of cmap/vmin/vmax/vmin2/vmax2/flip_y that
        weren't passed explicitly.
    cmap : str, optional
        Name of the 2D colormap PNG. Required if `vol` isn't given or
        doesn't carry metadata.
    vmin, vmax, vmin2, vmax2 : float, optional
        Axis ranges for the legend. Must match what went into the vol.
    xlabel, ylabel : str
        Axis labels on the legend inset.
    position : (x, y, w, h)
        Inset position in figure-fraction coords. Nudge this if the
        legend overlaps the flatmap.
    flip_y : bool, optional
        Must match the `flip_y` used when building the vol. Inferred
        from the vol when not given.
    labelsize, ticksize : int
        Font sizes for axis labels and ticks.

    Returns
    -------
    matplotlib.axes.Axes : the inset axes.
    """
    if vol is not None:
        cmap = cmap if cmap is not None else getattr(vol, '_cmap2d', None)
        vmin = vmin if vmin is not None else getattr(vol, '_vmin', 0)
        vmax = vmax if vmax is not None else getattr(vol, '_vmax', 1)
        vmin2 = vmin2 if vmin2 is not None else getattr(vol, '_vmin2', 0)
        vmax2 = vmax2 if vmax2 is not None else getattr(vol, '_vmax2', 1)
        if flip_y is None:
            flip_y = getattr(vol, '_flip_y', True)
    if flip_y is None:
        flip_y = True
    if cmap is None:
        raise ValueError(
            "add_2d_colorbar: need a cmap (either pass `vol` from "
            "pycortex_make_vol2D_rgba, or pass `cmap` explicitly)."
        )

    cmap_dir = cortex.options.config.get('webgl', 'colormaps')
    cmap_img = imread(os.path.join(cmap_dir, cmap + '.png'))

    ax = fig.add_axes(list(position))
    ax.imshow(
        cmap_img,
        origin='upper' if flip_y else 'lower',
        extent=[vmin, vmax, vmin2, vmax2],
        aspect='auto',
    )
    ax.set_xlabel(xlabel, fontsize=labelsize)
    ax.set_ylabel(ylabel, fontsize=labelsize)
    ax.tick_params(labelsize=ticksize)
    return ax


# ---------------------------------------------------------------------------
# Generic flatmap plotter (works for Volume, Volume2D, VolumeRGB)
# ---------------------------------------------------------------------------

def pycortex_plot_flatmap_from_vol(
    vol,
    title='',
    figsize=(16, 4),
    quickflat_kwargs=None,
):
    """
    Plot any pycortex Volume-like object (Volume, Volume2D, VolumeRGB) on
    a flatmap. Pair with any of the `pycortex_make_*_rgba` helpers, and
    with `add_2d_colorbar` for 2D maps.

    Parameters
    ----------
    vol : cortex.Volume | cortex.Volume2D | cortex.VolumeRGB
        Pre-built pycortex volume data object.
    title : str
        Axes title.
    figsize : tuple
        Figure size passed to `plt.subplots`.
    quickflat_kwargs : dict, optional
        Extra kwargs for `cortex.quickflat.make_figure`. Defaults to
        `{'with_curvature': True}`.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes.
    """
    quickflat_kwargs = dict(quickflat_kwargs or {})
    quickflat_kwargs.setdefault('with_curvature', True)

    fig, ax = plt.subplots(figsize=figsize)
    quickflat.make_figure(vol, fig=fig, **quickflat_kwargs)
    ax.set_title(title)
    ax.set_axis_off()
    return fig, ax