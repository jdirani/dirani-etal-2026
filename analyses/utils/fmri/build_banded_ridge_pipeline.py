from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from himalaya.kernel_ridge import MultipleKernelRidgeCV, ColumnKernelizer, Kernelizer
from voxelwise_tutorials.delayer import Delayer


def build_banded_ridge_pipeline(feature_spaces, banded_ridge_params, cv, delays):
    """
    Build a kernelized banded ridge regression pipeline, with Finite Impulse Response (FIR).
    Implementation was based on https://gallantlab.org/voxelwise_tutorials/notebooks/shortclips/09_fit_banded_ridge_model.html

    Parameters
    ----------
    feature_spaces : dict
        Dictionary mapping feature space names to column slices or indices.
    banded_ridge_params : dict
        Parameters for MultipleKernelRidgeCV (e.g., alpha grid, batch sizes).
    cv : scikit-learn cross-validation object
        Cross-validation strategy (e.g., KFold, GroupKFold).
    delays : list of int
        List of FIR delays to apply (e.g., [1, 2, 3, 4]).

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        Full sklearn pipeline that includes kernelization and model fitting.
    column_kernelizer : ColumnKernelizer
        The kernelizer step (can be inspected or reused separately).
    """

    # Step 1: Create preprocessing pipeline for one feature space
    preprocess_pipeline = make_pipeline(
        StandardScaler(with_mean=True, with_std=False),
        Delayer(delays=delays),
        Kernelizer(kernel="linear")
    )

    # Step 2: Create a kernelizer for each feature space.
    # - Each entry in kernelizers_tuples defines how to process one feature space
    # - Slices specify the columns of the input matrix belonging to each feature space
    kernelizers_tuples = [
        (name, preprocess_pipeline, slice_)
        for name, slice_ in feature_spaces.items()
    ]

    column_kernelizer = ColumnKernelizer(kernelizers_tuples)

    # Step 3: Define the banded ridge model
    mkr_model = MultipleKernelRidgeCV(
        kernels="precomputed",
        solver="random_search",
        solver_params=banded_ridge_params,
        cv=cv
    )

    # Step 4: Build the full pipeline
    # - First step computes and stacks kernels for all feature spaces
    # - Second step performs cross-validated banded ridge regression
    pipeline = make_pipeline(column_kernelizer, mkr_model)

    return pipeline, column_kernelizer
