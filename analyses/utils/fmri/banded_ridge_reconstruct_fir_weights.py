import numpy as np


def banded_ridge_reconstruct_fir_weights(banded_ridge_results, verbose=True):
    '''
    adapted from  https://gallantlab.org/voxelwise_tutorials/notebooks/shortclips/05_fit_wordnet_model.html
    
    Reconstruct finite impulse response (FIR) feature weights from a fitted 
    banded ridge regression model, averaging across delays and scaling by 
    voxel performance.

    This function extracts primal coefficients from the trained pipeline, 
    rescales them according to model scores (to account for voxel prediction 
    accuracy), reshapes them by delay, and averages across delays to obtain 
    a single weight per feature. The final coefficient matrix has shape 
    (n_features, n_voxels), aligned with the provided feature legend.

    Parameters
    ----------
    banded_ridge_results : dict
        Output dictionary from banded ridge fitting, expected to contain:
          - 'pipeline' : fitted sklearn Pipeline with ColumnKernelizer and MultipleKernelRidgeCV
          - 'feature_space_names' : list of str, names of feature spaces
          - 'feats_legend' : list of str, feature labels across all spaces
          - 'mask_modelled_voxels' : boolean mask of voxels modeled
          - 'scores' : ndarray of voxel prediction scores
          - 'delays' : list of int. Delays used in the FIR model. Only used here for shape sanity checks.
    Returns
    -------
    coefs : ndarray, shape (n_features, n_voxels)
        Delay-averaged, score-scaled feature weights for each voxel.
    '''

    

    pipeline = banded_ridge_results['pipeline']
    feature_space_names = banded_ridge_results['feature_space_names']
    feats_legend = banded_ridge_results['feats_legend']
    mask_modelled_voxels = banded_ridge_results['mask_modelled_voxels']
    scores = banded_ridge_results['scores']
    delays = banded_ridge_results['delays']

    # ----- Get the primal_coefs: list of len=number of feature spaces, each list contains n_feats*delays for the corresponding feature space.
    X_fit = pipeline.named_steps['columnkernelizer'].get_X_fit() # this returns the feature matrices after delay/scaling/kernelizing.
    primal_coefs = pipeline.named_steps['multiplekernelridgecv'].get_primal_coef(X_fit)

    # some paranoid sanity checks
    assert len(primal_coefs) == len(feature_space_names), 'Number of coefficient groups dont match number of feature spaces'
    total_coefs_across_groups = 0
    for _primal_coef, _featspace_name in zip(primal_coefs, feature_space_names):
        if verbose:
            print(f'{_featspace_name} : {_primal_coef.shape}')
        n_feats_in_current_feature_space = _primal_coef.shape[0]
        total_coefs_across_groups += n_feats_in_current_feature_space

        # make sure the feature_space (names) are in fact aligned witht the primal_coefs, because paranoia
        feature_space_prefix = _featspace_name.split('_')[0]
        expected_n_feats_in_current_feature_space = len([i for i in feats_legend if i.startswith(feature_space_prefix)]) * len(delays)
        assert expected_n_feats_in_current_feature_space == n_feats_in_current_feature_space
    if verbose:
        print('\ntotal number of weights = ', total_coefs_across_groups)
        print('expected number of weights = ', len(feats_legend)* len(delays))


    # ------ For each feature space, we need to (1) Scale the coefficients (2) average over delays
    # We need to fetch the model scores to scale the coefficients
    # Scale the weights
    masked_scores = scores[mask_modelled_voxels]


    coefs = []
    for idx_featspace, (_primal_coef, _featspace_name) in enumerate(zip(primal_coefs, feature_space_names)):
        
        # (1) Scale the coefficients
        if verbose:
            print(_featspace_name)
        _primal_coef /= np.linalg.norm(_primal_coef, axis=0)[None]
        _primal_coef *= np.sqrt(np.maximum(0, masked_scores))[None]

        # (2) Average over delays
        pipeline_featspace_kernel = pipeline.named_steps['columnkernelizer'].transformers[idx_featspace] # get the step name from the pipeline
        kernel_name = pipeline_featspace_kernel[0]

        # make sure we have the right step in the pipeline: i.e. make sure primal_coefs and feature_space_names are aligned
        assert kernel_name == _featspace_name, f"Mismatch between pipeline step and feature space: {kernel_name} vs {_featspace_name}"
        
        transformer_pipeline = pipeline_featspace_kernel[1] # Access the Pipeline object
        kernel_delayer = transformer_pipeline.named_steps.get('delayer')


        primal_coef_per_delay = kernel_delayer.reshape_by_delays(_primal_coef, axis=0)
        if verbose:
            print("    (n_delays, n_features, n_voxels) =", primal_coef_per_delay.shape)

        # average over delays
        average_coef = np.mean(primal_coef_per_delay, axis=0)
        if verbose:
            print("    (n_features, n_voxels) =", average_coef.shape)

        # Collect the averaged coefficients for each feature space
        coefs.append(average_coef)  
        # At this point: 
        #   - coefs is a list of arrays
        #   - each element has shape (n_features_in_space, n_voxels)


    # After the loop, stack everything along the first axis 
    coefs = np.vstack(coefs)  # Final shape: (total_n_features, n_voxels)
                                #   - total_n_features = sum of n_features_in_space across all feature spaces
                                #   - n_voxels = number of modeled voxels

    assert coefs.shape[0] == len(feats_legend)
    if verbose:
        print(f"\nFinal shape of coefs: (n_features, n_voxels) = {coefs.shape}")
        print(f"len(feats_legend) = {len(feats_legend)}")

    return coefs