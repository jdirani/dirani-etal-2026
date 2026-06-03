import operator
from functools import reduce
import numpy as np

def average_feature_weights_over_delays(wt, ndelays):

    '''
    Adapted from the SpeechModelTutorial jupyter notebook (https://github.com/HuthLab/speechmodeltutorial/tree/master)
    '''
    try:
        # Undelay voxel weights (average across delays)
        udwt = reduce(operator.add, np.split(wt / ndelays, ndelays))
        return udwt
    except ValueError as e:
        if "array split does not result in an equal division" in str(e):
            raise ValueError("Shapes are wrong, did you forget to remove weights associated with non-delayed features (e.g. movement covariates)?") from e
        else:
            raise
