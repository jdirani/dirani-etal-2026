import numpy as np
from scipy.stats import zscore

def average_repeated_runs(init_runs_data, train_runs, test_runs):
    """
    Average repeated fMRI runs across sessions for training and testing runs.

    For training runs, this function averages corresponding runs from both sessions (if available)
    for each run number separately (results in one averaged run per run_number).
    For testing, it combines all specified test runs from both sessions into a single averaged run.
    
    Parameters
    ----------
    init_runs_data : dict
        Nested dictionary of processed runs: {session_number: {run_number: ndarray}}.
    train_runs : list of int
        List of run numbers to use for training. Averaging is done per run across sessions.
    test_runs : list of int
        List of run numbers to use for testing. Averaging is done across all test runs and sessions.

    Returns
    -------
    runs_data : dict
        Dictionary of averaged and z-scored runs.
        Keys are run numbers (training runs) and the string 'test_avg' (test runs).
        Values are np.ndarray of shape (n_TRs, n_voxels).
    """
    runs_data = {}
    session_numbers = sorted(init_runs_data.keys())

    # ---- For train runs
    for run_number in train_runs:
        run_ses1 = init_runs_data[1][run_number]
        if 2 in init_runs_data and run_number in init_runs_data[2]:    # Check if session 2 exists and contains this specific run number 
            run_ses2 = init_runs_data[2][run_number]
            avg_repeat = np.mean([run_ses1, run_ses2], axis=0)
        else:
            avg_repeat = run_ses1  # use only session 1 if session 2 is missing

        # z-score
        z_avg_repeat = zscore(avg_repeat, axis=0)
        runs_data[run_number] = z_avg_repeat

    # ---- For test runs
    all_test_data = []
    for run_number in test_runs:
        for ses_number in session_numbers:
            if ses_number in init_runs_data and run_number in init_runs_data[ses_number]:
                all_test_data.append(init_runs_data[ses_number][run_number])

    if all_test_data:
        all_test_data = np.array(all_test_data)
        test_avg_repeat = np.mean(all_test_data, axis=0)
        z_test_avg_repeat = zscore(test_avg_repeat, axis=0)
        runs_data["test_avg"] = z_test_avg_repeat
    else:
        print("Warning: No test data found.")

    return runs_data


