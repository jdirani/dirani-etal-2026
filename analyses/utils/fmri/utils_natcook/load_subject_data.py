import numpy as np
import nibabel as nib

def load_subject_data(subj, path_config, session_numbers=[1, 2], run_numbers=[1, 2, 3, 4, 5, 6]):
    '''
    Loads subject data for all available sessions. If no data was found for a session,
    all_data won't have this session as a key.

    Parameters:
    ----------
    subj : str
        Subject identifier.
    path_config : FMRIPathConfig or None
        Configuration object for path definitions.
    session_numbers : list of int, optional
        Sessions to include (default is [1, 2]).
    run_numbers : list of int, optional
        Run indices to include (default is [1, 2, 3, 4, 5, 6]).

    Returns:
    -------
    all_data : dict
        Nested dictionary with structure {session_number: {run_number: ndarray}}.
    brainmask_img : Nifti1Image or None
        Brain mask image used (or None if not found or not applicable).
    '''

    all_data = {}

    # Try to load the brainmask image if available
    try:
        brainmask_path = path_config.get_brainmask_path(subj)
        brainmask_img = nib.load(brainmask_path)
    except Exception as e:
        print(f"Could not load brainmask for {subj}: {e}")
        brainmask_img = None

    def load_run(subj, session_number, run_number):
        run_path = path_config.get_postproc_path(subj, session_number, run_number)
        try:
            run_data = np.load(run_path)
            return run_data
        except Exception as e:
            print(f"Could not load run {subj} - {run_number} - ses-0{session_number}: {e}")
            return None

    for session_number in session_numbers:
        session_data = {}
        for run_number in run_numbers:
            run_data = load_run(subj, session_number, run_number)
            if run_data is not None:
                session_data[run_number] = run_data
        if session_data:  # Only add session if it has valid data
            all_data[session_number] = session_data

    
    print(f'    -> {subj} : {len(all_data)} sessions found ({list(all_data.keys())})')

    return all_data, brainmask_img
