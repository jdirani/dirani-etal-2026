from os.path import join

class FMRIPathConfig:
    """
    A configuration class to manage file paths and naming conventions for fMRI data processing.

    Attributes:
    ----------
    fmri_dir : str
        Root directory containing fMRI data files.
    patterns : dict
        Dictionary of path patterns for run files, brain masks, movement files, and anatomical scans.

    
    Class can be initialized with default path config: e.g. path_config = FMRIPathConfig()
    Alternatively, new paths patterns can be used by passing a dict that matches the format below:
        
        {
        'run_file': "{subj}/ses-0{session_number}/niftis/{subj}_natcook_run{run_number}_ses-0{session_number}+tlrc.nii.gz",
        'brainmask': "{subj}/anat/{subj}_ses-01_natcook_brainmask+tlrc.nii.gz",
        'movement': "{subj}/niftis/{subj}_natcook_movement_run{run_number}_ses-0{session_number}.1D",
        'anatomy': "{subj}/anat/{subj}_ses-01_anat_final+tlrc.nii.gz"
        }

    """
    
    def __init__(self, fmri_dir, file_patterns=None):
        self.fmri_dir = fmri_dir
        
        # Default file patterns - can be overridden
        self.patterns = file_patterns or {
            'run_file': "{subj}/ses-0{session_number}/niftis/{subj}_natcook_run{run_number}_ses-0{session_number}+tlrc.nii.gz",
            'brainmask': "{subj}/anat/{subj}_ses-01_natcook_brainmask_dilated+tlrc.nii.gz",
            'movement': "{subj}/niftis/{subj}_natcook_movement_run{run_number}_ses-0{session_number}.1D",
            'anatomy': "{subj}/anat/{subj}_ses-01_anat_final+tlrc.nii.gz",
            'postproc': "{subj}/ses-0{session_number}/postproc/{subj}_natcook_run{run_number}_ses-0{session_number}.npy",
            'cortical_mask': "{subj}/anat/{subj}_cortical_mask.npy"
        }
    
    def get_postproc_path(self, subj, session_number, run_number):
        return join(self.fmri_dir, 
                   self.patterns['postproc'].format(
                       subj=subj, session_number=session_number, run_number=run_number))

    def get_run_path(self, subj, session_number, run_number):
        return join(self.fmri_dir, 
                   self.patterns['run_file'].format(
                       subj=subj, session_number=session_number, run_number=run_number))
    
    def get_brainmask_path(self, subj):
        return join(self.fmri_dir, 
                   self.patterns['brainmask'].format(subj=subj))
    
    def get_movement_path(self, subj, session_number, run_number):
        return join(self.fmri_dir,
                   self.patterns['movement'].format(
                       subj=subj, session_number=session_number, run_number=run_number))
    
    def get_anat_path(self, subj):
        return join(self.fmri_dir,
                   self.patterns['anatomy'].format(subj=subj))

    def get_cortical_mask_path(self, subj):
        return join(self.fmri_dir,
                   self.patterns['cortical_mask'].format(subj=subj))