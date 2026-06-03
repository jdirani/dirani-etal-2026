import pandas as pd
import numpy as np
from interpdata import lanczosinterp2D
import matplotlib.pyplot as plt





def pad_runs_with_fixation(
    runs_feats,
    run_numbers,
    start_fixation_len_secs,
    end_fixation_len_secs,
    annot_sampling_hz,
    verbose=True
):
    """
    Pads all runs with fixation (zero-valued) samples at the start and end.

    Parameters
    ----------
    runs_feats : dict[int, np.ndarray]
        Dictionary mapping run numbers to annotation arrays (n_samples x n_features).
    run_numbers : list[int]
        List of run numbers to pad.
    start_fixation_len_secs : float
        Length of padding at the start (in seconds).
    end_fixation_len_secs : float
        Length of padding at the end (in seconds).
    annot_sampling_hz : float
        Sampling rate (e.g., 4.0 Hz).
    verbose : bool
        Whether to print detailed info per run.

    Returns
    -------
    runs_feats_padded : dict[int, np.ndarray]
        Dictionary with padded feature arrays.
    """
    def pad_features(features, n_samples_start, n_samples_end):
        n_features = features.shape[1]
        start_pad = np.zeros((n_samples_start, n_features))
        end_pad = np.zeros((n_samples_end, n_features))
        return np.vstack([start_pad, features, end_pad])
    
    def secs_to_time(secs):
        minutes = int(secs // 60)
        seconds = int(secs % 60)
        milliseconds = int((secs - int(secs)) * 1000)
        return f'{minutes}:{seconds:02d}.{milliseconds:03d}'

    runs_feats_padded = {}

    for run_number in run_numbers:
        features = runs_feats[run_number]
        num_samples = features.shape[0]
        num_features = features.shape[1]
        num_secs = num_samples / annot_sampling_hz

        start_fix_samples = int(start_fixation_len_secs * annot_sampling_hz)
        end_fix_samples = int(end_fixation_len_secs * annot_sampling_hz)

        if verbose:
            print(f'-------------- RUN {run_number} ------------')
            print(f'Original length: {num_samples} samples, {num_secs:.2f} secs')
            print(f'Time formatted: {secs_to_time(num_secs)}')
            print(f'Features: {num_features}')
            print(f'Padding: {start_fixation_len_secs} secs ({start_fix_samples} samples) at START')
            print(f'         {end_fixation_len_secs} secs ({end_fix_samples} samples) at END')

        # Pad and store
        padded = pad_features(features, start_fix_samples, end_fix_samples)
        runs_feats_padded[run_number] = padded

        if verbose:
            new_samples = padded.shape[0]
            new_secs = new_samples / annot_sampling_hz
            print(f'After padding: {new_samples} samples, {new_secs:.2f} secs')
            print(f'Time formatted: {secs_to_time(new_secs)}')
            print('')

    return runs_feats_padded






def crop_annotations_to_match_scans(runs_feats, runs_data, annot_sampling_hz, TR, verbose=True):
    """
    Crop annotation time series to match the duration of corresponding fMRI scans.
    This assumes the fmri data lengths are always shorter than their corresponding features lengths.

    Parameters
    ----------
    runs_feats : dict
        Dictionary of annotations per run: {run_number: np.ndarray of shape (n_timepoints, n_features)}
    runs_data : dict
        Dictionary of fMRI data per run: {run_number: np.ndarray of shape (n_TRs, ...)}
    annot_sampling_hz : float
        Sampling frequency of the annotations (in Hz).
    TR : float
        Repetition time of the fMRI data (in seconds).
    verbose : bool, default=True
        If True, print before/after annotation lengths and scan duration.

    Returns
    -------
    runs_feats_cropped : dict
        Dictionary with cropped annotation arrays, matching the duration of each scan.
    """
    import numpy as np

    runs_feats_cropped = {}
    all_run_numbers = runs_feats.keys()

    for run_number in all_run_numbers:
        # Compute scan duration in seconds
        scan_len_sec = runs_data[run_number].shape[0] * TR

        # Convert to number of annotation samples
        target_annot_len = int(scan_len_sec * annot_sampling_hz)

        # Crop annotation to match scan
        runs_feats_cropped[run_number] = runs_feats[run_number][:target_annot_len]

    if verbose:
        print("\n--- Annotation Cropping Summary ---")
        for run_number in all_run_numbers:
            # Original and cropped lengths
            original_samples = len(runs_feats[run_number])
            original_sec = original_samples / annot_sampling_hz

            cropped_samples = len(runs_feats_cropped[run_number])
            cropped_sec = cropped_samples / annot_sampling_hz

            # Scan duration
            scan_samples = runs_data[run_number].shape[0]
            scan_sec = scan_samples * TR

            print(f"Run {run_number}:")
            print(f"  Annotation length (before): {original_samples} samples ({original_sec:.2f} seconds)")
            print(f"  Annotation length (after):  {cropped_samples} samples ({cropped_sec:.2f} seconds)")
            print(f"  Scan length:                {scan_samples} TRs     ({scan_sec:.2f} seconds)")
            print(f"  Match: {'[GOOD]' if np.isclose(cropped_sec, scan_sec, atol=0.01) else '[BAD]'}\n")

    return runs_feats_cropped





def lanczos_downsample_annotations_to_TR(
    runs_feats,
    runs_data,
    annot_sampling_hz,
    TR,
    lanczos_window=3,
    lanczos_cutoff_mult=1.0,
    rectify=False,
    plot_result_sample=False
):
    """
    Downsamples annotation features to match fMRI TR resolution using Lanczos interpolation.

    Parameters:
    - runs_feats: dict {run_number: np.array}, high-res annotation features
    - runs_data: dict {run_number: np.array}, fMRI data to determine TR count
    - annot_sampling_hz: float, sampling rate of annotation features
    - TR: float, fMRI repetition time (in seconds)
    - lanczos_window: int, size of the Lanczos window (default: 3)
    - lanczos_cutoff_mult: float, frequency cutoff multiplier (default: 1.0)
    - rectify: bool, whether to rectify the signal (default: False)

    Returns:
    - downsampled_feats: dict {run_number: np.array}, features aligned with TR sampling
    """

    downsampled_feats = {}

    for run_number, feat_data in runs_feats.items():
        num_trs = runs_data[run_number].shape[0]
        feature_times_secs = np.arange(feat_data.shape[0]) / annot_sampling_hz
        tr_times_secs = np.arange(num_trs) * TR

        downsampled_data = lanczosinterp2D(
            data=feat_data,
            oldtime=feature_times_secs,
            newtime=tr_times_secs,
            window=lanczos_window,
            cutoff_mult=lanczos_cutoff_mult,
            rectify=rectify
        )

        downsampled_feats[run_number] = downsampled_data

        # Optionally plot the first feature dimension before and after downsampling
        if plot_result_sample:
            plt.figure(figsize=(12, 6))

            # Plot original data
            feature_idx_to_plot = 0
            plt.subplot(2, 1, 1)
            plt.plot(feature_times_secs, feat_data[:, feature_idx_to_plot])
            plt.title(f"Run {run_number} - Original Feature {feature_idx_to_plot} ({annot_sampling_hz} Hz)")
            plt.xlabel("Time (s)")
            plt.ylabel("Value")

            # Plot downsampled data
            plt.subplot(2, 1, 2)
            plt.plot(tr_times_secs, downsampled_data[:, feature_idx_to_plot], 'r.-')
            plt.title(f"Run {run_number} - Downsampled Feature {feature_idx_to_plot} (TR resolution)")
            plt.xlabel("Time (s)")
            plt.ylabel("Value")

            plt.tight_layout()
            plt.show()


    print("Downsampling complete.")
    return downsampled_feats



# ===================================== Viz ========================================
def plot_feature_with_TRs(
    runs_feats, 
    runs_data, 
    run_number, 
    feature_index=0, 
    annot_sampling_hz=4, 
    TR=2, 
    time_window=(0, 20)
):
    """
    Plots a sample of a single feature's values over time alongside TR onset times.

    Parameters:
    - runs_feats: dict, contains feature arrays per run (shape: [n_times, n_features])
    - runs_data: dict, contains original run data to determine TR count (shape: [n_TRs, ...])
    - run_number: int, which run to visualize
    - feature_index: int, index of the feature to visualize (default: 0)
    - annot_sampling_hz: float, sampling rate of feature data (e.g., 4 Hz)
    - TR: float, repetition time in seconds (default: 2)
    - time_window: tuple, time range (in seconds) to display on x-axis
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    # Extract feature values
    feat_values = runs_feats[run_number][:, feature_index]

    # Compute TR times 
    nTRs = runs_data[run_number].shape[0]
    TR_times_secs = np.arange(nTRs) * TR

    # Generate x-axis time points for feature values
    xaxis_times_secs = np.arange(len(feat_values)) / annot_sampling_hz 

    # Plot feature values
    ax.stem(xaxis_times_secs, feat_values, linefmt='k-', markerfmt='ko', basefmt='k', label='Feature value')

    # Plot TR onset times
    ax.stem(TR_times_secs, np.ones(nTRs) * 0.1, linefmt='r-', markerfmt='rx', basefmt='r', label='TR onset')

    # Formatting
    ax.set_xlim(time_window)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Feature Value')
    ax.legend()
    ax.set_title(f'Run {run_number} - Feature {feature_index}')

    plt.tight_layout()
    plt.show()
    
    return fig, ax
