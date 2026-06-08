import os
import pandas as pd
import librosa
import numpy as np
from tqdm import tqdm
from glob import glob
import scipy.stats # For skewness and kurtosis
from math import ceil, log2 # For calculating num_octaves

# --- spafe Import ---
try:
    from spafe.features.cqcc import cqcc as spafe_cqcc_extractor
    # from spafe.utils.cepstral import deltas # spafe has its own deltas, or use librosa's
except ImportError as e:
    print(f"ImportError: {e}. Could not import from 'spafe'.")
    print("Please ensure spafe is installed correctly: `pip install spafe`")
    exit()

# --- Configuration for CQCC Extraction using spafe ---
SPAFE_CQCC_NUM_CEPS = 20      # num_ceps: Number of cepstra to return
SPAFE_LOW_FREQ = 30           # low_freq (Hz) for CQT
SPAFE_HIGH_FREQ = 8000        # high_freq (Hz) for CQT or None for Nyquist
SPAFE_BINS_PER_OCTAVE_TARGET = 36 # Our desired number of bins per octave for CQT

# Local audio data directory
AUDIO_DIR_LOCAL = "ASVspoof5/flac_T"

# Output files
OUTPUT_DIR_LOCAL = "csv/"
os.makedirs(OUTPUT_DIR_LOCAL, exist_ok=True)
LOCAL_SPAFE_CQCC_FEATURES_CSV_PATH = os.path.join(OUTPUT_DIR_LOCAL, "local_test_spafe_cqcc_features.csv")


def get_utterance_level_functionals_spafe(frame_level_features_matrix, base_feature_name="feature"):
    """Applies functionals to a NumPy matrix of frame-level features."""
    if frame_level_features_matrix is None or frame_level_features_matrix.shape[0] == 0:
        return pd.Series(dtype='float64'), []

    num_coeffs_per_frame = frame_level_features_matrix.shape[1]
    mean_vals = np.mean(frame_level_features_matrix, axis=0)
    std_vals = np.std(frame_level_features_matrix, axis=0)
    skew_vals = scipy.stats.skew(frame_level_features_matrix, axis=0)
    kurt_vals = scipy.stats.kurtosis(frame_level_features_matrix, axis=0)

    utterance_features = []
    utterance_feature_names = []
    for i in range(num_coeffs_per_frame):
        base_col_name = f"{base_feature_name}_{i}"
        utterance_features.extend([mean_vals[i], std_vals[i], skew_vals[i], kurt_vals[i]])
        utterance_feature_names.extend([
            f"{base_col_name}_mean", f"{base_col_name}_std",
            f"{base_col_name}_skew", f"{base_col_name}_kurt"
        ])
    return pd.Series(utterance_features, index=utterance_feature_names), utterance_feature_names


def process_local_audio_files_for_spafe_cqcc(list_of_audio_paths, output_csv_path):
    all_utterance_features_list = []
    print(f"\nStarting spafe CQCC feature extraction for {len(list_of_audio_paths)} local files...")
    first_successful_column_names = None

    for audio_file_path in tqdm(list_of_audio_paths, desc="Processing audio with spafe"):
        utterance_id = os.path.splitext(os.path.basename(audio_file_path))[0]
        try:
            sig, sr = librosa.load(audio_file_path, sr=None)

            current_high_freq = SPAFE_HIGH_FREQ if SPAFE_HIGH_FREQ is None or SPAFE_HIGH_FREQ < sr/2 else sr//2
            
            # Calculate nfilts based on target bins_per_octave, low_freq, high_freq
            if SPAFE_LOW_FREQ <= 0: # log2 undefined for non-positive
                 raise ValueError("low_freq must be positive for octave calculation.")
            num_octaves = ceil(log2(current_high_freq / SPAFE_LOW_FREQ))
            nfilts_for_call = num_octaves * SPAFE_BINS_PER_OCTAVE_TARGET
            # Ensure nfilts is at least some minimum, e.g., spafe's default for num_ceps or more
            nfilts_for_call = max(nfilts_for_call, SPAFE_CQCC_NUM_CEPS * 2, 24) # Ensure enough filters


            # --- Call spafe's CQCC function ---
            frame_level_cqccs = spafe_cqcc_extractor(
                sig,
                fs=sr,
                num_ceps=SPAFE_CQCC_NUM_CEPS,
                low_freq=SPAFE_LOW_FREQ,
                high_freq=current_high_freq,
                number_of_bins_per_octave = SPAFE_BINS_PER_OCTAVE_TARGET
                # And then `nfilts` might be calculated differently or take a default.
                # For now, controlling via `nfilts` derived from `bins_per_octave_target`.
            )
            # spafe's cqcc output is (num_frames, num_ceps)

            if frame_level_cqccs is None or frame_level_cqccs.shape[0] == 0:
                print(f"Warning: No spafe CQCC features extracted for {utterance_id} (empty matrix).")
                continue

            cqccs_t = frame_level_cqccs.T
            delta_cqccs = librosa.feature.delta(cqccs_t, width=9)
            delta2_cqccs = librosa.feature.delta(cqccs_t, order=2, width=9)
            combined_frame_features = np.hstack((frame_level_cqccs, delta_cqccs.T, delta2_cqccs.T))
            
            utterance_stats_series, current_col_names = get_utterance_level_functionals_spafe(
                combined_frame_features, base_feature_name="spafe_cqcc_sdd"
            )
            
            if utterance_stats_series.empty:
                print(f"Warning: No utterance level features generated for {utterance_id}")
                continue

            if first_successful_column_names is None and current_col_names:
                first_successful_column_names = ['name'] + current_col_names

            utterance_stats_series['name'] = utterance_id
            all_utterance_features_list.append(utterance_stats_series)

        except Exception as e:
            print(f"ERROR processing {utterance_id} for spafe CQCC: {e}")
            import traceback
            traceback.print_exc()

    if all_utterance_features_list:
        final_df = pd.DataFrame(all_utterance_features_list)
        if first_successful_column_names and 'name' in final_df.columns:
            valid_cols_for_reindex = [col for col in first_successful_column_names if col in final_df.columns]
            final_df = final_df[valid_cols_for_reindex]
        elif 'name' in final_df.columns:
            cols = ['name'] + [col for col in final_df.columns if col != 'name']
            final_df = final_df[cols]

        # Add parameters to filename for clarity
        param_string = f"_numceps{SPAFE_CQCC_NUM_CEPS}_lowf{SPAFE_LOW_FREQ}_highf{SPAFE_HIGH_FREQ}_bpo{SPAFE_BINS_PER_OCTAVE_TARGET}"
        output_csv_path_with_params = output_csv_path.replace(".csv", param_string + ".csv")

        print(f"\nSaving combined spafe CQCC features to: {output_csv_path_with_params}")
        final_df.to_csv(output_csv_path_with_params, index=False, sep=';')
        print(f"spafe CQCC feature extraction complete. Output: {output_csv_path_with_params}")
        print("Final DataFrame head:")
        print(final_df.head())
    else:
        print("No spafe CQCC features were successfully extracted.")

# --- Main Execution ---
if __name__ == "__main__":
    local_flac_files = []
    if os.path.exists(AUDIO_DIR_LOCAL) and os.path.isdir(AUDIO_DIR_LOCAL):
        print(f"Searching for audio files in: {os.path.abspath(AUDIO_DIR_LOCAL)}")
        flac_found = sorted(glob(os.path.join(AUDIO_DIR_LOCAL, "*.flac")))
        wav_found = sorted(glob(os.path.join(AUDIO_DIR_LOCAL, "*.wav")))
        print(f"Found {len(flac_found)} FLAC files.")
        print(f"Found {len(wav_found)} WAV files.")
        local_flac_files.extend(flac_found)
        local_flac_files.extend(wav_found)
    else:
        print(f"Warning: Audio directory not found or not a directory: {os.path.abspath(AUDIO_DIR_LOCAL)}")

    if local_flac_files:
        num_test_files = min(5, len(local_flac_files))
        print(f"Processing a subset of {num_test_files} local files for testing with spafe.")
        process_local_audio_files_for_spafe_cqcc(local_flac_files[:num_test_files], LOCAL_SPAFE_CQCC_FEATURES_CSV_PATH)
    else:
        print(f"No audio files found in {AUDIO_DIR_LOCAL}. Please add some .flac or .wav files for testing.")