import pandas as pd
import os

# --- Configuration ---
MERGE_TRAIN = False
MERGE_DEV = False
MERGE_EVAL = True

# Paths to your input CSV files
GEMAPS_TRAIN_CSV = './csv/asvspoof5_T_FULL_egemaps_features_withprotocol.csv'
CQCC_TRAIN_CSV = './csv/asvspoof5_T_FULL_spafe_cqcc.csv' # Make sure filename matches exactly
COMBINED_TRAIN_CSV = './csv/asvspoof5_T_FULL_gemaps_cqcc_combined_features.csv'

GEMAPS_DEV_CSV = './csv/asvspoof5_D_FULL_egemaps_features_withprotocol.csv'
CQCC_DEV_CSV = './csv/asvspoof5_D_FULL_spafe_cqcc.csv' # Make sure filename matches
COMBINED_DEV_CSV = './csv/asvspoof5_D_FULL_gemaps_cqcc_combined_features.csv'

GEMAPS_EVAL_CSV = './csv/eGemaps/asvspoof5_E_aa_egemaps_features_withprotocol.csv'
CQCC_EVAL_CSV = './csv/CQCC/asvspoof5_E_aa_spafe_cqcc.csv' # Make sure filename matches
COMBINED_EVAL_CSV = './csv/combined/asvspoof5_E_aa_gemaps_cqcc_combined_features.csv'

# The common column to merge on
MERGE_COLUMN = 'name'

# Delimiter used in your CSV files (likely ';')
CSV_DELIMITER = ';'

# --- Main Script ---
def merge_feature_csvs(gemaps_path, cqcc_path, output_path, merge_on_col, delimiter=';'):
    """
    Loads GeMAPS and CQCC feature CSVs, merges them on a common column,
    and saves the combined DataFrame.
    """
    print(f"Loading GeMAPS features from: {gemaps_path}")
    try:
        df_gemaps = pd.read_csv(gemaps_path, delimiter=delimiter)
        print(f"GeMAPS data loaded. Shape: {df_gemaps.shape}")
        if merge_on_col not in df_gemaps.columns:
            print(f"ERROR: Merge column '{merge_on_col}' not found in GeMAPS CSV!")
            return None
    except FileNotFoundError:
        print(f"ERROR: GeMAPS file not found at {gemaps_path}")
        return None
    except Exception as e:
        print(f"ERROR loading GeMAPS CSV: {e}")
        return None

    print(f"\nLoading CQCC features from: {cqcc_path}")
    try:
        df_cqcc = pd.read_csv(cqcc_path, delimiter=delimiter)
        print(f"CQCC data loaded. Shape: {df_cqcc.shape}")
        if merge_on_col not in df_cqcc.columns:
            print(f"ERROR: Merge column '{merge_on_col}' not found in CQCC CSV!")
            return None
    except FileNotFoundError:
        print(f"ERROR: CQCC file not found at {cqcc_path}")
        return None
    except Exception as e:
        print(f"ERROR loading CQCC CSV: {e}")
        return None

    # --- Perform the Merge ---
    print(f"\nMerging DataFrames on column: '{merge_on_col}'")
    # Using an inner merge: only rows with matching 'name' in both DataFrames will be kept.
    # If a 'name' is in one file but not the other, that row will be dropped.
    # If you want to keep all rows from GeMAPS and add CQCC where available (or vice-versa),
    # you could use how='left' or how='right' and then handle potential NaNs.
    # For feature sets, an inner merge is usually what you want if you expect full correspondence.
    try:
        df_combined = pd.merge(df_gemaps, df_cqcc, on=merge_on_col, how='inner')
        # Pandas automatically handles non-merge columns. If there are identically named
        # feature columns (other than 'name'), it would add suffixes like '_x', '_y'.
        # This shouldn't happen if GeMAPS and CQCC features have unique names.
        print(f"Merge successful. Combined DataFrame shape: {df_combined.shape}")

        # Sanity check:
        if len(df_combined) < min(len(df_gemaps), len(df_cqcc)):
            print("WARNING: The number of rows in the merged DataFrame is less than in one of the input files.")
            print("This means some 'name' entries were not found in both files for the inner merge.")
            print(f"GeMAPS rows: {len(df_gemaps)}, CQCC rows: {len(df_cqcc)}, Merged rows: {len(df_combined)}")
        elif len(df_combined) == 0 and (len(df_gemaps) > 0 or len(df_cqcc) > 0):
             print("WARNING: Merged DataFrame is empty! No common 'name' entries found.")


    except Exception as e:
        print(f"ERROR during merge operation: {e}")
        return None

    # --- Save the Combined DataFrame ---
    if df_combined is not None and not df_combined.empty:
        print(f"\nSaving combined features to: {output_path}")
        try:
            df_combined.to_csv(output_path, index=False, sep=delimiter)
            print(f"Successfully saved combined data to {output_path}")
            return df_combined
        except Exception as e:
            print(f"ERROR saving combined CSV: {e}")
            return None
    elif df_combined is not None and df_combined.empty :
        print("Skipping save as merged DataFrame is empty.")
        return None
    return None


if __name__ == "__main__":
    print("--- Starting CSV Merge Script ---")

    if MERGE_TRAIN:
        # Example for Training Data
        print("\n--- Merging TRAINING Data ---")
        
        df_train_merged = merge_feature_csvs(
            GEMAPS_TRAIN_CSV,
            CQCC_TRAIN_CSV,
            COMBINED_TRAIN_CSV,
            MERGE_COLUMN,
            delimiter=CSV_DELIMITER
        )
        if df_train_merged is not None:
            print("\nSample of merged TRAINING data:")
            print(df_train_merged.head())
            print(f"Total columns in merged training data: {len(df_train_merged.columns)}")
            # Expected columns: 1 (name) + 2 (label, attackType from gemaps_withprotocol) + 88-2 (GeMAPS actual features) + 240 (CQCC)
            # If your gemaps_withprotocol has 'name', 'label', 'attackType', then (88 GeMAPS feature cols) + 240 (CQCC) + 3 (name, label, attackType)
            # So, if gemaps_df had N_gemaps_features + 3 id/label cols, and cqcc_df had N_cqcc_features + 1 id col,
            # merged will have N_gemaps_features + N_cqcc_features + 3 id/label cols.
            # If gemaps was 91 cols (name, label, attackType, 88 features)
            # If cqcc was 241 cols (name, 240 features)
            # Merged: name, label, attackType, 88 gemaps_features, 240 cqcc_features = 1+2+88+240 = 331 columns.
            # Check: df_gemaps.shape[1] -1 (for name) + df_cqcc.shape[1] -1 (for name) + 1 (for common name)
            # + 2 (for label, attackType that are only in GeMAPS df)
            # = (len(df_gemaps.columns) -1) + (len(df_cqcc.columns) -1) + 1
            # No, simpler: len(df_gemaps.columns) + (len(df_cqcc.columns) - 1) because 'name' is merged.
            if df_train_merged is not None: # Re-check because it might be None after call
                expected_cols = (pd.read_csv(GEMAPS_TRAIN_CSV, delimiter=CSV_DELIMITER, nrows=0).shape[1] + 
                                pd.read_csv(CQCC_TRAIN_CSV, delimiter=CSV_DELIMITER, nrows=0).shape[1] - 1)
                print(f"Expected columns if no overlap other than 'name': {expected_cols}")

    if MERGE_DEV:
        # Example for Development Data (Uncomment and adjust paths when ready)
        print("\n\n--- Merging DEVELOPMENT Data ---")
        

        df_dev_merged = merge_feature_csvs(
            GEMAPS_DEV_CSV,
            CQCC_DEV_CSV,
            COMBINED_DEV_CSV,
            MERGE_COLUMN,
            delimiter=CSV_DELIMITER
        )
        if df_dev_merged is not None:
            print("\nSample of merged DEVELOPMENT data:")
            print(df_dev_merged.head())
            print(f"Total columns in merged development data: {len(df_dev_merged.columns)}")

    if MERGE_EVAL:
        # Example for Evaluation Data (Uncomment and adjust paths when ready)
        print("\n\n--- Merging EVALUATION Data ---")

        df_eval_merged = merge_feature_csvs(
            GEMAPS_EVAL_CSV,
            CQCC_EVAL_CSV,
            COMBINED_EVAL_CSV,
            MERGE_COLUMN,
            delimiter=CSV_DELIMITER
        )
        if df_eval_merged is not None:
            print("\nSample of merged EVALUATION data:")
            print(df_eval_merged.head())
            print(f"Total columns in merged evaluation data: {len(df_eval_merged.columns)}")

    print("\n--- CSV Merge Script Finished ---")