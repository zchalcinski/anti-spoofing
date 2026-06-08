# ==============================================================================
# --- SECTION 1: SETUP AND DEPENDENCIES ---
# ==============================================================================
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns

print("--- [1/6] Dependencies imported successfully. ---")


# ==============================================================================
# --- SECTION 2: CONFIGURATION & PATHS ---
# ==============================================================================
# Define input paths for extracted acoustic features (eGeMAPS + CQCC)
TRAIN_FEATURES_CSV_PATH = '../csv/combined/asvspoof5_T_FULL_gemaps_cqcc_combined_features.csv' 
DEV_FEATURES_CSV_PATH = '../csv/combined/asvspoof5_D_FULL_gemaps_cqcc_combined_features.csv'
EVAL_FEATURES_CSV_PATH = '../csv/combined/asvspoof5_E_aa_gemaps_cqcc_combined_features.csv'

# Define output paths for serialized model artifacts and scalers
MODEL_SAVE_PATH = 'models/model_rf_optimized_gemaps_TRAIN_DEV_combined.pkl'
SCALER_SAVE_PATH = 'scaler_gemaps_TRAIN_DEV_combined.pkl'
ENCODER_SAVE_PATH = 'label_encoder_TRAIN_DEV_combined.pkl'

# Define output directory for scientific metrics and graphs
EXPORT_DIR = 'scientific_exports_rf_eval'

# Protocol metadata columns
ID_COLUMN = 'name'
LABEL_COLUMN = 'label'
ATTACK_TYPE_COLUMN = 'attackType'

# Ensure target directories exist
os.makedirs('models', exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

print(f"--- [2/6] Configuration loaded. Output directory: {EXPORT_DIR} ---")


# ==============================================================================
# --- SECTION 3: LOAD AND COMBINE TRAINING DATA (TRAIN + DEV) ---
# ==============================================================================
print("\n--- [3/6] Loading and combining TRAIN and DEV datasets ---")
# Loading base feature partitions
df_train = pd.read_csv(TRAIN_FEATURES_CSV_PATH, delimiter=';')
df_dev = pd.read_csv(DEV_FEATURES_CSV_PATH, delimiter=';')

# Merge Train and Dev sets to maximize the data manifold for final model training
df_train_combined = pd.concat([df_train, df_dev], ignore_index=True)
df_train_combined.dropna(subset=[LABEL_COLUMN], inplace=True)
print(f"  > Combined Training data shape (Train + Dev): {df_train_combined.shape}")

# Encode textual class labels into numerical format
label_encoder = LabelEncoder()
df_train_combined['true_label_encoded'] = label_encoder.fit_transform(df_train_combined[LABEL_COLUMN])

# Map discrete classes to establish strict binary targets (Bonafide: 0, Spoof: 1)
classes = label_encoder.classes_
encoded_classes = label_encoder.transform(classes)
class_mapping = dict(zip(classes, encoded_classes))
SPOOF_LABEL = class_mapping.get('spoof', 1)
BONAFIDE_LABEL = class_mapping.get('bonafide', 0)
print(f"  > Label mapping established: {class_mapping}")

# Isolate feature vectors from metadata
feature_columns = [col for col in df_train_combined.columns if col not in [ID_COLUMN, LABEL_COLUMN, 'true_label_encoded', ATTACK_TYPE_COLUMN]]
X_train = df_train_combined[feature_columns]
y_train = df_train_combined['true_label_encoded']

# Impute NaN/Inf values often caused by silent audio frames during feature extraction
if X_train.isnull().values.any() or np.isinf(X_train.values).any():
    print("  > Warning: NaN/Inf values found in combined training features. Imputing with mean...")
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    train_mean = X_train.mean()
    X_train = X_train.fillna(train_mean)
else:
    train_mean = X_train.mean() # Persist mean for downstream evaluation

# Fit standard scaler strictly on training data to prevent data leakage
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
print("  > StandardScaler fitted and applied to combined TRAIN+DEV data.")


# ==============================================================================
# --- SECTION 4: TRAIN AND SERIALIZE FINAL RANDOM FOREST MODEL ---
# ==============================================================================
print("\n--- [4/6] Training Final Random Forest Model ---")

# Pre-optimized hyperparameters derived from prior cross-validation grid search
best_params_rf = {
    'class_weight': None, 
    'criterion': 'entropy', 
    'max_depth': 20,
    'max_features': 'sqrt', 
    'min_samples_leaf': 2, 
    'min_samples_split': 5,
    'n_estimators': 300, 
    'random_state': 42, 
    'n_jobs': -1
}

final_rf_model = RandomForestClassifier(**best_params_rf)
final_rf_model.fit(X_train_scaled, y_train)
print("  > Model training complete.")

# Serialize model, scaler, and encoder to disk for reproducibility
joblib.dump(final_rf_model, MODEL_SAVE_PATH)
joblib.dump(scaler, SCALER_SAVE_PATH)
joblib.dump(label_encoder, ENCODER_SAVE_PATH)
print(f"  > Artifacts saved to: {MODEL_SAVE_PATH}")


# ==============================================================================
# --- SECTION 5: LOAD & PREPROCESS EVALUATION DATA ---
# ==============================================================================
print(f"\n--- [5/6] Loading EVALUATION data from: {EVAL_FEATURES_CSV_PATH} ---")
df_eval = pd.read_csv(EVAL_FEATURES_CSV_PATH, delimiter=';')
df_eval.dropna(subset=[LABEL_COLUMN], inplace=True)
print(f"  > Evaluation data shape: {df_eval.shape}")

# Map evaluation labels using the fitted encoder
df_eval['true_label_encoded'] = label_encoder.transform(df_eval[LABEL_COLUMN])

X_eval = df_eval[feature_columns]
y_eval = df_eval['true_label_encoded']

# Impute missing values using the mean derived strictly from the training partition
if X_eval.isnull().values.any() or np.isinf(X_eval.values).any():
    print("  > Warning: NaN/Inf values found in EVAL features. Imputing with TRAIN+DEV mean...")
    X_eval = X_eval.replace([np.inf, -np.inf], np.nan)
    X_eval = X_eval.fillna(train_mean.fillna(0))

# Scale evaluation features using the pre-fitted training scaler
X_eval_scaled = scaler.transform(X_eval)


# ==============================================================================
# --- SECTION 6: INFERENCE AND SCIENTIFIC METRICS EVALUATION ---
# ==============================================================================
print("\n--- [6/6] Running Inference and Calculating Metrics ---")
# Predict probabilities for the target class (Spoof)
y_scores = final_rf_model.predict_proba(X_eval_scaled)[:, SPOOF_LABEL]
df_eval['score'] = y_scores

def get_eer_stats(y_true, y_prob, pos_label=SPOOF_LABEL):
    """
    Computes Equal Error Rate (EER) and extracts the exact operational threshold
    where False Acceptance Rate (FPR) roughly equals False Rejection Rate (FNR).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=pos_label)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2 * 100.0
    eer_threshold = thresholds[eer_idx]
    return eer, eer_threshold, fpr, tpr, thresholds

overall_eer, eer_threshold, fpr, tpr, thresholds = get_eer_stats(y_eval, y_scores)
overall_auc = roc_auc_score(y_eval, y_scores)

# Assign hard predictions based strictly on the scientifically derived EER threshold
df_eval['prediction_eer_thresh'] = (df_eval['score'] >= eer_threshold).astype(int)
y_pred_optimal = df_eval['prediction_eer_thresh'].values

# Calculate global classification metrics
overall_acc = accuracy_score(y_eval, y_pred_optimal)
overall_prec = precision_score(y_eval, y_pred_optimal, pos_label=SPOOF_LABEL, zero_division=0)
overall_rec = recall_score(y_eval, y_pred_optimal, pos_label=SPOOF_LABEL, zero_division=0)
overall_f1 = f1_score(y_eval, y_pred_optimal, pos_label=SPOOF_LABEL, zero_division=0)

print(f"\n[GLOBAL METRICS - EVAL SET]")
print(f"  > Optimal EER Threshold: {eer_threshold:.4f}")
print(f"  > Overall EER:           {overall_eer:.2f}%")
print(f"  > Overall AUC:           {overall_auc:.4f}")
print(f"  > Accuracy (@ EER_th):   {overall_acc * 100:.2f}%")
print(f"  > F1-Score (@ EER_th):   {overall_f1:.4f}")

# Generate and export Visual Confusion Matrix
cm = confusion_matrix(y_eval, y_pred_optimal, labels=[0, 1])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Bonafide', 'Spoof'], yticklabels=['Bonafide', 'Spoof'])
plt.title(f'RF Confusion Matrix on EVAL (Threshold: {eer_threshold:.4f})')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(os.path.join(EXPORT_DIR, "03_confusion_matrix.png"), dpi=300)
plt.close()

# Evaluate system vulnerability against specific attack algorithms (Per-Attack Analysis)
print("\n[PER-ATTACK EER ANALYSIS]")
bonafide_df = df_eval[df_eval['true_label_encoded'] == BONAFIDE_LABEL]
spoof_df = df_eval[df_eval['true_label_encoded'] == SPOOF_LABEL]

attack_results = []
for attack in spoof_df[ATTACK_TYPE_COLUMN].unique():
    # Skip unclassified or empty attack labels
    if pd.isna(attack) or str(attack).strip() == '-': continue
        
    specific_attack_df = spoof_df[spoof_df[ATTACK_TYPE_COLUMN] == attack]
    
    # Isolate attack context: All bonafide samples vs target spoofing algorithm
    combined_df = pd.concat([bonafide_df, specific_attack_df])
    
    att_y_true = combined_df['true_label_encoded'].values
    att_y_scores = combined_df['score'].values
    
    try:
        att_eer, _, _, _, _ = get_eer_stats(att_y_true, att_y_scores)
        attack_results.append({
            'Attack_Type': attack, 
            'EER_%': att_eer, 
            'Spoof_Samples': len(specific_attack_df)
        })
    except ValueError:
        pass # Handle cases where dataset splits lack positive/negative samples

attack_metrics_df = pd.DataFrame(attack_results).sort_values(by='EER_%', ascending=False)
print("Hardest attacks to detect (Highest EER):")
print(attack_metrics_df.head(10).to_string(index=False))

# --- EXPORT SCIENTIFIC ARTIFACTS ---
print(f"\n--- Exporting Scientific Data to '{EXPORT_DIR}' ---")

# 1. Global Metrics
pd.DataFrame({
    'Metric': ['EER_%', 'EER_Threshold', 'ROC_AUC', 'Accuracy', 'Precision', 'Recall', 'F1_Score'],
    'Value': [overall_eer, eer_threshold, overall_auc, overall_acc, overall_prec, overall_rec, overall_f1]
}).to_csv(os.path.join(EXPORT_DIR, "01_global_metrics.csv"), index=False)

# 2. ROC Curve Mapping
pd.DataFrame({
    'FPR': fpr, 
    'TPR': tpr, 
    'Threshold': thresholds
}).to_csv(os.path.join(EXPORT_DIR, "02_roc_curve_data.csv"), index=False)

# 3. Confusion Matrix Raw Data
pd.DataFrame(
    cm, index=['True_Bonafide', 'True_Spoof'], columns=['Pred_Bonafide', 'Pred_Spoof']
).to_csv(os.path.join(EXPORT_DIR, "03_confusion_matrix_data.csv"))

# 4. Granular Per-Attack Analysis
attack_metrics_df.to_csv(os.path.join(EXPORT_DIR, "04_per_attack_eer.csv"), index=False)

# 5. Full Pipeline Output for Post-Hoc Error Analysis
df_eval.to_csv(os.path.join(EXPORT_DIR, "05_full_predictions.csv"), index=False)

print("\n--- Execution Complete. Model trained (T+D) and formally evaluated. ---")