# ==============================================================================
# --- SECTION 1: SETUP AND DEPENDENCIES ---
# ==============================================================================

import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoFeatureExtractor, WavLMModel
import torch.nn as nn
import librosa
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, f1_score, confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns

print("--- [1/5] Dependencies installed and imported successfully. ---")


# ==============================================================================
# --- SECTION 2: CONFIGURATION ---
# ==============================================================================
class EvalConfig:
    """
    Encapsulates all evaluation hyperparameters and file paths. 
    Isolating these variables ensures reproducible inference pipelines.
    """
    IS_DEBUG_RUN = False                  # Set to True to test the pipeline on a small subset of data
    USE_CHECKPOINT = True                 # Toggle to load fine-tuned weights vs testing the raw backbone
    DEBUG_SAMPLES = 2000
    
    # Path configuration
    TRAINED_MODEL_DIR = Path("/kaggle/input/wavlm-train")
    BEST_MODEL_FILENAME = Path("/kaggle/input/wavlm-checkpoint/wavlm-base-plus-antispoof.pt")
    OUTPUT_DIR = Path("/kaggle/working/")
    MODEL_ARCH_PATH = "/kaggle/input/wavlm-base-plus/transformers/default/1"
    
    # Dataset references
    FILE_MAP_PATH = "/kaggle/input/asvspoof-filemap/audio_filepath_map.feather"
    AUDIO_BASE_PATH = Path("/kaggle/input/asvspoof5/")
    EVAL_PROTOCOL_PATH = AUDIO_BASE_PATH / 'ASVspoof5_protocols/ASVspoof5.eval.track_1.tsv'
    
    # Processing parameters
    BATCH_SIZE = 36
    NUM_WORKERS = 2
    TARGET_SR = 16000                     # Standard sample rate required by WavLM

# --- Initialize Environment ---
config = EvalConfig()
config.BEST_MODEL_PATH = config.BEST_MODEL_FILENAME
config.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Automatically select hardware accelerator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- [2/5] Configuration loaded. Running on: {device} ---")


# ==============================================================================
# --- SECTION 3: DATA PIPELINES AND HELPER FUNCTIONS ---
# ==============================================================================
def get_filepaths_and_labels(protocol_path, file_map_df):
    """
    Parses the ASVspoof evaluation protocol and merges it with the absolute 
    file paths. Retains metadata (filename, attack_type) critical for 
    post-evaluation per-attack vulnerability analysis.
    """
    protocol_df = pd.read_csv(protocol_path, sep='\s+', header=None,
                              names=['speaker_id', 'filename', 'gender', 'sys_id_1', 'sys_id_2', 
                                     'sys_id_3', 'codec', 'attack_type', 'label', 'blank'])
    
    # Retain only recognized classes
    protocol_df = protocol_df[protocol_df['label'].isin(['bonafide', 'spoof'])]
    
    # Map filenames to their physical storage locations
    merged_df = pd.merge(protocol_df, file_map_df, left_on='filename', right_on='filename_stem')
    
    merged_df.rename(columns={'full_path': 'filepath'}, inplace=True)
    merged_df.dropna(subset=['filepath'], inplace=True)
    
    # Binarize labels: Bonafide = 0, Spoof = 1
    merged_df['label_numeric'] = (merged_df['label'] == 'spoof').astype(int)
    
    return merged_df[['filepath', 'label_numeric', 'filename', 'attack_type']]

class AudioDataset(Dataset):
    """
    Minimalist Dataset tailored for inference. Strips out training-specific 
    augmentations and sequence truncations to evaluate the full audio signal.
    """
    def __init__(self, df, target_sr): 
        self.df = df
        self.target_sr = target_sr
        
    def __len__(self): 
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            # Load full audio sequence as a mono signal
            waveform, _ = librosa.load(row['filepath'], sr=self.target_sr, mono=True)
            return {
                "waveform": waveform, 
                "label": torch.tensor(row['label_numeric'], dtype=torch.float),
                "filename": row['filename'],
                "attack_type": row['attack_type']
            }
        except Exception: 
            return None


# ==============================================================================
# --- SECTION 4: NEURAL NETWORK ARCHITECTURE ---
# ==============================================================================
class AttentivePooling(nn.Module):
    """
    Calculates dynamic attention weights across the temporal dimension 
    to aggregate frame-level features into a single utterance-level vector.
    """
    def __init__(self, in_features):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(in_features, in_features // 2), 
            nn.Tanh(), 
            nn.Linear(in_features // 2, 1)
        )
        
    def forward(self, x, mask):
        attn_weights = self.attention_net(x).squeeze(-1)
        # Apply strict masking to ignore zero-padded frames
        attn_weights.data.masked_fill_(~mask.bool(), -float('inf'))
        attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(1)
        return torch.bmm(attn_weights, x).squeeze(1)

class WavLMClassifier(nn.Module):
    """
    The complete model architecture consisting of the WavLM feature extractor
    and a custom attentive pooling classification head.
    """
    def __init__(self, model_path):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained(model_path)
        hidden_size = self.wavlm.config.hidden_size
        self.pooling = AttentivePooling(hidden_size)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_values, attention_mask):
        # Extract features through the SSL backbone
        outputs = self.wavlm(input_values, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        
        # Calculate correct spatial dimensions after WavLM's convolutional downsampling
        output_lengths = self.wavlm._get_feat_extract_output_lengths(attention_mask.sum(-1))
        downsampled_attention_mask = torch.zeros(
            last_hidden_state.shape[:2], dtype=torch.long, device=last_hidden_state.device
        )
        for i, length in enumerate(output_lengths):
            downsampled_attention_mask[i, :length] = 1
        
        # Apply pooling and final linear projection
        pooled_output = self.pooling(last_hidden_state, downsampled_attention_mask)
        return self.classifier(pooled_output).squeeze(-1)

# ==============================================================================
# --- SECTION 5: BATCHING AND METRICS EVALUATION ---
# ==============================================================================
feature_extractor = AutoFeatureExtractor.from_pretrained(config.MODEL_ARCH_PATH)

def collate_fn(batch):
    """
    Prepares evaluation batches by applying padding via the HuggingFace feature 
    extractor to handle variable length audio within the same tensor block.
    """
    batch = [item for item in batch if item is not None]
    if not batch: return None

    waveforms = [item['waveform'] for item in batch]
    labels = [item['label'] for item in batch]
    filenames = [item['filename'] for item in batch]
    attack_types = [item['attack_type'] for item in batch]
    
    inputs = feature_extractor(waveforms, sampling_rate=config.TARGET_SR, padding=True, return_tensors="pt")
    
    return {
        "input_values": inputs.input_values,
        "attention_mask": inputs.attention_mask,
        "labels": torch.stack(labels),
        "filenames": filenames,
        "attack_types": attack_types
    }

def get_eer_stats(y_true, y_scores):
    """
    Computes the Equal Error Rate (EER) and extracts the exact operational 
    probability threshold where the False Acceptance Rate roughly equals 
    the False Rejection Rate. Returns curve vectors for downstream plotting.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    
    # Calculate exact midpoint for high precision EER
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2 * 100.0
    eer_threshold = thresholds[eer_idx]
    
    return eer, eer_threshold, fpr, tpr, thresholds

print("--- [3/5] Helper functions and classes defined. ---")


# ==============================================================================
# --- SECTION 6: MAIN EVALUATION EXECUTION LOOP ---
# ==============================================================================
if __name__ == '__main__':
    # --- Data Loading ---
    print("\n--- [4/5] Loading data sources... ---")
    FILE_MAP_INPUT_PATH = Path(config.FILE_MAP_PATH)
    if not FILE_MAP_INPUT_PATH.exists():
        print(f"FATAL: File map not found at {FILE_MAP_INPUT_PATH}.")
        exit()
        
    filepath_map_df = pd.read_feather(FILE_MAP_INPUT_PATH)
    
    # Load evaluation protocol
    full_eval_protocol = pd.read_csv(config.EVAL_PROTOCOL_PATH, sep='\s+', header=None,
                                     names=['speaker_id', 'filename', 'gender', 'sys_id_1', 'sys_id_2', 
                                            'sys_id_3', 'codec', 'attack_type', 'label', 'blank'])
    
    eval_df = get_filepaths_and_labels(config.EVAL_PROTOCOL_PATH, filepath_map_df)

    if config.IS_DEBUG_RUN:
        print(f"\n--- RUNNING IN DEBUG MODE: Using {config.DEBUG_SAMPLES} samples ---")
        eval_df = eval_df.sample(n=min(len(eval_df), config.DEBUG_SAMPLES), random_state=42)

    # Initialize DataLoader
    eval_dataset = AudioDataset(eval_df, config.TARGET_SR)
    eval_loader = DataLoader(eval_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, collate_fn=collate_fn)

    # --- Model Initialization ---
    print(f"\n--- [5/5] Initializing Model... ---")
    model = WavLMClassifier(config.MODEL_ARCH_PATH).to(device)
    if torch.cuda.device_count() > 1: 
        model = nn.DataParallel(model)

    # Conditionally inject fine-tuned weights
    if config.USE_CHECKPOINT and config.BEST_MODEL_PATH.exists():
        print(f"Loading fine-tuned checkpoint from: {config.BEST_MODEL_PATH}")
        state_dict = torch.load(config.BEST_MODEL_PATH, map_location=device)
        
        model_to_load = model.module if isinstance(model, nn.DataParallel) else model
        model_to_load.load_state_dict(state_dict, strict=False)
        print("Fine-tuned weights loaded successfully.")
    else:
        print("----------------------------------------------------------------")
        print("NOTICE: Running in BASELINE MODE (No fine-tuned checkpoint loaded)")
        print("Model is using pre-trained WavLM weights and a RANDOM classifier head.")
        print("----------------------------------------------------------------")

    model.eval()

    # --- Forward Pass / Inference ---
    print("\n--- Running Inference on Eval Set ---")
    all_labels, all_scores, all_filenames, all_attacks = [], [], [], []
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            if batch is None: continue
            
            inputs, masks = batch['input_values'].to(device), batch['attention_mask'].to(device)
            # Mixed precision inference for VRAM optimization
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                logits = model(inputs, masks)
            
            all_labels.extend(batch['labels'].cpu().numpy())
            # Convert raw logits to probabilities [0, 1]
            all_scores.extend(torch.sigmoid(logits).cpu().numpy())
            all_filenames.extend(batch['filenames'])
            all_attacks.extend(batch['attack_types'])
    
    # Aggregate results into analytical dataframe
    results_df = pd.DataFrame({
        'filename': all_filenames,
        'attack_type': all_attacks,
        'label_numeric': all_labels,
        'score': all_scores
    })

    # --- Result Analytics ---
    # RESULT 1: Overall Performance Metrics
    print("\n--- [RESULT 1] Overall Performance on EVAL Set ---")
    y_true = results_df['label_numeric'].values
    y_scores = results_df['score'].values
    
    overall_eer, eer_threshold, fpr, tpr, thresholds = get_eer_stats(y_true, y_scores)
    overall_auc = roc_auc_score(y_true, y_scores)
    
    results_df['prediction_eer_thresh'] = (results_df['score'] >= eer_threshold).astype(int)
    y_pred_optimal = results_df['prediction_eer_thresh'].values

    overall_acc = accuracy_score(y_true, y_pred_optimal)
    overall_f1 = f1_score(y_true, y_pred_optimal, zero_division=0)
    
    print(f"  > Optimal EER Threshold: {eer_threshold:.4f}")
    print(f"  > Overall EER:           {overall_eer:.2f}%")
    print(f"  > Overall AUC:           {overall_auc:.4f}")
    print(f"  > Accuracy (@EER thresh):{overall_acc * 100:.2f}%")
    print(f"  > F1-Score (@EER thresh):{overall_f1:.4f}")

    # RESULT 2: Confusion Matrix Construction
    print("\n--- [RESULT 2] Confusion Matrix ---")
    cm = confusion_matrix(y_true, y_pred_optimal, labels=[0, 1])
    print(f"  True Negatives (Bonafide -> Bonafide): {cm[0][0]}")
    print(f"  False Positives (Bonafide -> Spoof):   {cm[0][1]}")
    print(f"  False Negatives (Spoof -> Bonafide):   {cm[1][0]}")
    print(f"  True Positives (Spoof -> Spoof):       {cm[1][1]}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['bonafide', 'spoof'], yticklabels=['bonafide', 'spoof'])
    plt.title(f'Confusion Matrix (@ Threshold {eer_threshold:.4f})')
    plt.xlabel('Predicted Label'); plt.ylabel('True Label')
    plt.savefig(config.OUTPUT_DIR / "final_confusion_matrix.png", dpi=300)
    plt.close()

    # RESULT 3: Per-Attack Breakdown
    print("\n--- [RESULT 3] Per-Attack EER Analysis ---")
    bonafide_df = results_df[results_df['label_numeric'] == 0]
    spoof_df = results_df[results_df['label_numeric'] == 1]
    
    attack_results = []
    # Calculate unique EER for each spoofing algorithm against all bonafide data
    for attack in spoof_df['attack_type'].unique():
        if pd.isna(attack) or attack == '-': continue
            
        specific_attack_df = spoof_df[spoof_df['attack_type'] == attack]
        combined_df = pd.concat([bonafide_df, specific_attack_df])
        
        att_y_true = combined_df['label_numeric'].values
        att_y_scores = combined_df['score'].values
        
        try:
            att_eer, _, _, _, _ = get_eer_stats(att_y_true, att_y_scores)
            attack_results.append({'Attack_Type': attack, 'EER_%': att_eer})
        except ValueError:
            pass # Skips iteration if sample distribution is invalid

    attack_metrics_df = pd.DataFrame(attack_results).sort_values(by='EER_%', ascending=False)
    print("Hardest attacks to detect (Highest EER):")
    print(attack_metrics_df.head(10).to_string(index=False))

    # RESULT 4: Data Structuring & Export
    print("\n--- [RESULT 4] Exporting Scientific Data ---")
    EXPORT_DIR = config.OUTPUT_DIR / "scientific_exports"
    EXPORT_DIR.mkdir(exist_ok=True)
    
    # ROC Curve coordinates
    roc_df = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'Threshold': thresholds})
    roc_df.to_csv(EXPORT_DIR / "roc_curve_data.csv", index=False)
    
    # Baseline statistical profile
    global_metrics_df = pd.DataFrame({
        'Metric': ['EER_%', 'EER_Threshold', 'AUC', 'Accuracy', 'F1_Score'],
        'Value': [overall_eer, eer_threshold, overall_auc, overall_acc, overall_f1]
    })
    global_metrics_df.to_csv(EXPORT_DIR / "global_metrics.csv", index=False)
    
    # Granular vulnerability map
    attack_metrics_df.to_csv(EXPORT_DIR / "per_attack_eer.csv", index=False)
    
    # Raw scoring dump for potential external validation
    results_df.to_csv(EXPORT_DIR / "full_predictions.csv", index=False)

    print(f"\n--- All scientific artifacts saved to {EXPORT_DIR} ---")