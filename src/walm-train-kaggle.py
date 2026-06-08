"""
wavlm-train-kaggle.py

Fine-tunes the WavLM Base+ model for voice spoofing detection using ASVspoof 5 dataset.
Utilizes Gradient Accumulation, Mixed Precision (AMP), and on-the-fly audio augmentations.
Designed to be executed in resource-constrained cloud environments (e.g., Kaggle).
"""

# ==============================================================================
# SECTION 1: SETUP, DEPENDENCIES AND ENVIRONMENT INITIALIZATION
# ==============================================================================

import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoFeatureExtractor, WavLMModel, get_linear_schedule_with_warmup
import torch.nn as nn
import torch.optim as optim
import librosa
import pandas as pd
from pathlib import Path
from tqdm.notebook import tqdm
import numpy as np
import gc
import os
from sklearn.metrics import roc_auc_score, roc_curve
from glob import glob
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift
import warnings
import soundfile as sf
import psutil

# Global runtime environment adjustments
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# SECTION 2: GLOBAL HYPERPARAMETERS AND EXPERIMENT CONFIGURATION
# ==============================================================================
class Config:
    """
    Encapsulates all experiment hyperparameters, directory paths, and training control states 
    to ensure reproducibility across different training sessions and hardware architectures.
    """
    # Execution mode flag
    IS_DEBUG_RUN = False            # True triggers a fast sanity-check execution on a minimal subset

    # Directory and file mapping paths
    MODEL_PATH = "/kaggle/input/wavlm-base-plus/transformers/default/1"
    FILE_MAP_INPUT_PATH = "/kaggle/input/asvspoof-filemap/audio_filepath_map.feather"
    AUDIO_BASE_PATH = Path("/kaggle/input/asvspoof5/")
    
    # Checkpoint configuration for training resilience and session continuation
    INPUT_CHECKPOINT_PATH = "/kaggle/input/wavlm-train/training_checkpoint.pt"
    OUTPUT_DIR = Path("/kaggle/working/")
    RESUMABLE_CHECKPOINT_PATH = OUTPUT_DIR / "training_checkpoint.pt"
    BEST_MODEL_PATH = OUTPUT_DIR / "best_model.pt"
    
    # Audio signal processing parameters
    TARGET_SR = 16000               # Standardized sampling rate for pre-trained WavLM architectures
    FREEZE_ENCODER = False          # False enables full fine-tuning of the SSL transformer backbone
    
    # Step-driven optimization and evaluation boundaries
    TARGET_TOTAL_STEPS = 15000      # Total expected lifespan for the learning rate scheduler
    TOTAL_OPTIMIZER_STEPS_PER_SESSION = 3600  # Execution constraint for the current runtime loop
    VALIDATION_INTERVAL_STEPS = 900 # Frequency of model evaluation against validation subset
    TRAINING_DATA_FRACTION = 0.4    # Fraction of the unified train-dev partition allocated for optimization
    VALIDATION_DATA_FRACTION = 0.4  # Fraction of the dev partition allocated for unbiased validation
    
    # Optimization and regularization hyperparameters
    BATCH_SIZE = 24
    ACCUMULATION_STEPS = 4          # Combined effective batch size = 24 * 4 = 96 samples per step
    MAX_AUDIO_SECONDS = 10          # Hard temporal ceiling for processing audio clips
    NUM_WORKERS = 0                 # Suppresses multi-processing overhead inside specific notebook kernels
    LEARNING_RATE = 1e-6 if not FREEZE_ENCODER else 1e-4
    WARMUP_RATIO = 0.1              # Initial proportion of target steps allocated to linear learning rate warmup
    GRADIENT_CLIP_VAL = 1.0         # Threshold for clipping gradient norms to preserve numerical stability

    # Evaluation controls for isolated debugging
    AUGMENTATION_PROB = 0.5
    DEBUG_TRAIN_SAMPLES = 100
    DEBUG_DEV_SAMPLES = 50

# Instantiate experiment configuration state
config = Config()
config.MAX_AUDIO_LENGTH = config.MAX_AUDIO_SECONDS * config.TARGET_SR
config.TRAIN_PROTOCOL_PATH = config.AUDIO_BASE_PATH / 'ASVspoof5_protocols/ASVspoof5.train.tsv'
config.DEV_PROTOCOL_PATH = config.AUDIO_BASE_PATH / 'ASVspoof5_protocols/ASVspoof5.dev.track_1.tsv'
config.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Select target acceleration device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\n--- Configuration Initialized ---")
print(f"--- Running on device: {device} | Debug Mode: {config.IS_DEBUG_RUN} ---")
print(f"--- Effective Batch Size: {config.BATCH_SIZE * config.ACCUMULATION_STEPS} (Batch: {config.BATCH_SIZE} * Accum: {config.ACCUMULATION_STEPS}) ---")


# ==============================================================================
# SECTION 3: DATASETS, DATA PREPARATION, AND AUGMENTATION PIPELINES
# ==============================================================================
def get_filepaths_and_labels(protocol_path, file_map_df):
    """
    Parses metadata protocols from the ASVspoof5 database structure and joins them
    with an absolute system file map to reconstruct full disk locations.
    """
    print(f"  > Loading protocol from: {protocol_path}")
    protocol_df = pd.read_csv(protocol_path, sep='\s+', header=None,
                              names=['speaker_id', 'filename', 'gender', 'sys_id_1', 'sys_id_2', 
                                     'sys_id_3', 'codec', 'attack_type', 'label', 'blank'])
    
    # Filter out ambiguous data rows and maintain binary distribution
    protocol_df = protocol_df[protocol_df['label'].isin(['bonafide', 'spoof'])]
    merged_df = pd.merge(protocol_df, file_map_df, left_on='filename', right_on='filename_stem', how='inner')
    merged_df.rename(columns={'full_path': 'filepath'}, inplace=True)
    
    # Eliminate rows pointing to non-existent or corrupted index lookups
    initial_count = len(merged_df)
    merged_df.dropna(subset=['filepath'], inplace=True)
    if len(merged_df) < initial_count:
        print(f"  > WARNING: Dropped {initial_count - len(merged_df)} entries with no matching file path.")
        
    # Map text classes into discrete numeric labels for binary cross entropy
    merged_df['label_numeric'] = (merged_df['label'] == 'spoof').astype(int)
    print(f"  > Loaded {len(merged_df)} samples.")
    return merged_df[['filepath', 'label_numeric']]

# Digital signal processing augmentations to counter acoustic overfitting
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.2),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.2),
])

class AudioDataset(Dataset):
    """
    Custom Map-style Dataset wrapping file IO operations and stochastic runtime augmentation.
    """
    def __init__(self, df, target_sr, max_length, augmentations=None, aug_prob=0.5):
        self.df = df
        self.target_sr = target_sr
        self.max_length = max_length
        self.augmentations = augmentations
        self.aug_prob = aug_prob

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            # Low-overhead file-system I/O reading raw array buffers directly
            waveform, original_sr = sf.read(row['filepath'], dtype='float32')
            
            # Match standard frequency if source sample mismatch occurs
            if original_sr != self.target_sr:
                waveform = librosa.resample(waveform, orig_sr=original_sr, target_sr=self.target_sr)

            # Consolidate multi-channel feeds to downmixed mono format
            if waveform.ndim > 1:
                waveform = np.mean(waveform, axis=1)

            # Enforce deterministic sequence ceiling limits
            if len(waveform) > self.max_length:
                waveform = waveform[:self.max_length]

            # Inject generalized transformations based on configured probability
            if self.augmentations and np.random.rand() < self.aug_prob:
                waveform = self.augmentations(samples=waveform, sample_rate=self.target_sr)
                
            return {"waveform": waveform, "label": torch.tensor(row['label_numeric'], dtype=torch.float)}
        
        except Exception as e:
            print(f"WARNING: Skipping corrupt file at index {idx}, path: {row['filepath']}. Error: {e}")
            return None


# ==============================================================================
# SECTION 4: ARCHITECTURE HEADS AND NEURAL NETWORK DEFINITIONS
# ==============================================================================
class AttentivePooling(nn.Module):
    """
    Aggregates temporal frame-level sequences into static context vectors
    by applying a self-attention scoring metric over non-padded temporal tokens.
    """
    def __init__(self, in_features):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.Tanh(),
            nn.Linear(in_features // 2, 1)
        )

    def forward(self, x, mask):
        # x shape: (batch, seq_len, features)
        # mask shape: (batch, seq_len)
        attn_weights = self.attention_net(x).squeeze(-1)
        
        # Enforce zero-probability contribution for structural padding tokens
        attn_weights.data.masked_fill_(~mask.bool(), -float('inf'))
        attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(1)
        
        # Batch matrix multiplication to collapse temporal dimension
        return torch.bmm(attn_weights, x).squeeze(1)

class WavLMClassifier(nn.Module):
    """
    End-to-end classification system pairing a pre-trained SSL transformer core 
    with a temporal pooling network and a logit output projection layer.
    """
    def __init__(self, model_path, freeze_encoder=False):
        super().__init__()
        print("--- Initializing model ---")
        self.wavlm = WavLMModel.from_pretrained(model_path)
        
        if freeze_encoder:
            print("  > Freezing WavLM encoder weights.")
            for param in self.wavlm.parameters():
                param.requires_grad = False
        
        hidden_size = self.wavlm.config.hidden_size
        self.pooling = AttentivePooling(hidden_size)
        self.classifier = nn.Linear(hidden_size, 1)
        print("--- Model initialized successfully ---")

    def forward(self, input_values, attention_mask):
        outputs = self.wavlm(input_values, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        
        # Compute exact representation changes downsampled by convolutional feature extraction
        output_lengths = self.wavlm._get_feat_extract_output_lengths(attention_mask.sum(-1))
        downsampled_attention_mask = torch.zeros(last_hidden_state.shape[:2], dtype=torch.long, device=last_hidden_state.device)
        for i, length in enumerate(output_lengths):
            downsampled_attention_mask[i, :length] = 1
            
        pooled_output = self.pooling(last_hidden_state, downsampled_attention_mask)
        return self.classifier(pooled_output).squeeze(-1)


# ==============================================================================
# SECTION 5: PIPELINE VALIDATION AND AUXILIARY ENGINEERING FUNCTIONS
# ==============================================================================
print("--- Loading feature extractor ---")
feature_extractor = AutoFeatureExtractor.from_pretrained(config.MODEL_PATH)

def collate_fn(batch):
    """
    Filters runtime loading anomalies and converts variable-length lists 
    into standard padded batch tensors via the feature extractor wrapper.
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    waveforms = [item['waveform'] for item in batch]
    labels = [item['label'] for item in batch]
    inputs = feature_extractor(waveforms, sampling_rate=config.TARGET_SR, padding=True, return_tensors="pt")
    return {"input_values": inputs.input_values, "attention_mask": inputs.attention_mask, "labels": torch.stack(labels)}

def calculate_eer(y_true, y_scores):
    """
    Calculates Equal Error Rate (EER) by extracting the point minimizing the distance
    between false negative rates and false positive rates along the ROC curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.absolute(fnr - fpr))
    return fpr[eer_index] * 100.0

def validate(model, loader, criterion, device):
    """
    Runs model evaluation across the specified validation partition to calculate metrics
    without tracking operational backward graphs.
    """
    print("\n--- Starting Validation ---")
    model.eval()
    total_loss = 0.0
    all_labels, all_preds = [], []
    
    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Validating", leave=False)
        for batch in progress_bar:
            if batch is None:
                continue
                
            inputs = batch['input_values'].to(device)
            masks = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                logits = model(inputs, masks)
                loss = criterion(logits, labels)
            
            total_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(torch.sigmoid(logits).cpu().numpy())
            
    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
    
    if len(np.unique(all_labels)) < 2:
        print("--- Validation Warning: Only one class present in validation set. Cannot compute EER/AUC. ---")
        return avg_loss, float('nan'), float('nan')
        
    eer = calculate_eer(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    print("--- Validation Finished ---")
    return avg_loss, eer, auc

def get_system_stats():
    """
    Fetches process monitor statistics to maintain a clear trace of RAM, CPU, 
    and multi-GPU memory behavior across training steps.
    """
    stats_list = []
    ram_used_gb = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
    
    try:
        with open("/sys/fs/cgroup/memory/memory.limit_in_bytes") as f:
            ram_limit_gb = int(f.read()) / (1024**3)
        ram_percent = (ram_used_gb / ram_limit_gb) * 100
        stats_list.append(f"RAM: {ram_used_gb:.1f}/{ram_limit_gb:.1f} GB ({ram_percent:.1f}%)")
    except FileNotFoundError:
        ram = psutil.virtual_memory()
        ram_total_gb = ram.total / (1024**3)
        stats_list.append(f"RAM: {ram_used_gb:.1f}/{ram_total_gb:.1f} GB")
    
    stats_list.append(f"CPU: {psutil.cpu_percent()}%")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)
            mem_allocated_gb = torch.cuda.memory_allocated(i) / (1024**3)
            mem_reserved_gb = torch.cuda.memory_reserved(i) / (1024**3)
            stats_list.append(f"GPU_{i}: {mem_allocated_gb:.1f}/{mem_reserved_gb:.1f} GB")
            
    return " | ".join(stats_list)

def infinite_dataloader(loader):
    """
    Memory-isolated generator that yields training items continuously, preventing 
    iterator destruction and state leaking inherent to simple nested collections.
    """
    while True:
        for batch in loader:
            yield batch
        print("\n--- DataLoader exhausted. Restarting iteration. ---\n")


# ==============================================================================
# SECTION 6: HIGH-LEVEL PIPELINE EXECUTION AND TRAINING LOOP
# ==============================================================================
if __name__ == '__main__':
    # --- Structural File Mapping ---
    print("\n--- Loading File Map ---")
    FILE_MAP_INPUT_PATH = Path(config.FILE_MAP_INPUT_PATH)
    if FILE_MAP_INPUT_PATH.exists():
        filepath_map_df = pd.read_feather(FILE_MAP_INPUT_PATH)
        print(f"  > File map loaded successfully from {FILE_MAP_INPUT_PATH}")
    else:
        print(f"FATAL: File map not found at {FILE_MAP_INPUT_PATH}. Execution terminated.")
        exit()

    # --- Loading Protocol Targets ---
    print("\n--- Loading Protocol Files ---")
    train_df = get_filepaths_and_labels(config.TRAIN_PROTOCOL_PATH, filepath_map_df)
    dev_df = get_filepaths_and_labels(config.DEV_PROTOCOL_PATH, filepath_map_df)
    
    # --- Strategic Dataset Splitting ---
    if config.IS_DEBUG_RUN:
        print(f"\n--- RUNNING IN DEBUG MODE ---")
        train_data_to_use = train_df.sample(n=min(len(train_df), config.DEBUG_TRAIN_SAMPLES), random_state=42)
        validation_data_to_use = dev_df.sample(n=min(len(dev_df), config.DEBUG_DEV_SAMPLES), random_state=42)
    else:
        print(f"\n--- RUNNING IN PRODUCTION MODE ---")
        full_train_df = pd.concat([train_df, dev_df], ignore_index=True)
        
        # Partition data under specified fractions to adjust memory footprints
        print(f"  > Original training data size: {len(full_train_df):,}")
        train_data_to_use = full_train_df.sample(frac=config.TRAINING_DATA_FRACTION, random_state=42)
        print(f"  > Using {config.TRAINING_DATA_FRACTION*100}% of data: {len(train_data_to_use):,} samples.")

        print(f"  > Original validation data size: {len(dev_df):,}")
        validation_data_to_use = dev_df.sample(frac=config.VALIDATION_DATA_FRACTION, random_state=42)
        print(f"  > Using {config.VALIDATION_DATA_FRACTION*100}% of data for validation: {len(validation_data_to_use):,} samples.")
        
    print(f"  > Training with {len(train_data_to_use):,} samples.")
    print(f"  > Validating with {len(validation_data_to_use):,} samples.")

    # --- Dataloader Instantiations ---
    print("\n--- Creating Datasets and DataLoaders ---")
    train_dataset = AudioDataset(train_data_to_use, config.TARGET_SR, config.MAX_AUDIO_LENGTH, augmentations=augment, aug_prob=config.AUGMENTATION_PROB)
    dev_dataset = AudioDataset(validation_data_to_use, config.TARGET_SR, config.MAX_AUDIO_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config.BATCH_SIZE * 2, shuffle=False, num_workers=config.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
    print("--- DataLoaders created ---")

    # --- Network Compilation ---
    model = WavLMClassifier(config.MODEL_PATH, freeze_encoder=config.FREEZE_ENCODER).to(device)

    # Memory optimization to discard activations and recompute during the backward path
    model_to_configure = model.module if isinstance(model, nn.DataParallel) else model
    model_to_configure.wavlm.config.gradient_checkpointing = True
    print("--- Gradient Checkpointing Enabled ---")
    
    if torch.cuda.device_count() > 1:
        print(f"--- Using {torch.cuda.device_count()} GPUs with DataParallel ---")
        model = nn.DataParallel(model)
        
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    # Scheduler Configuration
    if config.IS_DEBUG_RUN:
        total_scheduler_steps = len(train_loader) // config.ACCUMULATION_STEPS
    else:
        total_scheduler_steps = config.TARGET_TOTAL_STEPS
        warmup_steps = int(total_scheduler_steps * config.WARMUP_RATIO)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_scheduler_steps)
        print(f"--- Optimizer and Scheduler created. Scheduler Lifespan: {total_scheduler_steps} steps, Warmup: {warmup_steps} steps ---")
    
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    # --- Checkpoint Restorations ---
    start_step = 0
    best_eer = float('inf')
    input_checkpoint = Path(config.INPUT_CHECKPOINT_PATH) if config.INPUT_CHECKPOINT_PATH else None
    
    print("\n--- Checking for Resumable Checkpoint ---")
    if input_checkpoint and input_checkpoint.exists() and not config.IS_DEBUG_RUN:
        print(f"  > Found checkpoint at {input_checkpoint}. Loading...")
        checkpoint = torch.load(input_checkpoint, map_location=device)
        model_to_load = model.module if isinstance(model, nn.DataParallel) else model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_step = checkpoint.get('global_step', 0)
        best_eer = checkpoint.get('best_eer', float('inf'))
        print(f"  > Resuming training from step {start_step + 1}. Best EER so far: {best_eer:.2f}%")
    else:
        print("  > No checkpoint found or in debug mode. Starting training from scratch.")
    
    # --- Step-Based Active Loop Execution ---
    train_iterator = infinite_dataloader(train_loader)
    global_step = start_step
    max_steps = start_step + config.TOTAL_OPTIMIZER_STEPS_PER_SESSION if not config.IS_DEBUG_RUN else len(train_loader) // config.ACCUMULATION_STEPS
    
    print(f"\n--- Starting Training Session for {max_steps - start_step} Steps ---")
    progress_bar = tqdm(range(start_step, max_steps), desc="Session Progress", initial=start_step, total=max_steps)
    
    session_start_time = time.time()
    log_interval = 100 # Log every 100 steps for active monitoring

    for step in progress_bar:
        model.train()
        accumulated_loss = 0.0
        
        # Sequenced Gradient Accumulation Step
        for _ in range(config.ACCUMULATION_STEPS):
            batch = next(train_iterator)
            if batch is None:
                print("Warning: Skipping a None batch from dataloader.")
                continue
            
            inputs, masks, labels = batch['input_values'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                logits = model(inputs, masks)
                loss = criterion(logits, labels)
                loss = loss / config.ACCUMULATION_STEPS
                
            scaler.scale(loss).backward()
            accumulated_loss += loss.item()
        
        # Apply standard weights updates across clipped parameter fields
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_VAL)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        global_step += 1

        # --- Active Metrics Reporting ---
        if global_step % log_interval == 0 or global_step == max_steps:
            time_since_start = time.time() - session_start_time
            steps_done = global_step - start_step
            avg_step_time = time_since_start / steps_done if steps_done > 0 else 0
            steps_remaining = max_steps - global_step
            eta_seconds = steps_remaining * avg_step_time
            eta_mins, eta_secs = divmod(eta_seconds, 60)
            
            current_lr = scheduler.get_last_lr()[0]
            avg_log_loss = (accumulated_loss / log_interval) * config.ACCUMULATION_STEPS
            system_stats_str = get_system_stats()
            
            print(f"Step {global_step}/{max_steps} | Loss: {avg_log_loss:.4f} | LR: {current_lr:.2e} | ETA: {int(eta_mins)}m {int(eta_secs)}s | {system_stats_str}")
            accumulated_loss = 0.0
        
        # --- Periodical Evaluation and Serialization ---
        if (global_step % config.VALIDATION_INTERVAL_STEPS == 0 or global_step == max_steps) and not config.IS_DEBUG_RUN:
            val_loss, val_eer, val_auc = validate(model, dev_loader, criterion, device)
            print(f"Validation @ Step {global_step} | Val Loss: {val_loss:.4f} | Val EER: {val_eer:.2f}% | Val AUC: {val_auc:.4f}")
            
            if val_eer < best_eer:
                best_eer = val_eer
                print(f"  -> New best EER! Saving best model to {config.BEST_MODEL_PATH}")
                torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), config.BEST_MODEL_PATH)

            print(f"  -> Saving resumable checkpoint at step {global_step}...")
            checkpoint = {
                'global_step': global_step,
                'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_eer': best_eer
            }
            torch.save(checkpoint, config.RESUMABLE_CHECKPOINT_PATH)
            print(f"  -> Checkpoint saved to {config.RESUMABLE_CHECKPOINT_PATH}")

            # Explicit cache flushes to maintain system health
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # --- Final isolated debug reporting ---
    if config.IS_DEBUG_RUN:
        print("\n--- Debug run finished. Performing final validation. ---")
        val_loss, val_eer, val_auc = validate(model, dev_loader, criterion, device)
        print(f"\n--- DEBUG RUN COMPLETE ---")
        print(f"Final Validation Loss: {val_loss:.4f}")
        print(f"Final Validation EER:  {val_eer:.2f}%")
        print(f"Final Validation AUC:  {val_auc:.4f}")
        
    print(f"\n--- Training Session Finished ---")