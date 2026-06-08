"""
wavlm-eval-ood.py

Evaluates the fine-tuned WavLM model in an Out-of-Domain (OOD) "in-the-wild" scenario.
Calculates Cross-Corpus EER using an external bonafide reference and measures 
Spoof Detection Rate (SDR) for exclusively synthetic datasets (e.g., ElevenLabs, OpenAI).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoFeatureExtractor, WavLMModel
import librosa
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.metrics import roc_curve
import warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
PROTOCOL_PATH = "../FakeAudio/in_the_wild_protocol.csv"
MODEL_PATH = "../models/wavlm-base-plus-antispoof.pt" 
MODEL_ARCH_PATH = "microsoft/wavlm-base-plus" 

TARGET_SR = 16000
BATCH_SIZE = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DATASET & MODEL ---
class WildDataset(Dataset):
    def __init__(self, df): self.df = df
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            waveform, _ = librosa.load(row['filepath'], sr=TARGET_SR, mono=True)
            return {"waveform": waveform, "label": row['label'], "tts_engine": row['tts_engine']}
        except: return None

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ARCH_PATH)

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None
    waveforms = [item['waveform'] for item in batch]
    inputs = feature_extractor(waveforms, sampling_rate=TARGET_SR, padding=True, return_tensors="pt")
    return {
        "input_values": inputs.input_values,
        "attention_mask": inputs.attention_mask,
        "labels": torch.tensor([item['label'] for item in batch], dtype=torch.float),
        "tts_engines": [item['tts_engine'] for item in batch]
    }

class AttentivePooling(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.attention_net = nn.Sequential(nn.Linear(in_features, in_features // 2), nn.Tanh(), nn.Linear(in_features // 2, 1))
    def forward(self, x, mask):
        attn_weights = self.attention_net(x).squeeze(-1)
        attn_weights.data.masked_fill_(~mask.bool(), -float('inf'))
        attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(1)
        return torch.bmm(attn_weights, x).squeeze(1)

class WavLMClassifier(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained(model_path)
        self.pooling = AttentivePooling(self.wavlm.config.hidden_size)
        self.classifier = nn.Linear(self.wavlm.config.hidden_size, 1)
    def forward(self, input_values, attention_mask):
        outputs = self.wavlm(input_values, attention_mask=attention_mask).last_hidden_state
        out_lens = self.wavlm._get_feat_extract_output_lengths(attention_mask.sum(-1))
        down_mask = torch.zeros(outputs.shape[:2], dtype=torch.long, device=outputs.device)
        for i, l in enumerate(out_lens): down_mask[i, :l] = 1
        return self.classifier(self.pooling(outputs, down_mask)).squeeze(-1)

def calc_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute(fnr - fpr))
    return (fpr[idx] + fnr[idx]) / 2 * 100.0

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    df_prot = pd.read_csv(PROTOCOL_PATH)
    loader = DataLoader(WildDataset(df_prot), batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=0)

    print("[INFO] Initializing OOD Evaluation Model...")
    model = WavLMClassifier(MODEL_ARCH_PATH).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
    model.eval()

    all_labels, all_scores, all_engines = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating In-The-Wild"):
            if batch is None: continue
            inputs, masks = batch['input_values'].to(device), batch['attention_mask'].to(device)
            
            logits = model(inputs, masks) if device.type == 'cpu' else torch.amp.autocast('cuda')(model)(inputs, masks)
            probs = torch.sigmoid(logits)
            
            all_labels.extend(batch['labels'].numpy())
            all_scores.extend(probs.cpu().numpy())
            all_engines.extend(batch['tts_engines'])

    df_res = pd.DataFrame({'tts_engine': all_engines, 'label': all_labels, 'score': all_scores})
    
    # Cross-Corpus Evaluation
    bonafide_scores = df_res[df_res['label'] == 0]['score'].values
    results_table = []
    
    for engine in df_res['tts_engine'].unique():
        if engine == 'real_samples': continue
            
        spoof_scores = df_res[df_res['tts_engine'] == engine]['score'].values
        combined_labels = np.concatenate([np.zeros(len(bonafide_scores)), np.ones(len(spoof_scores))])
        combined_scores = np.concatenate([bonafide_scores, spoof_scores])
        
        cross_eer = calc_eer(combined_labels, combined_scores)
        sdr = (spoof_scores >= 0.5).mean() * 100 
        
        results_table.append({"TTS_Engine": engine, "Samples": len(spoof_scores), "EER_Percentage": cross_eer, "SDR_Percentage": sdr})

    results_df = pd.DataFrame(results_table).sort_values(by="EER_Percentage", ascending=False)
    results_df.to_csv("../scientific_exports/cross_corpus_tts_results.csv", index=False)
    print("\n[SUCCESS] Cross-corpus evaluation complete. Results saved to scientific_exports.")