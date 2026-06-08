"""
build-ood-protocol.py

This script generates a standardized CSV protocol for Out-of-Domain (OOD) evaluation.
It scans external datasets (e.g., Kaggle, HuggingFace, Mendeley), assigns binary labels 
(bonafide=0, spoof=1), and records the specific Text-to-Speech (TTS) engine used.
"""

import pandas as pd
from pathlib import Path

# --- CONFIGURATION ---
DATA_ROOT = Path("../FakeAudio") 
OUTPUT_CSV = "in_the_wild_protocol.csv"

def build_protocol(data_root: Path, output_file: str):
    protocol_data = []

    # 1. KAGGLE: AudioDeepfakeDetectionDataset (Contains both Bonafide and Spoof)
    kaggle_dir = data_root / "AudioDeepfakeDetectionDataset"
    if kaggle_dir.exists():
        for subfolder in kaggle_dir.iterdir():
            if not subfolder.is_dir(): 
                continue
            
            label = 0 if subfolder.name == "real_samples" else 1
            label_str = "bonafide" if label == 0 else "spoof"
                
            for f in subfolder.rglob("*.wav"):
                protocol_data.append({
                    "filepath": str(f.resolve()),
                    "label_str": label_str,
                    "label": label,
                    "tts_engine": subfolder.name,
                    "dataset_source": "Kaggle_ADD"
                })

    # 2. MENDELEY: Fake_ElevenLabs_Respeecher (Spoof only)
    mendeley_dir = data_root / "Fake_ElevenLabs_Respeecher" / "wav"
    if mendeley_dir.exists():
        for f in mendeley_dir.rglob("*.wav"):
            protocol_data.append({
                "filepath": str(f.resolve()),
                "label_str": "spoof",
                "label": 1,
                "tts_engine": "Mendeley_ElevenLabs_Respeecher",
                "dataset_source": "Mendeley"
            })

    # 3. HUGGINGFACE: skypro1111-elevenlabs_dataset (Spoof only)
    hf_dir = data_root / "skypro1111-elevenlabs_dataset" / "wavs"
    if hf_dir.exists():
        for f in hf_dir.rglob("*.wav"):
            protocol_data.append({
                "filepath": str(f.resolve()),
                "label_str": "spoof",
                "label": 1,
                "tts_engine": "HF_ElevenLabs",
                "dataset_source": "HuggingFace"
            })

    # Save to CSV
    df_protocol = pd.DataFrame(protocol_data)
    df_protocol.to_csv(output_file, index=False)

    print(f"[SUCCESS] Protocol generated with {len(df_protocol)} samples -> {output_file}")
    print("\nDataset Summary:")
    print(df_protocol.groupby(['dataset_source', 'tts_engine', 'label_str']).size().to_string())

if __name__ == "__main__":
    build_protocol(DATA_ROOT, OUTPUT_CSV)