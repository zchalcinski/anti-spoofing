# Voice Synthesis and Cloning Detection

This repository contains the source code, training pipelines, evaluation scripts, and deployment utilities for the thesis project concerning the detection of synthetically generated speech (*voice spoofing*). The research investigates the catastrophic generalization gap in self-supervised learning (SSL) models when confronted with zero-shot, in-the-wild generative architectures (e.g., LLM-based TTS and Neural Audio Codecs).

## 🗂️ Project Structure

* **`src/`** - Core Python scripts for dataset preparation, training, and cross-corpus evaluation (WavLM-based pipelines).
* **`utils/`** - Auxiliary scripts for classical machine learning baselines (Random Forest, SVM), feature extraction (openSMILE, CQCC), visualization, and model quantization.
* **`scientific_exports/`** - Automatically generated evaluation artifacts (CSV reports, Confusion Matrices, EER bar plots).
* **`mobile_app/`** - Android application source code integrating the quantized PyTorch Lite model for on-device inference.

## ⚙️ Setup & Installation

1. Clone the repository and navigate to the project root.
2. Create a virtual environment and install the required dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   ```

**Data & Checkpoint Notice:**
Due to GitHub file size limits and licensing restrictions, the raw audio datasets (ASVspoof 5, In-The-Wild corpora) and pre-trained `.pt` / `.ptl` model weights are excluded from this repository. 
* The **ASVspoof 5** dataset can be requested from the official challenge organizers.
* External "in-the-wild" datasets (e.g., ElevenLabs, OpenAI generated samples) must be placed in the `FakeAudio/` directory.

## 🚀 Execution Pipelines

### 1. Deep Learning Training (WavLM Base+)
*Designed for GPU-accelerated environments (e.g., Kaggle, AWS).* Fine-tunes the SSL backbone using gradient accumulation, mixed precision (AMP), and on-the-fly audio augmentations (RawBoost/Gaussian Noise).
```bash
python src/wavlm-train-kaggle.py
```

### 2. In-Domain Evaluation (ASVspoof 5)
*Designed for GPU-accelerated environments (e.g., Kaggle, AWS).* Calculates Equal Error Rate (EER), plots confusion matrices, and exports per-attack vulnerability metrics.
```bash
python src/wavlm-eval-kaggle.py
```

### 3. In-The-Wild Protocol Generation
*Runs locally* To evaluate models against commercial generators, first build the unified CSV protocol:
```bash
python src/build-ood-protocol.py
```

### 4. Cross-Corpus Evaluation (Generalization Test)
*Runs locally* Evaluates the fine-tuned model against the `in_the_wild_protocol.csv`. Extracts Cross-Corpus EER and Spoof Detection Rate (SDR) to demonstrate the hyperrealism effect and channel mismatch vulnerabilities.
```bash
python src/wavlm-eval-ood.py
```

## 📱 Mobile Deployment (PyTorch Lite)
*Runs locally* To convert the trained PyTorch model (`.pt`) into an optimized, quantized TorchScript format (`.ptl`) for the Android app, run:
```bash
python utils/mobile_conversion/convert_to_ptlite.py
```

## 📜 License
This project is licensed under the **MIT License** - see the `LICENSE` file for details.
```

***