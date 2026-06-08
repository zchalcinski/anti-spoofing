# ==============================================================================
# --- SECTION 1: SETUP AND DEPENDENCIES ---
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from transformers import AutoFeatureExtractor, WavLMModel
from torch.utils.mobile_optimizer import optimize_for_mobile

print("--- [1/4] Dependencies imported successfully. ---")


# ==============================================================================
# --- SECTION 2: NEURAL NETWORK ARCHITECTURE FOR JIT COMPILATION ---
# ==============================================================================
class AttentivePooling(nn.Module):
    """
    Implements attentive pooling to weigh and aggregate frame-level features.
    Designed specifically to be TorchScript-compatible.
    """
    def __init__(self, in_features):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.Tanh(),
            nn.Linear(in_features // 2, 1)
        )

    def forward(self, x, mask):
        # x: (batch, seq_len, features), mask: (batch, seq_len)
        attn_weights = self.attention_net(x).squeeze(-1)
        attn_weights.data.masked_fill_(~mask.bool(), -float('inf')) # Mask padding frames
        attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(1)
        return torch.bmm(attn_weights, x).squeeze(1)

class WavLMClassifier(nn.Module):
    """
    The main classification model pairing WavLM with Attentive Pooling.
    Refactored to eliminate dynamic Python control flows (e.g., loops) 
    that break JIT tracing mechanisms.
    """
    def __init__(self, model_path, freeze_encoder=False):
        super().__init__()
        print("  > Initializing WavLMClassifier backbone...")
        self.wavlm = WavLMModel.from_pretrained(model_path)
        
        if freeze_encoder:
            print("  > Freezing WavLM encoder weights.")
            for param in self.wavlm.parameters():
                param.requires_grad = False
        
        hidden_size = self.wavlm.config.hidden_size
        self.pooling = AttentivePooling(hidden_size)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_values, attention_mask):
        outputs = self.wavlm(input_values, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        
        # --- VECTORIZED MASK CREATION FOR JIT TRACING COMPATIBILITY ---
        # Replaces Python 'for' loops with vectorized tensor operations.
        # This is strictly required because torch.jit.trace unrolls loops based on 
        # the specific sample input, which breaks dynamic sequence length handling.
        max_len = last_hidden_state.shape[1]
        output_lengths = self.wavlm._get_feat_extract_output_lengths(attention_mask.sum(-1))
        
        arange_tensor = torch.arange(max_len, device=last_hidden_state.device)
        # Broadcasting automatically generates the correct 2D boolean mask
        downsampled_attention_mask = arange_tensor < output_lengths.unsqueeze(-1)
        # --------------------------------------------------------------
        
        pooled_output = self.pooling(last_hidden_state, downsampled_attention_mask)
        return self.classifier(pooled_output).squeeze(-1)


# ==============================================================================
# --- SECTION 3: LOCAL CONFIGURATION AND PATHS ---
# ==============================================================================
MODEL_ARCH_PATH = "./machine_learning/models/wavlm-base-plus" 
TRAINED_WEIGHTS_PATH = "./machine_learning/models/wavlm-base-plus-antispoof.pt"

# Defined outputs for PyTorch Lite (.ptl) mobile binaries
MOBILE_OUTPUT_PATH = "./machine_learning/models/wavlm-base-plus-antispoof.ptl"
MOBILE_QUANTIZED_OUTPUT_PATH = "./machine_learning/models/wavlm-base-plus-antispoof-quantized.ptl"


# ==============================================================================
# --- SECTION 4: EXPORT AND OPTIMIZATION PIPELINE ---
# ==============================================================================
if __name__ == '__main__':
    print("\n--- [2/4] Loading Trained PyTorch Model ---")
    model = WavLMClassifier(MODEL_ARCH_PATH)
    
    # Force loading onto CPU to ensure cross-platform compatibility for mobile edge devices
    state_dict = torch.load(TRAINED_WEIGHTS_PATH, map_location=torch.device('cpu'))
    
    # Strip 'module.' prefix if the model was serialized using DataParallel
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
    model.load_state_dict(state_dict)
    
    # Critical step: Set model to evaluation mode to disable dropout and lock batch norm layers
    model.eval()
    print("  > PyTorch model weights loaded and set to Eval mode.")

    # Create dummy tensors representative of the expected input format (1 sample, 4 seconds @ 16kHz)
    sample_input = (
        torch.randn(1, 16000 * 4),
        torch.ones(1, 16000 * 4)
    )

    # --- Standard (Float32) PyTorch Mobile Compilation ---
    print(f"\n--- [3/4] Tracing and Optimizing Float32 model for Mobile AI... ---")
    
    # Step A: Trace execution to freeze the computation graph
    print("  > Tracing model execution graph...")
    traced_model = torch.jit.trace(model, sample_input)
    
    # Step B: Apply operator fusion and mobile-specific bytecode optimizations
    print("  > Applying mobile operator optimizations...")
    optimized_model = optimize_for_mobile(traced_model)
    
    # Step C: Serialize to lightweight format
    print("  > Saving PyTorch Lite binary...")
    optimized_model._save_for_lite_interpreter(MOBILE_OUTPUT_PATH)
    
    size_float32 = Path(MOBILE_OUTPUT_PATH).stat().st_size / 1024**2
    print(f"  > Standard PyTorch Mobile model saved to: {MOBILE_OUTPUT_PATH} (Size: {size_float32:.2f} MB)")

    # --- Quantized (Int8) PyTorch Mobile Compilation ---
    print(f"\n--- [4/4] Quantizing, Tracing, and Optimizing Int8 model... ---")
    
    # Step A: Apply dynamic post-training quantization to reduce memory footprint
    # Targets nn.Linear layers which dominate the parameter count in Transformers
    print("  > Applying dynamic Int8 quantization...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    # Step B: Trace the new quantized execution graph
    print("  > Tracing quantized execution graph...")
    traced_quantized_model = torch.jit.trace(quantized_model, sample_input)
    
    # Step C: Optimize the traced quantized model for ARM/Edge hardware
    print("  > Optimizing quantized model for mobile execution...")
    optimized_quantized_model = optimize_for_mobile(traced_quantized_model)
    
    # Step D: Serialize the highly compressed format
    print("  > Saving quantized PyTorch Lite binary...")
    optimized_quantized_model._save_for_lite_interpreter(MOBILE_QUANTIZED_OUTPUT_PATH)
    
    size_quantized = Path(MOBILE_QUANTIZED_OUTPUT_PATH).stat().st_size / 1024**2
    print(f"  > Quantized PyTorch Mobile model saved to: {MOBILE_QUANTIZED_OUTPUT_PATH} (Size: {size_quantized:.2f} MB)")

    print("\n--- Pipeline Execution Complete: Edge AI binaries successfully generated! ---")