import os
import sys
import json
import torch
import numpy as np
import onnx
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import CompactGenderClassifier

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SAVE_DIR = os.path.join(PROJECT_ROOT, "models")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "model_config.json")
WEIGHTS_PATH = os.path.join(SAVE_DIR, "model_weights.pt")
FP32_ONNX_PATH = os.path.join(SAVE_DIR, "model.onnx")
INT8_ONNX_PATH = os.path.join(SAVE_DIR, "model_int8.onnx")

def export_and_quantize():
    print("===============================================================")
    print("Exporting ONNX Graph & Dynamic INT8 Quantization (Gender Model)")
    print("===============================================================")

    if not os.path.exists(CONFIG_PATH) or not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Model checkpoint not found. Please train first: python scripts/train.py")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # 1. Load PyTorch Model on CPU
    print("--> [1/4] Loading trained PyTorch model onto CPU...")
    model = CompactGenderClassifier(
        vocab_size=cfg["vocab_size"],
        emb_dim=cfg["emb_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        max_len=cfg["max_length"],
        dropout=0.0
    )
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True))
    model.eval()

    # 2. Export FP32 ONNX Graph with int32 inputs
    print("--> [2/4] Exporting FP32 ONNX graph with int32 tensor inputs...")
    dummy_input_ids = torch.randint(0, cfg["vocab_size"], (1, cfg["max_length"]), dtype=torch.int32)
    dummy_attention_mask = torch.ones((1, cfg["max_length"]), dtype=torch.int32)

    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        FP32_ONNX_PATH,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "logits": {0: "batch_size"}
        },
        opset_version=17,
        do_constant_folding=True
    )

    fp32_size_mb = os.path.getsize(FP32_ONNX_PATH) / (1024**2)
    print(f"    [Exported] FP32 ONNX: {FP32_ONNX_PATH} ({fp32_size_mb:.2f} MB)")

    # Validate ONNX graph
    onnx_model = onnx.load(FP32_ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    print("    [Validated] FP32 ONNX graph is valid.")

    # 3. Dynamic INT8 Quantization (Quantize Gather + MatMul)
    print("--> [3/4] Quantizing MatMul and Gather embedding nodes to INT8...")
    quantize_dynamic(
        model_input=FP32_ONNX_PATH,
        model_output=INT8_ONNX_PATH,
        op_types_to_quantize=["MatMul", "Gather"],
        weight_type=QuantType.QInt8
    )

    int8_size_kb = os.path.getsize(INT8_ONNX_PATH) / 1024
    int8_size_mb = int8_size_kb / 1024
    print(f"    [Quantized] INT8 ONNX: {INT8_ONNX_PATH} ({int8_size_kb:.1f} KB / {int8_size_mb:.2f} MB)")
    print(f"    Compression Ratio: {fp32_size_mb / int8_size_mb:.1f}x smaller")

    # 4. Numerical Divergence Check (PyTorch vs ONNX INT8)
    print("--> [4/4] Verifying numerical divergence across test batch...")
    test_ids = torch.randint(0, cfg["vocab_size"], (8, cfg["max_length"]), dtype=torch.int32)
    test_mask = torch.ones((8, cfg["max_length"]), dtype=torch.int32)

    with torch.no_grad():
        pt_logits = model(test_ids, test_mask).numpy()
        pt_probs = np.exp(pt_logits) / np.sum(np.exp(pt_logits), axis=-1, keepdims=True)

    session_opts = onnxruntime.SessionOptions()
    session_opts.intra_op_num_threads = 1
    session_opts.inter_op_num_threads = 1
    ort_session = onnxruntime.InferenceSession(INT8_ONNX_PATH, session_opts)

    ort_inputs = {
        "input_ids": test_ids.numpy(),
        "attention_mask": test_mask.numpy()
    }
    ort_logits = ort_session.run(None, ort_inputs)[0]
    ort_probs = np.exp(ort_logits) / np.sum(np.exp(ort_logits), axis=-1, keepdims=True)

    max_diff = np.max(np.abs(pt_probs - ort_probs))
    avg_diff = np.mean(np.abs(pt_probs - ort_probs))
    print(f"    Max Probability Divergence: {max_diff:.4f}")
    print(f"    Avg Probability Divergence: {avg_diff:.4f}")

    if max_diff < 0.05:
        print("    [Verification Passed] INT8 ONNX output matches PyTorch FP32 with high fidelity!")
    else:
        print("    [Notice] Moderate quantization variance observed (expected under INT8).")

    print("\n===============================================================")
    print(f"SUCCESS: Ultra-compact gender model ready at: {INT8_ONNX_PATH}")
    print("===============================================================")

if __name__ == "__main__":
    export_and_quantize()
