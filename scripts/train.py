import os
import sys
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import PreTrainedTokenizerFast, get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import CompactGenderClassifier

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
VOCAB_SIZE = 4000
EMB_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 2
NUM_HEADS = 4
MAX_LENGTH = 20
BATCH_SIZE = 512
EPOCHS = 6
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.05

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SCRATCH_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "..", ".model.scratch"))
MASTER_DATA_DIR = os.path.join(SCRATCH_DIR, "master_datasets")
TOKENIZER_DIR = os.path.join(PROJECT_ROOT, "models", "tokenizer")
SAVE_DIR = os.path.join(PROJECT_ROOT, "models")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

# Device Configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"--> Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"--> Available VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")
    print("--> Warning: CUDA not available, using CPU")

GENDER_MAP = {"MALE": 0, "FEMALE": 1}

def pre_tokenize_dataset(tokenizer, texts, max_length=MAX_LENGTH):
    """Pre-tokenize all texts in memory into contiguous int32 tensors."""
    t0 = time.time()
    print(f"--> Fast batch encoding {len(texts):,} strings into RAM...")
    enc = tokenizer(
        texts,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(torch.int32)
    attention_mask = enc["attention_mask"].to(torch.int32)
    ram_mb = (input_ids.element_size() * input_ids.nelement() + attention_mask.element_size() * attention_mask.nelement()) / (1024**2)
    print(f"    Encoding completed in {time.time() - t0:.2f}s (Tensor size in RAM: {ram_mb:.1f} MB)")
    return input_ids, attention_mask

def main():
    print("===============================================================")
    print("Training Dedicated Ultra-Compact Gender Classifier (100% Uncased)")
    print("===============================================================")
    print(f"Vocab Size:    {VOCAB_SIZE}")
    print(f"Embedding Dim: {EMB_DIM} (Factorized -> {HIDDEN_DIM})")
    print(f"Hidden Dim:    {HIDDEN_DIM} (Pre-LN Transformer, {NUM_LAYERS} layers, {NUM_HEADS} heads)")
    print(f"Dual Pooling:  MeanPool (128) + MaxPool (128) = 256 dims")
    print(f"Batch Size:    {BATCH_SIZE}")
    print(f"Epochs:        {EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print("---------------------------------------------------------------")

    # 1. Load Tokenizer
    tok_json_path = os.path.join(TOKENIZER_DIR, "tokenizer.json")
    if not os.path.exists(tok_json_path):
        raise FileNotFoundError(f"Tokenizer not found at: {tok_json_path}. Please run train_tokenizer.py first.")

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tok_json_path,
        do_lower_case=True,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]"
    )

    # 2. Load Datasets from .model.scratch (Strictly First Names with Known Gender)
    train_parquet = os.path.join(MASTER_DATA_DIR, "train.parquet")
    val_parquet = os.path.join(MASTER_DATA_DIR, "val.parquet")

    print("--> Loading training and validation parquets...")
    df_train = pd.read_parquet(train_parquet)
    df_val = pd.read_parquet(val_parquet)

    # Filter strictly for valid gender labels
    df_train_gender = df_train[df_train['gender'].isin(['MALE', 'FEMALE'])].copy()
    df_val_gender = df_val[df_val['gender'].isin(['MALE', 'FEMALE'])].copy()

    name_col_train = 'text' if 'text' in df_train_gender.columns else 'name'
    name_col_val = 'text' if 'text' in df_val_gender.columns else 'name'

    train_texts = df_train_gender[name_col_train].dropna().astype(str).str.lower().str.strip().tolist()
    train_labels = [GENDER_MAP[g] for g in df_train_gender['gender'].tolist()]

    val_texts = df_val_gender[name_col_val].dropna().astype(str).str.lower().str.strip().tolist()
    val_labels = [GENDER_MAP[g] for g in df_val_gender['gender'].tolist()]

    print(f"--> Master Labeled Train Records: {len(train_texts):,} (Male: {train_labels.count(0):,}, Female: {train_labels.count(1):,})")
    print(f"--> Master Labeled Val Records:   {len(val_texts):,} (Male: {val_labels.count(0):,}, Female: {val_labels.count(1):,})")

    # 3. Pre-Tokenize Datasets
    train_ids, train_mask = pre_tokenize_dataset(tokenizer, train_texts, MAX_LENGTH)
    val_ids, val_mask = pre_tokenize_dataset(tokenizer, val_texts, MAX_LENGTH)

    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
    val_labels_tensor = torch.tensor(val_labels, dtype=torch.long)

    train_dataset = TensorDataset(train_ids, train_mask, train_labels_tensor)
    val_dataset = TensorDataset(val_ids, val_mask, val_labels_tensor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # 4. Initialize Model
    model = CompactGenderClassifier(
        vocab_size=VOCAB_SIZE,
        emb_dim=EMB_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dim_feedforward=256,
        max_len=MAX_LENGTH,
        dropout=0.1
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"--> Model Initialized: {total_params:,} trainable parameters")

    # 5. Loss & Optimization (Class-weighted for maximum precision)
    num_m = train_labels.count(0)
    num_f = train_labels.count(1)
    total_samples = num_m + num_f
    weight_m = total_samples / (2.0 * max(1, num_m))
    weight_f = total_samples / (2.0 * max(1, num_f))
    class_weights = torch.tensor([weight_m, weight_f], dtype=torch.float, device=device)
    print(f"--> Loss Class Weights: MALE={weight_m:.3f}, FEMALE={weight_f:.3f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_f1 = 0.0
    best_val_acc = 0.0
    best_epoch = -1

    # 6. Training Loop (Mixed-Precision GPU accelerated)
    print(f"\n--> Starting GPU Training ({EPOCHS} epochs, batch_size={BATCH_SIZE}, mixed-precision FP16)...")
    for epoch in range(1, EPOCHS + 1):
        t_epoch_start = time.time()
        model.train()
        total_loss = 0.0

        for b_ids, b_mask, b_labels in train_loader:
            b_ids = b_ids.to(device, non_blocking=True)
            b_mask = b_mask.to(device, non_blocking=True)
            b_labels = b_labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(b_ids, b_mask)
                loss = criterion(logits, b_labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        val_loss = 0.0

        with torch.no_grad():
            for b_ids, b_mask, b_labels in val_loader:
                b_ids = b_ids.to(device, non_blocking=True)
                b_mask = b_mask.to(device, non_blocking=True)
                b_labels = b_labels.to(device, non_blocking=True)

                with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                    logits = model(b_ids, b_mask)
                    loss = criterion(logits, b_labels)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(b_labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        acc = accuracy_score(val_targets, val_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(val_targets, val_preds, average="weighted")
        duration = time.time() - t_epoch_start

        print(
            f"Epoch {epoch:02d}/{EPOCHS:02d} [{duration:.1f}s] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {acc*100:.2f}% | "
            f"Val F1: {f1*100:.2f}%"
        )

        if f1 > best_val_f1:
            best_val_f1 = f1
            best_val_acc = acc
            best_epoch = epoch
            weights_path = os.path.join(SAVE_DIR, "model_weights.pt")
            torch.save(model.state_dict(), weights_path)
            print(f"    --> [Checkpoint Saved] Best Model Checkpoint at: {weights_path} (F1: {best_val_f1*100:.2f}%)")

    print("\n===============================================================")
    print(f"Training Complete! Best Epoch: {best_epoch} | Acc: {best_val_acc*100:.2f}% | F1: {best_val_f1*100:.2f}%")
    print("===============================================================")

    # 7. Update Model Config
    config_path = os.path.join(CONFIG_DIR, "model_config.json")
    model_config = {
        "model_type": "CompactGenderClassifier",
        "vocab_size": VOCAB_SIZE,
        "emb_dim": EMB_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS,
        "max_length": MAX_LENGTH,
        "lowercase": True,
        "labels": {
            "0": "MALE",
            "1": "FEMALE"
        },
        "best_val_accuracy": float(best_val_acc),
        "best_val_f1": float(best_val_f1)
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)
    print(f"--> [Updated] Config at: {config_path}")

if __name__ == "__main__":
    main()
