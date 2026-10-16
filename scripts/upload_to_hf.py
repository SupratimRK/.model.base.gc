import os
import sys
import json
import shutil
import tempfile
from huggingface_hub import HfApi

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_ID = os.environ.get("HF_REPO_ID", "SupratimRK/compact-gender-classifier")
HF_TOKEN = os.environ.get("HF_TOKEN")

def build_hf_bundle(stage_dir):
    """Assembles all files required for the Hugging Face Model Hub repository."""
    print(f"--> Staging distribution bundle in: {stage_dir}")

    # 1. Model weights
    shutil.copy2(os.path.join(PROJECT_ROOT, "models", "model.onnx"), stage_dir)
    shutil.copy2(os.path.join(PROJECT_ROOT, "models", "model_int8.onnx"), stage_dir)
    shutil.copy2(os.path.join(PROJECT_ROOT, "models", "model_weights.pt"), stage_dir)

    # 2. Tokenizer files
    shutil.copy2(os.path.join(PROJECT_ROOT, "models", "tokenizer", "tokenizer.json"), stage_dir)
    shutil.copy2(os.path.join(PROJECT_ROOT, "models", "tokenizer", "tokenizer_config.json"), stage_dir)
    shutil.copy2(os.path.join(PROJECT_ROOT, "models", "tokenizer", "vocab.txt"), stage_dir)

    # 3. Model & predictor scripts
    shutil.copy2(os.path.join(PROJECT_ROOT, "scripts", "model.py"), stage_dir)
    shutil.copy2(os.path.join(PROJECT_ROOT, "scripts", "predict.py"), stage_dir)

    # 4. Generate Hugging Face standard config.json
    config_path = os.path.join(PROJECT_ROOT, "config", "model_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    hf_config = {
        "model_type": "CompactGenderClassifier",
        "architectures": ["CompactGenderClassifier"],
        "vocab_size": cfg.get("vocab_size", 4000),
        "emb_dim": cfg.get("emb_dim", 64),
        "hidden_dim": cfg.get("hidden_dim", 128),
        "num_layers": cfg.get("num_layers", 2),
        "num_heads": cfg.get("num_heads", 4),
        "dim_feedforward": 256,
        "max_length": cfg.get("max_length", 20),
        "dropout": 0.1,
        "id2label": {
            "0": "MALE",
            "1": "FEMALE"
        },
        "label2id": {
            "MALE": 0,
            "FEMALE": 1
        },
        "metrics": {
            "best_val_accuracy": cfg.get("best_val_accuracy", 0.8538),
            "best_val_f1": cfg.get("best_val_f1", 0.8538)
        }
    }
    with open(os.path.join(stage_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(hf_config, f, indent=2)

    # 5. .gitattributes (LFS tracking)
    gitattributes_content = """*.7z filter=lfs diff=lfs merge=lfs -text
*.arrow filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.bz2 filter=lfs diff=lfs merge=lfs -text
*.ckpt filter=lfs diff=lfs merge=lfs -text
*.ftz filter=lfs diff=lfs merge=lfs -text
*.gz filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.joblib filter=lfs diff=lfs merge=lfs -text
*.lfs.* filter=lfs diff=lfs merge=lfs -text
*.mlmodel filter=lfs diff=lfs merge=lfs -text
*.model filter=lfs diff=lfs merge=lfs -text
*.msgpack filter=lfs diff=lfs merge=lfs -text
*.npy filter=lfs diff=lfs merge=lfs -text
*.npz filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
*.ot filter=lfs diff=lfs merge=lfs -text
*.parquet filter=lfs diff=lfs merge=lfs -text
*.pb filter=lfs diff=lfs merge=lfs -text
*.pickle filter=lfs diff=lfs merge=lfs -text
*.pkl filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.rar filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text
saved_model/**/* filter=lfs diff=lfs merge=lfs -text
*.tar.* filter=lfs diff=lfs merge=lfs -text
*.tar filter=lfs diff=lfs merge=lfs -text
*.tflite filter=lfs diff=lfs merge=lfs -text
*.tgz filter=lfs diff=lfs merge=lfs -text
*.wasm filter=lfs diff=lfs merge=lfs -text
*.xz filter=lfs diff=lfs merge=lfs -text
*.zip filter=lfs diff=lfs merge=lfs -text
*.zst filter=lfs diff=lfs merge=lfs -text
*tfevents* filter=lfs diff=lfs merge=lfs -text
"""
    with open(os.path.join(stage_dir, ".gitattributes"), "w", encoding="utf-8") as f:
        f.write(gitattributes_content)

    # 6. README.md
    shutil.copy2(os.path.join(PROJECT_ROOT, "README.md"), stage_dir)

def upload_to_hub():
    print("===============================================================")
    print(f"Uploading to Hugging Face Hub: {REPO_ID}")
    print("===============================================================")

    api = HfApi(token=HF_TOKEN)

    # Check authentication
    try:
        user_info = api.whoami()
        print(f"--> Authenticated as: {user_info.get('name')}")
    except Exception as e:
        print(f"--> Error checking authentication: {e}")
        if not HF_TOKEN:
            print("    Please set HF_TOKEN environment variable or run 'hf auth login'.")
        sys.exit(1)

    # Ensure repository exists
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True, private=False)

    with tempfile.TemporaryDirectory() as tmp_dir:
        build_hf_bundle(tmp_dir)

        print(f"--> Uploading files to {REPO_ID}...")
        commit_info = api.upload_folder(
            folder_path=tmp_dir,
            repo_id=REPO_ID,
            repo_type="model",
            commit_message="Sync model from GitHub repository",
            commit_description="Automated sync via GitHub Action / script"
        )
        print(f"--> Commit URL: {commit_info}")

    print("\n===============================================================")
    print(f"SUCCESS: Synced to https://huggingface.co/{REPO_ID}")
    print("===============================================================")

if __name__ == "__main__":
    upload_to_hub()
