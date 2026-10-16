import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import CompactGenderClassifier

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

def find_first_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]

CONFIG_PATH = find_first_existing(
    os.path.join(SCRIPT_DIR, "config.json"),
    os.path.join(SCRIPT_DIR, "config", "model_config.json"),
    os.path.join(PROJECT_ROOT, "config", "model_config.json"),
    os.path.join(PROJECT_ROOT, "config.json")
)

TOKENIZER_DIR = find_first_existing(
    os.path.join(SCRIPT_DIR, "tokenizer"),
    os.path.join(PROJECT_ROOT, "models", "tokenizer"),
    SCRIPT_DIR
)

ONNX_PATH = find_first_existing(
    os.path.join(SCRIPT_DIR, "model_int8.onnx"),
    os.path.join(PROJECT_ROOT, "models", "model_int8.onnx"),
    os.path.join(SCRIPT_DIR, "models", "model_int8.onnx")
)

WEIGHTS_PATH = find_first_existing(
    os.path.join(SCRIPT_DIR, "model_weights.pt"),
    os.path.join(PROJECT_ROOT, "models", "model_weights.pt"),
    os.path.join(SCRIPT_DIR, "models", "model_weights.pt")
)

TITLE_PREFIXES = {
    'mr', 'mrs', 'ms', 'miss', 'dr', 'prof', 'sri', 'shri', 'smt', 'master',
    'md', 'mohd', 'sk', 'sekh', 'sheikh', 'syed', 'kazi', 'adv', 'er', 'ca'
}

class GenderClassifier:
    """
    Dedicated binary Gender predictor (MALE vs FEMALE).
    Supports ultra-fast ONNX INT8 CPU inference or PyTorch.
    """
    def __init__(self, use_onnx=True):
        if not os.path.exists(CONFIG_PATH):
            raise FileNotFoundError(f"Configuration not found: {CONFIG_PATH}. Please train the model first.")

        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        self.max_length = self.cfg.get("max_length", 20)
        self.labels = {int(k): v for k, v in self.cfg.get("labels", {"0": "MALE", "1": "FEMALE"}).items()}

        # Load Tokenizer
        tok_json = os.path.join(TOKENIZER_DIR, "tokenizer.json")
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tok_json,
            pad_token="[PAD]",
            unk_token="[UNK]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]"
        )

        self.use_onnx = use_onnx and os.path.exists(ONNX_PATH)
        if self.use_onnx:
            import onnxruntime
            session_opts = onnxruntime.SessionOptions()
            session_opts.intra_op_num_threads = 1
            session_opts.inter_op_num_threads = 1
            session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = onnxruntime.InferenceSession(ONNX_PATH, session_opts)
            self.backend_name = "ONNX Runtime INT8 (Single-Thread)"
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = CompactGenderClassifier(
                vocab_size=self.cfg["vocab_size"],
                emb_dim=self.cfg["emb_dim"],
                hidden_dim=self.cfg["hidden_dim"],
                num_layers=self.cfg["num_layers"],
                num_heads=self.cfg["num_heads"],
                max_len=self.max_length,
                dropout=0.0
            ).to(self.device)
            self.model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=self.device, weights_only=True))
            self.model.eval()
            self.backend_name = f"PyTorch ({self.device})"

    def parse_name_tokens(self, text):
        words = [w for w in str(text).strip().split() if len(w) > 0]
        if not words:
            return "", []

        idx = 0
        while idx < len(words):
            cleaned = "".join([c for c in words[idx].lower() if c.isalpha()])
            if cleaned in TITLE_PREFIXES:
                idx += 1
            else:
                break

        remaining = words[idx:]
        if not remaining:
            return words[0], []
        return remaining[0], remaining[1:]

    def softmax(self, logits):
        e_x = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        parsed_items = [self.parse_name_tokens(t) for t in texts]
        query_names = [given if given else raw for raw, (given, _) in zip(texts, parsed_items)]

        enc = self.tokenizer(
            query_names,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="np"
        )
        input_ids = enc["input_ids"].astype(np.int32)
        attention_mask = enc["attention_mask"].astype(np.int32)

        t0 = time.time()
        if self.use_onnx:
            feeds = {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
            logits = self.session.run(None, feeds)[0]
        else:
            with torch.no_grad():
                ids_t = torch.from_numpy(input_ids).to(self.device)
                mask_t = torch.from_numpy(attention_mask).to(self.device)
                logits = self.model(ids_t, mask_t).cpu().numpy()

        elapsed_ms = (time.time() - t0) * 1000 / len(texts)
        probs = self.softmax(logits)

        results = []
        for i, text in enumerate(texts):
            p = probs[i]
            given_name, surnames = parsed_items[i]
            pred_idx = int(np.argmax(p))
            label = self.labels.get(pred_idx, "UNKNOWN")
            conf = float(p[pred_idx])

            results.append({
                "text": text,
                "gender": label,
                "confidence": conf,
                "given_name": given_name,
                "surnames": surnames,
                "raw_probs": {
                    "male": float(p[0]),
                    "female": float(p[1])
                },
                "latency_ms": elapsed_ms
            })

        return results

def main():
    parser = argparse.ArgumentParser(description="CompactGenderClassifier Prediction Tool")
    parser.add_argument("names", nargs="*", help="Name string(s) to classify as positional arguments")
    parser.add_argument("--text", nargs="+", help="Name string(s) to classify")
    parser.add_argument("-i", "--interactive", action="store_true", help="Start interactive CLI loop")
    parser.add_argument("--use-pytorch", action="store_true", help="Force using PyTorch model instead of ONNX")
    args = parser.parse_args()

    clf = GenderClassifier(use_onnx=not args.use_pytorch)
    print(f"--> Initialized {clf.backend_name}")

    if args.interactive:
        print("\n===============================================================")
        print("CompactGenderClassifier Interactive Mode")
        print("Type a name and press Enter. Type 'exit' or 'q' to quit.")
        print("===============================================================\n")
        while True:
            try:
                raw = input("Enter Name > ").strip()
                if not raw:
                    continue
                if raw.lower() in ["exit", "quit", "q"]:
                    print("Exiting interactive mode.")
                    break
                r = clf.predict(raw)[0]
                probs = r["raw_probs"]
                given_str = f" [Given Name: {r['given_name']}]" if r["surnames"] else ""
                print(f"--> Result: {r['gender']} ({r['confidence']*100:.1f}%) | Male: {probs['male']*100:.1f}% | Female: {probs['female']*100:.1f}%{given_str}\n")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting interactive mode.")
                break
        return

    input_names = args.names if args.names else args.text
    if input_names:
        names = input_names
    else:
        names = [
            "Supratim",
            "Sayantika",
            "Supratim Mondal",
            "Sayantika Biswas",
            "Alexander Hamilton",
            "Priya Sharma",
            "John Smith",
            "Emily Watson",
            "Dr. Rahul Verma",
            "Ms. Fatima Sheikh"
        ]

    print("\n" + "=" * 95)
    print(f"{'Input Name':<24} | {'Gender':<8} | {'Conf':<6} | {'Given Name':<14} | {'Probabilities'}")
    print("=" * 95)

    results = clf.predict(names)
    for r in results:
        probs = r['raw_probs']
        prob_str = f"Male: {probs['male']*100:.1f}% | Female: {probs['female']*100:.1f}%"
        print(f"{r['text']:<24} | {r['gender']:<8} | {r['confidence']*100:5.1f}% | {r['given_name']:<14} | {prob_str}")

    print("=" * 95)

if __name__ == "__main__":
    main()
