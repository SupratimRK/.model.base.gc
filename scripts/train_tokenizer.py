import os
import sys
import string
import pandas as pd
from tokenizers import Tokenizer, normalizers
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Sequence, Whitespace, Punctuation, Digits
from tokenizers.trainers import WordPieceTrainer
from tokenizers.processors import TemplateProcessing
from tokenizers.decoders import WordPiece as WordPieceDecoder
from transformers import PreTrainedTokenizerFast

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SCRATCH_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "..", ".model.scratch"))
MASTER_DATA_DIR = os.path.join(SCRATCH_DIR, "master_datasets")
TOKENIZER_DIR = os.path.join(PROJECT_ROOT, "models", "tokenizer")

os.makedirs(TOKENIZER_DIR, exist_ok=True)

def train_custom_tokenizer(vocab_size=4000):
    print("===============================================================")
    print("Training Custom Domain WordPiece Tokenizer For Gender Classifier (100% Uncased)")
    print("===============================================================")
    print(f"Target Vocabulary Size: {vocab_size}")
    print(f"Normalizer: NFKC + Lowercase + StripAccents")
    print(f"Pre-Tokenizer: Whitespace + Punctuation + Digits(individual_digits=True)")

    # 1. Load Texts from Master Dataset (unique names)
    unique_names_path = os.path.join(MASTER_DATA_DIR, "unique_names.parquet")
    train_parquet = os.path.join(MASTER_DATA_DIR, "train.parquet")
    val_parquet = os.path.join(MASTER_DATA_DIR, "val.parquet")

    texts_set = set()
    if os.path.exists(unique_names_path):
        print(f"--> Reading names from: {unique_names_path}")
        df_un = pd.read_parquet(unique_names_path)
        texts_set.update(df_un['name'].dropna().astype(str).str.lower().str.strip().tolist())

    if os.path.exists(train_parquet):
        df_tr = pd.read_parquet(train_parquet)
        if 'name' in df_tr.columns:
            texts_set.update(df_tr['name'].dropna().astype(str).str.lower().str.strip().tolist())
        elif 'text' in df_tr.columns:
            texts_set.update(df_tr['text'].dropna().astype(str).str.lower().str.strip().tolist())

    if os.path.exists(val_parquet):
        df_va = pd.read_parquet(val_parquet)
        if 'name' in df_va.columns:
            texts_set.update(df_va['name'].dropna().astype(str).str.lower().str.strip().tolist())
        elif 'text' in df_va.columns:
            texts_set.update(df_va['text'].dropna().astype(str).str.lower().str.strip().tolist())

    all_texts = sorted(list(texts_set))
    print(f"--> Loaded {len(all_texts):,} unique lowercase strings for tokenizer training")

    # 2. Configure Tokenizer Architecture
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))

    # Built-in Normalizer: Automatically converts any input to clean lowercase
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.Lowercase(),
        normalizers.StripAccents()
    ])

    # Pre-tokenizer: Whitespace, Punctuation, and Individual Digits
    tokenizer.pre_tokenizer = Sequence([
        Whitespace(),
        Punctuation(),
        Digits(individual_digits=True)
    ])

    alphabet = list(string.ascii_lowercase + string.digits + "-._'\"/:@#$%&*()+=[]{}|\\<>!?~`^,;")
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
        initial_alphabet=alphabet,
        show_progress=True
    )

    # 3. Train Tokenizer
    print("--> Training WordPiece model on unique names corpus...")
    tokenizer.train_from_iterator(all_texts, trainer=trainer)

    # 4. Post-Processor: Add [CLS] and [SEP]
    cls_token_id = tokenizer.token_to_id("[CLS]")
    sep_token_id = tokenizer.token_to_id("[SEP]")
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", cls_token_id),
            ("[SEP]", sep_token_id),
        ]
    )
    tokenizer.decoder = WordPieceDecoder(prefix="##")

    # 5. Save Rust Tokenizer JSON
    tok_json_path = os.path.join(TOKENIZER_DIR, "tokenizer.json")
    tokenizer.save(tok_json_path)

    # Export vocab.txt
    vocab_dict = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
    vocab_txt_path = os.path.join(TOKENIZER_DIR, "vocab.txt")
    with open(vocab_txt_path, "w", encoding="utf-8") as f:
        for token, _ in sorted_vocab:
            f.write(token + "\n")

    print(f"--> [Saved] tokenizer.json: {tok_json_path}")
    print(f"--> [Saved] vocab.txt:       {vocab_txt_path} ({len(sorted_vocab)} tokens)")

    # 6. Save Hugging Face PreTrainedTokenizerFast wrapper
    fast_tok = PreTrainedTokenizerFast(
        tokenizer_file=tok_json_path,
        do_lower_case=True,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]"
    )
    fast_tok.save_pretrained(TOKENIZER_DIR)
    print(f"--> [Saved] HuggingFace PreTrainedTokenizerFast in: {TOKENIZER_DIR}")

    # 7. Verification Tests
    test_samples = [
        "Supratim",
        "supratim",
        "SUPRATIM",
        "Sayantika",
        "sayantika",
        "Priya",
        "Alexander",
        "John",
        "Emily",
        "Aarav"
    ]
    print("\n--- Verified Tokenizer Output ---")
    for s in test_samples:
        tokens = fast_tok.tokenize(s)
        has_unk = "[UNK]" in tokens
        status = "⚠️ HAS UNK" if has_unk else "✅ CLEAN"
        print(f"[{status}] '{s}' -> {tokens}")

    print("\n--> Custom uncased domain tokenizer ready for gender classifier training!")

if __name__ == "__main__":
    v_size = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    train_custom_tokenizer(vocab_size=v_size)
