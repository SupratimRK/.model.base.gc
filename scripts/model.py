import torch
import torch.nn as nn

class CompactGenderClassifier(nn.Module):
    """
    Ultra-compact transformer binary gender classifier trained from scratch.
    Features:
      - Factorized Embeddings (vocab_size -> emb_dim -> hidden_dim) for ~50% embedding weight reduction
      - Pre-LN TransformerEncoder for fast, stable training dynamics
      - Masked Dual-Pooling: Concat(MeanPool, MaxPool) for subword affix sensitivity
      - Single-task binary classification head: MALE (0) vs FEMALE (1)
      - Zero BertPooler / token_type_ids bloat for clean ONNX execution
    """
    def __init__(
        self,
        vocab_size=4000,
        emb_dim=64,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        dim_feedforward=256,
        max_len=20,
        dropout=0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        # Factorized Token Embeddings: cuts parameters from 4000x128=512k to 4000x64+64x128=264k
        self.token_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.emb_proj = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)

        # Transformer Encoder with Pre-LN (norm_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Dual Pooling: 128 (MeanPool) + 128 (MaxPool) = 256
        self.classifier = nn.Linear(hidden_dim * 2, 2)

        self._init_weights()

    def _init_weights(self):
        """Standard truncated normal initialization for training from scratch."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    with torch.no_grad():
                        m.weight[m.padding_idx].fill_(0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, input_ids, attention_mask):
        # input_ids: [batch_size, seq_len] (int32 or int64)
        # attention_mask: [batch_size, seq_len] (int32 or int64, 1 for real tokens, 0 for pad)
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        # Embeddings: token + positional
        x = self.emb_proj(self.token_emb(input_ids.long())) + self.pos_emb(positions)

        # PyTorch TransformerEncoder expects src_key_padding_mask=True for PAD tokens
        padding_mask = (attention_mask == 0)
        hidden = self.encoder(x, src_key_padding_mask=padding_mask)
        hidden = self.norm(hidden) # [batch_size, seq_len, hidden_dim]

        # 1. Masked Mean Pooling
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = torch.sum(hidden * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        # 2. Masked Max Pooling (padded positions set to -1e9 so they are never max)
        hidden_masked = hidden.masked_fill(padding_mask.unsqueeze(-1), -1e9)
        max_pooled = torch.max(hidden_masked, dim=1).values

        # Concatenate features: 128 + 128 = 256
        features = torch.cat([mean_pooled, max_pooled], dim=-1)
        features = self.dropout(features)
        logits = self.classifier(features)
        return logits
