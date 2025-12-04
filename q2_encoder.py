import torch
import torch.nn as nn

class SimpleTransformerEncoder(nn.Module):
    def __init__(self, d_model=128, num_heads=8, ff_hidden=512):
        super(SimpleTransformerEncoder, self).__init__()

        # Multi-head self-attention layer
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model,
                                               num_heads=num_heads,
                                               batch_first=True)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden, d_model)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)

        # ---- Multi-head self attention ----
        attn_output, _ = self.self_attn(x, x, x)

        # Residual + Normalization
        x = self.norm1(x + attn_output)

        # ---- Feed-forward network ----
        ffn_output = self.ffn(x)

        # Residual + Normalization again
        out = self.norm2(x + ffn_output)

        return out


# Testing output shape
if __name__ == "__main__":
    batch_size = 32
    seq_len = 10
    d_model = 128

    x = torch.randn(batch_size, seq_len, d_model)
    encoder = SimpleTransformerEncoder()

    output = encoder(x)
    print("Output shape:", output.shape)
