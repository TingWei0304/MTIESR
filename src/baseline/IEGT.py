import torch
import torch.nn as nn
import torch.nn.functional as F


class IEGT(nn.Module):
    """
    Interval-Enhanced Graph Transformer (简化复现版)
    """

    def __init__(self, num_items, hidden_size=100, num_heads=4, num_layers=2):
        super(IEGT, self).__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # item embedding
        self.item_embedding = nn.Embedding(num_items, hidden_size)

        # time embedding projection
        self.time_linear = nn.Linear(1, hidden_size)

        # transformer layers
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_size, num_heads)
            for _ in range(num_layers)
        ])

        # attention for long-term preference
        self.attn_linear = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Parameter(torch.Tensor(hidden_size))

        # output
        self.output = nn.Linear(hidden_size, num_items)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.time_linear.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, session_items, time_intervals):
        """
        session_items: [B, L]
        time_intervals: [B, L]
        """

        # item embedding
        item_emb = self.item_embedding(session_items)  # [B, L, D]

        # time embedding (exponential decay)
        time_intervals = time_intervals.unsqueeze(-1)  # [B, L, 1]
        time_feat = torch.exp(-time_intervals)
        time_emb = self.time_linear(time_feat)

        # concat
        x = item_emb + time_emb  # [B, L, D]

        # transformer
        for layer in self.layers:
            x = layer(x)

        # short-term preference (last item)
        h_s = x[:, -1, :]  # [B, D]

        # long-term preference (attention)
        attn_scores = torch.matmul(x, self.gate)
        attn_weights = F.softmax(attn_scores, dim=1)
        h_l = torch.sum(attn_weights.unsqueeze(-1) * x, dim=1)

        # fusion
        h = 0.5 * h_l + 0.5 * h_s

        # prediction
        logits = self.output(h)

        return logits


class GraphTransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()

        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # self-attention
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)

        # feedforward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x
