import torch
import torch.nn as nn
import torch.nn.functional as F


class LastAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, short_interest, long_interest):
        # Query: 短期兴趣, Key/Value: 长期兴趣
        q = self.query(short_interest).unsqueeze(1)  # [B, 1, H]
        k = self.key(long_interest).unsqueeze(1)  # [B, 1, H]
        v = self.value(long_interest).unsqueeze(1)  # [B, 1, H]

        # 注意力得分
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / self.scale  # [B, 1, 1]
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 加权聚合
        final_interest = torch.bmm(attn_weights, v).squeeze(1)  # [B, H]
        return final_interest
