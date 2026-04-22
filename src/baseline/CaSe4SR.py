import torch
import torch.nn as nn
import torch.nn.functional as F


class CaSe4SR(nn.Module):
    def __init__(self, num_items, num_cats, emb_dim=128, dropout=0.2):
        super().__init__()

        self.emb_dim = emb_dim

        # ===== Embedding =====
        self.item_embedding = nn.Embedding(num_items, emb_dim)
        self.cat_embedding = nn.Embedding(num_cats, emb_dim)

        # ===== Attention =====
        self.w1 = nn.Linear(emb_dim * 2, emb_dim * 2)
        self.w2 = nn.Linear(emb_dim * 2, emb_dim * 2, bias=False)
        self.q = nn.Linear(emb_dim * 2, 1)

        self.dropout = nn.Dropout(dropout)

        # ===== 输出 =====
        self.fc = nn.Linear(emb_dim * 2, num_items)

    def forward(self, items, cats, hypergraphs=None, times=None, masks=None):

        # ===== Embedding =====
        item_emb = self.item_embedding(items)   # [B, L, D]
        cat_emb = self.cat_embedding(cats)      # [B, L, D]

        # 拼接 item + category
        x = torch.cat([item_emb, cat_emb], dim=-1)  # [B, L, 2D]

        # ===== last item =====
        last = x[:, -1, :].unsqueeze(1).repeat(1, x.size(1), 1)

        # ===== Attention =====
        alpha = self.q(torch.sigmoid(self.w1(x) + self.w2(last))).squeeze(-1)  # [B, L]

        if masks is not None:
            alpha = alpha.masked_fill(~masks, -1e9)

        alpha = torch.softmax(alpha, dim=1).unsqueeze(-1)

        # ===== session 表示 =====
        session_emb = torch.sum(alpha * x, dim=1)  # [B, 2D]

        session_emb = self.dropout(session_emb)

        # ===== 输出 =====
        logits = self.fc(session_emb)

        return logits, logits

    def loss(self, item_pred, cat_pred, item_targets, cat_targets):
        return F.cross_entropy(item_pred, item_targets)
