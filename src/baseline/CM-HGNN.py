import torch
import torch.nn as nn
import torch.nn.functional as F


class CMHGNN(nn.Module):
    def __init__(self, num_items, emb_dim=100, dropout=0.2):
        super().__init__()

        self.num_items = num_items
        self.emb_dim = emb_dim

        # ===== Embedding =====
        self.embedding = nn.Embedding(num_items, emb_dim)

        # =====  GNN=====
        self.gnn = nn.Linear(emb_dim, emb_dim)

        self.w_1 = nn.Linear(2 * emb_dim, emb_dim)
        self.w_2 = nn.Linear(emb_dim, 1)

        self.dropout = nn.Dropout(dropout)

        # ===== 输出层 =====
        self.fc = nn.Linear(emb_dim, num_items)

    def forward(self, items, cats=None, hypergraphs=None, times=None, masks=None):
        """
        items: [B, L]
        masks: [B, L]
        """

        # ===== Embedding =====
        emb = self.embedding(items)  # [B, L, D]

        # ===== GNN=====
        h = self.gnn(emb)

        # ===== Attention Pooling =====
        attn = torch.tanh(self.w_1(torch.cat([emb, h], dim=-1)))
        attn = self.w_2(attn).squeeze(-1)  # [B, L]

        if masks is not None:
            attn = attn.masked_fill(~masks, -1e9)

        alpha = torch.softmax(attn, dim=1).unsqueeze(-1)

        session_rep = torch.sum(alpha * emb, dim=1)  # [B, D]

        session_rep = self.dropout(session_rep)

        # ===== 输出 =====
        logits = self.fc(session_rep)

        return logits, logits

    def loss(self, item_pred, cat_pred, item_targets, cat_targets):
        return F.cross_entropy(item_pred, item_targets)
