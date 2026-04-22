import torch
import torch.nn as nn
import torch.nn.functional as F


class SRGNN(nn.Module):


    def __init__(self, num_items, num_cats, hidden_size=128, step=1, dropout=0.2):
        super(SRGNN, self).__init__()

        self.hidden_size = hidden_size
        self.step = step

        # ===== Embedding =====
        self.item_embedding = nn.Embedding(num_items, hidden_size)
        self.cat_embedding = nn.Embedding(num_cats, hidden_size)

        # ===== GNN (简化版) =====
        self.gnn_linear = nn.Linear(hidden_size, hidden_size)

        # ===== Attention =====
        self.linear_one = nn.Linear(hidden_size, hidden_size)
        self.linear_two = nn.Linear(hidden_size, hidden_size)
        self.linear_three = nn.Linear(hidden_size, 1, bias=False)

        self.dropout = nn.Dropout(dropout)

        # ===== 输出 =====
        self.item_fc = nn.Linear(hidden_size, num_items)
        self.cat_fc = nn.Linear(hidden_size, num_cats)

    def forward(self, items, cats, hypergraphs=None, times=None, masks=None):
        """
        items: [B, L]
        masks: [B, L]
        """

        # ===== embedding =====
        h = self.item_embedding(items)  # [B, L, D]

        # ===== 简化 GNN（邻接信息用线性近似）=====
        for _ in range(self.step):
            h = self.gnn_linear(h)

        # ===== last item =====
        ht = h[:, -1, :]  # [B, D]

        # ===== attention =====
        q1 = self.linear_one(ht).unsqueeze(1)   # [B,1,D]
        q2 = self.linear_two(h)                # [B,L,D]

        alpha = self.linear_three(torch.sigmoid(q1 + q2)).squeeze(-1)  # [B,L]

        if masks is not None:
            alpha = alpha.masked_fill(~masks, -1e9)

        alpha = torch.softmax(alpha, dim=1).unsqueeze(-1)

        # ===== session 表示 =====
        a = torch.sum(alpha * h, dim=1)

        # ===== 融合 =====
        a = torch.cat([a, ht], dim=-1)
        a = nn.Linear(self.hidden_size * 2, self.hidden_size).to(a.device)(a)

        a = self.dropout(a)

        # ===== 输出 =====
        item_logits = self.item_fc(a)
        cat_logits = self.cat_fc(a)

        return item_logits, cat_logits

    def loss(self, item_pred, cat_pred, item_targets, cat_targets):
        return F.cross_entropy(item_pred, item_targets)
