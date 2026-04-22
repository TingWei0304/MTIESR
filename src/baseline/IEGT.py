import torch
import torch.nn as nn
import torch.nn.functional as F


class IEGT(nn.Module):

    def __init__(self, num_items, num_cats, hidden_size=128, num_heads=4, num_layers=2, dropout=0.2):
        super(IEGT, self).__init__()

        self.hidden_size = hidden_size

        # ===== Embedding =====
        self.item_embedding = nn.Embedding(num_items, hidden_size)
        self.cat_embedding = nn.Embedding(num_cats, hidden_size)

        # ===== 时间编码 =====
        self.time_linear = nn.Linear(1, hidden_size)

        # ===== Transformer =====
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ===== 长期兴趣 attention =====
        self.attn_linear = nn.Linear(hidden_size, 1)

        # ===== 融合 =====
        self.dropout = nn.Dropout(dropout)

        # ===== 输出 =====
        self.item_fc = nn.Linear(hidden_size, num_items)
        self.cat_fc = nn.Linear(hidden_size, num_cats)

    def forward(self, items, cats, hypergraphs=None, times=None, masks=None):
        """
        items: [B, L]
        cats: [B, L]
        times: [B, L]
        masks: [B, L]
        """

        # ===== Embedding =====
        item_emb = self.item_embedding(items)   # [B, L, D]
        cat_emb = self.cat_embedding(cats)      # [B, L, D]

        x = item_emb + cat_emb   # 简化融合（论文可写 early fusion）

        # ===== 时间信息 =====
        if times is not None:
            t = times.float().unsqueeze(-1)   # [B, L, 1]
            t = torch.exp(-t)                # exponential decay
            t_emb = self.time_linear(t)
            x = x + t_emb

        # ===== Transformer（加 mask）=====
        if masks is not None:
            # True=有效 → transformer需要False=mask
            src_key_padding_mask = ~masks
        else:
            src_key_padding_mask = None

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # ===== Short-term =====
        h_s = x[:, -1, :]   # [B, D]

        # ===== Long-term =====
        attn_score = self.attn_linear(x).squeeze(-1)  # [B, L]

        if masks is not None:
            attn_score = attn_score.masked_fill(~masks, -1e9)

        attn_weight = torch.softmax(attn_score, dim=1)
        h_l = torch.sum(attn_weight.unsqueeze(-1) * x, dim=1)

        # ===== 融合 =====
        h = 0.5 * h_s + 0.5 * h_l
        h = self.dropout(h)

        # ===== 输出 =====
        item_logits = self.item_fc(h)
        cat_logits = self.cat_fc(h)

        return item_logits, cat_logits

    def loss(self, item_pred, cat_pred, item_targets, cat_targets):
        loss_item = F.cross_entropy(item_pred, item_targets)
        loss_cat = F.cross_entropy(cat_pred, cat_targets)
        return loss_item + 0.5 * loss_cat
