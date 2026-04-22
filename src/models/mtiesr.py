import torch
import torch.nn as nn
import torch.nn.functional as F
from .hgnn import HGNN


class MTIESR(nn.Module):
    def __init__(
        self,
        num_items,
        num_cats,
        emb_dim=128,
        hgnn_layers=2,
        gru_dim=128,
        num_heads=4,
        dropout=0.2,
        alpha=0.5
    ):
        super().__init__()

        self.alpha = alpha

        # ===== Embedding =====
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.cat_emb = nn.Embedding(num_cats, emb_dim)

        # ===== HGNN（全局建模）=====
        self.hgnn = HGNN(emb_dim, num_layers=hgnn_layers)

        # ===== GRU（短期兴趣）=====
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=gru_dim,
            batch_first=True
        )

        # ===== Transformer（长期兴趣）=====
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        # ===== 门控融合 =====
        self.gate = nn.Linear(gru_dim + emb_dim, gru_dim)

        # ===== 输出层 =====
        self.item_fc = nn.Linear(gru_dim, num_items)
        self.cat_fc = nn.Linear(gru_dim, num_cats)

        self.dropout = nn.Dropout(dropout)

    #统一接口
    def forward(self, batch):
        """
        batch:
        {
            "items": [B, L]
            "cats": [B, L]
            "mask": [B, L]
            "times": [B, L] (optional)
            "hypergraphs": dict (optional)
        }
        """

        items = batch["items"]
        cats = batch["cats"]
        masks = batch.get("mask", None)
        times = batch.get("times", None)
        hypergraphs = batch.get("hypergraphs", None)

        # ===== Embedding =====
        item_e = self.item_emb(items)  # [B, L, D]

        # ===== HGNN（全局图）=====
        if hypergraphs is not None:
            all_item_emb = self.item_emb.weight  # [num_items, D]
            all_item_emb = self.hgnn(all_item_emb, hypergraphs)
            item_e = all_item_emb[items]  # 替换局部embedding

        # ===== GRU（短期兴趣）=====
        gru_out, _ = self.gru(item_e)  # [B, L, H]
        h_gru = gru_out[:, -1, :]      # [B, H]

        # ===== Transformer（长期兴趣）=====
        trans_out = self.transformer(item_e)  # [B, L, D]
        h_trans = trans_out[:, -1, :]         # [B, D]

        # ===== 门控融合 =====
        fusion = torch.cat([h_gru, h_trans], dim=-1)
        gate = torch.sigmoid(self.gate(fusion))

        # 对齐维度
        h_trans_proj = h_trans[:, :h_gru.size(1)]

        h = gate * h_gru + (1 - gate) * h_trans_proj
        h = self.dropout(h)

        # ===== 输出 =====
        item_logits = self.item_fc(h)
        cat_logits = self.cat_fc(h)

        return {
            "item_logits": item_logits,
            "cat_logits": cat_logits
        }

    # 统一 loss 
    def loss(self, outputs, batch):
        item_pred = outputs["item_logits"]
        cat_pred = outputs["cat_logits"]

        item_targets = batch["item_targets"]
        cat_targets = batch["cat_targets"]

        loss_item = F.cross_entropy(item_pred, item_targets)
        loss_cat = F.cross_entropy(cat_pred, cat_targets)

        return loss_item + self.alpha * loss_cat
