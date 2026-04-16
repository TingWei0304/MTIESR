# import pickle
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class MTIESR(nn.Module):
#     def __init__(self, num_items, num_cats, emb_dim, hgnn_dim, gru_dim, item2cat_path):
#         super().__init__()
#         self.num_items = num_items
#         self.emb_dim = emb_dim
#
#         # 加载并转换类别映射为张量
#         with open(item2cat_path, 'rb') as f:
#             item2cat_dict = pickle.load(f)
#         # 将字典转换为连续索引张量 [重要修复]
#         self.register_buffer(
#             'item2cat',
#             torch.tensor([item2cat_dict[i] for i in range(num_items)], dtype=torch.long)
#         )
#
#         # 统一嵌入维度 [关键修复]
#         self.item_emb = nn.Embedding(num_items, emb_dim)
#         self.cat_emb = nn.Embedding(num_cats, emb_dim)
#
#         # 超图网络
#         self.hgnn = nn.Sequential(
#             nn.Linear(emb_dim * 2, hgnn_dim),  # 输入维度匹配修复后的嵌入
#             nn.ReLU(),
#             nn.Linear(hgnn_dim, hgnn_dim)
#         )
#
#         # GRU序列建模
#         self.gru = nn.GRU(
#             input_size=emb_dim * 2 + hgnn_dim,
#             hidden_size=gru_dim,
#             batch_first=True
#         )
#
#         # 预测层
#         self.item_fc = nn.Linear(gru_dim, num_items)
#         self.cat_fc = nn.Linear(gru_dim, num_cats)
#
#     def forward(self, session, categories, hypergraphs, time_slice_ids, masks=None):
#         B, L = session.shape
#
#         # 基础嵌入
#         item_emb = self.item_emb(session)  # [B, L, D]
#         cat_emb = self.cat_emb(categories)  # [B, L, D]
#         base_emb = torch.cat([item_emb, cat_emb], dim=-1)  # [B, L, 2D]
#
#         # 超图特征 [关键修复]
#         # 获取每个item对应的category嵌入
#         item_cat_ids = self.item2cat[session.flatten()]  # [B*L]
#         item_global_emb = self.item_emb.weight.unsqueeze(0)  # [1, num_items, D]
#         cat_global_emb = self.cat_emb(self.item2cat)  # [num_items, D]
#
#         # 拼接每个item及其对应category的嵌入
#         global_emb = torch.cat([item_global_emb, cat_global_emb.unsqueeze(0)], dim=-1)  # [1, num_items, 2D]
#         hgnn_feat = self.hgnn(global_emb.squeeze(0))  # [num_items, hgnn_dim]
#
#         # 匹配批次维度
#         batch_hgnn = hgnn_feat[session.flatten()].view(B, L, -1)  # [B, L, hgnn_dim]
#
#         # 特征拼接
#         enhanced_emb = torch.cat([base_emb, batch_hgnn], dim=-1)  # [B, L, 2D + hgnn_dim]
#
#         # GRU建模
#         gru_out, _ = self.gru(enhanced_emb)  # [B, L, gru_dim]
#         final_interest = gru_out[:, -1, :]  # [B, gru_dim]
#
#         # 预测
#         item_pred = self.item_fc(final_interest)
#         cat_pred = self.cat_fc(final_interest)
#         return item_pred, cat_pred
#
#     def loss(self, item_pred, cat_pred, item_target, cat_target):
#         alpha = 0.75  # 低频物品权重
#         gamma = 2.0   # 困难样本聚焦参数
#         ce_loss = F.cross_entropy(item_pred, item_target, reduction='none')
#         pt = torch.exp(-ce_loss)
#         focal_loss = (alpha * (1-pt)**gamma * ce_loss).mean()
#         return focal_loss + 0.2 * F.cross_entropy(cat_pred, cat_target)
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
        hgnn_dim=128,
        gru_dim=128,
        num_heads=4,
        dropout=0.2,
        alpha=0.5,
        num_layers=2
    ):
        super().__init__()

        self.alpha = alpha

        # ===== Embedding =====
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.cat_emb = nn.Embedding(num_cats, emb_dim)

        # ===== HGNN =====
        self.hgnn = HGNN(emb_dim, num_layers)

        # ===== GRU（短期兴趣）=====
        self.gru = nn.GRU(emb_dim, gru_dim, batch_first=True)

        # ===== Transformer（长期兴趣）=====
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # ===== 门控融合（核心创新）=====
        self.gate = nn.Linear(gru_dim + emb_dim, gru_dim)

        # ===== 预测层 =====
        self.item_fc = nn.Linear(gru_dim, num_items)
        self.cat_fc = nn.Linear(gru_dim, num_cats)

        self.dropout = nn.Dropout(dropout)

    def forward(self, items, cats, hypergraphs, times=None, masks=None):

        item_e = self.item_emb(items)

        # ===== HGNN增强 =====
        all_item_emb = self.item_emb.weight
        all_item_emb = self.hgnn(all_item_emb, hypergraphs)
        item_e = all_item_emb[items]

        # ===== GRU =====
        gru_out, _ = self.gru(item_e)
        h_gru = gru_out[:, -1, :]

        # ===== Transformer =====
        trans_out = self.transformer(item_e)
        h_trans = trans_out[:, -1, :]

        # ===== 门控融合 =====
        fusion = torch.cat([h_gru, h_trans], dim=-1)
        gate = torch.sigmoid(self.gate(fusion))

        h = gate * h_gru + (1 - gate) * h_trans[:, :h_gru.size(1)]

        h = self.dropout(h)

        # ===== 输出 =====
        item_logits = self.item_fc(h)
        cat_logits = self.cat_fc(h)

        return item_logits, cat_logits

    def loss(self, item_pred, cat_pred, item_targets, cat_targets):
        loss_item = F.cross_entropy(item_pred, item_targets)
        loss_cat = F.cross_entropy(cat_pred, cat_targets)

        return loss_item + self.alpha * loss_cat