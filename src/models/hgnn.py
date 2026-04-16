import torch
import torch.nn as nn
import torch.nn.functional as F


class HGNNLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, X, H, DV_inv_sqrt, DE_inv):
        """
        X: [N, D]
        H: sparse [N, E]
        """

        # ===== 左归一化 =====
        X_norm = X * DV_inv_sqrt.unsqueeze(1)

        # ===== H^T X =====
        HX = torch.sparse.mm(H.t(), X_norm)   # [E, D]

        # ===== 边归一化 =====
        HX = HX * DE_inv.unsqueeze(1)

        # ===== H @ HX =====
        X_out = torch.sparse.mm(H, HX)        # [N, D]

        # ===== 右归一化 =====
        X_out = X_out * DV_inv_sqrt.unsqueeze(1)

        # ===== 线性变换 =====
        X_out = self.linear(X_out)

        return X_out


class HypergraphAttentionFusion(nn.Module):
    """
    ⭐ 多超图注意力融合（论文核心点）
    """
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, X_list):
        """
        X_list: list of [N, D]
        """

        # [num_graph, N, D]
        X_stack = torch.stack(X_list, dim=0)

        # attention score
        scores = self.attn(X_stack)  # [G, N, 1]
        weights = torch.softmax(scores, dim=0)

        # 加权融合
        X = (weights * X_stack).sum(dim=0)

        return X


class HGNN(nn.Module):
    def __init__(self, dim, num_layers=2, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            HGNNLayer(dim) for _ in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # ⭐ Attention融合
        self.fusion = HypergraphAttentionFusion(dim)

    def forward(self, X, hypergraphs):
        """
        hypergraphs:
        {
            name: {
                "H": sparse tensor,
                "DV_inv_sqrt": tensor,
                "DE_inv": tensor
            }
        }
        """

        graph_outputs = []

        for name, data in hypergraphs.items():
            H = data["H"]
            DV_inv_sqrt = data["DV_inv_sqrt"]
            DE_inv = data["DE_inv"]

            X_h = X

            for i, layer in enumerate(self.layers):
                residual = X_h

                X_h = layer(X_h, H, DV_inv_sqrt, DE_inv)

                # ===== Residual =====
                X_h = X_h + residual

                # ===== LayerNorm =====
                X_h = self.norms[i](X_h)

                # ===== 激活 + Dropout =====
                X_h = F.relu(X_h)
                X_h = self.dropout(X_h)

            graph_outputs.append(X_h)

        # ===== Attention融合 =====
        X = self.fusion(graph_outputs)

        return X