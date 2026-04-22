import torch
import torch.nn as nn
import torch.nn.functional as F


class STEM(nn.Module):
    def __init__(
        self,
        num_items,
        num_cats,
        emb_dim=128,
        hidden_dim=128,
        num_shared_experts=2,
        num_specific_experts=2,
        num_tasks=2,
        dropout=0.2
    ):
        super().__init__()

        self.num_tasks = num_tasks
        self.num_shared_experts = num_shared_experts
        self.num_specific_experts = num_specific_experts

        # ===== Embedding =====
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.cat_emb = nn.Embedding(num_cats, emb_dim)

        input_dim = emb_dim * 2

        # ===== Experts =====
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for _ in range(num_shared_experts)
        ])

        self.specific_experts = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
                for _ in range(num_specific_experts)
            ])
            for _ in range(num_tasks)
        ])

        # ===== Gate =====
        self.gates = nn.ModuleList([
            nn.Linear(input_dim, num_shared_experts + num_specific_experts * num_tasks)
            for _ in range(num_tasks)
        ])

        # ===== Tower =====
        self.towers = nn.ModuleList([
            nn.Linear(hidden_dim, num_items if i == 0 else num_cats)
            for i in range(num_tasks)
        ])

    def forward(self, items, cats, hypergraphs=None, times=None, masks=None):
        """
        items: [B, L]
        cats:  [B, L]
        """

        # ===== Embedding =====
        item_e = self.item_emb(items)
        cat_e = self.cat_emb(cats)

        x = torch.cat([item_e, cat_e], dim=-1)  # [B, L, 2D]

        if masks is not None:
            mask = masks.unsqueeze(-1).float()
            x = (x * mask).sum(1) / mask.sum(1)
        else:
            x = x.mean(dim=1)

        # ===== Experts =====
        shared_outs = [expert(x) for expert in self.shared_experts]

        specific_outs = []
        for t in range(self.num_tasks):
            task_outs = [expert(x) for expert in self.specific_experts[t]]
            specific_outs.append(task_outs)

        # ===== Gate + Fusion =====
        outputs = []
        for t in range(self.num_tasks):
            gate = torch.softmax(self.gates[t](x), dim=-1)

            all_experts = []
            for s in specific_outs:
                all_experts.extend(s)
            all_experts.extend(shared_outs)

            all_experts = torch.stack(all_experts, dim=1)  # [B, E, D]

            out = torch.sum(gate.unsqueeze(-1) * all_experts, dim=1)
            outputs.append(out)

        # ===== Prediction =====
        item_logits = self.towers[0](outputs[0])
        cat_logits = self.towers[1](outputs[1])

        return item_logits, cat_logits

    def loss(self, item_pred, cat_pred, item_targets, cat_targets):
        loss_item = F.cross_entropy(item_pred, item_targets)
        loss_cat = F.cross_entropy(cat_pred, cat_targets)
        return loss_item + 0.5 * loss_cat
