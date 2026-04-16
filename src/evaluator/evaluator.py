import numpy as np
import torch

class Evaluator:
    def __init__(self, topk_list=[10, 20], multi_target=False):
        self.topk_list = sorted(topk_list)
        self.max_k = max(topk_list)
        self.multi_target = multi_target  # 建议 False（单目标）

    def evaluate(self, model, dataloader, device, hypergraphs):
        model.eval()

        metrics = {
            k: {"hr": 0.0, "ndcg": 0.0, "mrr": 0.0, "precision": 0.0}
            for k in self.topk_list
        }

        total_samples = 0
        category_correct = 0
        category_total = 0

        with torch.no_grad():
            for batch in dataloader:
                items_seq, cats_seq, item_targets, cat_targets, masks, time_ids = batch

                items_seq = items_seq.to(device)
                cats_seq = cats_seq.to(device)
                item_targets = item_targets.to(device)
                cat_targets = cat_targets.to(device)

                batch_size = item_targets.size(0)
                total_samples += batch_size

                # forward
                item_logits, cat_logits = model(items_seq, cats_seq, hypergraphs, time_ids, masks)

                _, topk_indices = item_logits.topk(self.max_k, dim=1)
                topk_indices = topk_indices.cpu()
                targets = item_targets.cpu()

                for k in self.topk_list:
                    topk = topk_indices[:, :k]
                    hit_mask = (topk == targets.unsqueeze(1))

                    # ===== HR =====
                    metrics[k]["hr"] += hit_mask.any(dim=1).float().sum().item()

                    # ===== Precision（修复）=====
                    metrics[k]["precision"] += hit_mask.sum().item()

                    # ===== MRR =====
                    for i in range(batch_size):
                        hits = hit_mask[i].nonzero(as_tuple=False)
                        if len(hits) > 0:
                            rank = hits[0].item() + 1
                            metrics[k]["mrr"] += 1.0 / rank

                    # ===== NDCG（单目标）=====
                    for i in range(batch_size):
                        hits = hit_mask[i].nonzero(as_tuple=False)
                        if len(hits) > 0:
                            rank = hits[0].item() + 1
                            metrics[k]["ndcg"] += 1.0 / np.log2(rank + 1)

                # ===== Category Accuracy =====
                pred_cat = cat_logits.argmax(dim=1)
                category_correct += (pred_cat == cat_targets).sum().item()
                category_total += batch_size

        # ===== Final Metrics =====
        final_metrics = {}
        for k in self.topk_list:
            final_metrics[f"HR@{k}"] = metrics[k]["hr"] / total_samples
            final_metrics[f"MRR@{k}"] = metrics[k]["mrr"] / total_samples
            final_metrics[f"NDCG@{k}"] = metrics[k]["ndcg"] / total_samples
            final_metrics[f"P@{k}"] = metrics[k]["precision"] / (total_samples * k)

        final_metrics["Category_ACC"] = category_correct / category_total

        return final_metrics