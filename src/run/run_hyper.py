# import os
# import pickle
# import numpy as np
# import scipy.sparse as sp
# import torch
# import pandas as pd
#
# from src.models.mtiesr import MTIESR
# from src.utils.data_utils import get_dataloaders
# from src.evaluator.evaluator import Evaluator
# from src.trainer.trainer import Trainer
# from src.utils.utils import set_seed
#
# set_seed(42)
#
# dataset_name = "2019-oct"
# data_dir = f"/root/Documents/wtt/second/datasets/{dataset_name}"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # ===== 加载映射 =====
# with open(os.path.join(data_dir, 'item2idx.pkl'), 'rb') as f:
#     item2idx = pickle.load(f)
# with open(os.path.join(data_dir, 'cat2idx.pkl'), 'rb') as f:
#     cat2idx = pickle.load(f)
#
# # ===== 加载超图 =====
# hypergraph_dict = {}
# for hyper_type in ['time', 'session', 'category', 'user']:
#     path = os.path.join(data_dir, f'hypergraph_{hyper_type}.npz')
#     h = sp.load_npz(path).tocoo()
#     indices = torch.tensor(np.vstack([h.row, h.col]), dtype=torch.long)
#     values = torch.tensor(h.data, dtype=torch.float32)
#     H = torch.sparse_coo_tensor(indices, values, h.shape).coalesce().to(device)
#     hypergraph_dict[hyper_type] = H
#
# # ===== 数据 =====
# train_loader, val_loader, test_loader, _ = get_dataloaders(
#     data_dir=data_dir,
#     batch_size=100,
#     num_workers=4
# )
#
# evaluator = Evaluator(topk_list=[10, 20])
#
# # =========================
# # ⭐ 六个超参数
# # =========================
# param_configs = {
#     "gru_dim": [64, 128, 192, 256, 320, 384],
#     "emb_dim": [64, 128, 192, 256, 320],
#     "lr": [1e-3, 5e-4, 1e-4, 5e-5],
#     "batch_size": [64, 128, 256],
#     "dropout": [0.1, 0.2, 0.3, 0.4],
#     "epochs": [20, 40, 60, 80]
# }
#
# # ===== 结果保存 =====
# os.makedirs("results", exist_ok=True)
#
# def run_experiment(param_name, values):
#     all_results = []
#
#     for val in values:
#         print(f"\n===== {param_name} = {val} =====")
#
#         # 默认参数
#         emb_dim = 128
#         gru_dim = 128
#         lr = 1e-4
#         batch_size = 100
#         epochs = 30
#
#         # 替换当前参数
#         if param_name == "gru_dim":
#             gru_dim = val
#             emb_dim = val
#         elif param_name == "emb_dim":
#             emb_dim = val
#             gru_dim = val
#         elif param_name == "lr":
#             lr = val
#         elif param_name == "batch_size":
#             batch_size = val
#         elif param_name == "epochs":
#             epochs = val
#
#         # 重新加载数据（batch_size变化）
#         train_loader, val_loader, test_loader, _ = get_dataloaders(
#             data_dir=data_dir,
#             batch_size=batch_size,
#             num_workers=4
#         )
#
#         model = MTIESR(
#             num_items=len(item2idx),
#             num_cats=len(cat2idx),
#             emb_dim=emb_dim,
#             hgnn_dim=emb_dim,
#             gru_dim=gru_dim,
#             item2cat_path=os.path.join(data_dir, 'item2cat.pkl')
#         ).to(device)
#
#         trainer = Trainer(model=model, device=device, hypergraphs=hypergraph_dict)
#
#         trainer.train(
#             train_loader=train_loader,
#             val_loader=val_loader,
#             test_loader=test_loader,
#             evaluator=evaluator,
#             epochs=epochs,
#             lr=lr,
#             save_path=f"./save_model/{param_name}_{val}.pt"
#         )
#
#         result = evaluator.evaluate(model, test_loader, device, hypergraph_dict)
#         result[param_name] = val
#
#         all_results.append(result)
#
#     df = pd.DataFrame(all_results)
#     df.to_csv(f"results/{param_name}.csv", index=False)
#     print(f"Saved results/{param_name}.csv")
#
#
# # ===== 执行全部实验 =====
# for param_name, values in param_configs.items():
#     run_experiment(param_name, values)
import os
import pickle
import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd

from src.models.mtiesr import MTIESR
from src.utils.data_utils import get_dataloaders
from src.evaluator.evaluator import Evaluator
from src.trainer.trainer import Trainer
from src.utils.utils import set_seed

set_seed(42)

dataset_name = "2019-oct"
data_dir = f"/root/Documents/wtt/second/datasets/{dataset_name}"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 加载映射 =====
with open(os.path.join(data_dir, 'item2idx.pkl'), 'rb') as f:
    item2idx = pickle.load(f)
with open(os.path.join(data_dir, 'cat2idx.pkl'), 'rb') as f:
    cat2idx = pickle.load(f)

# ===== 超图 =====
hypergraph_dict = {}
for t in ['time', 'session', 'category', 'user']:
    h = sp.load_npz(os.path.join(data_dir, f'hypergraph_{t}.npz')).tocoo()
    indices = torch.tensor(np.vstack([h.row, h.col]), dtype=torch.long)
    values = torch.tensor(h.data, dtype=torch.float32)
    hypergraph_dict[t] = torch.sparse_coo_tensor(indices, values, h.shape).coalesce().to(device)

# ===== evaluator =====
evaluator = Evaluator(topk_list=[10, 20])

# ===== 6个参数 =====
param_configs = {
    "gru_dim": [64, 128, 192, 256, 320, 384],
    "trans_dim": [64, 128, 192, 256, 320],
    "num_heads": [1, 2, 4, 6, 8, 10],
    "hgnn_dim": [64, 128, 192, 256],
    "dropout": [0.1, 0.2, 0.3, 0.4],
    "lambda": [0.1, 0.3, 0.5, 0.7, 1.0]
}

os.makedirs("results", exist_ok=True)

def run_exp(param, values):
    results = []

    for v in values:
        print(f"\n===== {param} = {v} =====")

        # 默认值
        cfg = {
            "gru_dim": 128,
            "trans_dim": 128,
            "hgnn_dim": 128,
            "num_heads": 4,
            "dropout": 0.2,
            "lambda": 0.5
        }

        cfg[param] = v

        train_loader, val_loader, test_loader, _ = get_dataloaders(
            data_dir=data_dir,
            batch_size=100,
            num_workers=4
        )

        model = MTIESR(
            num_items=len(item2idx),
            num_cats=len(cat2idx),
            emb_dim=cfg["hgnn_dim"],
            hgnn_dim=cfg["hgnn_dim"],
            gru_dim=cfg["gru_dim"],
            trans_dim=cfg["trans_dim"],
            num_heads=cfg["num_heads"],
            dropout=cfg["dropout"],
            lambda_=cfg["lambda"]
        ).to(device)

        trainer = Trainer(model=model, device=device, hypergraphs=hypergraph_dict)

        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            evaluator=evaluator,
            epochs=30,
            lr=1e-4,
            save_path=f"./save_model/{param}_{v}.pt"
        )

        res = evaluator.evaluate(model, test_loader, device, hypergraph_dict)
        res[param] = v
        results.append(res)

    pd.DataFrame(results).to_csv(f"results/{param}.csv", index=False)


for p, v in param_configs.items():
    run_exp(p, v)