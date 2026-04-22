import os
import pickle
import argparse
import numpy as np
import scipy.sparse as sp
import torch

from src.models.mtiesr import MTIESR
from src.utils.data_utils import get_dataloaders
from src.trainer.trainer import Trainer
from src.evaluator.evaluator import Evaluator
from src.utils.utils import set_seed

def load_sparse(path):
    H = sp.load_npz(path).tocoo()
    indices = torch.from_numpy(
        np.vstack((H.row, H.col)).astype(np.int64)
    )
    values = torch.from_numpy(H.data.astype(np.float32))
    return torch.sparse_coo_tensor(indices, values, torch.Size(H.shape))


def prepare_hypergraph(H, device):
    H = H.coalesce().to(device)

    DV = torch.sparse.sum(H, dim=1).to_dense()
    DE = torch.sparse.sum(H, dim=0).to_dense()

    DV_inv_sqrt = torch.pow(DV + 1e-8, -0.5)
    DE_inv = 1.0 / (DE + 1e-8)

    return {
        "H": H,
        "DV_inv_sqrt": DV_inv_sqrt.to(device),
        "DE_inv": DE_inv.to(device)
    }


def main(args):
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    data_dir = os.path.join(base_dir, "datasets", args.dataset)

    print(f"使用数据集: {args.dataset}")
    print(f"数据路径: {data_dir}")
    print(f"模型: {args.model}")

    if args.dataset == "cosmetics":
        emb_dim = 64
        dropout = 0.3
    elif args.dataset == "multi-category":
        emb_dim = 96
        dropout = 0.2
    else:
        emb_dim = args.emb_dim
        dropout = args.dropout

    with open(os.path.join(data_dir, 'item2idx.pkl'), 'rb') as f:
        item2idx = pickle.load(f)

    with open(os.path.join(data_dir, 'cat2idx.pkl'), 'rb') as f:
        cat2idx = pickle.load(f)

    hypergraphs = {}
    for name in ["session", "category", "time", "user"]:
        path = os.path.join(data_dir, f"hypergraph_{name}.npz")
        if os.path.exists(path):
            H = load_sparse(path)
            hypergraphs[name] = prepare_hypergraph(H, device)

    print(f" 超图加载完成: {list(hypergraphs.keys())}")

    # ===== DataLoader =====
    train_loader, val_loader, test_loader, _ = get_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=4
    )

    if args.model == "mtiesr":
        model = MTIESR(
            num_items=len(item2idx),
            num_cats=len(cat2idx),
            emb_dim=emb_dim,
            hgnn_dim=args.hgnn_dim,
            gru_dim=args.gru_dim,
            num_heads=args.num_heads,
            dropout=dropout,
            alpha=args.alpha,
            num_layers=args.num_layers
        )

    elif args.model == "iegt":
        from src.models.iegt import IEGT
        model = IEGT(
            num_items=len(item2idx),
            hidden_size=emb_dim
        )

    elif args.model == "stem":
        from src.models.stem import STEM
        model = STEM(
            num_items=len(item2idx),
            num_cats=len(cat2idx),
            emb_dim=emb_dim
        )

    elif args.model == "case4sr":
        from src.models.case4sr import CaSe4SR
        model = CaSe4SR(
            num_item=len(item2idx),
            num_cat=len(cat2idx)
        )

    else:
        raise ValueError("未知模型")

    model = model.to(device)

    print("模型初始化完成")

    use_hypergraph = args.model in ["mtiesr"]

    trainer = Trainer(
        model,
        device,
        hypergraphs if use_hypergraph else None
    )

    evaluator = Evaluator(topk_list=[10, 20])

    # ===== 训练 =====
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        evaluator=evaluator,
        epochs=args.epochs,
        lr=args.lr,
        save_path=f"save_model/{args.dataset}_{args.model}.pt"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ===== 模型选择 =====
    parser.add_argument("--model", type=str, default="mtiesr",
                        choices=["mtiesr", "iegt", "stem", "case4sr"])

    # ===== 数据集 =====
    parser.add_argument("--dataset", type=str, default="2019-oct")

    # ===== 模型参数 =====
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--gru_dim", type=int, default=256)
    parser.add_argument("--hgnn_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--num_layers", type=int, default=2)

    # ===== 训练参数 =====
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()

    main(args)
