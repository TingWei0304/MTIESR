import torch
import numpy as np

def to_tensor(data, device):
    """ 将数据转换为 PyTorch 张量，并移动到指定设备 """
    if isinstance(data, np.ndarray):
        return torch.tensor(data, dtype=torch.float32, device=device)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    return data

def normalize_hypergraph_adj(adj):
    """
    归一化超图邻接矩阵
    - adj: 超图邻接矩阵 (torch.Tensor)
    """
    rowsum = adj.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    return adj * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]

def fuse_interest(short_term, long_term, alpha=0.5):
    """
    融合短期兴趣和长期兴趣
    - short_term: GRU/LSTM 输出的短期兴趣向量
    - long_term: Transformer 提取的长期兴趣向量
    - alpha: 融合权重（可学习）
    """
    return alpha * short_term + (1 - alpha) * long_term

def save_model(model, path):
    """ 保存模型 """
    torch.save(model.state_dict(), path)

def load_model(model, path, device):
    """ 加载模型 """
    model.load_state_dict(torch.load(path, map_location=device))
    return model

def set_seed(seed=42):
    """ 设置随机种子，保证实验可复现 """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
