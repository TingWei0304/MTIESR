import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
import hashlib
import pickle
from collections import defaultdict
from sklearn.model_selection import train_test_split
from datetime import timedelta
from scipy.special import softmax
import math


# --------------------------
# 哈希类别用于稳定类别划分
# --------------------------
def stable_hash(cat_id, num_bins=100, salt="hypergcn_salt"):
    hash_obj = hashlib.sha256(f"{salt}_{cat_id}".encode())
    return int(hash_obj.hexdigest(), 16) % num_bins + 1

def sample_sessions(df, sample_ratio=0.1, random_seed=42):
    """按会话分层采样，保持会话完整性"""
    unique_sessions = df['session_id'].unique()
    sampled_sessions = pd.Series(unique_sessions).sample(
        frac=sample_ratio,
        random_state=random_seed
    )
    return df[df['session_id'].isin(sampled_sessions)]


# --------------------------
# 加载数据 + 哈希类别 + 排序
# --------------------------
def load_data(file_path, sep=',',
              sample_ratio=0.1,  # 新增采样比例
              min_freq=5,
              min_session_len=2,
              max_session_len=50):
    # 加载全量数据
    df = pd.read_csv(file_path, sep=sep, parse_dates=['timestamp'])

    # 0. 先执行会话采样（减少数据规模）
    if sample_ratio < 1.0:
        df = sample_sessions(df, sample_ratio=sample_ratio)
        print(f"📦 采样后数据量: {len(df)} 行 (原始数据的 {sample_ratio:.0%})")

    # 1. 过滤低频item和category
    item_counts = df['item_id'].value_counts()
    valid_items = item_counts[item_counts >= min_freq].index
    df = df[df['item_id'].isin(valid_items)]

    cat_counts = df['category_id'].value_counts()
    valid_cats = cat_counts[cat_counts >= min_freq].index
    df = df[df['category_id'].isin(valid_cats)]

    # 2. 过滤会话长度
    df = df.groupby('session_id').filter(
        lambda x: min_session_len <= len(x) <= max_session_len
    )

    # 3. 哈希类别
    df["hashed_cat"] = df["category_id"].apply(lambda x: stable_hash(x, num_bins=100))

    # 排序
    df = df.sort_values('timestamp').reset_index(drop=True)

    # 打印统计
    print(f"✅ 最终有效数据量: {len(df)} 行")
    print(f"🔢 唯一会话: {df['session_id'].nunique()}")
    print(f"🔢 唯一物品: {df['item_id'].nunique()}, 唯一类别: {df['category_id'].nunique()}")
    print(f"📏 平均会话长度: {df.groupby('session_id').size().mean():.1f}")
    return df


# --------------------------
# Session 级别划分 train/val/test（时间无泄漏）
# --------------------------
def split_dataset(df, output_dir, test_size=0.2, val_size=0.1):
    # 划分会话ID
    session_ids = df['session_id'].unique()
    train_sessions, test_sessions = train_test_split(session_ids, test_size=test_size, random_state=42)
    train_sessions, val_sessions = train_test_split(train_sessions, test_size=val_size / (1 - test_size),
                                                    random_state=42)

    # 获取训练集物品
    train_items = df[df['session_id'].isin(train_sessions)]['item_id'].unique()

    # 过滤测试集和验证集
    test = df[df['session_id'].isin(test_sessions)].query("item_id in @train_items")
    val = df[df['session_id'].isin(val_sessions)].query("item_id in @train_items")
    train = df[df['session_id'].isin(train_sessions)]

    # 保存文件
    os.makedirs(output_dir, exist_ok=True)
    train.to_csv(os.path.join(output_dir, "train.txt"), sep="\t", index=False, header=False)
    val.to_csv(os.path.join(output_dir, "val.txt"), sep="\t", index=False, header=False)
    test.to_csv(os.path.join(output_dir, "test.txt"), sep="\t", index=False, header=False)

    print(f"📊 数据集统计:")
    print(f"- Train: {len(train)} 行, {len(train_sessions)} 会话")
    print(f"- Val  : {len(val)} 行, {len(val_sessions)} 会话")
    print(f"- Test : {len(test)} 行, {len(test_sessions)} 会话")
    print(f"- 测试集物品覆盖率: {test['item_id'].nunique() / len(train_items):.2%}")
    return train, val, test


# --------------------------
# 建立 item 和 category 映射
# --------------------------
def build_mappings(df):
    item2idx = {item: i for i, item in enumerate(df['item_id'].unique())}
    cat2idx = {cat: i for i, cat in enumerate(df['hashed_cat'].unique())}
    print(f"🔢 Unique items: {len(item2idx)}, categories: {len(cat2idx)}")
    return item2idx, cat2idx


# --------------------------
# 添加时间切片编号
# --------------------------
def add_time_slices(df, num_time_slices=8):
    start_time, end_time = df['timestamp'].min(), df['timestamp'].max()
    total_seconds = (end_time - start_time).total_seconds()
    time_unit = total_seconds / num_time_slices
    df['slice_id'] = df['timestamp'].apply(lambda t: int((t - start_time).total_seconds() / time_unit))
    df['slice_id'] = df['slice_id'].clip(0, num_time_slices - 1)
    return df


# --------------------------
# 多任务监督标签生成：预测下一物品 & 当前类别
# --------------------------
def generate_multi_task_labels(df):
    df = df.sort_values(['session_id', 'timestamp'])
    df['next_item'] = df.groupby('session_id')['item_id'].shift(-1)
    df['current_cat'] = df['hashed_cat']
    print("🧾 Multi-task label sample:\n", df[['item_id', 'current_cat', 'next_item']].dropna().head())
    return df[['session_id', 'user_id', 'item_id', 'current_cat', 'next_item', 'timestamp', 'slice_id']]


# --------------------------
# 构建多层次超边（时间切片、会话、类别、用户高频）
# --------------------------
def build_hyperedges(df, item2idx, cat2idx, num_time_slices=8):
    df = df.copy()
    df['item_idx'] = df['item_id'].map(item2idx)
    df['cat_idx'] = df['hashed_cat'].map(cat2idx)
    df = add_time_slices(df, num_time_slices=num_time_slices)

    hyperedges_dict = defaultdict(list)
    weights_dict = defaultdict(list)

    # 时间切片超边
    time_hyperedges = []
    time_weights = []
    for slice_id, group in df.groupby('slice_id'):
        items = group['item_idx'].unique().tolist()
        if len(items) >= 2:
            weight = 1.0 / math.log(slice_id + 2)
            time_hyperedges.append(items)
            time_weights.append(weight)
    hyperedges_dict['time'] = time_hyperedges
    weights_dict['time'] = softmax(time_weights)

    # 会话超边
    session_hyperedges = []
    for _, group in df.groupby('session_id'):
        items = group['item_idx'].unique().tolist()
        if len(items) >= 2:
            session_hyperedges.append(items)
    hyperedges_dict['session'] = session_hyperedges
    weights_dict['session'] = [1.0] * len(session_hyperedges)

    # 类别超边
    cat_hyperedges = []
    for (cat_idx, _), group in df.groupby(['cat_idx', 'session_id']):
        items = group['item_idx'].unique().tolist()
        if len(items) >= 2:
            cat_hyperedges.append(items)
    hyperedges_dict['category'] = cat_hyperedges
    weights_dict['category'] = [1.0] * len(cat_hyperedges)

    # 用户超边
    user_hyperedges = []
    for user_id, group in df.groupby('user_id'):
        top_items = group['item_idx'].value_counts().nlargest(5).index.tolist()
        if len(top_items) >= 3:
            user_hyperedges.append(top_items)
    hyperedges_dict['user'] = user_hyperedges
    weights_dict['user'] = [1.0] * len(user_hyperedges)

    # 打印统计信息
    for key in hyperedges_dict:
        print(f"🧩 {key}超边数量: {len(hyperedges_dict[key])}")
        print(f"  示例（前3个）: {hyperedges_dict[key][:3]}")

    return hyperedges_dict, weights_dict


# --------------------------
# 保存超图稀疏矩阵 H
# --------------------------
def save_hypergraph_by_type(hyperedges_dict, weights_dict, item_size, output_dir):
    for hyper_type in hyperedges_dict:
        hyperedges = hyperedges_dict[hyper_type]
        weights = weights_dict[hyper_type]

        row, col, data = [], [], []
        for eid, (edge, weight) in enumerate(zip(hyperedges, weights)):
            for item in edge:
                row.append(item)
                col.append(eid)
                data.append(weight)

        H = sp.coo_matrix((data, (row, col)), shape=(item_size, len(hyperedges)))
        output_path = os.path.join(output_dir, f"hypergraph_{hyper_type}.npz")
        sp.save_npz(output_path, H)
        print(f"✅ Saved {hyper_type} hypergraph ({H.shape}) to {output_path}")


# --------------------------
# 全流程入口函数
# --------------------------
def preprocess_and_save(input_path, output_dir):
    df = load_data(input_path)
    df = add_time_slices(df)
    train, val, test = split_dataset(df, output_dir)
    item2idx, cat2idx = build_mappings(pd.concat([train, val]))

    for split_df, name in [(train, 'train'), (val, 'val'), (test, 'test')]:
        labeled_df = generate_multi_task_labels(split_df)
        labeled_df.to_csv(f"{output_dir}/{name}_multitask.txt", index=False)

    all_df = pd.concat([train, val])
    hyperedges_dict, weights_dict = build_hyperedges(all_df, item2idx, cat2idx)

    os.makedirs(output_dir, exist_ok=True)

    save_hypergraph_by_type(hyperedges_dict, weights_dict, len(item2idx), output_dir)

    item2cat = {}
    for item_id, cat in df[['item_id', 'hashed_cat']].drop_duplicates(subset='item_id').values:
        if item_id in item2idx:  # 确保 item_id 在 item2idx 中
            item_idx = item2idx[item_id]
            item2cat[item_idx] = cat2idx[cat]  # 值也转换为 cat2idx 的索引

    # 保存 item2cat.pkl
    with open(f"{output_dir}/item2cat.pkl", 'wb') as f:
        pickle.dump(item2cat, f)
    print(list(item2cat.items())[:10])
    print(f"✅ Saved item2cat mapping (size: {len(item2cat)})")

    with open(f"{output_dir}/item2idx.pkl", 'wb') as f:
        pickle.dump(item2idx, f)
    with open(f"{output_dir}/cat2idx.pkl", 'wb') as f:
        pickle.dump(cat2idx, f)
    with open(f"{output_dir}/idx2item.pkl", 'wb') as f:
        pickle.dump({v: k for k, v in item2idx.items()}, f)
    with open(f"{output_dir}/idx2cat.pkl", 'wb') as f:
        pickle.dump({v: k for k, v in cat2idx.items()}, f)

    total_hyperedges = sum(len(edges) for edges in hyperedges_dict.values())
    print("✅ Preprocessing finished.")
    print(f"Train/Val/Test samples: {len(train)}, {len(val)}, {len(test)}")
    print(f"Total hyperedges: {total_hyperedges} (分类型统计: { {k: len(v) for k, v in hyperedges_dict.items()} })")
    print(f"Total items: {len(item2idx)}")
    print(f"Output saved to: {output_dir}")


# --------------------------
# 执行主函数
# --------------------------
if __name__ == "__main__":
    preprocess_and_save(
        input_path="../../datasets/mul-cate/mul-cate.txt",
        output_dir="../../datasets/mul-cate/",
    )

