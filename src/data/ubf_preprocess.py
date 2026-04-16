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

def sample_sessions(df, sample_ratio=0.1, random_state=42):
    """按会话ID分层采样，保留完整会话"""
    unique_sessions = df['session_id'].unique()
    sampled_sessions = pd.Series(unique_sessions).sample(
        frac=sample_ratio,
        random_state=random_state
    )
    return df[df['session_id'].isin(sampled_sessions)]
# --------------------------
# 加载数据 + 原始类别统计 + 哈希映射
# --------------------------
def load_data(file_path, sep=',', hash_ratio=1.5,
             min_item_freq=5,
             days_limit=30,
             sample_ratio=1.0,
             random_state=42):
    # 加载原始数据
    df = pd.read_csv(file_path, sep=sep, parse_dates=['timestamp'])
    print(f"📦 原始数据量: {len(df)}")

    # 1. 先限制时间窗口
    max_time = df['timestamp'].max()
    min_time = max_time - timedelta(days=days_limit)
    df = df[df['timestamp'] >= min_time]
    print(f"⏱ 时间窗口过滤: 最近 {days_limit} 天 ({min_time.date()} ~ {max_time.date()}), 剩余 {len(df)} 条记录")

    # 2. 再执行会话采样
    if sample_ratio < 1.0:
        df = sample_sessions(df, sample_ratio=sample_ratio, random_state=random_state)
        print(f"🎯 会话采样比例: {sample_ratio:.0%} → 剩余数据量: {len(df)}")

    # 3. 最后过滤低频物品
    item_freq = df['item_id'].value_counts()
    frequent_items = item_freq[item_freq >= min_item_freq].index
    df = df[df['item_id'].isin(frequent_items)]
    print(f"🔍 保留频次 ≥{min_item_freq} 的物品: {len(frequent_items)} 个，最终数据量: {len(df)}")

    # 哈希处理
    original_cat_count = df["category_id"].nunique()
    num_bins = max(int(original_cat_count * hash_ratio), 500)
    df["hashed_cat"] = df["category_id"].apply(lambda x: stable_hash(x, num_bins=num_bins))
    df = df.sort_values('timestamp').reset_index(drop=True)

    cat_dist = df['hashed_cat'].value_counts(normalize=True)
    print("\n🔎 类别分布分析:")
    print(f"- 总哈希类别数: {len(cat_dist)}")
    print(f"- Top-3类别占比: {cat_dist.head(3).to_dict()}")
    print(f"- 长尾系数 (基尼系数): {gini_coefficient(cat_dist.values):.3f}")

    # 2. 物品热度分布验证
    item_counts = df['item_id'].value_counts()
    print("\n🔎 物品热度分布:")
    print(f"- 总物品数: {len(item_counts)}")
    print(f"- 前1%物品占比: {item_counts.head(int(len(item_counts) * 0.01)).sum() / len(df):.2%}")
    print(f"- 长尾系数 (基尼系数): {gini_coefficient(item_counts.values):.3f}")

    # 3. 会话长度分析
    session_lengths = df.groupby('session_id').size()
    print("\n🔎 会话长度分布:")
    print(f"- 平均长度: {session_lengths.mean():.1f}")
    print(f"- 最大长度: {session_lengths.max()}")
    print(f"- 长度分布分位数:\n{session_lengths.quantile([0.1, 0.5, 0.9])}")

    return df


def gini_coefficient(values):
    sorted_values = np.sort(values)
    n = len(sorted_values)
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * sorted_values)) / (n * np.sum(sorted_values))

def split_dataset(df, output_dir, test_size=0.2, val_size=0.1):
    session_ids = df['session_id'].unique()
    val_ratio = val_size / (1 - test_size)

    train_sessions, test_sessions = train_test_split(session_ids, test_size=test_size, random_state=42)
    train_sessions, val_sessions = train_test_split(train_sessions, test_size=val_ratio, random_state=42)

    train = df[df['session_id'].isin(train_sessions)]
    val = df[df['session_id'].isin(val_sessions)]
    test = df[df['session_id'].isin(test_sessions)]

    os.makedirs(output_dir, exist_ok=True)
    train.to_csv(os.path.join(output_dir, "train.txt"), sep="\t", index=False, header=False)
    val.to_csv(os.path.join(output_dir, "val.txt"), sep="\t", index=False, header=False)
    test.to_csv(os.path.join(output_dir, "test.txt"), sep="\t", index=False, header=False)

    print(f"📁 Split sessions -> Train: {len(train_sessions)}, Val: {len(val_sessions)}, Test: {len(test_sessions)}")
    print(f"🕒 Train time: {train['timestamp'].min()} ~ {train['timestamp'].max()}")
    print(f"🕒 Val time  : {val['timestamp'].min()} ~ {val['timestamp'].max()}")
    print(f"🕒 Test time : {test['timestamp'].min()} ~ {test['timestamp'].max()}")
    return train, val, test


def build_mappings(df):
    item_counts = df['item_id'].value_counts()
    item2idx = {item: i for i, item in enumerate(item_counts.index)}
    cat2idx = {cat: i for i, cat in enumerate(df['hashed_cat'].unique())}
    print(f"🔢 Unique items: {len(item2idx)}, categories: {len(cat2idx)}")
    return item2idx, cat2idx


def add_time_slices(df, num_time_slices=8):
    start_time, end_time = df['timestamp'].min(), df['timestamp'].max()
    total_seconds = (end_time - start_time).total_seconds()
    time_unit = total_seconds / num_time_slices
    df['slice_id'] = df['timestamp'].apply(lambda t: int((t - start_time).total_seconds() / time_unit))
    df['slice_id'] = df['slice_id'].clip(0, num_time_slices - 1)
    return df


def generate_multi_task_labels(df):
    df = df.sort_values(['session_id', 'timestamp'])
    df['next_item'] = df.groupby('session_id')['item_id'].shift(-1)
    df['current_cat'] = df['hashed_cat']
    print("🧾 Multi-task label sample:\n", df[['item_id', 'current_cat', 'next_item']].dropna().head())
    return df[['session_id', 'user_id', 'item_id', 'current_cat', 'next_item', 'timestamp', 'slice_id']]


def build_hyperedges(df, item2idx, cat2idx, num_time_slices=8):
    df = df.copy()
    df['item_idx'] = df['item_id'].map(item2idx)
    df['cat_idx'] = df['hashed_cat'].map(cat2idx)
    df = add_time_slices(df, num_time_slices=num_time_slices)

    hyperedges_dict = defaultdict(list)
    weights_dict = defaultdict(list)

    for slice_id, group in df.groupby('slice_id'):
        items = group['item_idx'].unique().tolist()
        if len(items) >= 2:
            weight = 1.0 / math.log(slice_id + 2)
            hyperedges_dict['time'].append(items)
            weights_dict['time'].append(weight)
    weights_dict['time'] = softmax(weights_dict['time'])

    for _, group in df.groupby('session_id'):
        items = group['item_idx'].unique().tolist()
        if len(items) >= 2:
            hyperedges_dict['session'].append(items)
    weights_dict['session'] = [1.0] * len(hyperedges_dict['session'])

    for (cat_idx, _), group in df.groupby(['cat_idx', 'session_id']):
        items = group['item_idx'].unique().tolist()
        if len(items) >= 2:
            hyperedges_dict['category'].append(items)
    weights_dict['category'] = [1.0] * len(hyperedges_dict['category'])

    for user_id, group in df.groupby('user_id'):
        top_items = group['item_idx'].value_counts().nlargest(5).index.tolist()
        if len(top_items) >= 3:
            hyperedges_dict['user'].append(top_items)
    weights_dict['user'] = [1.0] * len(hyperedges_dict['user'])

    for key in hyperedges_dict:
        print(f"🧩 {key}超边数量: {len(hyperedges_dict[key])}")
        print(f"  示例（前3个）: {hyperedges_dict[key][:3]}")

    return hyperedges_dict, weights_dict


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


def preprocess_and_save(input_path, output_dir,
                       hash_ratio=1.5,
                       min_item_freq=5,
                       days_limit=30,
                       sample_ratio=0.1,    # 新增采样参数
                       random_state=42):    # 新增随机种子
    # 将参数传递到 load_data
    df = load_data(
        input_path,
        hash_ratio=hash_ratio,
        min_item_freq=min_item_freq,
        days_limit=days_limit,
        sample_ratio=sample_ratio,       # 传递采样比例
        random_state=random_state        # 传递随机种子
    )
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
        if item_id in item2idx:
            item_idx = item2idx[item_id]
            item2cat[item_idx] = cat2idx[cat]

    with open(f"{output_dir}/item2cat.pkl", 'wb') as f:
        pickle.dump(item2cat, f)
    with open(f"{output_dir}/item2idx.pkl", 'wb') as f:
        pickle.dump(item2idx, f)
    with open(f"{output_dir}/cat2idx.pkl", 'wb') as f:
        pickle.dump(cat2idx, f)
    with open(f"{output_dir}/idx2item.pkl", 'wb') as f:
        pickle.dump({v: k for k, v in item2idx.items()}, f)
    with open(f"{output_dir}/idx2cat.pkl", 'wb') as f:
        pickle.dump({v: k for k, v in cat2idx.items()}, f)

    print("✅ Preprocessing finished.")
    print(f"Train/Val/Test samples: {len(train)}, {len(val)}, {len(test)}")
    print(f"Total hyperedges: {sum(len(edges) for edges in hyperedges_dict.values())}")
    print(f"Total items: {len(item2idx)}")
    print(f"Output saved to: {output_dir}")


# --------------------------
# 执行主函数
# --------------------------
if __name__ == "__main__":
    preprocess_and_save(
        input_path="../../datasets/ubf/ubf_process.csv",
        output_dir="../../datasets/ubf/",
        hash_ratio=1.5,
        min_item_freq=4,
        days_limit=1500,
        sample_ratio=0.1,
        random_state=42
    )
