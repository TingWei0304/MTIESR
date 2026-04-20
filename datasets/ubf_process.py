import os
import pickle
import pandas as pd
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
import random

# ======================
# 路径配置
# ======================
DATA_DIR = "/root/Documents/wtt/second/datasets/ubf"

ITEM_PATH = os.path.join(DATA_DIR, "item_profile.csv")
BEHAVIOR_PATH = os.path.join(DATA_DIR, "user_item_behavior_history.csv")

SAVE_DIR = DATA_DIR

# ======================
# 过滤参数
# ======================
MIN_ITEM_FREQ = 5
MIN_SESSION_LEN = 3
SESSION_GAP = 30  # minutes

# ======================
# ⭐ 超图规模控制（关键！！）
# ======================
MAX_SESSION_EDGES = 50000
MAX_USER_EDGES = 30000
MAX_TIME_EDGES = 5000
MAX_CAT_EDGES = 20000

# ======================
# 1. 加载数据
# ======================
print("📦 加载数据...")

item_df = pd.read_csv(
    ITEM_PATH,
    header=None,
    names=["item_id", "category_id", "city", "tag"]
)

item_df = item_df[["item_id", "category_id"]].dropna()

behavior_df = pd.read_csv(
    BEHAVIOR_PATH,
    header=None,
    names=["user_id", "item_id", "behavior", "timestamp"]
)

behavior_df = behavior_df.drop_duplicates()
behavior_df = behavior_df[behavior_df["behavior"] == "clk"]

behavior_df["timestamp"] = pd.to_datetime(
    behavior_df["timestamp"], unit="s"
)

behavior_df = behavior_df.sort_values(["user_id", "timestamp"])

# ======================
# 2. Session划分
# ======================
print("🧩 构建Session...")

behavior_df["time_diff"] = (
    behavior_df.groupby("user_id")["timestamp"]
    .diff().dt.total_seconds().div(60)
)

behavior_df["new_session"] = (
    (behavior_df["time_diff"] > SESSION_GAP) |
    (behavior_df.groupby("user_id").cumcount() == 0)
)

behavior_df["session_id"] = (
    behavior_df.groupby("user_id")["new_session"]
    .cumsum().astype(str)
    + "_" + behavior_df["user_id"].astype(str)
)

# ======================
# 3. 合并商品信息
# ======================
print("🔗 合并商品信息...")

df = pd.merge(behavior_df, item_df, on="item_id", how="inner")

# ======================
# 4. 过滤低频 item
# ======================
print("🧹 过滤低频item...")

item_counts = df["item_id"].value_counts()
valid_items = item_counts[item_counts >= MIN_ITEM_FREQ].index
df = df[df["item_id"].isin(valid_items)]

# ======================
# 5. 过滤短session
# ======================
print("✂️ 过滤短Session...")

session_sizes = df.groupby("session_id").size()
valid_sessions = session_sizes[session_sizes >= MIN_SESSION_LEN].index
df = df[df["session_id"].isin(valid_sessions)]

# ======================
# 6. 编码
# ======================
print("🔢 编码...")

item2idx = {item: i for i, item in enumerate(df["item_id"].unique())}
cat2idx = {cat: i for i, cat in enumerate(df["category_id"].unique())}

df["item_idx"] = df["item_id"].map(item2idx)
df["cat_idx"] = df["category_id"].map(cat2idx)

# ======================
# 7. 构建序列数据
# ======================
print("📚 构建序列...")

sessions = defaultdict(list)
cats = defaultdict(list)
users = {}

for row in df.itertuples():
    sessions[row.session_id].append(row.item_idx)
    cats[row.session_id].append(row.cat_idx)
    users[row.session_id] = row.user_id

data = []

for sid in sessions:
    seq = sessions[sid]
    cat_seq = cats[sid]

    for i in range(1, len(seq)):
        data.append({
            "session": seq[:i],
            "target": seq[i],
            "cat_target": cat_seq[i]
        })

# ======================
# 8. 划分数据集
# ======================
print("📊 划分数据集...")

np.random.shuffle(data)

n = len(data)
train = data[:int(0.8 * n)]
val = data[int(0.8 * n):int(0.9 * n)]
test = data[int(0.9 * n):]

for name, d in zip(["train", "val", "test"], [train, val, test]):
    with open(os.path.join(SAVE_DIR, f"{name}.pkl"), "wb") as f:
        pickle.dump(d, f)

# ======================
# 9. 构建超图（核心优化）
# ======================
print("🌐 构建超图...")

num_items = len(item2idx)

def build_hypergraph(edges):
    rows, cols = [], []

    for eid, nodes in enumerate(edges):
        nodes = list(set(nodes))  # ⭐去重
        for n in nodes:
            rows.append(n)
            cols.append(eid)

    data = np.ones(len(rows))
    H = sp.coo_matrix((data, (rows, cols)), shape=(num_items, len(edges)))
    return H

# ===== session超图（采样）=====
session_edges = list(sessions.values())
if len(session_edges) > MAX_SESSION_EDGES:
    session_edges = random.sample(session_edges, MAX_SESSION_EDGES)

# ===== category超图 =====
cat_edges_dict = defaultdict(set)
for item, cat in zip(df["item_idx"], df["cat_idx"]):
    cat_edges_dict[cat].add(item)

cat_edges = [list(v) for v in cat_edges_dict.values()]
if len(cat_edges) > MAX_CAT_EDGES:
    cat_edges = cat_edges[:MAX_CAT_EDGES]

# ===== user超图 =====
user_edges_dict = defaultdict(set)
for sid in sessions:
    user_edges_dict[users[sid]].update(sessions[sid])

user_edges = [list(v) for v in user_edges_dict.values()]
if len(user_edges) > MAX_USER_EDGES:
    user_edges = random.sample(user_edges, MAX_USER_EDGES)

# ===== time超图 =====
df["time_bin"] = df["timestamp"].dt.hour // 4

time_edges_dict = defaultdict(set)
for row in df.itertuples():
    time_edges_dict[row.time_bin].add(row.item_idx)

time_edges = [list(v) for v in time_edges_dict.values()]
if len(time_edges) > MAX_TIME_EDGES:
    time_edges = time_edges[:MAX_TIME_EDGES]

# ===== 保存超图 =====
graphs = {
    "session": session_edges,
    "category": cat_edges,
    "user": user_edges,
    "time": time_edges
}

for name, edges in graphs.items():
    print(f"👉 {name} edges: {len(edges)}")
    H = build_hypergraph(edges)
    sp.save_npz(os.path.join(SAVE_DIR, f"hypergraph_{name}.npz"), H)

# ======================
# 10. 保存映射
# ======================
with open(os.path.join(SAVE_DIR, "item2idx.pkl"), "wb") as f:
    pickle.dump(item2idx, f)

with open(os.path.join(SAVE_DIR, "cat2idx.pkl"), "wb") as f:
    pickle.dump(cat2idx, f)

print("🎉 数据处理 + 超图构建 完成！")