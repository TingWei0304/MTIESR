import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import pickle
import os


class SessionDataset(Dataset):
    def __init__(self, filepath, item2idx_path, cat2idx_path, max_len=20):

        self.data = pd.read_csv(filepath)

        with open(item2idx_path, 'rb') as f:
            self.item2idx = pickle.load(f)

        with open(cat2idx_path, 'rb') as f:
            self.cat2idx = pickle.load(f)

        self.sessions = []
        self.targets = []
        self.cat_targets = []
        self.time_slice_seqs = []

        grouped = self.data.groupby('session_id')

        for _, group in grouped:
            group = group.dropna(subset=['next_item'])

            items = group['item_id'].tolist()
            cats = group['current_cat'].tolist()
            times = group['slice_id'].tolist()
            targets = group['next_item'].tolist()

            for i in range(1, len(items)):
                seq_items = items[:i][-max_len:]
                seq_cats = cats[:i][-max_len:]
                seq_times = times[:i][-max_len:]

                try:
                    seq_items_idx = [self.item2idx[int(x)] for x in seq_items]
                    seq_cats_idx = [self.cat2idx[int(x)] for x in seq_cats]

                    target_idx = self.item2idx[int(targets[i - 1])]
                    cat_target_idx = self.cat2idx[int(cats[i - 1])]

                except KeyError:
                    continue

                self.sessions.append((seq_items_idx, seq_cats_idx))
                self.time_slice_seqs.append(seq_times)
                self.targets.append(target_idx)
                self.cat_targets.append(cat_target_idx)

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        item_seq, cat_seq = self.sessions[idx]
        time_seq = self.time_slice_seqs[idx]

        return (
            torch.tensor(item_seq, dtype=torch.long),
            torch.tensor(cat_seq, dtype=torch.long),
            torch.tensor(self.targets[idx], dtype=torch.long),
            torch.tensor(self.cat_targets[idx], dtype=torch.long),
            len(item_seq),
            torch.tensor(time_seq, dtype=torch.long)
        )


def collate_fn(batch):

    item_seqs, cat_seqs, item_targets, cat_targets, lengths, time_ids = zip(*batch)

    max_len = max(lengths)

    padded_items = torch.zeros(len(batch), max_len, dtype=torch.long)
    padded_cats = torch.zeros(len(batch), max_len, dtype=torch.long)
    padded_times = torch.zeros(len(batch), max_len, dtype=torch.long)
    masks = torch.zeros(len(batch), max_len, dtype=torch.bool)

    for i, (item, cat, t_id, l) in enumerate(zip(item_seqs, cat_seqs, time_ids, lengths)):
        padded_items[i, :l] = item
        padded_cats[i, :l] = cat
        padded_times[i, :l] = t_id
        masks[i, :l] = True

    return (
        padded_items,
        padded_cats,
        torch.stack(item_targets),
        torch.stack(cat_targets),
        masks,
        padded_times
    )


def get_dataloaders(data_dir, batch_size=128, max_len=20, num_workers=4):

    item2idx_path = os.path.join(data_dir, 'item2idx.pkl')
    cat2idx_path = os.path.join(data_dir, 'cat2idx.pkl')

    train_set = SessionDataset(os.path.join(data_dir, 'train_multitask.txt'),
                               item2idx_path, cat2idx_path, max_len)

    val_set = SessionDataset(os.path.join(data_dir, 'val_multitask.txt'),
                             item2idx_path, cat2idx_path, max_len)

    test_set = SessionDataset(os.path.join(data_dir, 'test_multitask.txt'),
                              item2idx_path, cat2idx_path, max_len)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers
    )

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers
    )

    # ⚠️ 不再加载超图（统一由 run.py 负责）
    return train_loader, val_loader, test_loader, None