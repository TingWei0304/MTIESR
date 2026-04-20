# MTIESR
# MTIESR: Multi-Task Interest Evolution Hypergraph Model

## Overview

This repository provides the official implementation of our paper:

**“Multi-Task Learning based Interest Evolution Hypergraph Model for Session-based Recommendation”**

We propose:

* Multi-type hypergraph modeling (session, category, time, user)
* Interest evolution modeling via temporal hyperedges
* Multi-task learning (item + category prediction)

---

## Reproducibility Checklist (ACM TORS)

* ✔ Fixed random seed (42)
* ✔ Deterministic training (cudnn disabled benchmark)
* ✔ Full preprocessing scripts provided
* ✔ Exact dataset splits reproducible
* ✔ All hyperparameters reported
* ✔ One-command training pipeline

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

```bash
bash scripts/preprocess.sh
```

---

## Training

```bash
bash scripts/train.sh
```

---

## Evaluation

```bash
bash scripts/eval.sh
```

---

##  Hyperparameters

| Parameter     | Value |
| ------------- | ----- |
| Embedding Dim | 256   |
| GRU Dim       | 256   |
| HGNN Dim      | 256   |
| Heads         | 6     |
| Dropout       | 0.1   |
| LR            | 1e-4  |

---

## Special Issue Relevance

This work relates to modern recommender systems challenges:

* Hypergraph modeling → complex relational structures
* Temporal slices → dynamic preference evolution
* Multi-task learning → joint semantic modeling

---

## Notes

* All experiments are reproducible using provided scripts.
* Data preprocessing follows session-level splitting without leakage.

---
