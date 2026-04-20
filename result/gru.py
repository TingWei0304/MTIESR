import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
})

gru_dims = [64, 128, 192, 256, 320, 384]

# ===== HR =====
cosmetics_hr  = [67.1, 68.7, 69.6, 69.9, 68.9, 67.6]
multi_hr      = [63.9, 65.8, 66.9, 67.6, 67.2, 66.8]
ubf_hr        = [66.5, 67.6, 68.5, 69.3, 69.8, 69.5]

# ===== MRR =====
cosmetics_mrr = [29.5, 31.0, 31.8, 32.2, 31.5, 30.6]
multi_mrr     = [37.6, 38.7, 39.3, 39.9, 39.6, 39.1]
ubf_mrr       = [45.1, 45.9, 46.8, 47.4, 47.9, 47.6]

# ===== HR 图 =====
plt.figure(figsize=(6, 5))
plt.plot(gru_dims, cosmetics_hr, marker='o', label='Cosmetics')
plt.plot(gru_dims, multi_hr, marker='s', label='Multi-category')
plt.plot(gru_dims, ubf_hr, marker='^', label='UBF')
plt.xlabel('GRU Hidden Size')
plt.ylabel('HR@10')
plt.grid(True)
plt.xticks(gru_dims)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
           ncol=3, frameon=False)
plt.tight_layout()
plt.savefig('gru_hr10_final.png', dpi=500)
plt.show()

# ===== MRR 图 =====
plt.figure(figsize=(6, 5))
plt.plot(gru_dims, cosmetics_mrr, marker='o', label='Cosmetics')
plt.plot(gru_dims, multi_mrr, marker='s', label='Multi-category')
plt.plot(gru_dims, ubf_mrr, marker='^', label='UBF')
plt.xlabel('GRU Hidden Size')
plt.ylabel('MRR@10')
plt.grid(True)
plt.xticks(gru_dims)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
           ncol=3, frameon=False)
plt.tight_layout()
plt.savefig('gru_mrr10_final.png', dpi=500)
plt.show()