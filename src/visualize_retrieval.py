import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.flatimagedataset import FlatImageDataset
from src.extract_clip_embeddings import extract_clip_embeddings

import random

import torch
from torch.utils.data import DataLoader
from PIL import Image

import matplotlib.pyplot as plt


def visualise_retrieval(query_dir, gallery_dir, model, preprocess,
                         num_queries=3, top_k=5):
    """Shows query image + top-k retrieved gallery images side by side."""
    q_loader = DataLoader(FlatImageDataset(query_dir,   preprocess), batch_size=64)
    g_loader = DataLoader(FlatImageDataset(gallery_dir, preprocess), batch_size=64)

    q_embs, q_names = extract_clip_embeddings(q_loader, model, is_loader=True)
    g_embs, g_names = extract_clip_embeddings(g_loader, model, is_loader=True)

    sim      = torch.mm(q_embs, g_embs.t())
    q_sample = random.sample(range(len(q_names)), min(num_queries, len(q_names)))

    fig, axes = plt.subplots(len(q_sample), top_k + 1,
                              figsize=(3 * (top_k + 1), 3 * len(q_sample)))
    if len(q_sample) == 1:
        axes = [axes]

    for row, q_idx in enumerate(q_sample):
        _, top_idx = torch.topk(sim[q_idx], k=top_k)
        q_img = Image.open(os.path.join(query_dir, q_names[q_idx])).convert('RGB')
        axes[row][0].imshow(q_img)
        axes[row][0].set_title(f'QUERY\n{q_names[q_idx][:15]}', fontsize=8)
        axes[row][0].axis('off')
        for col, g_idx in enumerate(top_idx.tolist()):
            g_img = Image.open(os.path.join(gallery_dir, g_names[g_idx])).convert('RGB')
            score = sim[q_idx][g_idx].item()
            axes[row][col+1].imshow(g_img)
            axes[row][col+1].set_title(f'#{col+1} sim={score:.2f}\n{g_names[g_idx][:15]}', fontsize=7)
            axes[row][col+1].axis('off')

    plt.suptitle('Query → Top-K Retrieved Gallery Images', fontsize=12)
    plt.tight_layout()
    plt.show()