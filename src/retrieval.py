import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.datasets import FlatImageDataset
from src.extract_clip_embeddings import extract_clip_embeddings

import random

import torch
from torch.utils.data import DataLoader
import PIL.Image as PILImage 

import matplotlib.pyplot as plt

def evaluate_retrieval(clip_model, loader, device):
    embs, labels = extract_clip_embeddings(loader, clip_model, is_loader=True)
    sim = torch.mm(embs, embs.t())
    sim.fill_diagonal_(-1)
    labels_t      = torch.tensor(labels)
    top1_acc      = (labels_t[sim.argmax(dim=1)] == labels_t).float().mean().item()
    top10_correct = (labels_t[sim.topk(10, dim=1).indices] == labels_t.unsqueeze(1)).any(dim=1)
    top10_acc     = top10_correct.float().mean().item()
    return top1_acc, top10_acc


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

        # FIXED HERE: Changed Image.open to PILImage.open
        q_img = PILImage.open(os.path.join(query_dir, q_names[q_idx])).convert('RGB')
        axes[row][0].imshow(q_img)
        axes[row][0].set_title(f'QUERY\n{q_names[q_idx][:15]}', fontsize=8)
        axes[row][0].axis('off')

        for col, g_idx in enumerate(top_idx.tolist()):
            # FIXED HERE: Changed Image.open to PILImage.open
            g_img = PILImage.open(os.path.join(gallery_dir, g_names[g_idx])).convert('RGB')
            score = sim[q_idx][g_idx].item()
            axes[row][col+1].imshow(g_img)
            axes[row][col+1].set_title(f'#{col+1} sim={score:.2f}\n{g_names[g_idx][:15]}', fontsize=7)
            axes[row][col+1].axis('off')

    plt.suptitle('Query → Top-K Retrieved Gallery Images', fontsize=12)
    plt.tight_layout()
    plt.show()

