# val_loader, clip_model, preprocess are assumed to be defined in the main script

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.extract_clip_embeddings import extract_clip_embeddings
from src.flatimagedataset import FlatImageDataset

import random
import torch
from torch import device
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import matplotlib.pyplot as plt



class FlatImageDataset(Dataset):
    """Loads images from a flat directory for CLIP embedding extraction."""
    def __init__(self, root_dir, preprocess):
        self.root_dir   = root_dir
        self.preprocess = preprocess
        self.image_paths = []
        self.image_names = []
        for filename in os.listdir(root_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                self.image_paths.append(os.path.join(root_dir, filename))
                self.image_names.append(filename)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image    = Image.open(img_path).convert('RGB')
        return self.preprocess(image), self.image_names[idx]

def extract_clip_embeddings(folder_or_loader, model, preprocess=None,
                             batch_size=64, is_loader=False):
    """Extract L2-normalised CLIP embeddings from a folder or DataLoader."""
    model.eval()
    all_embs, all_meta = [], []

    loader = folder_or_loader if is_loader else DataLoader(
        FlatImageDataset(folder_or_loader, preprocess),
        batch_size=batch_size, num_workers=2, pin_memory=True
    )

    with torch.no_grad():
        for imgs, meta in loader:
            imgs = imgs.to(device)
            embs = model.encode_image(imgs).float()
            embs = embs / embs.norm(dim=-1, keepdim=True)
            all_embs.append(embs.cpu())
            all_meta.extend(meta if isinstance(meta[0], str) else meta.tolist())

    return torch.cat(all_embs), all_meta


def generate_submission(query_dir, gallery_dir, model, preprocess, top_k=10):
    """Full retrieval pipeline: embed → cosine similarity → rank → dict."""
    print('Extracting query embeddings...')
    q_embs, q_names = extract_clip_embeddings(query_dir, model, preprocess)
    print('Extracting gallery embeddings...')
    g_embs, g_names = extract_clip_embeddings(gallery_dir, model, preprocess)

    sim = torch.mm(q_embs, g_embs.t())

    res = {}
    for i, qname in enumerate(q_names):
        _, top_idx = torch.topk(sim[i], k=top_k)
        res[qname]  = [g_names[j] for j in top_idx.tolist()]
    return res


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


# ── Zero-shot validation on LFW val set (skipped if LFW not loaded) ──────────
if val_loader is not None:
    print('Running zero-shot evaluation on LFW val set...')
    val_embs, val_labels = extract_clip_embeddings(val_loader, clip_model, is_loader=True)
    sim_matrix = torch.mm(val_embs, val_embs.t())
    sim_matrix.fill_diagonal_(-1)
    top1_idx     = sim_matrix.argmax(dim=1)
    val_labels_t = torch.tensor(val_labels)
    top1_acc     = (val_labels_t[top1_idx] == val_labels_t).float().mean().item()
    print(f'Zero-shot Top-1 accuracy: {top1_acc * 100:.2f}%')

    print('LFW not loaded — skipping zero-shot LFW eval. Functions are ready.')