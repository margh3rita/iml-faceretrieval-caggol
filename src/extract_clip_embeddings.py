import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.flatimagedataset import FlatImageDataset

import torch
# from torch import device
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  




def extract_clip_embeddings(folder_or_loader, model, preprocess=None,
                             batch_size=64, is_loader=False, device=device):
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


