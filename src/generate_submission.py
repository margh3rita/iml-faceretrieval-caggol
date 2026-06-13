# val_loader, clip_model, preprocess are assumed to be defined in the main script

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.extract_clip_embeddings import extract_clip_embeddings


import torch


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