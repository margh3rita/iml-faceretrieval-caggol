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

    import os
import random
import matplotlib.pyplot as plt
import PIL.Image as PILImage

def visualise_test_submission(query_dir, gallery_dir, submission_dict, num_queries=3, top_k=5):
    """
    Visualises submission dictionary mapping by looking up actual files
    inside the final query_dir and gallery_dir paths.
    """
    # Pick a few sample queries from the submission dictionary keys
    q_names = list(submission_dict.keys())
    if not q_names:
        print("Submission dictionary is empty! Run the generation block first.")
        return

    q_sample = random.sample(q_names, min(num_queries, len(q_names)))

    # Construct plotting grid
    fig, axes = plt.subplots(len(q_sample), top_k + 1,
                             figsize=(3 * (top_k + 1), 3 * len(q_sample)))

    # Handle single row formatting edge case
    if num_queries == 1 or len(q_sample) == 1:
        axes = [axes]

    for row, q_name in enumerate(q_sample):
        # 1. Render original test Query image
        q_path = os.path.join(query_dir, q_name)
        if os.path.exists(q_path):
            q_img = PILImage.open(q_path).convert('RGB')
            axes[row][0].imshow(q_img)
        axes[row][0].set_title(f'QUERY\n{q_name[:15]}', fontsize=8, color='blue')
        axes[row][0].axis('off')

        # 2. Render ranked predictions returned by the model
        retrieved_list = submission_dict[q_name]
        for col in range(top_k):
            ax = axes[row][col + 1]
            if col < len(retrieved_list):
                g_name = retrieved_list[col]
                g_path = os.path.join(gallery_dir, g_name)

                if os.path.exists(g_path):
                    g_img = PILImage.open(g_path).convert('RGB')
                    ax.imshow(g_img)
                ax.set_title(f'Rank #{col+1}\n{g_name[:15]}', fontsize=7)
            ax.axis('off')

    plt.suptitle('Submission Preview: Target Queries → Top Predicted Gallery Matches', fontsize=12, weight='bold')
    plt.tight_layout()
    plt.show()
