import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from typing import List, Optional, Dict, Tuple, defaultdict


from src.problematics_eval import QueryResult
from PIL import Image


def _load(path: str) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except:
        return Image.new("RGB", (112, 112), color=(40, 40, 40))


def plot_worst_retrievals(
    results:      List[QueryResult],
    n_queries:    int  = 10,       # quante query peggiori mostrare
    n_retrieved:  int  = 8,        # quante retrieved per riga
    figsize_w:    int  = 20,
    save_path:    Optional[str] = None,
):
    """
    Griglia:  1 colonna query  |  n_retrieved colonne risultati
    Verde  = identità corretta
    Rosso  = identità sbagliata
    """
    worst = results[:n_queries]
    n_cols = 1 + n_retrieved
    fig = plt.figure(figsize=(figsize_w, 2.8 * n_queries))
    fig.patch.set_facecolor("#111111")

    outer = gridspec.GridSpec(n_queries, 1, figure=fig, hspace=0.06)

    for row_i, res in enumerate(worst):
        inner = gridspec.GridSpecFromSubplotSpec(
            1, n_cols, subplot_spec=outer[row_i], wspace=0.03
        )

        # ── colonna 0: query ────────────────────────────────────────────
        ax_q = fig.add_subplot(inner[0])
        ax_q.imshow(_load(res.query_path))
        ax_q.set_xticks([]); ax_q.set_yticks([])
        for spine in ax_q.spines.values():
            spine.set_edgecolor("#FACC15"); spine.set_linewidth(3)
        ax_q.set_title(
            f"QUERY\n{res.query_id}\nAP={res.ap:.2f}  rank={res.rank_of_best}",
            fontsize=7, color="#FACC15", pad=4, fontweight="bold"
        )

        # ── colonne 1…n: retrieved ──────────────────────────────────────
        for col_i in range(n_retrieved):
            ax = fig.add_subplot(inner[col_i + 1])
            if col_i >= len(res.top_k_paths):
                ax.axis("off")
                continue

            img  = _load(res.top_k_paths[col_i])
            pred = res.top_k_ids[col_i]
            sim  = res.top_k_sims[col_i]
            correct = (pred == res.query_id)

            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            color = "#22C55E" if correct else "#EF4444"
            for spine in ax.spines.values():
                spine.set_edgecolor(color); spine.set_linewidth(2.5)

            marker = "✓" if correct else "✗"
            ax.set_title(
                f"{marker} #{col_i+1}\n{pred[:12]}\n{sim:.3f}",
                fontsize=6.5, color=color, pad=3
            )

        # separatore orizzontale
        if row_i < n_queries - 1:
            line_ax = fig.add_axes([0.01, 1 - (row_i+1)/n_queries, 0.98, 0.001])
            line_ax.set_facecolor("#333333")
            line_ax.axis("off")

    plt.suptitle(
        "Worst retrievals — difficoltà decrescente",
        color="white", fontsize=13, y=1.002, fontweight="bold"
    )
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"salvato in {save_path}")
    else:
        plt.show()
    plt.close()


def plot_confusion_identities(
    results: List[QueryResult],
    top_n_confused: int = 15,
    save_path: Optional[str] = None,
):
    """Heatmap: quali identità vengono confuse più spesso al rank 1."""
    confusions: Dict[Tuple[str,str], int] = defaultdict(int)
    for r in results:
        if not r.is_hit_at_1 and r.top_k_ids:
            confusions[(r.query_id, r.top_k_ids[0])] += 1

    if not confusions:
        print("nessuna confusione al rank 1")
        return

    # top coppie più confuse
    top_pairs = sorted(confusions.items(), key=lambda x: -x[1])[:top_n_confused]
    labels_q  = [p[0][0] for p in top_pairs]
    labels_r  = [p[0][1] for p in top_pairs]
    counts    = [p[1]    for p in top_pairs]

    identities = sorted(set(labels_q + labels_r))
    n = len(identities)
    idx = {k: i for i, k in enumerate(identities)}
    matrix = np.zeros((n, n))
    for (q, r), c in zip(zip(labels_q, labels_r), counts):
        matrix[idx[q], idx[r]] = c

    fig, ax = plt.subplots(figsize=(max(8, n*0.7), max(6, n*0.6)))
    im = ax.imshow(matrix, cmap="Reds", aspect="auto")
    ax.set_xticks(range(n)); ax.set_xticklabels(identities, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(identities, fontsize=8)
    ax.set_xlabel("Retrieved (rank 1)"); ax.set_ylabel("Query")
    ax.set_title("Confusion matrix — identità confuse al rank 1", fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label="n confusioni")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()





    # The necessary imports (torch, clip, FaceGallery, EmbeddingExtractor, RetrievalEvaluator, plot_worst_retrievals, plot_confusion_identities) are already handled by previous cells or are globally available.

# DEVICE and other global variables (QUERY_DIR, GALLERY_DIR) are already defined in previous cells.
# DEVICE is available as 'device'
# QUERY_DIR is available as '/content/celebrity_retrieval/test/query'
# GALLERY_DIR is available as '/content/celebrity_retrieval/test/gallery'
# CKPT_PATH is available as './celebrity_retrieval/best_model.pth'

# 1. Load model and checkpoint
# The clip_model and clip_preprocess are already loaded in a previous cell.
# arcface_head is also already instantiated.

print(f"Loading checkpoint from {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location=device)
clip_model.load_state_dict(ckpt["clip_state"])
clip_model.eval()

# Load ArcFace head state if available in the checkpoint
if "head_state" in ckpt:
    arcface_head.load_state_dict(ckpt["head_state"])
    arcface_head.eval()
else:
    print("Warning: 'head_state' not found in checkpoint. ArcFace head will not be loaded from checkpoint.")

print(f'Loaded checkpoint from epoch {ckpt["epoch"]} (Top-1 = {ckpt["top1"]*100:.2f}%)')

# 2. Load dataset
# Changed nested=True to nested=False to match the likely flat structure of the test data
query_gal   = FaceGallery(QUERY_DIR,   nested=False)
index_gal   = FaceGallery(GALLERY_DIR, nested=False)

# 3. Extract embedding
extractor = EmbeddingExtractor(clip_model, clip_preprocess, device)
q_emb = extractor.extract(query_gal, arcface_head=arcface_head)
i_emb = extractor.extract(index_gal, arcface_head=arcface_head)

# 4. Evaluate
evaluator = RetrievalEvaluator(query_gal, index_gal, q_emb, i_emb)
results   = evaluator.run(top_k=20, exclude_self=True)
metrics   = evaluator.global_metrics(results)

print("\n── global metrics ──────────────────")
for k, v in metrics.items():
    print(f"  {k:8s}  {v:.4f}")

# --- Inserting labels demonstration ---
print("\n── Sample Labels from Results ──────────────────")
for i, res in enumerate(results[:3]): # Print labels for the first 3 results
    print(f"Query {i+1}:")
    print(f"  Query ID: {res.query_id}")
    print(f"  Top-5 Retrieved IDs: {res.top_k_ids[:5]}")
    print(f"  Rank of Best Match: {res.rank_of_best}")
# ──────────────────────────────────────

# 5. Visualize
plot_worst_retrievals(
    results,
    n_queries   = 12,   # the 12 most problematic queries
    n_retrieved = 8,
    save_path   = "worst_retrievals.png"
)

plot_confusion_identities(
    results,
    top_n_confused = 20,
    save_path       = "confusion_matrix.png"
)
