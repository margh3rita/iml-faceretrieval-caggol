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