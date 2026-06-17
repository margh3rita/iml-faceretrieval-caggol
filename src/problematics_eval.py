import torch
import numpy as np
from pathlib import Path
from PIL import Image
from dataclasses import dataclass, field
from typing import List, Dict
from tqdm import tqdm



# ── struttura dati per una singola query ─────────────────────────────────────

@dataclass
class QueryResult:
    query_path:   str
    query_id:     str                    # identità ground-truth (nome cartella)
    rank_of_best: int                    # rank del primo positivo trovato (1-based)
    ap:           float                  # average precision
    top_k_paths:  List[str]  = field(default_factory=list)
    top_k_ids:    List[str]  = field(default_factory=list)
    top_k_sims:   List[float] = field(default_factory=list)
    is_hit_at_1:  bool = False
    is_hit_at_5:  bool = False

    @property
    def difficulty_score(self) -> float:
        """Più alto = predizione più sbagliata."""
        return self.rank_of_best + (1.0 - self.ap) * 10


# ── dataset flat / gerarchico ────────────────────────────────────────────────

class FaceGallery:
    """
    Supporta due strutture:
      flat   → root/img1.jpg  (label derivata dal nome file senza estensione)
      nested → root/identity/img.jpg  (label = nome cartella)
    """
    EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}

    def __init__(self, root: str, nested: bool = True):
        self.root   = Path(root)
        self.paths: List[Path] = []
        self.labels: List[str] = []

        if nested:
            for identity_dir in sorted(self.root.iterdir()):
                if not identity_dir.is_dir():
                    continue
                for img in sorted(identity_dir.iterdir()):
                    if img.suffix.lower() in self.EXTS:
                        self.paths.append(img)
                        self.labels.append(identity_dir.name)
        else:
            for img in sorted(self.root.iterdir()):
                if img.suffix.lower() in self.EXTS:
                    self.paths.append(img)
                    self.labels.append(img.stem)

        assert len(self.paths) > 0, f"nessuna immagine trovata in {root}"

    def __len__(self): return len(self.paths)


# ── estrazione embedding ─────────────────────────────────────────────────────

class EmbeddingExtractor:
    def __init__(self, clip_model, preprocess, device: str = "cuda"):
        self.model      = clip_model.eval()
        self.preprocess = preprocess
        self.device     = device

    @torch.no_grad()
    def extract(
        self,
        gallery: FaceGallery,
        batch_size: int = 64,
        arcface_head=None,        # opzionale: proietta con arcface
    ) -> torch.Tensor:
        """Restituisce (N, D) normalizzato su L2."""
        all_embs = []
        paths    = gallery.paths

        for i in tqdm(range(0, len(paths), batch_size), desc="embedding"):
            batch_paths = paths[i : i + batch_size]
            imgs = torch.stack([
                self.preprocess(Image.open(p).convert("RGB"))
                for p in batch_paths
            ]).to(self.device)

            feats = self.model.encode_image(imgs).float()

            if arcface_head is not None:
                feats = arcface_head(feats)          # proiezione opzionale

            feats = torch.nn.functional.normalize(feats, dim=-1)
            all_embs.append(feats.cpu())

        return torch.cat(all_embs, dim=0)


# ── evaluator principale ─────────────────────────────────────────────────────

class RetrievalEvaluator:
    def __init__(
        self,
        query_gallery:   FaceGallery,
        index_gallery:   FaceGallery,
        query_embeddings:  torch.Tensor,   # (Nq, D)
        index_embeddings:  torch.Tensor,   # (Ni, D)
    ):
        self.qg    = query_gallery
        self.ig    = index_gallery
        self.q_emb = query_embeddings
        self.i_emb = index_embeddings

    def run(self, top_k: int = 20, exclude_self: bool = True) -> List[QueryResult]:
        """
        Per ogni query, calcola ranking e metriche.
        exclude_self=True rimuove l'immagine query stessa dal ranking.
        """
        sims = self.q_emb @ self.i_emb.T        # (Nq, Ni) cosine similarity
        results: List[QueryResult] = []

        for qi, q_path in enumerate(self.qg.paths):
            q_label = self.qg.labels[qi]
            row     = sims[qi].clone()

            if exclude_self:
                # azzera se stessa nel gallery (stesso path)
                for ii, i_path in enumerate(self.ig.paths):
                    if i_path == q_path:
                        row[ii] = -2.0

            ranked_idx  = torch.argsort(row, descending=True)[:top_k]
            ranked_sims = row[ranked_idx].tolist()
            ranked_ids  = [self.ig.labels[i] for i in ranked_idx.tolist()]
            ranked_paths = [str(self.ig.paths[i]) for i in ranked_idx.tolist()]

            # ── rank del primo positivo (1-based) ───────────────────────
            rank_best = top_k + 1
            for r, lbl in enumerate(ranked_ids):
                if lbl == q_label:
                    rank_best = r + 1
                    break

            # ── average precision ────────────────────────────────────────
            positives_seen = 0
            ap_sum         = 0.0
            n_positives    = sum(1 for l in self.ig.labels if l == q_label)
            if exclude_self and q_path in self.ig.paths:
                n_positives = max(1, n_positives - 1)

            for r, lbl in enumerate(ranked_ids, start=1):
                if lbl == q_label:
                    positives_seen += 1
                    ap_sum += positives_seen / r
            ap = ap_sum / n_positives if n_positives > 0 else 0.0

            results.append(QueryResult(
                query_path   = str(q_path),
                query_id     = q_label,
                rank_of_best = rank_best,
                ap           = ap,
                top_k_paths  = ranked_paths,
                top_k_ids    = ranked_ids,
                top_k_sims   = ranked_sims,
                is_hit_at_1  = (rank_best == 1),
                is_hit_at_5  = (rank_best <= 5),
            ))

        results.sort(key=lambda r: r.difficulty_score, reverse=True)
        return results

    def global_metrics(self, results: List[QueryResult]) -> Dict[str, float]:
        return {
            "mAP":   np.mean([r.ap for r in results]),
            "Hit@1": np.mean([r.is_hit_at_1 for r in results]),
            "Hit@5": np.mean([r.is_hit_at_5 for r in results]),
            "MRR":   np.mean([1.0/r.rank_of_best for r in results]),
        }
    

    