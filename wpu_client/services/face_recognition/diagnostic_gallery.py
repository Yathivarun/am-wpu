"""Local face gallery for diagnostic (offline) mode.

A gallery is a directory of per-person sub-folders seeded by `tools/seed_face.py`:

    diagnostic_gallery/
      ekan/
        embedding.npy        # SFace 128-D L2-normalised vector
        meta.json            # {"name": "Ekan", "gender": "male"}

At runtime the face service matches a live probe embedding against these entries
by cosine distance — no server, no Triton. The matched person's `gender` selects
which `face_stock_images/<gender>/` sketches the slideshow shows.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)

EMBEDDING_FILE = "embedding.npy"
META_FILE = "meta.json"


@dataclass
class GalleryEntry:
    slug: str          # folder name / local identifier (used as visit_id)
    name: str
    gender: str        # "male" | "female"
    vector: np.ndarray  # 128-D L2-normalised


@dataclass
class Match:
    entry: GalleryEntry
    distance: float    # cosine distance (0 = identical)


class DiagnosticGallery:
    """In-memory set of seeded faces, matched by cosine distance."""

    def __init__(self, gallery_dir: str):
        self.gallery_dir = gallery_dir
        self._entries: list[GalleryEntry] = []

    def load(self) -> int:
        """Load every valid person sub-folder. Returns the number loaded."""
        self._entries = []
        if not os.path.isdir(self.gallery_dir):
            logger.warning("Diagnostic gallery dir not found: %s", self.gallery_dir)
            return 0

        for slug in sorted(os.listdir(self.gallery_dir)):
            person_dir = os.path.join(self.gallery_dir, slug)
            if not os.path.isdir(person_dir):
                continue
            emb_path = os.path.join(person_dir, EMBEDDING_FILE)
            meta_path = os.path.join(person_dir, META_FILE)
            if not (os.path.exists(emb_path) and os.path.exists(meta_path)):
                logger.warning("Skipping '%s' — missing embedding.npy or meta.json", slug)
                continue
            try:
                vec = np.load(emb_path).astype(np.float32).flatten()
                with open(meta_path) as f:
                    meta = json.load(f)
                gender = str(meta.get("gender", "")).lower()
                if gender not in ("male", "female"):
                    logger.warning("Skipping '%s' — gender must be male/female, got %r", slug, gender)
                    continue
                self._entries.append(GalleryEntry(
                    slug=slug,
                    name=meta.get("name", slug),
                    gender=gender,
                    vector=vec,
                ))
            except Exception as e:
                logger.error("Failed to load gallery entry '%s': %s", slug, e)

        logger.info(
            "Diagnostic gallery loaded: %d person(s) from %s [%s]",
            len(self._entries), self.gallery_dir,
            ", ".join(f"{e.name}/{e.gender}" for e in self._entries) or "none",
        )
        return len(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def match(self, vector, threshold: float) -> Optional[Match]:
        """Return the nearest entry within `threshold` cosine distance, else None."""
        if not self._entries:
            return None
        best: Optional[Match] = None
        for entry in self._entries:
            try:
                dist = float(cosine(vector, entry.vector))
            except Exception as e:
                logger.error("Cosine failed for '%s': %s", entry.slug, e)
                continue
            if best is None or dist < best.distance:
                best = Match(entry=entry, distance=dist)
        if best is not None and best.distance < threshold:
            return best
        return None
