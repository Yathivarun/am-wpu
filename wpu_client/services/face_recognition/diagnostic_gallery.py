"""Local face gallery for diagnostic (offline) mode.

A gallery is a directory of per-person sub-folders seeded by `scripts/seed_face.py`:

    data/embeddings/
      ekan/
        embedding_sface.npy      # SFace 128-D L2-normalised vector
        embedding_mobilenet.npy  # MobileFaceNet 512-D L2-normalised vector
        meta.json                # {"name": "Ekan", "models": [...]}
        sketches/                # this person's face-swap sketch image(s)
          01.png ...

Each person is seeded once with EVERY model; the vectors live in separate spaces
(128-D vs 512-D). At runtime the face service matches a live probe against the
entries IN THE SAME MODEL'S SPACE ONLY (it passes the configured model to
`match`). On a match, the slideshow shows that person's OWN sketches. Legacy
galleries that only have `embedding.npy` are loaded as the sface vector.
"""

import json
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)

LEGACY_EMBEDDING_FILE = "embedding.npy"  # pre-multi-model seeds (sface 128-D)
META_FILE = "meta.json"
SKETCHES_DIR = "sketches"


def _embedding_file(model: str) -> str:
    return f"embedding_{model}.npy"


@dataclass
class GalleryEntry:
    slug: str           # folder name / local identifier (used as registration_id)
    name: str
    # model name → L2-normalised vector (e.g. {"sface": 128-D, "mobilenet": 512-D}).
    vectors: dict[str, np.ndarray] = field(default_factory=dict)
    sketch_dir: str = ""  # data/embeddings/<slug>/sketches — this person's raw cutouts
    gender: Optional[str] = None  # optional metadata; used for scene gender filtering
    # The whole parsed meta.json, so local_assets can honour explicit
    # `sau`/`fru` filename keys without re-reading the file.
    meta: dict = field(default_factory=dict)


@dataclass
class Match:
    entry: GalleryEntry
    distance: float     # cosine distance (0 = identical)


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
            meta_path = os.path.join(person_dir, META_FILE)
            if not os.path.exists(meta_path):
                logger.warning("Skipping '%s' — missing meta.json", slug)
                continue
            try:
                with open(meta_path) as f:
                    meta = json.load(f)

                # Load one vector per model: embedding_<model>.npy, plus the
                # legacy embedding.npy (treated as the sface vector).
                vectors: dict[str, np.ndarray] = {}
                for fname in os.listdir(person_dir):
                    if fname.startswith("embedding_") and fname.endswith(".npy"):
                        model = fname[len("embedding_"):-len(".npy")]
                        vectors[model] = (
                            np.load(os.path.join(person_dir, fname))
                            .astype(np.float32)
                            .flatten()
                        )
                legacy = os.path.join(person_dir, LEGACY_EMBEDDING_FILE)
                if "sface" not in vectors and os.path.exists(legacy):
                    vectors["sface"] = np.load(legacy).astype(np.float32).flatten()

                if not vectors:
                    logger.warning("Skipping '%s' — no embedding_*.npy vectors", slug)
                    continue

                gender = meta.get("gender")
                self._entries.append(GalleryEntry(
                    slug=slug,
                    name=meta.get("name", slug),
                    vectors=vectors,
                    sketch_dir=os.path.join(person_dir, SKETCHES_DIR),
                    gender=str(gender).lower() if gender else None,
                    meta=meta,
                ))
            except Exception as e:
                logger.error("Failed to load gallery entry '%s': %s", slug, e)

        logger.info(
            "Diagnostic gallery loaded: %d person(s) from %s [%s]",
            len(self._entries), self.gallery_dir,
            ", ".join(
                f"{e.name}({'/'.join(sorted(e.vectors))})" for e in self._entries
            ) or "none",
        )
        return len(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def entries(self) -> list[GalleryEntry]:
        """Every loaded person, in gallery order."""
        return list(self._entries)

    def get_entry(self, slug: str) -> Optional[GalleryEntry]:
        """Look up a loaded entry by its slug (registration_id), or None."""
        for entry in self._entries:
            if entry.slug == slug:
                return entry
        return None

    def random_entry(self) -> Optional[GalleryEntry]:
        """A random seeded person — prefers one that actually has sketches.

        Used for the unknown-face preview: an unrecognised visitor is shown a
        random person's sketches to get a feel for what they look like.
        """
        if not self._entries:
            return None
        with_sketches = [
            e for e in self._entries
            if os.path.isdir(e.sketch_dir) and os.listdir(e.sketch_dir)
        ]
        return random.choice(with_sketches or self._entries)

    def match(self, vector, threshold: float, model: str) -> Optional[Match]:
        """Nearest entry within `threshold` cosine distance, in `model`'s space.

        Only entries that carry a vector for `model` are considered — vectors
        from different models live in different spaces and are never compared.
        """
        if not self._entries:
            return None
        best: Optional[Match] = None
        skipped = 0
        for entry in self._entries:
            ref = entry.vectors.get(model)
            if ref is None:
                skipped += 1
                continue
            try:
                dist = float(cosine(vector, ref))
            except Exception as e:
                logger.error("Cosine failed for '%s': %s", entry.slug, e)
                continue
            if best is None or dist < best.distance:
                best = Match(entry=entry, distance=dist)
        if skipped:
            logger.warning(
                "%d gallery entr(y/ies) have no '%s' vector — re-seed them to "
                "match in this model. Skipped from matching.", skipped, model,
            )
        if best is not None and best.distance < threshold:
            return best
        return None
