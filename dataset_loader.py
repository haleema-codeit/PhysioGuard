"""Filesystem loader for the checked-in real/fake dataset."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass
class VideoItem:
    path: str
    label: int
    dataset_source: str


def discover_dataset(root: str, max_per_class: Optional[int] = None):
    root = Path(root)
    items = []
    for label, folder in ((0, root / "real"), (1, root / "fake")):
        if not folder.exists():
            continue
        paths = sorted(folder.rglob("*.mp4"))
        if max_per_class:
            paths = paths[:max_per_class]
        for path in paths:
            relative = path.relative_to(folder)
            source = relative.parts[0] if len(relative.parts) > 1 else "unknown"
            items.append(VideoItem(str(path), label, source))
    return items


def split_known_unseen(items, held_out_source: Optional[str] = None):
    sources = sorted({item.dataset_source for item in items})
    held_out_source = held_out_source or (sources[-1] if sources else None)
    return ([x for x in items if x.dataset_source != held_out_source],
            [x for x in items if x.dataset_source == held_out_source], held_out_source)

