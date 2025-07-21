from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from torchvision import datasets, transforms

# -----------------------------------------------------------------------------
#   BenchmarkImageDataset – thin wrapper over torchvision ImageFolder
# -----------------------------------------------------------------------------


def build_default_transform(image_size: int = 224) -> transforms.Compose:  # type: ignore[name-defined]
    """Return default CLIP-style preprocessing transform."""
    normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    return transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])


class BenchmarkImageDataset(datasets.ImageFolder):
    """AIGC Detection Benchmark dataset loader.

    Expect directory with sub-folders representing *class labels*.
    Real images should be under `real/` (label 0) and fakes under any
    generator-specific folders (label 1), or follow the original benchmark
    hierarchy.  For flexibility we simply inherit from `ImageFolder` and let
    users prepare a root dir containing two high-level subfolders:

        root/real/...
        root/fake/...

    or arbitrary structure; `ImageFolder` will map each folder to an index.
    We additionally provide `label_map` to coerce those indices into
    {0: real, 1: fake} based on folder names.
    """

    def __init__(
        self,
        root: str | Path,
        transform: Optional[Callable] = None,
    ) -> None:
        transform = transform or build_default_transform()
        super().__init__(root=str(root), transform=transform)

        # Build mapping from ImageFolder targets → binary label
        # Heuristic: folder name containing "real" → 0 else 1.
        self.binary_targets = [0 if "real" in self.classes[y].lower() else 1 for y in self.targets]

    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
        label = self.binary_targets[index]
        return img, label 