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
        allowed_classes: Optional[list[str]] = None,
    ) -> None:
        """Parameters
        ----------
        root : str | Path
            Directory organised for ``torchvision.datasets.ImageFolder``.
        transform : Callable, optional
            Data augmentation / preprocessing.
        allowed_classes : list[str], optional
            If given, only samples whose *original class folder name* matches one
            of these strings will be kept.  Folder name matching is case-
            insensitive.  This is useful to restrict ProGAN training to the
            four LSUN categories (car, cat, chair, horse) required by ALEI.
        """
        transform = transform or build_default_transform()
        super().__init__(root=str(root), transform=transform)

        # If allowed_classes specified, filter indices
        if allowed_classes is not None:
            allowed_lower = {c.lower() for c in allowed_classes}
            keep_indices = [i for i, y in enumerate(self.targets) if self.classes[y].lower() in allowed_lower]
            # Subset samples & targets
            self.samples = [self.samples[i] for i in keep_indices]
            self.targets = [self.targets[i] for i in keep_indices]

        # Build mapping from ImageFolder targets → binary label
        self.binary_targets = [0 if "real" in self.classes[y].lower() else 1 for y in self.targets]

    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
        label = self.binary_targets[index]
        return img, label 