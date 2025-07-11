import torch, sys, os, pytest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.r_vfid.frequency_expert import FrequencyExpert

def test_frequency_patch_tokens():
    expert = FrequencyExpert(mode="npr", embed_dim=1024, patch_level=True)  # type: ignore[arg-type]
    images = torch.randn(2, 3, 224, 224)
    tokens = expert(images)
    B, L, d = tokens.shape
    assert tokens.ndim == 3 and L == 256 and d == 1024 