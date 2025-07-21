import torch, sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List
import torch.nn as nn

# Stub tokenizer that converts list[str] to random integer tensors

def _stub_tokenizer(prompts: List[str]):
    return torch.randint(0, 10000, (len(prompts), 10))


class _DummyClip(nn.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        self.token_embedding = nn.Embedding(10000, dim)
        self.dim = dim

    def encode_text(self, tokens):
        return torch.randn(tokens.size(0), self.dim)

    def encode_image(self, images):
        return torch.randn(images.size(0), self.dim)


def test_default_imagenet_loaded():
    from src.models.r_vfid.prompt_utils import PromptBuilder

    dummy_clip = _DummyClip(dim=32)
    builder = PromptBuilder(dummy_clip, _stub_tokenizer, classnames=None, prompt_length=5, top_c=5)
    assert len(builder.classnames) >= 1000  # ImageNet-1K or more
    assert builder.text_feature_bank.shape[0] == len(builder.classnames)


def test_update_classnames():
    from src.models.r_vfid.prompt_utils import PromptBuilder

    dummy_clip = _DummyClip(dim=32)
    builder = PromptBuilder(dummy_clip, _stub_tokenizer, classnames=["cat", "dog"], prompt_length=5)
    initial_n = len(builder.classnames)
    builder.update_classnames(["unicorn", "dragon"])
    assert len(builder.classnames) == initial_n + 2
    assert builder.text_feature_bank.shape[0] == len(builder.classnames) 