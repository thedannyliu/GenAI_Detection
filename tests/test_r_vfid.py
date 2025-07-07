import pytest
import torch

# Adjust Python path if running outside package installation
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.r_vfid.model import RvfidModel
from src.models.r_vfid.prompt_utils import PromptBuilder


def test_prompt_builder_shape():
    model = RvfidModel(num_experts=1)  # minimal model
    images = torch.randn(2, 3, 224, 224)
    prompt_tokens = model.prompt_builder.build(images)
    B, Lp, d = prompt_tokens.shape
    assert B == 2 and d == 768 and Lp == model.prompt_builder.top_c * 2 * model.prompt_builder.prompt_len


def test_rvfid_forward_shape():
    model = RvfidModel(num_experts=2)
    images = torch.randn(4, 3, 224, 224)
    logits = model(images)
    assert logits.shape == (4, 2) 