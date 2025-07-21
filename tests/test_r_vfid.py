# pyright: reportMissingImports=false
import torch, sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.r_vfid.model import RvfidModel


def test_query_head_adapter():
    model = RvfidModel(num_experts=1)
    initial_adapters = len(model.query_head.adapters)
    model.query_head.add_adapter()
    assert len(model.query_head.adapters) == initial_adapters + 1


def test_rvfid_forward_shape():
    model = RvfidModel(num_experts=2)
    images = torch.randn(4, 3, 224, 224)
    logits = model(images)
    assert logits.shape == (4, 2) 


def test_lora_injected():
    model = RvfidModel(num_experts=1)
    lora_layers = [m for m in model.modules() if m.__class__.__name__ == "LoRALinear"]
    assert len(lora_layers) > 0 