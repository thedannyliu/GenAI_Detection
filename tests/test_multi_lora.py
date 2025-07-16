# pyright: reportMissingImports=false
import torch, sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.r_vfid import RvfidModel
from src.models.r_vfid.multi_lora import MultiLoRALinear

def test_multi_lora_injection_and_switch():
    model = RvfidModel(num_experts=3)
    visual = model.clip_model.visual
    # 找第一個 MultiLoRALinear
    mll = None
    for m in visual.modules():
        if isinstance(m, MultiLoRALinear):
            mll = m
            break
    assert hasattr(visual, "set_expert")
    if mll is not None:
        assert mll.num_experts == 3

    # 先記錄 active_idx
    visual.set_expert(1)
    if mll is not None:
        assert mll.active_idx == 1
    visual.set_expert(2)
    if mll is not None:
        assert mll.active_idx == 2

    # forward 不應報錯
    logits = model(torch.randn(1,3,224,224))
    assert logits.shape == (1,2) 