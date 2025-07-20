# pyright: reportMissingImports=false
import torch, os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.r_vfid import RvfidModel


def test_add_domain_expert():
    model = RvfidModel(num_experts=2)
    prev_num = model.num_experts
    model.add_domain_prompt_and_expert(mode="dncnn")
    assert model.num_experts == prev_num + 1
    # Router output dim check
    assert model.router.num_experts == model.num_experts
    # Visual MultiLoRA branch count
    from src.models.r_vfid.multi_lora import MultiLoRALinear
    mll = next((m for m in model.clip_model.visual.modules() if isinstance(m, MultiLoRALinear)), None)
    assert mll is not None
    assert mll.num_experts == model.num_experts
    # Forward should run without error
    logits = model(torch.randn(1,3,224,224))
    assert logits.shape == (1,2) 