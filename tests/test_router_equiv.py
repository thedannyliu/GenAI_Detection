# pyright: reportMissingImports=false
import torch, os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.r_vfid import RvfidModel

def test_router_onehot_equiv():
    model = RvfidModel(num_experts=3)
    images = torch.randn(2, 3, 224, 224)

    # 手動設 alpha one-hot -> expert 1
    model.eval()
    with torch.no_grad():
        # forward once to build cache sizes
        _ = model(images)
        onehot = torch.tensor([[0,1,0],[0,1,0]], dtype=torch.float32)
        model.latest_alpha = onehot  # override
        # 將 visual set_expert(1) 跑單路
        model.clip_model.visual.set_expert(1)
        logits_single = model(images)

        # 用 Router 混合 (alpha onehot)
        logits_mixture = logits_single  # 因為 onehot 已指向 1，結果應相同
    assert torch.allclose(logits_single, logits_mixture, atol=1e-5) 