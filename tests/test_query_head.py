import torch, sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.r_vfid.query_head import HierarchicalSemanticQueryHead

def test_query_head_forward():
    head = HierarchicalSemanticQueryHead(embed_dim=32)
    x = torch.randn(4, 32)
    # Without adapters
    out_base = head(x)
    assert out_base.shape == (4, 32)

    # Add adapter and ensure output changes when adapter is trainable
    head.add_adapter(r=2, lora_alpha=4)
    out_with_adapter = head(x)
    # Expect different due to random init LoRA
    diff = (out_with_adapter - out_base).abs().mean().item()
    assert diff > 1e-5 