# pyright: reportMissingImports=false
"""Unit test for *zero forgetting* property after incremental expert addition.

The criterion: after calling `add_domain_prompt_and_expert`, parameters that
belonged to the original model (prior to addition) must remain bit-exactly
unchanged and frozen (requires_grad=False).  New parameters may be introduced
and trainable.
"""
import copy, os, sys, torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.r_vfid.model import RvfidModel


def _state_subset(state_dict, prefix_exclude="experts."):
    """Return a copy without new expert keys (those that include last index)."""
    # This helper is robust enough: we compare *all* keys present before.
    return {k: v.clone() for k, v in state_dict.items()}


def test_zero_forgetting_params_unchanged():
    model = RvfidModel(num_experts=2, gating_mode="sigmoid")
    # Snapshot old state_dict
    sd_before = _state_subset(model.state_dict())

    # Run expert addition (simulate new task)
    model.add_domain_prompt_and_expert(mode="npr")  # now num_experts==3

    # 1) Check original params unchanged
    sd_after = model.state_dict()
    for k, v_before in sd_before.items():
        v_after = sd_after[k]
        if v_before.shape == v_after.shape:
            assert torch.equal(v_before, v_after), f"Parameter {k} changed after expansion!"
        else:
            # Weight expanded in dim0 (e.g., router.fc.weight). Check prefix rows unchanged.
            min_shape = tuple(min(a, b) for a, b in zip(v_before.shape, v_after.shape))
            slices = tuple(slice(0, s) for s in min_shape)
            assert torch.equal(v_before[slices], v_after[slices]), f"Prefix of {k} changed after expansion!"

    # 2) Ensure old params are frozen
    for name, param in model.named_parameters():
        if name in sd_before:
            assert not param.requires_grad, f"Old param {name} should be frozen."

    # 3) New parameters exist and are trainable
    num_trainable_new = sum(p.requires_grad for n, p in model.named_parameters() if n not in sd_before)
    assert num_trainable_new > 0, "No new trainable parameters added." 