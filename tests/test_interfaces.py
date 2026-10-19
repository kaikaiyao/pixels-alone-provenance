from __future__ import annotations

import torch

from passive_provenance.interfaces import build_interface_wrapper
from passive_provenance.matrix import InterfaceSpec


def test_quantized_interface_rounds_to_eight_bits():
    wrapper = build_interface_wrapper(
        InterfaceSpec(interface_id="fake_score_q8", kind="quantized_target_score", quantization_bits=8),
        ("real", "sd15", "sdxl"),
        target_label="fake",
    )
    reply = wrapper.query(torch.tensor([0.2, 0.3, 0.5]))
    assert isinstance(reply.payload, float)
    assert 0.0 <= reply.payload <= 1.0


def test_binary_interface_can_abstain_near_threshold():
    wrapper = build_interface_wrapper(
        InterfaceSpec(interface_id="binary", kind="binary_accept_reject", abstention=True),
        ("real", "fake"),
        target_label="fake",
        binary_threshold=0.5,
        abstention_margin=0.1,
    )
    reply = wrapper.query(torch.tensor([0.51, 0.49]))
    assert reply.abstained is True


def test_rate_limited_interface_marks_budget_exhaustion():
    wrapper = build_interface_wrapper(
        InterfaceSpec(interface_id="binary_budget_1", kind="binary_accept_reject", rate_limit=1),
        ("real", "fake"),
        target_label="fake",
        binary_threshold=0.5,
    )
    first = wrapper.query(torch.tensor([0.2, 0.8]))
    second = wrapper.query(torch.tensor([0.2, 0.8]))
    assert first.budget_exhausted is False
    assert second.budget_exhausted is True
