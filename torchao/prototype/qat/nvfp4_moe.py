"""
NVFP4 QAT for MoE (Mixture of Experts) models.

This module provides fake quantization for MoE expert layers using decomposed
PyTorch ops with NVFP4 fake quantization (quantize->dequantize roundtrip via
``NVFP4Tensor``) inserted before each GEMM. Gradients flow through the STE
(Straight-Through Estimator) automatically since all ops are standard PyTorch
operations that autograd can trace.

Usage::

    from torchao.prototype.qat.nvfp4_moe import apply_nvfp4_moe_qat
    model = apply_nvfp4_moe_qat(model)
"""

import torch
import torch.nn as nn

from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor,
    per_tensor_amax_to_scale,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def nvfp4_fake_quantize(x: torch.Tensor) -> torch.Tensor:
    """Fake quantize tensor to NVFP4 and back, with STE for gradients.

    Computes a quantize->dequantize roundtrip using ``NVFP4Tensor`` and
    applies the Straight-Through Estimator so the forward uses the
    dequantized value while the backward treats quantization as identity.
    """
    amax = torch.max(torch.abs(x.detach()))
    per_tensor_scale = per_tensor_amax_to_scale(amax)
    x_nvfp4 = NVFP4Tensor.to_nvfp4(
        x.detach().contiguous(), per_tensor_scale=per_tensor_scale
    )
    x_dq = x_nvfp4.dequantize(x.dtype)
    # STE: forward uses dequantized value, backward treats quant as identity
    return x + (x_dq - x).detach()


def _fake_quantized_expert_forward(
    hidden_states: torch.Tensor,  # [T_expert, H]
    gate_up_weight: torch.Tensor,  # [H, 2*I]
    gate_up_bias: torch.Tensor,  # [2*I]
    down_weight: torch.Tensor,  # [I, H]
    down_bias: torch.Tensor,  # [H]
    alpha: float,
    limit: float,
) -> torch.Tensor:
    """Single-expert forward with NVFP4 fake quantization before each GEMM."""
    # Fake quantize activations and first GEMM weights
    fq_hidden = nvfp4_fake_quantize(hidden_states)
    fq_gate_up_w = nvfp4_fake_quantize(gate_up_weight)

    # GEMM1: [T, H] @ [H, 2*I] + bias -> [T, 2*I]
    gate_up = fq_hidden @ fq_gate_up_w + gate_up_bias

    # De-interleave and apply GptOss activation
    gate = gate_up[..., ::2]   # even columns
    up = gate_up[..., 1::2]    # odd columns
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    intermediate = (up + 1) * glu  # [T, I]

    # Fake quantize intermediate activations and second GEMM weights
    fq_intermediate = nvfp4_fake_quantize(intermediate)
    fq_down_w = nvfp4_fake_quantize(down_weight)

    # GEMM2: [T, I] @ [I, H] + bias -> [T, H]
    output = fq_intermediate @ fq_down_w + down_bias
    return output


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


class NVFP4FakeQuantizedGptOssExperts(nn.Module):
    """Drop-in replacement for ``GptOssExperts`` that inserts NVFP4 fake
    quantization before each GEMM in the expert forward pass.

    All operations are standard PyTorch ops, so autograd traces through
    them natively. The NVFP4 fake quantization (quantize->dequantize roundtrip)
    injects quantization noise in the forward pass, and the STE lets
    gradients flow through as if the quantization were identity.
    """

    def __init__(self, config):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size
        self.alpha = 1.702
        self.limit = 7.0

        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim)
        )
        self.gate_up_proj_bias = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.expert_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.expert_dim, self.hidden_size)
        )
        self.down_proj_bias = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size)
        )

    @classmethod
    def from_experts(cls, experts: nn.Module, config) -> "NVFP4FakeQuantizedGptOssExperts":
        """Create from an existing ``GptOssExperts`` module, sharing parameter storage."""
        new = cls.__new__(cls)
        nn.Module.__init__(new)

        new.intermediate_size = experts.intermediate_size
        new.num_experts = experts.num_experts
        new.hidden_size = experts.hidden_size
        new.expert_dim = experts.expert_dim
        new.alpha = experts.alpha
        new.limit = experts.limit

        # Share parameters (no copy)
        new.gate_up_proj = experts.gate_up_proj
        new.gate_up_proj_bias = experts.gate_up_proj_bias
        new.down_proj = experts.down_proj
        new.down_proj_bias = experts.down_proj_bias

        return new

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_indices=None,
        routing_weights=None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states_2d = hidden_states.reshape(-1, self.hidden_size)
        num_tokens, hidden_size = hidden_states_2d.shape

        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(
                router_indices, num_classes=self.num_experts + 1
            )
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(
                expert_mask.sum(dim=(-1, -2)), 0
            ).nonzero()

        next_states = torch.zeros(
            num_tokens, hidden_size, dtype=hidden_states_2d.dtype, device=hidden_states_2d.device
        )
        for expert_idx_item in expert_hit:
            expert_idx = expert_idx_item[0]
            if expert_idx == self.num_experts:
                continue
            with torch.no_grad():
                _, token_idx = torch.where(expert_mask[expert_idx])

            current_state = hidden_states_2d[token_idx]
            out = _fake_quantized_expert_forward(
                current_state,
                self.gate_up_proj[expert_idx],
                self.gate_up_proj_bias[expert_idx],
                self.down_proj[expert_idx],
                self.down_proj_bias[expert_idx],
                self.alpha,
                self.limit,
            )
            weighted_output = out * routing_weights[token_idx, expert_idx, None]

            padded = torch.zeros_like(next_states)
            padded[token_idx] = weighted_output.to(hidden_states_2d.dtype)
            next_states = next_states + padded

        return next_states.view(batch_size, -1, self.hidden_size)


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------


def apply_nvfp4_moe_qat(model: nn.Module) -> nn.Module:
    """Replace all ``GptOssExperts`` modules with ``NVFP4FakeQuantizedGptOssExperts``.

    This applies NVFP4 QAT to the MoE expert layers so that the forward pass
    uses decomposed PyTorch ops with NVFP4 fake quantization while backward
    uses the STE for gradient computation.

    Args:
        model: A HuggingFace model containing ``GptOssExperts`` modules.

    Returns:
        The same model with expert modules replaced in-place.
    """
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts

    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, GptOssExperts):
            replacements.append((name, module))

    for name, module in replacements:
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, attr_name = parts
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            attr_name = name

        new_module = NVFP4FakeQuantizedGptOssExperts.from_experts(
            module, model.config
        )
        setattr(parent, attr_name, new_module)

    return model
