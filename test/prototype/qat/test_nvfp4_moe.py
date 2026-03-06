"""
Tests for NVFP4 QAT with decomposed PyTorch ops for MoE models.

Includes:
- Unit tests for the QAT fake-quantization path (no flashinfer dependency).
- Reference-vs-kernel comparison tests that verify a decomposed PyTorch
  reference matches the flashinfer ``trtllm_fp4_block_scale_moe`` fused
  kernel (requires flashinfer + SM100+ GPU).
"""

import pytest
import torch
import torch.nn.functional as F

from torchao.prototype.qat.nvfp4_moe import (
    NVFP4FakeQuantizedGptOssExperts,
    _fake_quantized_expert_forward,
    nvfp4_fake_quantize,
)
from torchao.quantization.utils import compute_error

# ---------------------------------------------------------------------------
# Optional flashinfer imports (for reference-vs-kernel tests)
# ---------------------------------------------------------------------------
try:
    from flashinfer import (
        e2m1_and_ufp8sf_scale_to_float,
        fp4_quantize,
    )
    from flashinfer.fp4_quantization import block_scale_interleave
    from flashinfer.fused_moe import trtllm_fp4_block_scale_moe
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )
    from flashinfer.utils import get_compute_capability

    _has_flashinfer = True
except ImportError:
    _has_flashinfer = False


def _has_flashinfer_sm100():
    """Check if flashinfer is available and GPU supports SM100+."""
    if not _has_flashinfer:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        cc = get_compute_capability(torch.device("cuda"))
        return cc[0] >= 10
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


class MockConfig:
    """Minimal config matching GptOssExperts expectations."""

    def __init__(self, hidden_size=64, intermediate_size=64, num_local_experts=4):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_local_experts = num_local_experts


def _make_routing(num_tokens, num_experts, top_k, device):
    """Create synthetic routing inputs."""
    router_indices = torch.stack(
        [torch.randperm(num_experts, device=device)[:top_k] for _ in range(num_tokens)]
    )  # [T, top_k]
    routing_weights = torch.randn(num_tokens, num_experts, device=device).softmax(dim=-1)
    return router_indices, routing_weights


class TestFakeQuantizeRoundtrip:
    def test_shape_dtype_preserved(self):
        x = torch.randn(16, 64, dtype=torch.bfloat16, device="cuda")
        x_fq = nvfp4_fake_quantize(x)
        assert x_fq.shape == x.shape
        assert x_fq.dtype == x.dtype

    def test_sqnr(self):
        torch.manual_seed(42)
        x = torch.randn(16, 64, dtype=torch.bfloat16, device="cuda")
        x_fq = nvfp4_fake_quantize(x)
        sqnr = compute_error(x, x_fq)
        assert sqnr >= 8.0, f"SQNR {sqnr:.2f} dB below threshold 8.0 dB"

    def test_gradient_flows_through(self):
        x = torch.randn(16, 64, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        x_fq = nvfp4_fake_quantize(x)
        loss = x_fq.sum()
        loss.backward()
        assert x.grad is not None
        # STE: gradient should be all ones (from sum)
        assert torch.allclose(x.grad, torch.ones_like(x.grad))


class TestFakeQuantizedExpertForward:
    def test_vs_bf16_forward(self):
        T, H, I = 16, 64, 64
        torch.manual_seed(42)
        hidden = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
        gate_up_w = torch.randn(H, 2 * I, dtype=torch.bfloat16, device="cuda") * 0.02
        gate_up_b = torch.zeros(2 * I, dtype=torch.bfloat16, device="cuda")
        down_w = torch.randn(I, H, dtype=torch.bfloat16, device="cuda") * 0.02
        down_b = torch.zeros(H, dtype=torch.bfloat16, device="cuda")
        alpha, limit = 1.702, 7.0

        # bf16 reference (no fake quantization)
        gate_up = hidden @ gate_up_w + gate_up_b
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
        glu = gate * torch.sigmoid(gate * alpha)
        intermediate = (up + 1) * glu
        ref = intermediate @ down_w + down_b

        # Fake quantized forward
        fq_out = _fake_quantized_expert_forward(
            hidden, gate_up_w, gate_up_b, down_w, down_b, alpha, limit,
        )

        assert fq_out.shape == ref.shape
        assert fq_out.dtype == ref.dtype
        # Should be close but not identical (quantization noise)
        sqnr = compute_error(ref, fq_out)
        assert sqnr >= 8.0, f"SQNR {sqnr:.2f} dB below threshold 8.0 dB"
        # Verify not exactly equal (quantization noise is present)
        assert not torch.allclose(ref, fq_out, atol=0, rtol=0)


class TestGradientFlow:
    def test_all_params_get_gradients(self):
        E, H, I, T, top_k = 4, 64, 64, 16, 2
        config = MockConfig(hidden_size=H, intermediate_size=I, num_local_experts=E)

        torch.manual_seed(42)
        module = NVFP4FakeQuantizedGptOssExperts(config).to(
            device="cuda", dtype=torch.bfloat16
        )
        # Initialize with small random weights
        with torch.no_grad():
            module.gate_up_proj.normal_(0, 0.02)
            module.gate_up_proj_bias.zero_()
            module.down_proj.normal_(0, 0.02)
            module.down_proj_bias.zero_()

        hidden = torch.randn(1, T, H, dtype=torch.bfloat16, device="cuda")
        router_indices, routing_weights = _make_routing(T, E, top_k, "cuda")

        output = module(hidden, router_indices=router_indices, routing_weights=routing_weights)
        loss = output.sum()
        loss.backward()

        for name in ["gate_up_proj", "gate_up_proj_bias", "down_proj", "down_proj_bias"]:
            param = getattr(module, name)
            assert param.grad is not None, f"{name}.grad is None"
            assert torch.any(param.grad != 0), f"{name}.grad is all zeros"


# ==========================================================================
# Reference implementation matching trtllm_fp4_block_scale_moe numerics
# ==========================================================================
#
# The functions below depend on flashinfer and are only used by
# ``TestReferenceVsKernel``.  They are placed here (in the test file, not in
# ``nvfp4_moe.py``) so the module stays flashinfer-free.
# ==========================================================================


# ---- FP4 quantization helpers -------------------------------------------


def _calculate_fp4_global_scale_factor(tensor):
    """Compute FP4 global scale: ``(448 * 6) / amax``.

    448 is the max representable FP8-E4M3 value and 6 is the max FP4-E2M1
    value; their product bounds the NvFP4 dynamic range.
    """
    return (448 * 6) / tensor.float().abs().nan_to_num().max()


def _quant_fp4(a, a_global_sf, is_sf_swizzled_layout=True):
    """Quantize *a* to packed FP4 with a pre-computed global scale."""
    a_fp4, a_sf = fp4_quantize(
        a.cuda(), a_global_sf.cuda(), 16, False, is_sf_swizzled_layout
    )
    return a_fp4, a_sf, a_global_sf


def _quant_fp4_batches(a, num_experts, is_sf_swizzled_layout=True):
    """Per-expert FP4 quantization (independent global scale per expert)."""
    quant_a, sfs, global_sfs = [], [], []
    for i in range(num_experts):
        g = _calculate_fp4_global_scale_factor(a[i])
        a_fp4, a_sf, _ = _quant_fp4(a[i], g, is_sf_swizzled_layout)
        quant_a.append(a_fp4)
        sfs.append(a_sf)
        global_sfs.append(g)
    return torch.stack(quant_a), torch.stack(sfs), torch.stack(global_sfs)


def _quant_dequant_fp4(a):
    """FP4 quantize-then-dequantize roundtrip.

    Simulates the intermediate-activation quantisation error that occurs
    in the kernel between GEMM1+activation and GEMM2.
    """
    a_global_sf = _calculate_fp4_global_scale_factor(a)
    a_fp4, a_sf = fp4_quantize(a.cuda(), a_global_sf.cuda(), 16, False, True)
    a_pt = e2m1_and_ufp8sf_scale_to_float(
        a_fp4.cpu(),
        a_sf.cpu().reshape(-1),
        (1 / a_global_sf).cpu(),
        16,  # sf_vec_size
        1,   # ufp8_type (E4M3)
        True,  # is_sf_swizzled_layout
    )
    return a_pt.cuda(), a_global_sf


def _e2m1_and_ufp8_scale_batches(mat_fp4, scale_tensor, global_scale_tensor):
    """Batch dequantisation: packed-FP4 + block scales → float32.

    Iterates over the expert (batch) dimension, calling the flashinfer
    dequantisation function per expert.
    """
    num_batches = mat_fp4.size(0)
    scale_tensor = scale_tensor.view(num_batches, -1)
    tensors = [
        e2m1_and_ufp8sf_scale_to_float(
            mat_fp4[b].cpu(),
            scale_tensor[b].cpu().reshape(-1),
            global_scale_tensor[b].cpu(),
            16,  # sf_vec_size
            1,   # ufp8_type (E4M3)
            True,  # is_sf_swizzled_layout
        )
        for b in range(num_batches)
    ]
    return torch.stack(tensors)


# ---- Routing reference ---------------------------------------------------


def _routing_reference_renormalize(expert_logits, top_k, num_experts, padding):
    """TopK → Softmax routing reference (``RoutingMethodType.Renormalize``).

    1. Select the top-*k* experts per token.
    2. Softmax-normalise the selected logits (only across the *k* chosen).
    3. Build the permutation tables that map (token, k) → position in the
       padded, expert-sorted buffer.
    """
    device = expert_logits.device
    expert_logits_cpu = expert_logits.cpu()
    num_tokens = expert_logits_cpu.shape[0]

    # Step 1: TopK → Softmax normalisation
    topk_values, topk_idx = torch.topk(expert_logits_cpu, k=top_k, dim=-1)
    topk_values = F.softmax(topk_values.float(), dim=-1)

    # Build full-expert scores with only the selected entries non-zero.
    scores = torch.zeros_like(expert_logits_cpu)
    for i in range(num_tokens):
        for j in range(top_k):
            scores[i, topk_idx[i, j]] = topk_values[i, j]

    # Step 2: Compute permutation from the (sparse) score tensor.
    topKLogits, topKIndices = torch.topk(scores, top_k, dim=1)

    numTokensPerExpert = torch.zeros(num_experts, dtype=torch.int64)
    expandedTokenIdxToExpert = -torch.ones(num_tokens * top_k, dtype=torch.int64)
    expandedTokenIdxToIdxInExpert = -torch.ones(
        num_tokens * top_k, dtype=torch.int64
    )

    for tokenIdx in range(num_tokens):
        for k in range(top_k):
            expandedIdx = tokenIdx * top_k + k
            expertIndex = topKIndices[tokenIdx, k]
            expandedTokenIdxToExpert[expandedIdx] = expertIndex
            expandedTokenIdxToIdxInExpert[expandedIdx] = numTokensPerExpert[
                expertIndex
            ]
            numTokensPerExpert[expertIndex] += 1

    paddedPrefixSum = torch.zeros(num_experts + 1, dtype=torch.int64)
    for ii in range(num_experts):
        paddedPrefixSum[ii + 1] = paddedPrefixSum[ii] + (
            (numTokensPerExpert[ii] + padding - 1) // padding * padding
        )
    permutedBufferSize = paddedPrefixSum[num_experts].item()

    expandedTokenIdxToPermutedIdx = -torch.ones(
        num_tokens * top_k, dtype=torch.int64
    )
    for tokenIdx in range(num_tokens):
        for k in range(top_k):
            expandedIdx = tokenIdx * top_k + k
            expert = expandedTokenIdxToExpert[expandedIdx]
            offsetInExpert = expandedTokenIdxToIdxInExpert[expandedIdx]
            permutedIdx = paddedPrefixSum[expert] + offsetInExpert
            expandedTokenIdxToPermutedIdx[expandedIdx] = permutedIdx

    return {
        "permutedBufferSize": permutedBufferSize,
        "expandedTokenIdxToPermutedIdx": expandedTokenIdxToPermutedIdx.to(
            device
        ),
        "numTokensPerExpert": numTokensPerExpert.to(device),
        "topKLogits": topKLogits.to(device),
        "topKIndices": topKIndices.to(device),
    }


# ---- Reference MoE forward -----------------------------------------------


def _run_moe_reference(
    hidden_states_float,
    permute_info,
    gemm1_weights_float,
    gemm2_weights_float,
    num_experts,
    num_tokens,
    top_k,
    hidden_size,
    intermediate_size,
    padding,
    gemm1_bias=None,
    gemm2_bias=None,
):
    """Reference MoE forward in pure PyTorch.

    All arithmetic is in float32 (after dequantisation from FP4).  Follows
    ``run_moe_dequant`` from flashinfer's test suite for the
    ``FP4_NVFP4_NVFP4`` quant mode.

    Returns ``(output, c_global_sf)`` where *c_global_sf* is the global
    scale produced by the intermediate FP4 quant-dequant roundtrip (needed
    for the kernel's output-scale computation).
    """
    expanded_idx = permute_info["expandedTokenIdxToPermutedIdx"].cpu()
    num_tok_per_expert = permute_info["numTokensPerExpert"].cpu()
    total_padded = permute_info["permutedBufferSize"]

    # 1. Permute tokens into expert-sorted order.
    permute_out = torch.full(
        (total_padded, hidden_size), float("nan"), device="cuda"
    ).float()
    for i in range(num_tokens):
        for j in range(top_k):
            pid = expanded_idx[i * top_k + j]
            permute_out[pid] = hidden_states_float[i]

    # 2. GEMM1 — per-expert matmul: [T_e, H] @ [2*I, H]^T → [T_e, 2*I]
    gemm1_out = torch.full(
        (total_padded, 2 * intermediate_size), float("nan"), device="cuda"
    ).float()
    pos = 0
    for eidx in range(num_experts):
        n = num_tok_per_expert[eidx].item()
        if n == 0:
            continue
        gemm1_out[pos : pos + n] = (
            permute_out[pos : pos + n] @ gemm1_weights_float[eidx].t()
        )
        if gemm1_bias is not None:
            gemm1_out[pos : pos + n] += gemm1_bias[eidx]
        pos += n
        pos = (pos + padding - 1) // padding * padding

    # 3. SwiGLU activation: silu(gate) * value
    #    Weight layout: first I cols = value ("up"), next I cols = gate.
    act_out = torch.full(
        (total_padded, intermediate_size), float("nan"), device="cuda"
    ).float()
    pos = 0
    for eidx in range(num_experts):
        n = num_tok_per_expert[eidx].item()
        if n == 0:
            continue
        a = gemm1_out[pos : pos + n]
        value = a[:, :intermediate_size]
        gate = a[:, intermediate_size:]
        act_out[pos : pos + n] = F.silu(gate) * value
        pos += n
        pos = (pos + padding - 1) // padding * padding

    # 4. Intermediate FP4 quant-dequant roundtrip.
    act_out, c_global_sf = _quant_dequant_fp4(act_out.to(torch.bfloat16))
    act_out = act_out.float()

    # 5. GEMM2 — per-expert matmul: [T_e, I] @ [H, I]^T → [T_e, H]
    gemm2_out = torch.full(
        (total_padded, hidden_size), float("nan"), device="cuda"
    ).float()
    pos = 0
    for eidx in range(num_experts):
        n = num_tok_per_expert[eidx].item()
        if n == 0:
            continue
        gemm2_out[pos : pos + n] = (
            act_out[pos : pos + n] @ gemm2_weights_float[eidx].t()
        )
        if gemm2_bias is not None:
            gemm2_out[pos : pos + n] += gemm2_bias[eidx]
        pos += n
        pos = (pos + padding - 1) // padding * padding

    # 6. Finalise: weighted sum over each token's top-k experts.
    expert_weight = permute_info["topKLogits"].float()
    output = torch.zeros(num_tokens, hidden_size, dtype=torch.float32, device="cuda")
    for i in range(num_tokens):
        for k in range(top_k):
            pid = expanded_idx[i * top_k + k]
            output[i] += gemm2_out[pid] * expert_weight[i, k]

    return output, c_global_sf


# ---- Accuracy check -------------------------------------------------------


def _check_accuracy(a, b, atol, rtol, percent):
    """Assert that at least *percent* of elements are close.

    Mirrors ``check_accuracy`` from the flashinfer MoE test suite.
    """
    assert torch.isfinite(a).all(), "Non-finite values in reference output"
    assert torch.isfinite(b).all(), "Non-finite values in kernel output"
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"

    close = torch.isclose(a, b, atol=atol, rtol=rtol)
    match_ratio = close.float().mean().item()
    assert match_ratio >= percent, (
        f"Only {match_ratio:.4f} of elements match "
        f"(need {percent:.4f}, atol={atol}, rtol={rtol})"
    )


# ---- Kernel comparison tests ----------------------------------------------


@pytest.mark.skipif(
    not _has_flashinfer_sm100(),
    reason="Requires flashinfer and SM100+ GPU (Blackwell)",
)
class TestReferenceVsKernel:
    """Compare the decomposed-PyTorch reference against the flashinfer
    ``trtllm_fp4_block_scale_moe`` fused kernel."""

    def test_swiglu(self):
        """Standard SwiGLU activation, no bias."""
        E, H, I, T, top_k = 8, 1024, 512, 64, 2
        self._run_comparison(E, H, I, T, top_k)

    def test_swiglu_with_bias(self):
        """Standard SwiGLU activation with GEMM bias."""
        E, H, I, T, top_k = 8, 1024, 512, 64, 2
        torch.manual_seed(1)
        gemm1_bias = torch.randn(E, 2 * I, device="cuda", dtype=torch.float32)
        gemm2_bias = torch.randn(E, H, device="cuda", dtype=torch.float32)
        self._run_comparison(
            E, H, I, T, top_k, gemm1_bias=gemm1_bias, gemm2_bias=gemm2_bias,
        )

    # ------------------------------------------------------------------

    def _run_comparison(
        self, E, H, I, T, top_k, gemm1_bias=None, gemm2_bias=None,
    ):
        padding = 128
        sf_vec_size = 16
        epilogue_tile_m = 128
        torch.manual_seed(0)

        # ---- 1. Generate random inputs --------------------------------
        hidden_states = 2 * torch.randn(
            T, H, device="cuda", dtype=torch.bfloat16
        )
        gemm1_weights = torch.randn(
            E, 2 * I, H, device="cuda", dtype=torch.bfloat16
        )
        gemm2_weights = torch.randn(
            E, H, I, device="cuda", dtype=torch.bfloat16
        )
        expert_logits = torch.randn(
            T, E, device="cuda", dtype=torch.bfloat16
        )

        # ---- 2. Compute routing (Renormalize: TopK → Softmax) ---------
        permute_info = _routing_reference_renormalize(
            expert_logits, top_k, E, padding
        )

        # ---- 3. Quantize weights to FP4 (swizzled for reference) ------
        gemm1_fp4, gemm1_sf_swz, gemm1_gsf = _quant_fp4_batches(
            gemm1_weights, E, is_sf_swizzled_layout=True
        )
        gemm2_fp4, gemm2_sf_swz, gemm2_gsf = _quant_fp4_batches(
            gemm2_weights, E, is_sf_swizzled_layout=True
        )

        # Linear-layout scales (for kernel weight shuffling)
        _, gemm1_sf_lin, _ = _quant_fp4_batches(
            gemm1_weights, E, is_sf_swizzled_layout=False
        )
        _, gemm2_sf_lin, _ = _quant_fp4_batches(
            gemm2_weights, E, is_sf_swizzled_layout=False
        )

        # ---- 4. Quantize hidden states to FP4 -------------------------
        hs_gsf = _calculate_fp4_global_scale_factor(hidden_states)

        # Swizzled – for reference dequantisation
        hs_fp4_ref, hs_sf_ref, _ = _quant_fp4(
            hidden_states, hs_gsf, is_sf_swizzled_layout=True
        )
        # Non-swizzled – for the kernel
        hs_fp4_kern, hs_sf_kern, _ = _quant_fp4(
            hidden_states, hs_gsf, is_sf_swizzled_layout=False
        )
        hs_sf_kern = hs_sf_kern.view(torch.float8_e4m3fn).reshape(T, -1)

        # ---- 5. Dequantize everything for the reference ----------------
        hs_float = e2m1_and_ufp8sf_scale_to_float(
            hs_fp4_ref.cpu(),
            hs_sf_ref.cpu().reshape(-1),
            (1 / hs_gsf).cpu(),
            sf_vec_size,
            1,     # ufp8_type (E4M3)
            True,  # is_sf_swizzled_layout
        ).cuda()

        g1_float = _e2m1_and_ufp8_scale_batches(
            gemm1_fp4, gemm1_sf_swz, 1 / gemm1_gsf
        ).cuda()
        g2_float = _e2m1_and_ufp8_scale_batches(
            gemm2_fp4, gemm2_sf_swz, 1 / gemm2_gsf
        ).cuda()

        # ---- 6. Run reference ------------------------------------------
        ref_output, c_gsf = _run_moe_reference(
            hs_float,
            permute_info,
            g1_float,
            g2_float,
            E, T, top_k, H, I, padding,
            gemm1_bias=gemm1_bias,
            gemm2_bias=gemm2_bias,
        )

        # ---- 7. Prepare shuffled weights for the kernel ----------------
        g1_w = gemm1_fp4.view(torch.float8_e4m3fn).reshape(E, 2 * I, H // 2)
        g1_sf = gemm1_sf_lin.view(torch.float8_e4m3fn).reshape(
            E, 2 * I, H // sf_vec_size
        )
        g2_w = gemm2_fp4.view(torch.float8_e4m3fn).reshape(E, H, I // 2)
        g2_sf = gemm2_sf_lin.view(torch.float8_e4m3fn).reshape(
            E, H, I // sf_vec_size
        )

        cache = {}
        g1ws, g1ss, g2ws, g2ss = [], [], [], []
        for i in range(E):
            # GEMM1 weights
            pi = _maybe_get_cached_w3_w1_permute_indices(
                cache, g1_w[i].view(torch.uint8), epilogue_tile_m,
            )
            g1ws.append(
                g1_w[i].view(torch.uint8)[pi.to(g1_w.device)].contiguous()
            )
            # GEMM1 scales
            pi_sf = _maybe_get_cached_w3_w1_permute_indices(
                cache, g1_sf[i].view(torch.uint8), epilogue_tile_m,
                num_elts_per_sf=16,
            )
            g1ss.append(
                block_scale_interleave(
                    g1_sf[i]
                    .view(torch.uint8)[pi_sf.to(g1_sf.device)]
                    .contiguous()
                )
            )
            # GEMM2 weights
            pi = get_w2_permute_indices_with_cache(
                cache, g2_w[i].view(torch.uint8), epilogue_tile_m,
            )
            g2ws.append(
                g2_w[i].view(torch.uint8)[pi.to(g2_w.device)].contiguous()
            )
            # GEMM2 scales
            pi_sf = get_w2_permute_indices_with_cache(
                cache, g2_sf[i].view(torch.uint8), epilogue_tile_m,
                num_elts_per_sf=16,
            )
            g2ss.append(
                block_scale_interleave(
                    g2_sf[i]
                    .view(torch.uint8)[pi_sf.to(g2_sf.device)]
                    .contiguous()
                )
            )

        g1ws = torch.stack(g1ws)
        g1ss = (
            torch.stack(g1ss)
            .view(torch.float8_e4m3fn)
            .reshape(E, 2 * I, H // sf_vec_size)
        )
        g2ws = torch.stack(g2ws)
        g2ss = (
            torch.stack(g2ss)
            .view(torch.float8_e4m3fn)
            .reshape(E, H, I // sf_vec_size)
        )

        # Output scales (combine global scales for the kernel).
        scale_c_fc1 = c_gsf * (1.0 / gemm1_gsf) * (1.0 / hs_gsf)
        scale_gate_fc1 = (1.0 / gemm1_gsf) * (1.0 / hs_gsf)
        scale_c_fc2 = (1.0 / c_gsf) * (1.0 / gemm2_gsf)

        # ---- 8. Call the fused kernel ----------------------------------
        kernel_output = trtllm_fp4_block_scale_moe(
            routing_logits=expert_logits,
            routing_bias=None,
            hidden_states=hs_fp4_kern,
            hidden_states_scale=hs_sf_kern,
            gemm1_weights=g1ws,
            gemm1_weights_scale=g1ss,
            gemm1_bias=gemm1_bias,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=g2ws,
            gemm2_weights_scale=g2ss,
            gemm2_bias=gemm2_bias,
            output1_scale_scalar=scale_c_fc1,
            output1_scale_gate_scalar=scale_gate_fc1,
            output2_scale_scalar=scale_c_fc2,
            num_experts=E,
            top_k=top_k,
            n_group=None,
            topk_group=None,
            intermediate_size=I,
            local_expert_offset=0,
            local_num_experts=E,
            routed_scaling_factor=None,
            routing_method_type=1,  # Renormalize (TopK → Softmax)
            do_finalize=True,
        )
        kernel_output = kernel_output[0].float()

        # ---- 9. Compare -----------------------------------------------
        _check_accuracy(
            ref_output, kernel_output,
            atol=0.1, rtol=0.85, percent=0.925,
        )
