import pytest
import torch
import torch.nn.functional as F


K = 128
V = 128
H = 32
HV = 32
T = 4
SCALE = K**-0.5


def _make_inputs(n: int):
    torch.manual_seed(107)
    q = torch.randn(n, T, H, K, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(n, T, HV, V, device="cuda", dtype=torch.bfloat16)
    # beta_x = 100 overflows exp(float32). The thresholded softplus branch
    # must return x directly without evaluating that exponential.
    a = torch.full(
        (n, T, HV, K), 100.0, device="cuda", dtype=torch.bfloat16
    )
    b = torch.zeros(n, T, HV, device="cuda", dtype=torch.bfloat16)
    a_log = torch.zeros(HV, device="cuda", dtype=torch.float32)
    dt_bias = torch.zeros(HV, K, device="cuda", dtype=torch.float32)
    state = (
        torch.randn(n, HV, V, K, device="cuda", dtype=torch.float32) * 0.01
    )
    return q, k, v, a, b, a_log, dt_bias, state


def _reference(q, k, v, a, b, a_log, dt_bias, state):
    current = state.clone()
    outputs = []
    snapshots = []
    a_scale = torch.exp(a_log)[None, :, None]
    for step in range(T):
        x = a[:, step].float() + dt_bias[None]
        gate = torch.exp(
            -a_scale * F.softplus(x, beta=1.0, threshold=20.0)
        )
        qf = q[:, step].float()
        kf = k[:, step].float()
        q_norm = (
            qf
            * torch.rsqrt((qf * qf).sum(-1, keepdim=True) + 1e-6)
            * SCALE
        )
        k_norm = kf * torch.rsqrt(
            (kf * kf).sum(-1, keepdim=True) + 1e-6
        )
        state_dot_k = torch.einsum(
            "nhvk,nhk->nhv", current, gate * k_norm
        )
        update = torch.sigmoid(b[:, step].float()).unsqueeze(-1) * (
            v[:, step].float() - state_dot_k
        )
        current = (
            current * gate.unsqueeze(-2)
            + update.unsqueeze(-1) * k_norm.unsqueeze(-2)
        )
        outputs.append(torch.einsum("nhvk,nhk->nhv", current, q_norm))
        snapshots.append(current.clone())
    return torch.stack(outputs, 1), torch.stack(snapshots, 1)


def _assert_finite_close(actual, expected, *, atol=3e-3, rtol=2e-2):
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual.float(), expected.float(), atol=atol, rtol=rtol
    )


@pytest.mark.parametrize(
    ("mode", "n", "expected_route"),
    [
        ("recurrent", 2, "small_vk"),
        ("recurrent", 64, "large_ws"),
        ("replayssm", 1, "shuffle"),
        ("replayssm", 2, "tensor_core"),
    ],
)
def test_kda_cula_extreme_softplus_is_finite(mode, n, expected_route):
    from sglang.kernels.ops.attention.cula_kda.mtp import (
        _WS_WORK_UNIT_THRESHOLD,
        kda_decode_mtp,
    )
    from sglang.kernels.ops.attention.cula_kda.mtp_kvbuffer import (
        _select_kvb_variant,
        kda_decode_mtp_kvbuffer,
        kda_flush_kvbuffer,
    )

    q, k, v, a, b, a_log, dt_bias, initial_state = _make_inputs(n)
    expected_out, expected_states = _reference(
        q, k, v, a, b, a_log, dt_bias, initial_state
    )
    indices = torch.arange(n, device="cuda", dtype=torch.int32)
    state = initial_state.clone()

    common = dict(
        A_log=a_log,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        initial_state_source=state,
        initial_state_indices=indices,
        scale=SCALE,
        use_qk_l2norm_in_kernel=True,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        disable_state_update=True,
    )

    if mode == "recurrent":
        route = (
            "large_ws"
            if n * HV >= _WS_WORK_UNIT_THRESHOLD
            else "small_vk"
        )
        assert route == expected_route
        intermediate = torch.empty(
            n, T, HV, V, K, device="cuda", dtype=torch.float32
        )
        output = kda_decode_mtp(
            **common,
            state_layout="vk",
            intermediate_states_buffer=intermediate,
        )
        torch.cuda.synchronize()
        _assert_finite_close(output, expected_out)
        _assert_finite_close(
            intermediate, expected_states, atol=3e-5, rtol=2e-4
        )
        torch.testing.assert_close(state, initial_state, atol=0, rtol=0)
        return

    route = _select_kvb_variant(n, HV, T)
    assert route == expected_route
    scratch = (
        torch.empty(n, T, HV, V, device="cuda", dtype=torch.float32),
        torch.empty(n, T, HV, K, device="cuda", dtype=torch.float32),
        torch.empty(n, T, HV, K, device="cuda", dtype=torch.float32),
    )
    output = kda_decode_mtp_kvbuffer(
        **common,
        emit_output=True,
        d_buffer=scratch[0],
        k_buffer=scratch[1],
        g_buffer=scratch[2],
    )
    torch.cuda.synchronize()
    _assert_finite_close(output, expected_out)
    for tensor in scratch:
        assert torch.isfinite(tensor).all()
    torch.testing.assert_close(state, initial_state, atol=0, rtol=0)

    accept_len = torch.full(
        (n,), T, device="cuda", dtype=torch.int32
    )
    kda_flush_kvbuffer(
        state, indices, *scratch, accept_len=accept_len
    )
    torch.cuda.synchronize()
    _assert_finite_close(
        state, expected_states[:, -1], atol=3e-5, rtol=2e-4
    )
