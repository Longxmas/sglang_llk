"""Self-contained host-side helpers for the KDA MTP CuTe-DSL decode/verify kernels.

Vendored verbatim from cuLA (``cula/ops/kda/decode/cute.py``) so the kernel files
in this package depend only on ``torch`` and ``cuda.bindings`` / ``cutlass`` -- there
is no runtime dependency on the ``cula`` package.

State layout note: the kernels use the FLA-compatible "vk" state layout
``state.shape == (pool_size, HV, V, K)`` (V outer, K contiguous/last).
"""

import cuda.bindings.driver as cuda
import torch

# Stream cache: (device, torch_stream_id) -> cuda.bindings CUstream. The active
# torch stream is converted once per (device, stream) and reused across launches.
_stream_cache: dict[tuple, cuda.CUstream] = {}

# Kernel tuning constants. The decode/verify path is tuned around K=128.
TILE_K = 128
NUM_THREADS = 128


def _get_cached_stream(device: torch.device):
    """Convert the active torch stream to a cached cuda.bindings CUstream."""
    stream_id = int(torch.cuda.current_stream(device=device).cuda_stream)
    cache_key = (str(device), stream_id)
    if cache_key not in _stream_cache:
        _stream_cache[cache_key] = cuda.CUstream(stream_id)
    return _stream_cache[cache_key]


def _normalize_A_log(A_log: torch.Tensor, HV: int) -> torch.Tensor:
    if A_log.numel() != HV:
        raise ValueError(f"Unexpected A_log shape: {A_log.shape}; expected numel={HV}")
    return A_log.reshape(HV).contiguous()


def _normalize_dt_bias(dt_bias: torch.Tensor, HV: int, K: int) -> torch.Tensor:
    if dt_bias.numel() != HV * K:
        raise ValueError(f"Unexpected dt_bias shape: {dt_bias.shape}; expected numel={HV * K}")
    return dt_bias.reshape(HV, K).contiguous()


def _canonicalize_state_layout(state_layout: str | None) -> str:
    """Accept only the two explicit state layouts used by the kernel.

    Internal meaning:
        - "vk": state shape (..., V, K)
        - "kv": state shape (..., K, V)
    """
    if state_layout is None:
        return "vk"

    normalized = str(state_layout).strip().lower()
    if normalized not in ("vk", "kv"):
        raise ValueError(f"Unsupported state_layout={state_layout}; expected only 'vk' or 'kv'")
    return normalized


def _normalize_state_source(initial_state_source, *, N, HV, K, V, device, state_layout="vk"):
    """Validate that the incoming state already matches the requested layout."""
    if initial_state_source is None:
        if state_layout == "vk":
            h0_source = torch.zeros(N, HV, V, K, dtype=torch.float32, device=device)
            return h0_source, N, False
        h0_source = torch.zeros(N, HV, K, V, dtype=torch.float32, device=device)
        return h0_source, N, True

    if initial_state_source.dim() != 4:
        raise ValueError(f"Unexpected initial_state_source shape: {initial_state_source.shape}; expected a 4D state tensor")

    if initial_state_source.shape[1] != HV:
        raise ValueError(f"Unexpected initial_state_source shape: {initial_state_source.shape}; expected HV={HV}")

    if state_layout == "vk":
        if initial_state_source.shape[2:] != (V, K):
            raise ValueError(
                f"State layout mismatch for state_layout='vk': got {initial_state_source.shape}, expected (..., {HV}, {V}, {K})"
            )
        return initial_state_source, initial_state_source.shape[0], False

    if initial_state_source.shape[2:] != (K, V):
        raise ValueError(
            f"State layout mismatch for state_layout='kv': got {initial_state_source.shape}, expected (..., {HV}, {K}, {V})"
        )
    return initial_state_source, initial_state_source.shape[0], True


def _normalize_state_indices(initial_state_indices, *, N, pool_size, device):
    """Normalize state indices for decode.

    For compatibility callers, missing indices default to a sequential mapping.
    """
    if initial_state_indices is None:
        if pool_size < N:
            raise ValueError(f"initial_state_source only has pool_size={pool_size}, but N={N}")
        return torch.arange(N, device=device, dtype=torch.int32)

    indices = initial_state_indices.to(device=device, dtype=torch.int32)
    if indices.numel() != N:
        raise ValueError(f"Unexpected initial_state_indices shape: {initial_state_indices.shape}; expected numel={N}")
    return indices.contiguous()


def _prepare_output_tensor(q: torch.Tensor, out: torch.Tensor | None, shape: tuple[int, ...]) -> torch.Tensor:
    if out is None:
        return q.new_empty(shape, dtype=torch.bfloat16)
    if out.shape != shape:
        raise ValueError(f"Unexpected out shape: {out.shape}; expected {shape}")
    if out.device != q.device:
        raise ValueError(f"Unexpected out device: {out.device}; expected {q.device}")
    if out.dtype != torch.bfloat16:
        raise ValueError(f"Unexpected out dtype: {out.dtype}; expected torch.bfloat16")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")
    return out
