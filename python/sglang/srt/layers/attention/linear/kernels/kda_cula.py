from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)


class CulaKDAKernel(LinearAttnKernelBase):
    """cuLA CuTe DSL kernel for KDA (Kimi Delta Attention) speculative verify.

    ``target_verify`` runs the KDA sigmoid-gating delta-rule recurrence over the
    ``T`` draft tokens of each request, writing per-token intermediate SSM
    snapshots into ``intermediate_states_buffer`` (FLA-compatible "vk" state
    layout, ``(pool, HV, V, K)``). The committed state is never mutated during
    verify (``disable_state_update=True``); SGLang's rollback stage commits the
    accepted-length snapshot afterwards.

    ``decode`` (single-token, ``T=1``) is provided so the kernel is a complete
    ``LinearAttnKernelBase`` implementation, but the upstream KDA decode path
    stays on Triton / CuTe DSL -- this class is selected only for the
    ``verify_kernel`` slot (``--linear-attn-verify-backend cula``).

    The vendored CuTe DSL op (:mod:`sglang.jit_kernel.cula_kda`) is imported
    lazily so importing this module never requires the cutlass DSL toolchain.
    """

    def __init__(self):
        super().__init__()
        self._kda_decode_mtp = None
        self._kda_decode_mtp_replayssm = None

    def _op(self):
        if self._kda_decode_mtp is None:
            from sglang.jit_kernel.cula_kda import kda_decode_mtp

            self._kda_decode_mtp = kda_decode_mtp
        return self._kda_decode_mtp

    def _replayssm_op(self):
        if self._kda_decode_mtp_replayssm is None:
            from sglang.jit_kernel.cula_kda import kda_decode_mtp_kvbuffer

            self._kda_decode_mtp_replayssm = kda_decode_mtp_kvbuffer
        return self._kda_decode_mtp_replayssm

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        kda_decode_mtp = self._op()

        # SGLang decode layout: q/k [1, N, H, K], v [1, N, HV, V] (one tok/req).
        # cuLA wants batched [N, T=1, *]; move the request axis to front.
        N = q.shape[1]
        K = q.shape[3]
        HV, V = v.shape[2], v.shape[3]
        q_c = q.transpose(0, 1).contiguous()  # [N, 1, H, K]
        k_c = k.transpose(0, 1).contiguous()
        v_c = v.transpose(0, 1).contiguous()  # [N, 1, HV, V]
        a_c = a.reshape(N, 1, HV, K).contiguous()
        b_c = b.reshape(N, 1, HV).contiguous()

        o = kda_decode_mtp(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q_c,
            k=k_c,
            v=v_c,
            a=a_c,
            b=b_c,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            scale=K**-0.5,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            state_layout="vk",
            disable_state_update=False,  # decode commits the recurrent state
            lower_bound=kwargs.get("lower_bound"),
        )
        # cuLA o: [N, 1, HV, V] -> SGLang decode layout [1, N, HV, V].
        return o.transpose(0, 1)

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> tuple:
        raise NotImplementedError(
            "CulaKDAKernel provides KDA MTP decode / target_verify only; "
            "KDA prefill stays on the Triton chunk kernel."
        )

    def target_verify(
        self,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        intermediate_states_buffer: Optional[torch.Tensor] = None,
        cache_steps: Optional[int] = None,
        replayssm_d: Optional[torch.Tensor] = None,
        replayssm_k: Optional[torch.Tensor] = None,
        replayssm_g: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """MTP target-verify (topk=1 linear chain), over ``T = cache_steps`` draft
        tokens per request. The committed state is never mutated during verify
        (``disable_state_update=True``). Two scratch strategies:

        * **recurrent** (default): a per-token post-state snapshot is written into
          ``intermediate_states_buffer``; SGLang commits the accept-length prefix.
        * **ReplaySSM, ring-free** (when ``replayssm_d`` is provided): a compact
          ``(d, k, g)`` scratch triple is written per draft token and folded into
          the committed state at the rollback stage.
        """
        # SGLang verify layout: q/k [1, N*T, H, K], v [1, N*T, HV, V] (req-major),
        # a [N*T, HV*K], b [N*T, HV]. cuLA wants [N, T, *].
        T = int(cache_steps)
        seq = q.shape[1]
        N = seq // T
        K = q.shape[3]
        HV, V = v.shape[2], v.shape[3]
        H = q.shape[2]

        # Strided K-contiguous views; the cuLA kernel consumes the dynamic-layout
        # variant directly without a contiguous copy.
        q_c = q.reshape(N, T, H, K)
        k_c = k.reshape(N, T, H, K)
        v_c = v.reshape(N, T, HV, V)
        a_c = a.reshape(N, T, HV, K).contiguous()
        b_c = b.reshape(N, T, HV).contiguous()
        common = dict(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q_c,
            k=k_c,
            v=v_c,
            a=a_c,
            b=b_c,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            scale=K**-0.5,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            disable_state_update=True,  # never touch committed state in verify
            lower_bound=kwargs.get("lower_bound"),
        )

        if replayssm_d is not None:
            o = self._replayssm_op()(
                **common,
                emit_output=True,
                d_buffer=replayssm_d[:N],
                k_buffer=replayssm_k[:N],
                g_buffer=replayssm_g[:N],
            )
        else:
            o = self._op()(
                **common,
                state_layout="vk",
                intermediate_states_buffer=(
                    intermediate_states_buffer[:N]
                    if intermediate_states_buffer is not None
                    else None
                ),
            )
        # cuLA o: [N, T, HV, V] -> SGLang verify layout [1, N*T, HV, V].
        return o.reshape(1, seq, HV, V)
