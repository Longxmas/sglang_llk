from typing import Optional

import torch

from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)


class CulaKDAKernel(LinearAttnKernelBase):
    """cuLA CuTe DSL kernel for KDA (Kimi Delta Attention) MTP decode / verify.

    Provides two paths:

    * ``decode`` -- single-token (T=1) recurrence via ``cula.ops.kda_decode_mtp``.
    * ``target_verify`` -- speculative multi-token (T draft tokens) verify. Two
      kernel families, selected by whether the kvbuffer scratch buffers are
      supplied via kwargs:

        - **recurrent** (default): ``cula.ops.kda_decode_mtp`` writing per-token
          intermediate SSM snapshots into ``intermediate_states_buffer`` (VK
          layout, indexed by request 0..N-1), committed by SGLang's state scatter.
        - **kvbuffer** (``u_buffer``/``kinv_buffer``/``b_buffer``): the chunkwise
          ``cula.ops.kda_decode_mtp_kvbuffer`` writing those three scratch buffers,
          committed later by ``kda_flush_kvbuffer`` at the rollback stage.

    Prefill is intentionally NOT provided here: the KDA prefill stays on the
    Triton chunk kernel (the dispatcher keeps ``extend_kernel`` on Triton when
    the prefill backend is ``cula``). The cuLA MTP ops (``cula.ops.*``) are
    imported lazily so importing this module never requires the cutlass DSL
    toolchain.
    """

    def __init__(self):
        super().__init__()
        # cuLA MTP decode/verify ops, imported lazily.
        self._kda_decode_mtp = None
        self._kda_decode_mtp_kvbuffer = None

    def _recurrent(self):
        if self._kda_decode_mtp is None:
            from cula.ops.kda_decode_mtp import kda_decode_mtp

            self._kda_decode_mtp = kda_decode_mtp
        return self._kda_decode_mtp

    def _kvbuffer(self):
        if self._kda_decode_mtp_kvbuffer is None:
            from cula.ops.kda_decode_mtp_kvbuffer import kda_decode_mtp_kvbuffer

            self._kda_decode_mtp_kvbuffer = kda_decode_mtp_kvbuffer
        return self._kda_decode_mtp_kvbuffer

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
        kda_decode_mtp = self._recurrent()

        # SGLang decode layout: q/k [1, N, H, K], v [1, N, HV, V] (one tok/req).
        # cuLA wants batched [N, T=1, *]; move the request axis to front.
        N = q.shape[1]
        H, K = q.shape[2], q.shape[3]
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
            "CulaKDAKernel provides KDA MTP decode/target_verify only; "
            "KDA prefill stays on the Triton chunk kernel (the dispatcher keeps "
            "extend on Triton when the prefill backend is 'cula')."
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
        intermediate_state_indices: Optional[torch.Tensor] = None,
        cache_steps: Optional[int] = None,
        retrieve_parent_token: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """MTP target-verify (topk=1 linear chain only)."""
        # SGLang verify layout: q/k [1, N*T, H, K], v [1, N*T, HV, V] (req-major),
        # a [N*T, HV*K], b [N*T, HV]. cuLA wants [N, T, *].
        T = int(cache_steps)
        seq = q.shape[1]
        N = seq // T
        H, K = q.shape[2], q.shape[3]
        HV, V = v.shape[2], v.shape[3]

        q_c = q.reshape(N, T, H, K).contiguous()
        k_c = k.reshape(N, T, H, K).contiguous()
        v_c = v.reshape(N, T, HV, V).contiguous()
        a_c = a.reshape(N, T, HV, K).contiguous()
        b_c = b.reshape(N, T, HV).contiguous()
        scale = K**-0.5

        u_buffer = kwargs.get("u_buffer")
        kinv_buffer = kwargs.get("kinv_buffer")
        b_buffer = kwargs.get("b_buffer")

        if u_buffer is not None:
            # kvbuffer (chunkwise cg/tp) verify: writes u/kinv/b scratch, the full
            # state is reconstructed at rollback by kda_flush_kvbuffer.
            kda_decode_mtp_kvbuffer = self._kvbuffer()
            o = kda_decode_mtp_kvbuffer(
                A_log=A_log,
                dt_bias=dt_bias,
                q=q_c,
                k=k_c,
                v=v_c,
                a=a_c,
                b=b_c,
                initial_state_source=ssm_states,
                initial_state_indices=cache_indices,
                scale=scale,
                use_qk_l2norm_in_kernel=True,
                softplus_beta=1.0,
                softplus_threshold=20.0,
                disable_state_update=True,
                emit_output=True,
                u_buffer=u_buffer,
                kinv_buffer=kinv_buffer,
                b_buffer=b_buffer,
                lower_bound=kwargs.get("lower_bound"),
            )
        else:
            # recurrent (vk/ws) verify: per-token intermediate SSM snapshots into
            # intermediate_states_buffer (sliced to the current batch of N reqs).
            kda_decode_mtp = self._recurrent()
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
                scale=scale,
                use_qk_l2norm_in_kernel=True,
                softplus_beta=1.0,
                softplus_threshold=20.0,
                state_layout="vk",
                disable_state_update=True,  # never touch committed state in verify
                intermediate_states_buffer=(
                    intermediate_states_buffer[:N]
                    if intermediate_states_buffer is not None
                    else None
                ),
                lower_bound=kwargs.get("lower_bound"),
            )
        # cuLA o: [N, T, HV, V] -> SGLang verify layout [1, N*T, HV, V].
        return o.reshape(1, seq, HV, V)
