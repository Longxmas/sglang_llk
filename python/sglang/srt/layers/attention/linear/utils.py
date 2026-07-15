from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Optional

from sglang.srt.utils.common import rank0_log

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class LinearAttnKernelBackend(Enum):
    TRITON = "triton"
    CUTEDSL = "cutedsl"
    FLASHINFER = "flashinfer"
    FLASHKDA = "flashkda"
    CULA = "cula"
    # cuLA verify using a ring-free ReplaySSM scratch: a compact (d,k,g) triple
    # per draft token, folded into the state at rollback, instead of the
    # recurrent per-token full-state snapshot. (Same ReplaySSM idea as the decode
    # ring, but rewritten in full each verify pass rather than advanced by a
    # cursor.)
    CULA_REPLAYSSM = "cula-replayssm"
    CUSTOM = "custom"

    @classmethod
    def _missing_(cls, value):
        return cls.CUSTOM

    def is_triton(self):
        return self == LinearAttnKernelBackend.TRITON

    def is_cutedsl(self):
        return self == LinearAttnKernelBackend.CUTEDSL

    def is_flashinfer(self):
        return self == LinearAttnKernelBackend.FLASHINFER

    def is_flashkda(self):
        return self == LinearAttnKernelBackend.FLASHKDA

    def is_cula(self):
        return self == LinearAttnKernelBackend.CULA

    def is_cula_replayssm(self):
        return self == LinearAttnKernelBackend.CULA_REPLAYSSM

    def is_any_cula(self):
        """True for either cuLA verify variant (recurrent or ReplaySSM)."""
        return self in (
            LinearAttnKernelBackend.CULA,
            LinearAttnKernelBackend.CULA_REPLAYSSM,
        )

    def is_custom(self):
        return self == LinearAttnKernelBackend.CUSTOM


LINEAR_ATTN_DECODE_BACKEND: Optional[LinearAttnKernelBackend] = None
LINEAR_ATTN_PREFILL_BACKEND: Optional[LinearAttnKernelBackend] = None
LINEAR_ATTN_VERIFY_BACKEND: Optional[LinearAttnKernelBackend] = None


def initialize_linear_attn_config(server_args: ServerArgs):
    global LINEAR_ATTN_DECODE_BACKEND
    global LINEAR_ATTN_PREFILL_BACKEND
    global LINEAR_ATTN_VERIFY_BACKEND

    base = server_args.linear_attn_backend
    decode = server_args.linear_attn_decode_backend or base
    prefill = server_args.linear_attn_prefill_backend or base
    # The speculative verify kernel is selected independently of decode/prefill:
    # only Triton (default) and cuLA implement target_verify, so it does not
    # inherit `base` (which may be e.g. cutedsl, whose verify is unimplemented).
    verify = server_args.linear_attn_verify_backend or "triton"

    LINEAR_ATTN_DECODE_BACKEND = LinearAttnKernelBackend(decode)
    LINEAR_ATTN_PREFILL_BACKEND = LinearAttnKernelBackend(prefill)
    LINEAR_ATTN_VERIFY_BACKEND = LinearAttnKernelBackend(verify)

    rank0_log(
        f"Linear attention kernel backend: decode={decode}, prefill={prefill}, "
        f"verify={verify}"
    )


def get_linear_attn_decode_backend() -> LinearAttnKernelBackend:
    global LINEAR_ATTN_DECODE_BACKEND
    if LINEAR_ATTN_DECODE_BACKEND is None:
        logger.warning(
            "LINEAR_ATTN_DECODE_BACKEND is not initialized, using triton backend"
        )
        LINEAR_ATTN_DECODE_BACKEND = LinearAttnKernelBackend.TRITON
    return LINEAR_ATTN_DECODE_BACKEND


def get_linear_attn_prefill_backend() -> LinearAttnKernelBackend:
    global LINEAR_ATTN_PREFILL_BACKEND
    if LINEAR_ATTN_PREFILL_BACKEND is None:
        logger.warning(
            "LINEAR_ATTN_PREFILL_BACKEND is not initialized, using triton backend"
        )
        LINEAR_ATTN_PREFILL_BACKEND = LinearAttnKernelBackend.TRITON
    return LINEAR_ATTN_PREFILL_BACKEND


def get_linear_attn_verify_backend() -> LinearAttnKernelBackend:
    global LINEAR_ATTN_VERIFY_BACKEND
    if LINEAR_ATTN_VERIFY_BACKEND is None:
        logger.warning(
            "LINEAR_ATTN_VERIFY_BACKEND is not initialized, using triton backend"
        )
        LINEAR_ATTN_VERIFY_BACKEND = LinearAttnKernelBackend.TRITON
    return LINEAR_ATTN_VERIFY_BACKEND
