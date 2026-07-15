"""Vendored cuLA KDA MTP CuTe-DSL decode/verify kernels.

Self-contained: depends only on ``torch`` + ``cutlass`` (CuTe DSL); no dependency
on the ``cula`` package.

- ``kda_decode_mtp`` handles single-token decode (T=1) and the recurrent
  multi-token speculative verify (T>1, via ``intermediate_states_buffer``
  snapshots and ``disable_state_update``).
- ``kda_decode_mtp_kvbuffer`` is the chunkwise speculative verify: it writes a
  compact ``(d, k, g)`` scratch triple per draft token instead of a per-token
  full-state snapshot; ``kda_flush_kvbuffer_all_layers`` folds that scratch into
  the committed state at the rollback (accept-length) stage.
"""

from .mtp import kda_decode_mtp
from .mtp_kvbuffer import kda_decode_mtp_kvbuffer, kda_flush_kvbuffer_all_layers

__all__ = [
    "kda_decode_mtp",
    "kda_decode_mtp_kvbuffer",
    "kda_flush_kvbuffer_all_layers",
]
