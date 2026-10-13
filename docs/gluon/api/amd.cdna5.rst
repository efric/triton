AMD CDNA 5
==========

.. currentmodule:: triton.experimental.gluon.language.amd.cdna5

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: autosummary/gluon-module.rst

    async_copy
    cluster
    mbarrier
    tdm


.. autosummary::
    :toctree: generated
    :nosignatures:

    buffer_load
    buffer_store
    cooperative_atomic_load
    cooperative_atomic_store
    get_scaled_upcast_fp4_scale_layout
    get_wmma_scale_layout
    load_shared_fp4_repacked
    make_partitioned_dot_layouts
    scaled_downcast
    scaled_upcast
    wmma
    wmma_scaled
    PartitionedSharedLayout

Cooperative global memory accesses
---------------------------------

``cooperative_atomic_load`` and ``cooperative_atomic_store`` access independent,
naturally aligned 128-byte regions through tensors of global pointers. The
``bytes_per_lane`` argument fixes the LLVM intrinsic: 4, 8, or 16 bytes per lane
use groups of 32, 16, or 8 consecutive lanes in a wave32, respectively.

A mask may enable different complete regions within a wave. It must be provably
constant over every element of each region. For example, a layout for four
packets of 32-bit words can express a runtime packet count as follows inside a
Gluon kernel:

.. code-block:: python

    layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [1, 1], [1, 0])
    packet = gl.arange(0, 4, layout=gl.SliceLayout(1, layout))
    word = gl.arange(0, 32, layout=gl.SliceLayout(0, layout))
    pointers = base + packet[:, None] * 32 + word[None, :]
    values = gl.amd.cdna5.cooperative_atomic_load(
        pointers, mask=packet[:, None] < packet_count, other=0,
        bytes_per_lane=16, sem="relaxed", scope="sys",
    )

Only enabled packets need storage. A partial payload in an enabled packet still
requires all 128 bytes to be allocated and accessed. The intrinsic preserves
the complete packet snapshot; packet format, readiness checks, and storage
reuse belong to the communication protocol.
