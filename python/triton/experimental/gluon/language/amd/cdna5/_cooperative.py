from ..._core import builtin, distributed_type, tensor, _unwrap_if_constexpr
from ..._layouts import DistributedLayout
from ..._semantic import _check


def _cooperative_atomic_args(pointer, bytes_per_lane, sem, scope, is_store):
    _check(
        isinstance(pointer.type, distributed_type) and pointer.dtype.is_ptr(),
        lambda: "cooperative atomics require a distributed tensor of pointers")
    _check(isinstance(pointer.type.layout, DistributedLayout),
           lambda: "cooperative atomics require an explicit distributed layout")
    element_ty = pointer.dtype.element_ty
    _check((element_ty.is_int() or element_ty.is_floating()) and element_ty.primitive_bitwidth in (8, 16, 32, 64),
           lambda: "cooperative atomics require 8-, 16-, 32-, or 64-bit numeric elements")
    bytes_per_lane = _unwrap_if_constexpr(bytes_per_lane)
    _check(bytes_per_lane in (4, 8, 16) and bytes_per_lane * 8 >= element_ty.primitive_bitwidth,
           lambda: "bytes_per_lane must be 4, 8, or 16 and hold a whole element")
    sem = _unwrap_if_constexpr(sem)
    orderings = {"relaxed": 0, "release" if is_store else "acquire": 3 if is_store else 2, "seq_cst": 5}
    _check(sem in orderings, lambda: f"unsupported cooperative atomic memory semantic: {sem}")
    scope = _unwrap_if_constexpr(scope)
    scopes = {
        "sys": "",
        "gpu": "agent",
        "cta": "workgroup",
        "": "",
        "system": "",
        "agent": "agent",
        "workgroup": "workgroup",
        "cluster": "cluster",
        "wavefront": "wavefront",
        "one-as": "one-as",
        "agent-one-as": "agent-one-as",
        "workgroup-one-as": "workgroup-one-as",
        "cluster-one-as": "cluster-one-as",
        "wavefront-one-as": "wavefront-one-as",
    }
    _check(scope in scopes, lambda: f"unsupported cooperative atomic memory scope: {scope}")
    return bytes_per_lane, orderings[sem], scopes[scope]


@builtin
def cooperative_atomic_load(pointer, *, bytes_per_lane, sem="relaxed", scope="gpu", _semantic=None):
    """Load naturally aligned 128-byte regions cooperatively on CDNA5.

    ``pointer`` is a tensor of global-memory pointers with an explicit layout.
    Each lane loads ``bytes_per_lane`` contiguous bytes per instruction, using
    groups of 32, 16, or 8 consecutive lanes for 4, 8, or 16 bytes respectively.
    Each group must address a contiguous, 128-byte-aligned region. All lanes
    participate, and every accessed byte must be backed by storage. These
    address and allocation requirements are caller preconditions.

    Each 128-byte region is loaded atomically with respect to matching
    cooperative stores. Different regions can be observed independently. The
    result preserves the pointer tensor's shape and layout. Element values are
    transported without numeric conversion.

    Only wave32 and supported global memory/fabric paths are valid. System
    scope does not make host memory reached over PCIe a supported target.

    Args:
        pointer (tensor): Tensor of pointers to 8-, 16-, 32-, or 64-bit numbers.
        bytes_per_lane (constexpr): Fixed access size per lane: 4, 8, or 16.
        sem (constexpr): ``"relaxed"``, ``"acquire"``, or ``"seq_cst"``.
        scope (constexpr): ``"gpu"``, ``"cta"``, or ``"sys"``, or an AMD LLVM
            scope (``agent``, ``workgroup``, ``cluster``, ``wavefront``, or their
            ``-one-as`` forms; ``one-as`` for system scope).

    Returns:
        tensor: The loaded values in the pointer tensor's layout.
    """
    pointer = _semantic.to_tensor(pointer)
    bytes_per_lane, ordering, scope = _cooperative_atomic_args(pointer, bytes_per_lane, sem, scope, False)
    result_ty = pointer.type.with_element_ty(pointer.dtype.element_ty)
    handle = _semantic.builder.create_cooperative_atomic_load(result_ty.to_ir(_semantic.builder), pointer.handle,
                                                              bytes_per_lane, ordering, scope)
    return tensor(handle, result_ty)


@builtin
def cooperative_atomic_store(pointer, value, *, bytes_per_lane, sem="relaxed", scope="gpu", _semantic=None):
    """Store naturally aligned 128-byte regions cooperatively on CDNA5.

    The layout, participation, address, and target requirements are the same as
    :func:`cooperative_atomic_load`. Each region is stored atomically with
    respect to matching cooperative loads and stores. ``value`` is broadcast
    and converted to the pointer's element type, as for ordinary stores.

    Args:
        pointer (tensor): Tensor of global-memory pointers.
        value (tensor or scalar): Values to store.
        bytes_per_lane (constexpr): Fixed access size per lane: 4, 8, or 16.
        sem (constexpr): ``"relaxed"``, ``"release"``, or ``"seq_cst"``.
        scope (constexpr): Synchronization scope; see :func:`cooperative_atomic_load`.
    """
    pointer = _semantic.to_tensor(pointer)
    bytes_per_lane, ordering, scope = _cooperative_atomic_args(pointer, bytes_per_lane, sem, scope, True)
    _check(not pointer.dtype.is_const(), lambda: "cannot store through a constant pointer")
    value = _semantic.to_tensor(value)
    pointer, value, _ = _semantic._broadcast_ptr_val_mask(pointer, value, None)
    value = _semantic.cast(value, pointer.dtype.element_ty)
    _semantic.builder.create_cooperative_atomic_store(pointer.handle, value.handle, bytes_per_lane, ordering, scope)
