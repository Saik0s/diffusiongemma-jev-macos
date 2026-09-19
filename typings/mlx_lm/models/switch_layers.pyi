import mlx.core as mx

def _gather_sort(
    x: mx.array, indices: mx.array
) -> tuple[mx.array, mx.array, mx.array]: ...

def _scatter_unsort(
    x: mx.array, inv_order: mx.array, shape: tuple[int, ...] | None = None
) -> mx.array: ...
