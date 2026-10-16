import functools
import operator
from collections.abc import Callable, Sequence
from typing import Any

import dask
import dask.array as da
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import skimage.measure


def block_ndi_label_delayed(block: da.Array) -> tuple[np.ndarray, np.ndarray]:
    """Label a Dask block using scikit-image's label function.

    :param block: Dask array block.
    :return: Tuple containing labeled block and the number of labels in the block.

    Note:
        This function is intended to be used as part of Dask blockwise operations for labeling.
    """
    label_delayed = dask.delayed(skimage.measure.label)
    labeled_block = label_delayed(block)
    n = labeled_block.max()
    n = dask.delayed(np.int32)(n)
    labeled = da.from_delayed(labeled_block, shape=block.shape, dtype=np.int32)
    n = da.from_delayed(n, shape=(), dtype=np.int32)
    return labeled, n


def renumber_labels(image: da.Array) -> da.Array:
    """Renumber labels in a Dask array to ensure global uniqueness.

    :param image: Dask array representing labeled blocks.
    :return: Dask array with globally unique labels.
    """
    # First, label each block independently, incrementing the labels in that
    # block by the total number of labels from previous blocks. This way, each
    # block's labels are globally unique.
    labeled_blocks = np.empty(image.numblocks, dtype=object)

    block_iter = zip(
        np.ndindex(*image.numblocks),
        map(
            functools.partial(operator.getitem, image),
            da.core.slices_from_chunks(image.chunks),
        ),
    )
    index, input_block = next(block_iter)
    labeled_blocks[index], total = block_ndi_label_delayed(input_block)
    for index, input_block in block_iter:
        labeled_block, n = block_ndi_label_delayed(input_block)
        block_label_offset = da.where(labeled_block > 0, total, 0)
        labeled_block += block_label_offset
        labeled_blocks[index] = labeled_block
        total += n

    # Put all the blocks together
    block_labeled = da.block(labeled_blocks.tolist())
    return block_labeled


def _segment_block(
    *blocks: np.ndarray,
    seg_fn: Callable,
    block_depth: tuple[int, ...],
    label_dtype: np.dtype,
    block_info: dict | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Segment an overlapped block, keeping only labels present in the block core.

    Labels that only occur in the overlap are removed (they belong to a neighboring
    block) and the remaining labels are renumbered sequentially starting at 1.
    """
    labels = np.asarray(seg_fn(*blocks, **kwargs))
    spatial_dims = len(block_depth)
    chunk_location = block_info[None]["chunk-location"][-spatial_dims:]
    num_chunks = block_info[None]["num-chunks"][-spatial_dims:]
    core = tuple(
        slice(d if loc > 0 else 0, -d if (loc < n - 1 and d > 0) else None)
        for loc, n, d in zip(chunk_location, num_chunks, block_depth)
    )
    present = np.zeros(labels.max() + 1 if labels.size > 0 else 1, dtype=bool)
    present[labels[core]] = True
    present[0] = False
    lookup = np.cumsum(present, dtype=label_dtype)
    lookup[~present] = 0
    return lookup[labels]


def _block_count(labels: np.ndarray) -> np.ndarray:
    """Number of labels in a sequentially labeled block."""
    count = labels.max() if labels.size > 0 else 0
    return np.array(count, dtype=np.int64).reshape((1,) * labels.ndim)


def _match_labels(a: np.ndarray, b: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Find pairs of labels in two segmentations of the same region with IoU >= threshold.

    :param a: Labels for the shared region from one block.
    :param b: Labels for the shared region from the adjoining block.
    :param iou_threshold: Minimum intersection over union to consider labels identical.
    :return: Array of shape (n, 2) of matching label pairs.
    """
    mask = (a > 0) & (b > 0)
    if not mask.any():
        return np.empty((0, 2), dtype=np.int64)
    a_overlap = a[mask].astype(np.int64)
    b_overlap = b[mask].astype(np.int64)
    n = b_overlap.max() + 1
    keys, intersection = np.unique(a_overlap * n + b_overlap, return_counts=True)
    pairs = np.stack([keys // n, keys % n], axis=1)

    a_ids, a_area = np.unique(a[a > 0], return_counts=True)
    b_ids, b_area = np.unique(b[b > 0], return_counts=True)
    a_area = a_area[np.searchsorted(a_ids, pairs[:, 0])]
    b_area = b_area[np.searchsorted(b_ids, pairs[:, 1])]
    iou = intersection / (a_area + b_area - intersection)
    return pairs[iou >= iou_threshold]


def _match_block_labels(
    a: np.ndarray,
    b: np.ndarray,
    offset_a: np.ndarray,
    offset_b: np.ndarray,
    axis: int,
    block_depth: int,
    iou_threshold: float,
) -> np.ndarray:
    """Match labels in the region shared by overlapped block `a` and its neighbor `b`
    along `axis`.

    :return: Object array containing an array of shape (n, 2) of matching global label
        pairs.
    """
    a = a[(slice(None),) * axis + (slice(-2 * block_depth, None),)]
    b = b[(slice(None),) * axis + (slice(0, 2 * block_depth),)]
    pairs = _match_labels(a, b, iou_threshold)
    pairs[:, 0] += offset_a.item()
    pairs[:, 1] += offset_b.item()
    result = np.empty((1,) * a.ndim, dtype=object)
    result.flat[0] = pairs
    return result


def _label_mapping(
    pairs: np.ndarray | list, total: np.ndarray, dtype: np.dtype
) -> np.ndarray:
    """Compute a mapping from global block labels to merged, sequential labels.

    :param pairs: Object array with one element per pair of adjoining blocks, each
        containing an array of shape (n, 2) of global labels to merge.
    :param total: Total number of labels across all blocks.
    :param dtype: Label data type.
    """
    total = int(total)
    pairs = np.concatenate([np.empty((0, 2), np.int64)] + list(pairs))
    graph = scipy.sparse.coo_matrix(
        (np.ones(len(pairs), dtype=bool), (pairs[:, 0], pairs[:, 1])),
        shape=(total + 1, total + 1),
    )
    # Background (0) is never merged, so it is always component 0, and every label
    # 1..total is present, so the components are numbered 1..k without gaps.
    _, mapping = scipy.sparse.csgraph.connected_components(graph, directed=False)
    return mapping.astype(dtype)


def _relabel_block(
    labels: np.ndarray, offset: np.ndarray, mapping: np.ndarray
) -> np.ndarray:
    """Map block labels to global labels using the block offset."""
    result = mapping[labels + offset.item()]
    result[labels == 0] = 0
    return result


def segment_blocks(
    image: da.Array,
    seg_fn: Callable,
    *args: da.Array,
    depth: int | Sequence[int] = 50,
    spatial_dims: int = 2,
    iou_threshold: float = 0.5,
    dtype: np.dtype = np.int32,
    **kwargs: Any,
) -> da.Array:
    """Segment a Dask array block-wise using overlapping blocks.

    Each block is padded with `depth` elements from its neighbors (via
    :func:`dask.array.map_overlap`) and segmented independently with `seg_fn`.
    A block keeps only the labels that are present in its core (unpadded) region. Labels
    from adjoining blocks that represent the same object are identified by comparing both
    segmentations in the region the blocks share (the overlap on either side of their
    common border) and are merged when their intersection over union is at least
    `iou_threshold`. Finally, labels are renumbered sequentially starting at 1.

    :param image: Image to segment. The last `spatial_dims` dimensions are the spatial
        dimensions; any leading dimensions (e.g. channels) are passed to `seg_fn` whole.
    :param seg_fn: Segmentation function that takes a NumPy block of `image` (followed
        by blocks of `args`) and `kwargs` and returns integer labels with the spatial
        shape of the block.
    :param args: Additional Dask arrays (e.g. nuclei labels or a mask) to pass block-wise
        to `seg_fn`. Their spatial dimensions must match those of `image`.
    :param depth: Number of elements each block shares with its neighbors in each
        spatial dimension. Should be at least as large as the largest object.
    :param spatial_dims: Number of spatial dimensions.
    :param iou_threshold: Minimum intersection over union in the shared region between
        labels from adjoining blocks to merge them into a single label. Values > 0.5
        guarantee that a label is merged with at most one label in each neighbor.
    :param dtype: Label data type.
    :param kwargs: Additional keyword arguments to pass to `seg_fn`.
    :return: Dask array of sequential labels with the spatial shape of `image`.

    :Example:

    .. code-block:: python

        import dask.array as da
        import numpy as np
        import skimage.measure

        image = da.from_array(np.random.random((1000, 1000)) > 0.99, chunks=250)
        labels = segment_blocks(image, skimage.measure.label, depth=10)
    """
    if isinstance(depth, int):
        depth = (depth,) * spatial_dims
    depth = tuple(depth)
    if len(depth) != spatial_dims:
        raise ValueError("Length of depth must equal spatial_dims")

    spatial_chunks = image.chunks[-spatial_dims:]
    arrays = []
    for array in (image,) + args:
        if array.shape[-spatial_dims:] != image.shape[-spatial_dims:]:
            raise ValueError("Spatial shape of all arrays must match image")
        # Non-spatial dimensions are passed whole to seg_fn
        leading_chunks = tuple((s,) for s in array.shape[:-spatial_dims])
        arrays.append(array.rechunk(leading_chunks + spatial_chunks))
    depths = [(0,) * (a.ndim - spatial_dims) + depth for a in arrays]

    # Segment overlapped blocks. Output blocks retain the overlap. Output chunks must be
    # specified explicitly as map_overlap reports the non-overlapped chunks when
    # trim=False.
    overlapped_chunks = da.overlap.overlap(
        arrays[0], depth=depths[0], boundary="none"
    ).chunks[-spatial_dims:]
    labels = da.map_overlap(
        _segment_block,
        *arrays,
        depth=depths,
        boundary="none",
        trim=False,
        align_arrays=False,
        drop_axis=tuple(range(image.ndim - spatial_dims)),
        chunks=overlapped_chunks,
        seg_fn=seg_fn,
        block_depth=depth,
        label_dtype=dtype,
        dtype=dtype,
        meta=np.empty((0,) * spatial_dims, dtype=dtype),
        **kwargs,
    )

    # Labels are made globally unique by offsetting each block by the number of labels
    # in preceding blocks.
    counts = labels.map_blocks(
        _block_count,
        chunks=tuple((1,) * n for n in labels.numblocks),
        dtype=np.int64,
    ).ravel()
    total = counts.sum()
    offsets = (counts.cumsum(axis=0) - counts).reshape(labels.numblocks)
    offsets = offsets.rechunk((1,) * spatial_dims)

    # Match labels between adjoining blocks in the region they share: the last
    # 2 * depth elements of a block and the first 2 * depth elements of its neighbor.
    pairs = []
    for axis in range(spatial_dims):
        if depth[axis] == 0 or labels.numblocks[axis] == 1:
            continue
        first = tuple(
            slice(None, -1) if i == axis else slice(None) for i in range(spatial_dims)
        )
        second = tuple(
            slice(1, None) if i == axis else slice(None) for i in range(spatial_dims)
        )
        axis_pairs = da.map_blocks(
            _match_block_labels,
            labels.blocks[first],
            labels.blocks[second],
            offsets.blocks[first],
            offsets.blocks[second],
            axis=axis,
            block_depth=depth[axis],
            iou_threshold=iou_threshold,
            chunks=tuple((1,) * n for n in offsets.blocks[first].numblocks),
            dtype=object,
            meta=np.empty((0,) * spatial_dims, dtype=object),
        )
        pairs.append(axis_pairs.ravel())
    # Gather all pairs into a single block. Convert to delayed without optimization so
    # that tasks are shared with the rest of the graph instead of being duplicated.
    if len(pairs) > 0:
        pairs = da.concatenate(pairs).rechunk(-1).to_delayed(optimize_graph=False)
        pairs = pairs.item()
    total = total.to_delayed(optimize_graph=False).item()
    mapping = dask.delayed(_label_mapping)(pairs, total, dtype)

    labels = da.overlap.trim_internal(labels, dict(enumerate(depth)), boundary="none")
    return da.map_blocks(
        _relabel_block,
        labels,
        offsets,
        mapping,
        dtype=dtype,
        meta=np.empty((0,) * spatial_dims, dtype=dtype),
    )
