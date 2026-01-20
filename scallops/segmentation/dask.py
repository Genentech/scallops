import functools
import operator

import dask
import dask.array as da
import numpy as np
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
