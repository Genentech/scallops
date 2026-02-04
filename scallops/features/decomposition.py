import logging

import anndata
import dask
import dask.array as da
from array_api_compat import get_namespace
from sklearn.utils import gen_batches

logger = logging.getLogger("scallops")


def pca(
    adata: anndata.AnnData,
    n_components: int | float | None = None,
    min_std: float | None = 0,
    standardize: bool = True,
    max_value: float | None = None,
    batch_size: int | None = None,
    chunks: tuple[int, int] | bool = True,
    gpu: bool | None = None,
    whiten: bool = False,
) -> anndata.AnnData:
    """Embed data using PCA.

    :param adata: AnnData object.
    :param standardize: Whether to standardize the data.
    :param n_components: Number of PCA components.
    :param min_std: Remove features with standard deviation <= `min_std`.
    :param max_value: Truncate to this value after scaling
    :param batch_size: Batch size for incremental PCA.
    :param chunks: Rechunk dask array.
    :param gpu: Whether to use GPU.
    :param whiten: Whether to use whitening.
    :return: PCA Embedding
    """
    X = adata.X
    is_dask = isinstance(X, da.Array)
    if gpu is None:
        try:
            import torch

            gpu = torch.cuda.is_available()
        except ModuleNotFoundError:
            gpu = False
    means = None
    stds = None
    if standardize or min_std is not None:
        means = X.mean(axis=0, keepdims=True)
        stds = X.std(axis=0, keepdims=True)

    if min_std is not None:
        if is_dask:
            means, stds = dask.compute(means, stds)
        features_keep = stds > min_std
        features_keep = features_keep.squeeze()
        logger.info(f"# of features {features_keep.sum():,} / {X.shape[1]:,}")
        X = X[:, features_keep]
        stds = stds[:, features_keep]
        means = means[:, features_keep]
    if standardize:
        X = (X - means) / stds
        if max_value is not None:
            X[X > max_value] = max_value
            X[X < -max_value] = -max_value
    if is_dask and chunks:
        if isinstance(chunks, bool):
            if batch_size is not None:
                chunks = (batch_size, -1)
            else:
                chunks = (100000, -1)
                # TODO determine chunk size using available memory
                # total_memory = torch.cuda.get_device_properties(0).total_memory
        X = X.rechunk(chunks)
    if batch_size is not None:
        if gpu:
            from cuml.decomposition import IncrementalPCA
        else:
            from sklearn.decomposition import IncrementalPCA

        d = IncrementalPCA(n_components=n_components, whiten=whiten, copy=not is_dask)
        batches = list(gen_batches(X.shape[0], batch_size, min_batch_size=n_components))
        try:
            from tqdm import tqdm
        except ImportError:

            def tqdm(iterator, *args, **kwargs):
                return iterator

        for batch in tqdm(batches):
            X_batch = X[batch]
            if is_dask:
                X_batch = X_batch.compute()
            d.partial_fit(X_batch)

        #  x = d.transform(X)  # loads everything into memory

    else:
        if not is_dask:
            if gpu:
                from cuml.decomposition import PCA
            else:
                from sklearn.decomposition import PCA
        else:
            if gpu:
                # needs distributed
                from cuml.dask.decomposition import PCA
            else:
                from dask_ml.decomposition import PCA

        import inspect

        sig = inspect.signature(PCA)
        kwargs = dict(n_components=n_components, whiten=whiten)
        if "random_state" in sig.parameters.keys():
            kwargs["random_state"] = 239753
        d = PCA(**kwargs)
        d.fit(X)

    components_ = d.components_
    mean_ = d.mean_
    variance_ratio = d.explained_variance_ratio_
    variance = d.explained_variance_
    X_transformed = X @ components_.T  # (n_components, n_features)
    X_transformed -= get_namespace(mean_).reshape(mean_, (1, -1)) @ components_.T

    uns = {
        "pca": {
            "variance_ratio": variance_ratio,
            "variance": variance,
            "mean": mean_,
            "PCs": components_,
        }
    }

    return anndata.AnnData(X=X_transformed, obs=adata.obs.copy(), uns=uns)
