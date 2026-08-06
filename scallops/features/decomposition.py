import logging

import anndata
import dask.array as da
from array_api_compat import get_namespace
from sklearn.utils import gen_batches

from scallops.utils import tqdm_func

logger = logging.getLogger("scallops")


def pca(
    data: anndata.AnnData,
    n_components: int | float | None = None,
    batch_size: int | None = None,
    gpu: bool | None = None,
    whiten: bool = False,
    progress: bool = True,
) -> anndata.AnnData:
    """Embed data using PCA.

    :param data: AnnData object.
    :param n_components: Number of PCA components.
    :param batch_size: Batch size for incremental PCA.
    :param gpu: Whether to use GPU.
    :param whiten: Whether to use whitening.
    :param progress: Whether to show progress bar for incremental PCA.
    :return: PCA Embedding
    """
    X = data.X
    is_dask = isinstance(data.X, da.Array)
    if gpu is None:
        try:
            import torch

            gpu = torch.cuda.is_available()
            if gpu:
                logger.info("Using GPU for PCA")
        except ModuleNotFoundError:
            gpu = False
    X_transformed = None
    if batch_size is not None:
        if gpu:
            from cuml.decomposition import IncrementalPCA
        else:
            from sklearn.decomposition import IncrementalPCA

        d = IncrementalPCA(n_components=n_components, whiten=whiten, copy=not is_dask)
        batches = list(gen_batches(X.shape[0], batch_size, min_batch_size=n_components))
        tqdm, progress_args = tqdm_func(progress)
        for batch in tqdm(batches, **progress_args):
            X_batch = X[batch]
            if is_dask:
                X_batch = X_batch.compute()
            d.partial_fit(X_batch)

        # x = d.transform(X)  # loads everything into memory

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
        X_transformed = d.fit_transform(X)

    components_ = d.components_
    mean_ = d.mean_
    variance_ratio = d.explained_variance_ratio_
    variance = d.explained_variance_
    if X_transformed is None:
        if mean_ is not None:
            X = X - mean_
        X_transformed = X @ components_.T  # (n_components, n_features)
        if whiten:
            X_transformed /= get_namespace(variance).sqrt(variance)

    uns = {
        "pca": {
            "variance_ratio": variance_ratio,
            "variance": variance,
            "mean": mean_,
            "PCs": components_,
            "features": data.var.index.values,
        }
    }

    return anndata.AnnData(X=X_transformed, obs=data.obs, uns=uns)
