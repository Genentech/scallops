import logging
from collections.abc import Sequence
from functools import partial

import anndata
import dask
import dask.array as da
from array_api_compat import get_namespace
from sklearn.utils import gen_batches

from scallops.features.util import _anndata_to_xr

logger = logging.getLogger("scallops")


def pca(
    adata: anndata.AnnData,
    n_components: int | float | None = None,
    min_std: float | None = 0,
    standardize: bool = True,
    standardize_by: str | Sequence[str] | None = None,
    max_value: float | None = None,
    batch_size: int | None = None,
    gpu: bool | None = None,
    whiten: bool = False,
    progress: bool = True,
) -> anndata.AnnData:
    """Embed data using PCA.

    :param adata: AnnData object.
    :param standardize: Whether to standardize the data.
    :param standardize_by: Standardize the data specified groups
    :param n_components: Number of PCA components.
    :param min_std: Remove features with standard deviation <= `min_std`.
    :param max_value: Clip to this value after standardizing
    :param batch_size: Batch size for incremental PCA.
    :param gpu: Whether to use GPU.
    :param whiten: Whether to use whitening.
    :param progress: Whether to show progress bar for incremental PCA.
    :return: PCA Embedding
    """

    is_dask = isinstance(adata.X, da.Array)
    if gpu is None:
        try:
            import torch

            gpu = torch.cuda.is_available()
            if gpu:
                logger.info("Using GPU for PCA")
        except ModuleNotFoundError:
            gpu = False
    xp = get_namespace(adata.X)
    if standardize_by is not None:
        xdata = _anndata_to_xr(adata)

        def _standardize(x, min_std, max_value):
            std = x.std(dim="obs")
            if min_std is not None and min_std > 0:
                std = std.where(std.data > min_std)
            x = (x - x.mean(dim="obs")) / std
            if max_value is not None:
                x = x.clip(-max_value, max_value)
            return x

        xdata = xdata.groupby(standardize_by).map(
            partial(_standardize, min_std, max_value)
        )
        X = xdata.data
        non_nan_features = ~xp.isnan(X, axis=0)
        if is_dask:
            non_nan_features = non_nan_features.compute()
        X = X[:, non_nan_features]
        logger.info(f"# of features {X.shape[1]:,} / {adata.X.shape[1]:,}")
        obs = adata.obs.loc[xdata.coords["obs"].values]
    else:
        X = adata.X
        obs = adata.obs
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

            X = X[:, features_keep]
            logger.info(f"# of features {X.shape[1]:,} / {adata.X.shape[1]:,}")
            stds = stds[:, features_keep]
            means = means[:, features_keep]
        if standardize:
            X = (X - means) / stds
            if max_value is not None:
                X = xp.clip(X, -max_value, max_value)

    if batch_size is not None:
        if gpu:
            from cuml.decomposition import IncrementalPCA
        else:
            from sklearn.decomposition import IncrementalPCA

        d = IncrementalPCA(n_components=n_components, whiten=whiten, copy=not is_dask)
        batches = list(gen_batches(X.shape[0], batch_size, min_batch_size=n_components))

        if progress:
            try:
                from tqdm import tqdm
            except ImportError:
                from scallops.utils import _tqdm_shim as tqdm
        else:
            from scallops.utils import _tqdm_shim as tqdm
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

    return anndata.AnnData(X=X_transformed, obs=obs, uns=uns)
