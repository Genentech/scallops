import logging

import dask.array as da
import numpy as np
from anndata import AnnData
from array_api_compat import get_namespace
from sklearn.utils import gen_batches

from scallops.utils import tqdm_func

logger = logging.getLogger("scallops")


class PCA:
    def __init__(
        self,
        n_components: int | float | None = None,
        batch_size: int | None = None,
        gpu: bool | None = None,
        whiten: bool = False,
        progress: bool = True,
    ):
        """Embed data using PCA

        :param n_components: Number of PCA components.
        :param batch_size: Batch size for incremental PCA.
        :param gpu: Whether to use GPU.
        :param whiten: Whether to use whitening.
        :param progress: Whether to show progress bar for incremental PCA.
        """
        self.n_components = n_components
        self.batch_size = batch_size
        self.gpu = gpu
        self.whiten = whiten
        self.progress = progress

    @property
    def components_(self):
        return self.d.components_

    @property
    def mean_(self):
        return self.d.mean_

    @property
    def explained_variance_ratio_(self):
        return self.d.explained_variance_ratio_

    @property
    def explained_variance_(self):
        return self.d.explained_variance_

    def fit(self, X: np.ndarray | da.Array, y: None = None):
        """Fit the model.

        :param X: Training data.
        :param y: Not used, present for API consistency by convention.
        :return: The instance itself.
        """

        is_dask = isinstance(X, da.Array)
        if self.gpu is None:
            try:
                import torch

                gpu = torch.cuda.is_available()
                if gpu:
                    logger.info("Using GPU for PCA")
            except ModuleNotFoundError:
                gpu = False

        if self.batch_size is not None:
            if gpu:
                from cuml.decomposition import IncrementalPCA
            else:
                from sklearn.decomposition import IncrementalPCA

            d = IncrementalPCA(
                n_components=self.n_components, whiten=self.whiten, copy=not is_dask
            )
            batches = list(
                gen_batches(
                    X.shape[0], self.batch_size, min_batch_size=self.n_components or 0
                )
            )
            tqdm, progress_args = tqdm_func(self.progress)
            for batch in tqdm(batches, **progress_args):
                X_batch = X[batch]
                if is_dask:
                    X_batch = X_batch.compute()
                d.partial_fit(X_batch)

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
            kwargs = dict(n_components=self.n_components, whiten=self.whiten)
            if "random_state" in sig.parameters.keys():
                kwargs["random_state"] = 239753
            d = PCA(**kwargs)
            d.fit(X)
        self.d = d
        return self

    def add_uns(self, data: AnnData):
        """Add metadata for storing PCA parameters in uns slot
        :param data: Data to add metadata to.
        """
        data.uns["pca"] = {
            "variance_ratio": self.explained_variance_ratio_,
            "variance": self.explained_variance_,
            "mean": self.mean_,
            "PCs": self.components_,
            "features": data.var.index.values,
        }

    def transform(
        self, X: np.ndarray | da.Array, y: None = None
    ) -> np.ndarray | da.Array:
        """Apply dimensionality reduction.

        :param X: Data to project.
        :param y: Not used, present for API consistency by convention.
        :return: Projection of data
        """

        d = self.d
        components_ = d.components_
        mean_ = d.mean_
        variance = d.explained_variance_

        if mean_ is not None:
            X = X - mean_
        X_transformed = X @ components_.T  # (n_components, n_features)
        if self.whiten:
            X_transformed /= get_namespace(variance).sqrt(variance)
        return X_transformed
