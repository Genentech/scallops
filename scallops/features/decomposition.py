import logging
import warnings
from collections.abc import Sequence
from functools import partial

import anndata
import dask
import dask.array as da
import numpy as np
from array_api_compat import get_namespace
from sklearn.utils import gen_batches

from scallops.features.util import _anndata_to_xr, _slice_anndata

logger = logging.getLogger("scallops")


def _centerscale(
    data: anndata.AnnData,
    min_std: float | None = 0,
    standardize: bool = True,
    standardize_by: str | Sequence[str] | None = None,
    max_value: float | None = None,
) -> anndata.AnnData:
    is_dask = isinstance(data.X, da.Array)
    xp = get_namespace(data.X)
    if standardize and standardize_by is not None:
        xdata = _anndata_to_xr(data, standardize_by)

        def _standardize(x, min_std, max_value):
            std = x.std(dim="obs")
            if min_std is not None and min_std > 0:
                std = std.where(std.data > min_std)
            x = (x - x.mean(dim="obs")) / std
            if max_value is not None:
                x = x.clip(-max_value, max_value)
            return x

        xdata = xdata.groupby(standardize_by).map(
            partial(_standardize, min_std=min_std, max_value=max_value)
        )
        X = xdata.data
        no_nans_per_feature = xp.isnan(X).sum(axis=0) == 0

        if is_dask:
            no_nans_per_feature = no_nans_per_feature.compute()
        X = X[:, no_nans_per_feature]
        logger.info(f"# of features {X.shape[1]:,} / {data.X.shape[1]:,}")
        return anndata.AnnData(
            X=X,
            obs=data.obs.loc[xdata.coords["obs"].values],
            var=data.var[no_nans_per_feature],
        )
    else:
        X = data.X
        var = data.var
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

            stds = stds[:, features_keep]
            means = means[:, features_keep]
            var = data.var[features_keep]
            logger.info(f"# of features {X.shape[1]:,} / {data.X.shape[1]:,}")
        if standardize:
            X = (X - means) / stds
            if max_value is not None:
                X = xp.clip(X, -max_value, max_value)

        return anndata.AnnData(X=X, obs=data.obs.copy(), var=var)


def pca(
    data: anndata.AnnData,
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

    :param data: AnnData object.
    :param standardize: Whether to standardize the data.
    :param standardize_by: Standardize the data specified groups
    :param n_components: Number of PCA components.
    :param min_std: Remove features with standard deviation <= `min_std` after
     standardization.
    :param max_value: Clip to this value after standardizing
    :param batch_size: Batch size for incremental PCA.
    :param gpu: Whether to use GPU.
    :param whiten: Whether to use whitening.
    :param progress: Whether to show progress bar for incremental PCA.
    :return: PCA Embedding
    """
    if standardize:
        data = _centerscale(
            data=data,
            min_std=min_std,
            standardize=standardize,
            standardize_by=standardize_by,
            max_value=max_value,
        )
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


# ---------------------------------------------------------------------------
# Sphering / whitening
# ---------------------------------------------------------------------------


def sphere(
    data: anndata.AnnData,
    by: str | Sequence[str] | None = None,
    epsilon: float = 1e-5,
) -> anndata.AnnData:
    """Apply a sphering (ZCA whitening) transformation.

    Centres and rescales the data so that the sample covariance of the output
    is the identity matrix (up to the regularisation *epsilon*).  When *by*
    is provided the transform is fitted and applied independently within each
    group (e.g. per plate/condition).

    The whitening matrix is ``W = U diag(1/sqrt(s + epsilon)) U^T`` where
    ``U, s, _`` are from the SVD of the (centred) sample covariance.

    :param data: AnnData object.  Must be in memory (dask arrays not
        supported for this step; materialise with ``data.X = data.X.compute()``
        beforehand).
    :param by: Column(s) in ``obs`` to apply the transform per group.
    :param epsilon: Small regularisation constant added to singular values
        before inversion.
    :return: Sphered AnnData with the same shape and ``var`` as *data*.
    """

    def _sphere_block(X: np.ndarray) -> np.ndarray:
        orig_dtype = X.dtype
        Xf = X.astype(np.float64)
        x_centered = Xf - Xf.mean(axis=0)
        cov = np.cov(x_centered, rowvar=False, ddof=1)
        if cov.ndim == 0:
            cov = cov.reshape(1, 1)
        u, s, _ = np.linalg.svd(cov)
        W = u @ np.diag(1.0 / np.sqrt(s + epsilon)) @ u.T
        return (x_centered @ W).astype(orig_dtype)

    if by is not None:
        def _sphere_xr(xdata):
            return xdata.copy(data=_sphere_block(xdata.values), deep=False)

        xdata = _anndata_to_xr(data, by)
        result = xdata.groupby(by).map(_sphere_xr)
        return anndata.AnnData(
            X=result.data,
            obs=data.obs.loc[result.coords["obs"].values],
            var=data.var.copy(),
        )

    # Non-grouped: skip xarray to avoid obs-coordinate index type issues
    X_sphered = _sphere_block(np.asarray(data.X))
    return anndata.AnnData(X=X_sphered, obs=data.obs.copy(), var=data.var.copy())


# ---------------------------------------------------------------------------
# PCA component selection via Tracy-Widom
# ---------------------------------------------------------------------------


def largest_variance_from_random_matrix(
    n_obs: int,
    n_features: int,
    pval: str = "0.05",
) -> float:
    """Return the expected maximum eigenvalue of a random Wishart matrix.

    Uses the Tracy-Widom distribution (order 1) to compute the threshold
    above which a PCA eigenvalue is unlikely to be noise.  See
    *Johnstone (2001)* and *Shekhar et al. (2022)* for derivations.

    :param n_obs: Number of observations (cells).
    :param n_features: Number of features used to fit PCA.
    :param pval: Significance level, either ``"0.05"`` or ``"0.01"``.
    :return: Eigenvalue threshold; components whose explained variance exceeds
        this are considered statistically significant.
    """
    quantiles = {"0.01": 2.023335, "0.05": 0.9792895}
    if pval not in quantiles:
        raise ValueError(f"pval must be '0.01' or '0.05', got {pval!r}")
    val1 = (n_obs - 1) ** 0.5
    val2 = n_features ** 0.5
    mu = (val1 + val2) ** 2
    sigma = (val1 + val2) * (1.0 / val1 + 1.0 / val2) ** (1.0 / 3.0)
    return (quantiles[pval] * sigma + mu) / (n_obs - 1)


def _permutation_eigenvalue_thresholds(
    X: np.ndarray,
    n_components: int,
    n_perms: int,
    pval: float,
    seed: int,
) -> np.ndarray:
    """Estimate per-rank eigenvalue thresholds by column-independent permutation.

    Each column of *X* is shuffled independently, destroying feature-feature
    correlations while preserving each feature's marginal distribution.  This
    gives an empirical null that accounts for non-Gaussian marginals (unlike
    Tracy-Widom which assumes Gaussian entries) while removing the between-
    feature signal.

    :param X: Data matrix ``(n_obs, n_features)`` in z-score space.
    :param n_components: Number of PCA components to evaluate.
    :param n_perms: Number of permutation replicates.
    :param pval: Quantile for the threshold (e.g. 0.05 → 95th percentile).
    :param seed: Random seed.
    :return: Array of shape ``(n_components,)`` with per-rank thresholds.
    """
    from sklearn.decomposition import PCA as skPCA

    rng = np.random.default_rng(seed)
    n, p = X.shape
    n_fit = min(n - 1, p, n_components)
    perm_variances = np.empty((n_perms, n_fit), dtype=np.float64)

    for k in range(n_perms):
        # Shuffle each column independently → destroys feature-feature correlations
        X_perm = rng.permuted(X, axis=0)  # permutes each column independently
        d = skPCA(n_components=n_fit)
        d.fit(X_perm)
        perm_variances[k] = d.explained_variance_

    return np.quantile(perm_variances, 1.0 - pval, axis=0)


def select_pca_components(
    data: anndata.AnnData,
    method: str = "variance",
    min_variance_fraction: float = 0.95,
    pval: float | str = 0.05,
    n_perms: int = 100,
    seed: int = 0,
    max_components: int | None = None,
    pca_uns_key: str = "pca",
    n_features: int | None = None,
) -> anndata.AnnData:
    """Retain only statistically significant or informative PCA components.

    Three methods are provided, with different suitability for morphological
    profiling data (Cell Painting, OPS):

    **``"variance"`` (recommended for correlated features)**
        Keep the minimum number of PCs needed to explain *min_variance_fraction*
        of the total explained variance.  Completely immune to the feature-
        correlation problem because the threshold is relative to the data's own
        variance.

    **``"permutation"`` (statistically principled for non-Gaussian features)**
        Each feature column is independently shuffled *n_perms* times, which
        destroys between-feature correlations while preserving each feature's
        marginal distribution.  The ``(1 − pval)``-quantile of the per-rank
        permutation eigenvalues is used as the threshold.  More accurate than
        Tracy-Widom when feature distributions are non-Gaussian, but
        computationally slower (requires *n_perms* PCA fits).

    **``"tracy_widom"`` (original implementation — use with caution)**
        Applies the Tracy-Widom distribution from *Johnstone (2001)*.  This
        test assumes that, under the null, the data matrix has i.i.d. Gaussian
        entries — an assumption **strongly violated** by morphological profiling
        data where features are highly correlated (e.g. all intensity features
        in the same compartment).  For CellPainting-scale data the test will
        typically retain all PCs regardless of noise level, making it
        effectively useless as a selection criterion.  Use ``"variance"`` or
        ``"permutation"`` instead.  Tracy-Widom is retained for compatibility
        with the gould pipeline and for datasets where features are known to be
        approximately uncorrelated.

    .. warning::
       For optical / morphological profiling data, **do not use
       ``method="tracy_widom"``** as the primary selection criterion.
       Feature correlations (e.g. ρ ≈ 0.7–0.9 within CellProfiler compartments)
       inflate all eigenvalues above the Tracy-Widom threshold even under a
       pure-noise null, causing the test to retain every component.  Use
       ``method="variance"`` (fast, interpretable) or ``method="permutation"``
       (slower, non-parametric null).

    :param data: AnnData in PCA space, as produced by :func:`pca`.  Must
        contain ``uns[pca_uns_key]["variance"]`` and
        ``uns[pca_uns_key]["variance_ratio"]``.
    :param method: Selection strategy.  One of ``"variance"`` (default),
        ``"permutation"``, or ``"tracy_widom"``.
    :param min_variance_fraction: Minimum cumulative variance fraction to retain
        (used by ``method="variance"``).  Default ``0.95``.
    :param pval: Significance level used by ``"permutation"`` and
        ``"tracy_widom"``.  For ``"permutation"`` this is a float (e.g.
        ``0.05``); for ``"tracy_widom"`` it is a string (``"0.05"`` or
        ``"0.01"``).  Default ``0.05``.
    :param n_perms: Number of permutation replicates for
        ``method="permutation"``.  Default ``100``.
    :param seed: Random seed for ``method="permutation"``.
    :param max_components: Hard upper cap on the number of retained components,
        applied after all other selection logic.  *None* means no cap.
    :param pca_uns_key: Key in ``uns`` where PCA metadata is stored.
    :param n_features: Number of original features used to fit PCA (used by
        ``"tracy_widom"``).  Inferred from ``uns[pca_uns_key]["features"]``
        when *None*.
    :return: Sliced AnnData retaining only the selected PC columns.
    """
    pca_info = data.uns.get(pca_uns_key, {})
    variance = np.asarray(pca_info.get("variance", []))
    variance_ratio = np.asarray(pca_info.get("variance_ratio", []))
    if variance.size == 0:
        raise ValueError(
            f"uns['{pca_uns_key}']['variance'] is missing or empty.  "
            "Run map-pca (or scallops.features.decomposition.pca) first."
        )

    n_total = int(variance.size)

    if method == "variance":
        # ------------------------------------------------------------------ #
        # Cumulative variance fraction — robust to feature correlations
        # ------------------------------------------------------------------ #
        if variance_ratio.size == 0:
            # Fall back: normalize variance to sum-to-1
            variance_ratio = variance / variance.sum()
        cum = np.cumsum(variance_ratio)
        n_keep = int(np.searchsorted(cum, min_variance_fraction)) + 1
        n_keep = min(n_keep, n_total)
        logger.info(
            f"select_pca_components (method=variance): retaining {n_keep} / {n_total} "
            f"components to explain ≥{min_variance_fraction:.0%} of variance"
        )

    elif method == "permutation":
        # ------------------------------------------------------------------ #
        # Column-independent permutation null — accounts for non-Gaussianity
        # ------------------------------------------------------------------ #
        pca_components = pca_info.get("PCs")
        pca_mean = pca_info.get("mean")
        if pca_components is None:
            raise ValueError(
                f"uns['{pca_uns_key}']['PCs'] is required for method='permutation'."
            )
        # Back-project to feature space for a meaningful permutation
        X = np.asarray(data.X, dtype=np.float64)
        PCs = np.asarray(pca_components, dtype=np.float64)   # (n_pcs, n_features)
        if pca_mean is not None:
            X_feat = X @ PCs + np.asarray(pca_mean)          # (n_obs, n_features)
        else:
            X_feat = X @ PCs

        pval_float = float(pval)
        thresholds = _permutation_eigenvalue_thresholds(
            X_feat, n_components=n_total, n_perms=n_perms, pval=pval_float, seed=seed
        )
        keep_mask = variance[:len(thresholds)] > thresholds
        n_keep = max(int(keep_mask.sum()), 1)
        logger.info(
            f"select_pca_components (method=permutation, n_perms={n_perms}, "
            f"p={pval_float}): retaining {n_keep} / {n_total} components"
        )

    elif method == "tracy_widom":
        # ------------------------------------------------------------------ #
        # Tracy-Widom — assumes i.i.d. Gaussian entries: AVOID for correlated
        # features such as Cell Painting / morphological profiling data.
        # ------------------------------------------------------------------ #
        warnings.warn(
            "select_pca_components: method='tracy_widom' assumes i.i.d. Gaussian "
            "entries under the null.  Morphological profiling features are highly "
            "correlated (e.g. ρ≈0.7–0.9 within CellProfiler compartments), which "
            "inflates all eigenvalues above the Tracy-Widom threshold even in pure "
            "noise.  Consider method='variance' or method='permutation' instead.",
            UserWarning,
            stacklevel=2,
        )
        pval_str = str(pval) if isinstance(pval, float) else pval
        if pval_str not in ("0.05", "0.01"):
            pval_str = "0.05"
        if n_features is None:
            feats = pca_info.get("features")
            n_features = int(len(feats)) if feats is not None else data.shape[1]
        threshold = largest_variance_from_random_matrix(data.shape[0], n_features, pval_str)
        n_keep = max(int((variance > threshold).sum()), 1)

        retention_rate = n_keep / n_total
        if retention_rate > 0.8:
            warnings.warn(
                f"select_pca_components (tracy_widom): {n_keep}/{n_total} components "
                f"retained ({retention_rate:.0%}).  This high retention rate is a "
                "strong signal that feature correlations are inflating eigenvalues "
                "above the Tracy-Widom threshold.  The test is likely unreliable — "
                "switch to method='variance' or method='permutation'.",
                UserWarning,
                stacklevel=2,
            )
        logger.info(
            f"select_pca_components (method=tracy_widom): retaining {n_keep} / "
            f"{n_total} components (threshold={threshold:.4f}, p={pval_str})"
        )

    else:
        raise ValueError(
            f"Unknown method {method!r}. "
            "Choose from 'variance', 'permutation', or 'tracy_widom'."
        )

    if max_components is not None:
        if n_keep > max_components:
            logger.info(
                f"select_pca_components: capping {n_keep} → {max_components} "
                "components (max_components limit)"
            )
        n_keep = min(n_keep, int(max_components))

    n_keep = max(n_keep, 1)
    return _slice_anndata(data, None, slice(0, n_keep))
