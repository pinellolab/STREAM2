"""UMAP (Uniform Manifold Approximation and Projection)"""

import umap as umap_learn
from sklearn.manifold import (
    LocallyLinearEmbedding,
    Isomap,
    TSNE,
    SpectralEmbedding,
)


def dimension_reduction(
    adata,
    n_neighbors=15,
    n_components=2,
    random_state=2020,
    layer=None,
    obsm=None,
    n_dim=None,
    method="umap",
    eigen_solver="auto",
    **kwargs,
):
    """perform UMAP
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    method: `str`, optional (default: 'umap')
        Choose from {{'umap','se','mlle','tsne','isomap'}}
        Method used for dimension reduction.
        'umap': Uniform Manifold Approximation and Projection
        'se': Spectral embedding algorithm
        'mlle': Modified locally linear embedding algorithm
        'tsne': T-distributed Stochastic Neighbor Embedding
        'isomap': Isomap Embedding


    Returns
    -------
    updates `adata` with the following fields:
    `.obsm['X_umap']` : `numpy.ndarray`
        UMAP coordinates of samples.
    """

    if sum(list(map(lambda x: x is not None, [layer, obsm]))) == 2:
        raise ValueError("Only one of `layer` and `obsm` can be used")
    elif obsm is not None:
        X = adata.obsm[obsm]
    elif layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X
    if n_dim is not None:
        X = X[:, :n_dim]

    if method == "umap":
        reducer = umap_learn.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            random_state=random_state,
            **kwargs,
        )
    elif method == "se":
        reducer = SpectralEmbedding(
            n_neighbors=n_neighbors,
            n_components=n_components,
            random_state=random_state,
            **kwargs,
        )
    elif method == "mlle":
        reducer = LocallyLinearEmbedding(
            n_neighbors=n_neighbors,
            n_components=n_components,
            eigen_solver=eigen_solver,
            random_state=random_state,
            **kwargs,
        )
    elif method == "tsne":
        reducer = TSNE(
            n_components=n_components,
            random_state=random_state,
            **kwargs,
        )
    elif method == "isomap":
        reducer = Isomap(
            n_neighbors=n_neighbors,
            n_components=n_components,
            eigen_solver=eigen_solver,
            **kwargs,
        )

    reducer.fit(X)
    adata.obsm["X_dr"] = reducer.embedding_
