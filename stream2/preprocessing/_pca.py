"""Principal component analysis"""

import numpy as np
from sklearn.decomposition import TruncatedSVD
from ._utils import (
    locate_elbow,
)


def pca(adata,
        n_components=50,
        algorithm='randomized',
        n_iter=5,
        random_state=2021,
        tol=0.0,
        feature=None,
        **kwargs,
        ):
    """perform Principal Component Analysis (PCA)
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.

    Returns
    -------
    updates `adata` with the following fields:
    `.obsm['X_pca']` : `array`
        PCA transformed X.
    `.uns['pca']['PCs']` : `array`
        Principal components in feature space,
        representing the directions of maximum variance in the data.
    `.uns['pca']['variance']` : `array`
        The variance of the training samples transformed by a
        projection to each component.
    `.uns['pca']['variance_ratio']` : `array`
        Percentage of variance explained by each of the selected components.
    """
    if(feature is None):
        X = adata.X.copy()
    else:
        mask = adata.var[feature]
        X = adata[:,mask].X.copy()
    svd = TruncatedSVD(n_components=n_components,
                       algorithm=algorithm,
                       n_iter=n_iter,
                       random_state=random_state,
                       tol=tol,
                       **kwargs)
    svd.fit(X)
    adata.obsm['X_pca'] = svd.transform(X)
    adata.uns['pca'] = dict()
    adata.uns['pca']['PCs'] = svd.components_.T
    adata.uns['pca']['variance'] = svd.explained_variance_
    adata.uns['pca']['variance_ratio'] = svd.explained_variance_ratio_


def select_pcs(adata,
               n_pcs=None,
               S=1,
               curve='convex',
               direction='decreasing',
               online=False,
               min_elbow=None,
               **kwargs):
    """select top PCs based on variance_ratio
    """
    if(n_pcs is None):
        n_components = adata.obsm['X_pca'].shape[1]
        if(min_elbow is None):
            min_elbow = n_components/10
        n_pcs = locate_elbow(range(n_components),
                             adata.uns['pca']['variance_ratio'],
                             S=S,
                             curve=curve,
                             min_elbow=min_elbow,
                             direction=direction,
                             online=online,
                             **kwargs)
        adata.uns['pca']['n_pcs'] = n_pcs
    else:
        adata.uns['pca']['n_pcs'] = n_pcs


def select_pcs_features(adata,
                        S=1,
                        curve='convex',
                        direction='decreasing',
                        online=False,
                        min_elbow=None,
                        **kwargs):
    """select features that contribute to the top PCs
    """
    n_pcs = adata.uns['pca']['n_pcs']
    n_features = adata.uns['pca']['PCs'].shape[0]
    if(min_elbow is None):
        min_elbow = n_features/10
    adata.uns['pca']['features'] = dict()
    ids_features = list()
    for i in range(n_pcs):
        elbow = locate_elbow(range(n_features),
                             np.sort(np.abs(adata.uns['pca']['PCs'][:, i],))[::-1],
                             S=S,
                             min_elbow=min_elbow,
                             **kwargs)
        ids_features_i = \
            list(np.argsort(np.abs(adata.uns['pca']['PCs'][:, i],))[::-1][:elbow])
        adata.uns['pca']['features'][f'pc_{i}'] = ids_features_i
        ids_features = ids_features + ids_features_i
    adata.var['top_pcs'] = False
    adata.var.loc[adata.var_names[np.unique(ids_features)], 'top_pcs'] = True
