"""Functions to calculate principal graph"""

import elpigraph

from .._settings import settings


def learn_graph(adata,
                method='principal_curve',
                obsm=None,
                layer=None,
                n_nodes=30,
                epg_lambda=0.01,
                epg_mu=0.1,
                epg_alpha=0,
                n_jobs=None,
                **kwargs,
                ):
    """Learn principal graph

    Parameters
    ----------
    adata: `AnnData`
        Anndata object.
    method: `str`, (default: 'directed_graph');
        Method used to calculate clonal distances.
        Possible methods:
        - 'directed_graph': shortest-path-based directed graph
        - 'mnn':
        - 'wasserstein'
    layer: `str`, optional (default: None)
        The layer used to perform UMAP
    obsm: `str`, optional (default: None)
        The multi-dimensional annotation of observations used to perform UMAP
    **kwargs:
        Additional arguments to each method

    Returns
    -------
    updates `adata.uns['clone']` with the following field.
    distance: `sparse matrix`` (`.uns['clone']['distance']`)
        A condensed clone distance matrix.
        It can be converted into a redundant square matrix using `squareform`
        from Scipy.
    """

    if(sum(list(map(lambda x: x is not None,
                    [layer, obsm]))) == 2):
        raise ValueError("Only one of `layer` and `obsm` can be used")
    elif obsm is not None:
        if obsm in adata.obsm_keys():
            mat = adata.obsm[obsm]
        else:
            raise ValueError(
                f'could not find {obsm} in `adata.obsm_keys()`')
    elif layer is not None:
        if layer in adata.layers.keys():
            mat = adata.layers[layer]
        else:
            raise ValueError(
                f'could not find {layer} in `adata.layers.keys()`')
    else:
        mat = adata.X
    if n_jobs is None:
        n_jobs = settings.n_jobs

    if method == 'principal_curve':
        epg = elpigraph.computeElasticPrincipalCurve(
            X=mat,
            NumNodes=n_nodes,
            n_cores=n_jobs,
            Do_PCA=False,
            CenterData=False,
            Lambda=epg_lambda,
            Mu=epg_mu,
            alpha=epg_alpha,
            **kwargs)
    else:
        raise ValueError(
            f'Method "{method}" is not supported')
    return epg
