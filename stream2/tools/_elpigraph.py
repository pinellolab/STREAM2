"""Functions to calculate principal graph"""

import numpy as np
import elpigraph
import networkx as nx

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
    updates `adata.uns['epg']` with the following field.
    conn: `sparse matrix` (`.uns['epg']['conn']`)
        A connectivity sparse matrix.
    node_pos: `array` (`.uns['epg']['node_pos']`)
        Node positions.
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
        dict_epg = elpigraph.computeElasticPrincipalCurve(
            X=mat,
            NumNodes=n_nodes,
            n_cores=n_jobs,
            Do_PCA=False,
            CenterData=False,
            Lambda=epg_lambda,
            Mu=epg_mu,
            alpha=epg_alpha,
            **kwargs)[0]
    else:
        raise ValueError(
            f'Method "{method}" is not supported')

    G = nx.Graph()
    G.add_edges_from(dict_epg['Edges'][0].tolist(), weight=1)
    mat_conn = nx.to_scipy_sparse_matrix(G,
                                         nodelist=np.arange(n_nodes),
                                         weight='weight')
    adata.uns['epg'] = dict()
    adata.uns['epg']['conn'] = mat_conn
    adata.uns['epg']['node_pos'] = dict_epg['NodePositions']
    adata.uns['epg']['params'] = {
        'obsm': obsm,
        'layer': layer,
        'n_nodes': n_nodes,
        'epg_lamba': epg_lambda,
        'epg_mu': epg_mu,
        'epg_alpha': epg_alpha,
    }
