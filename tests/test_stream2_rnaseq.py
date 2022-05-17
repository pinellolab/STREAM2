import stream2 as st
import pytest


@pytest.fixture
def adata():
    return st.read_h5ad(
        "tests/data/rnaseq_paul15.h5ad")


def test_stream2_rnaseq_paul15(adata, tmp_path):
    st.settings.set_workdir(tmp_path / "result_rnaseq_paul15")
    st.settings.set_figure_params(dpi=80,
                                  style='white',
                                  fig_size=[5, 5],
                                  rc={'image.cmap': 'viridis'})
    st.pp.filter_genes(adata, min_n_cells=3)
    st.pp.cal_qc_rna(adata)
    st.pl.violin(adata,
                 list_obs=['n_counts', 'n_genes', 'pct_mt'])
    st.pp.normalize(adata, method='lib_size')
    st.pp.log_transform(adata)
    st.pp.select_variable_genes(adata, n_top_genes=500)
    st.pl.variable_genes(adata, show_texts=False)
    st.pp.pca(adata, feature='highly_variable', n_components=40)
    st.pl.pca_variance_ratio(adata, log=True)

    st.tl.umap(adata, obsm='X_pca', n_dim=40, n_jobs=1)
    st.pl.umap(adata, color=['paul15_clusters', 'n_genes'],
               dict_drawing_order={
                   'paul15_clusters': 'random',
                   'n_genes': 'sorted'},
               fig_legend_ncol=2,
               fig_size=(5.5, 5))

    st.tl.seed_graph(adata, obsm='X_umap')
    st.tl.learn_graph(adata, obsm='X_umap')
    st.pl.graph(adata,
                color=['paul15_clusters', 'n_genes'],
                show_text=True,
                show_node=True)
    st.tl.infer_pseudotime(adata,
                           source=0,
                           target=4)
    st.pl.graph(adata,
                color=['epg_pseudotime'],
                show_text=False,
                show_node=False)
    st.tl.infer_pseudotime(adata, source=0)
    st.pl.graph(adata,
                color=['epg_pseudotime'],
                show_text=False,
                show_node=False)
