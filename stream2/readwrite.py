"""reading and writing"""
import anndata as ad
import os
import pandas as pd

from anndata import (
    AnnData,
    read_h5ad,
    read_csv,
    read_excel,
    read_hdf,
    read_loom,
    read_mtx,
    read_text,
    read_umi_tools,
    read_zarr,
)


def read_10X_output(file_path, assay='RNA', **kwargs):
    if(file_path is None):
        file_path = ''
    _fp = lambda f:  os.path.join(file_path,f)
    
    adata = ad.read_mtx(_fp('matrix.mtx'),**kwargs).T 
    adata.X = adata.X
    adata.obs_names = pd.read_csv(_fp('barcodes.tsv'), header=None)[0]
    if assay =='ATAC':
        features = pd.read_csv(_fp('peaks.bed'), header=None, sep='\t')
        features.columns = ["seqnames", "start","end"]
        features.index = features['seqnames'].astype(str) + '_' + features['start'].astype(str) +'_' + features['end'].astype(str)
    else:
        features = pd.read_csv(_fp('genes.tsv'), header=None, sep='\t')
        features.index = features.index.astype('str')
    adata.var = features
    
    return adata
