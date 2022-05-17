"""Preprocess."""

import anndata as ad
import pandas as pd
from scipy import io
import os
import subprocess
from scipy.sparse import coo_matrix
from sklearn import preprocessing


def anndata2file(adata, outdir):
    isExist = os.path.exists(outdir)
    if not isExist:
        os.makedirs(outdir)
        print(outdir, " is created!")

    # Save region
    region = adata.var[["seqnames", "start", "end"]]
    region.to_csv(
        outdir + "/region_file.bed", sep="\t", header=None, index=None
    )

    # Save sample
    sample = pd.DataFrame(adata.obs.index.tolist())
    sample.to_csv(outdir + "/sample_file.tsv", header=None, index=None)

    # Save count
    count = adata.X.T
    count = coo_matrix(count)
    io.mmwrite(outdir + "/count_file.mtx", count)


def file2anndata(indir, scale):
    print("Read zscored motif...")
    df_zscores = pd.read_csv(
        indir + "/zscores.tsv.gz", sep="\t", compression="gzip", index_col=0
    )
    if scale:
        print("Scale zscored motif...")
        df_zscores_scaled = preprocessing.scale(df_zscores, axis=1)
        df_zscores_scaled = pd.DataFrame(
            df_zscores_scaled,
            index=df_zscores.index,
            columns=df_zscores.columns,
        )
        df_zscores_scaled.to_csv(
            os.path.join(indir, "zscores_scaled.tsv.gz"),
            sep="\t",
            compression="gzip",
        )
        motifs = df_zscores_scaled
    else:
        motifs = df_zscores

    print("Rename TF names...")
    TFs = motifs.index.tolist()
    TF_new = [x.split("_")[1] for x in TFs]
    motifs.index = TF_new
    motifs = motifs.fillna(0)
    motifs = motifs.T
    motifs.to_csv(indir + "/zscores_renamed.csv", sep="\t")

    print("Save to anndata...")
    adata = ad.read_csv(
        filename=indir + "/zscores_renamed.csv", delimiter="\t"
    )
    return adata


def atac_chromVar(
    adata,
    species,
    genome,
    feature,
    scale=False,
    n_jobs=1,
    env="stream2_chromVar",
    outdir="./stream2_atac",
):
    # save files from anndata
    anndata2file(adata, outdir)

    # check whether atac env exists (stream2_chromVar)
    # not, "conda create -n stream2_chromVar R chronmVar,..."
    # run R script
    subprocess.run(
        args=[
            "./check_env.sh",
            env,
            outdir,
            species,
            genome,
            feature,
            str(n_jobs),
        ]
    )

    # read in motif matrix and save in anndata
    adata_motif = file2anndata(outdir, scale)
    adata_motif.obs = adata.obs
    adata_motif.uns = adata.uns
    adata_motif.obsm = adata.obsm
    return adata_motif
