[![CI](https://github.com/pinellolab/stream2/actions/workflows/CI.yml/badge.svg)](https://github.com/pinellolab/stream2/actions/workflows/CI.yml)

# STREAM2 (Latest version v0.1)

STREAM2: Fast, scalable, and interactive trajectory analysis of single-cell omics data

![simba](./docs/source/_static/img/logo.png?raw=true)

[![install with bioconda](https://img.shields.io/badge/install%20with-bioconda-brightgreen.svg?style=flat-square)](http://bioconda.github.io/recipes/stream/README.html)

<!-- [![Build Status](https://travis-ci.org/pinellolab/STREAM.svg)](https://travis-ci.org/pinellolab/STREAM) -->

[![CI](https://github.com/pinellolab/STREAM/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/pinellolab/STREAM/actions/workflows/python-package-conda.yml)

Latest News
-----------

> May 5, 2022  

Alpha Version 0.1 is now available.

Introduction
------------

STREAM2 (**S**ingle-cell **T**rajectories **R**econstruction, **E**xploration **A**nd **M**apping) is an interactive pipeline capable of disentangling and visualizing complex trajectories from for single-cell omics data.

The four key innovations of STREAM2 are: 
1) STREAM2 can learn more biologically meaningful trajectories in a semi-supervised way by leveraging external information (e.g. time points, FACS labels, predefined relations of clusters, etc.); 
2) STREAM2 is able to learn not only linear or tree-like structures but also more complex graphs with loops or disconnected components; 
3) STREAM2 supports trajectory inference for various single-cell assays such as gene expression, chromatin accessibility, protein expression level, and DNA methylation; 
4) STREAM2 introduces a flexible and powerful path-based marker detection procedure. In addition, we provide a scalable and fast python package along with a comprehensive documentation website to facilitate STREAM2 analysis. 
We also provide an accompanying interactive website to allow non-expert users to freely and easily compute and explore the obtained trajectories.

![simba](./docs/source/_static/img/Fig1_V2.1.jpg?raw=true)

Installation with Github with latest branch (Recommended)
----------------------------------------
```sh
$ pip install git+https://github.com/pinellolab/STREAM2.git@latest
```

If you are new to conda environment:

1)	If Anaconda (or miniconda) is already installed with **Python 3**, skip to 2) otherwise please download and install Python3 Anaconda from here: https://www.anaconda.com/download/

2)	Open a terminal and add the Bioconda channel with the following commands:

```sh
$ conda config --add channels defaults
$ conda config --add channels bioconda
$ conda config --add channels conda-forge
```

3)	Create an environment named `env_stream2` , install **stream2**, **jupyter**, and activate it with the following commands:

```sh
$ conda create -n env_stream2 python=3.8 jupyter
$ conda activate env_stream2
$ pip install git+https://github.com/pinellolab/STREAM2.git@latest
```

4)  To perform STREAM2 analyis in Jupyter Notebook as shown in **Tutorial**, type `jupyter notebook` within `env_stream2`:

```sh
$ jupyter notebook
```

You should see the notebook open in your browser.

Tutorial
--------

Tutorials for the usage of STREAM2 can be found at **STREAM2_tutorial** repositories [https://github.com/pinellolab/STREAM2_tutorials] It includes tutorials for scRNA-seq data, scATAC-seq data, protenomics data, DNA methelyation data, and multiomics data.
