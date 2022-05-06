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

Installation with Bioconda (Recommended)
----------------------------------------
```sh
$ conda install -c bioconda stream2
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

* *For single cell **RNA-seq** analysis*:
```sh
$ conda create -n env_stream2 python=3.8 stream=0.1 jupyter
$ conda activate env_stream2
```

4)  To perform STREAM2 analyis in Jupyter Notebook as shown in **Tutorial**, type `jupyter notebook` within `env_stream2`:

```sh
$ jupyter notebook
```

You should see the notebook open in your browser.

Tutorial
--------

* Example for scRNA-seq: [1.1-STREAM_scRNA-seq (Bifurcation).ipynb](https://nbviewer.jupyter.org/github/pinellolab/STREAM/blob/master/tutorial/1.1.STREAM_scRNA-seq%20%28Bifurcation%29.ipynb?flush_cache=true)

* Example for scATAC-seq: [1.2-STREAM_scRNA-seq (Multifurcation) on 2D visulization.ipynb](https://nbviewer.jupyter.org/github/pinellolab/STREAM/blob/master/tutorial/1.2.use_vis.ipynb?flush_cache=true)

* Example for sc proteomics: [1.3-STREAM_scRNA-seq (Multifurcation) on original embedding.ipynb](https://nbviewer.jupyter.org/github/pinellolab/STREAM/blob/master/tutorial/1.2.STREAM_scRNA-seq%20%28Multifurcation%29.ipynb?flush_cache=true)

* Example for sc DNA methylation: [2.1-STREAM_scATAC-seq_peaks.ipynb](https://nbviewer.jupyter.org/github/pinellolab/STREAM/blob/master/tutorial/2.1-STREAM_scATAC-seq_peaks.ipynb?flush_cache=true)

* Example for share-seq: [2.2-STREAM_scATAC-seq_k-mers.ipynb](https://nbviewer.jupyter.org/github/pinellolab/STREAM/blob/master/tutorial/2.2.STREAM_scATAC-seq_k-mers.ipynb?flush_cache=true)

* Example for modifying trajectories: [4-STREAM_complex_trajectories.ipynb](https://nbviewer.jupyter.org/github/pinellolab/STREAM/blob/master/tutorial/4.STREAM_complex_trajectories.ipynb?flush_cache=true)

Tutorials for v0.4.1 and earlier versions can be found [here](https://github.com/pinellolab/STREAM/tree/master/tutorial/archives/v0.4.1_and_earlier_versions)

Installation with Docker
------------------------

With Docker no installation is required, the only dependence is Docker itself. Users will completely get rid of all the installation and configuration issues. Docker will do all the dirty work for you!

Docker can be downloaded freely from here: [https://store.docker.com/search?offering=community&type=edition](https://store.docker.com/search?offering=community&type=edition)

To get an image of STREAM, simply execute the following command:

```sh
$ docker pull pinellolab/stream2
```

>Basic usage of *docker run* 
>```sh
>$ docker run [OPTIONS] IMAGE [COMMAND] [ARG...]
> ```
>Options:  
>```
>--publish , -p	Publish a container’s port(s) to the host  
>--volume , -v	Bind mount a volume  
>--workdir , -w	Working directory inside the container  
>```

To use STREAM2 inside the docker container:
* Mount your data folder and enter STREAM2 docker container:

```bash
$ docker run --entrypoint /bin/bash -v /your/data/file/path/:/data -w /data -p 8888:8888 -it pinellolab/stream2:0.1
```
* Inside the container, launch Jupyter notebook:
```
root@46e09702ce87:/data# jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root
```
Access the notebook through your desktops browser on http://127.0.0.1:8888. The notebook will prompt you for a token which was generated when you create the notebook.


STREAM2 interactive website
--------------------------

In order to make STREAM user friendly and accessible to non-bioinformatician, we have created an interactive website: [http://stream.pinellolab.org](https://stream.pinellolab.partners.org/)

The website can also run on a local machine. More details can be found [https://github.com/pinellolab/STREAM_web](https://github.com/pinellolab/STREAM_web)



Credits: 
