# Interpretable Ensembles for Uncertain Categorical Data

- [Overview](#overview)
- [Datasets](#datasets)
  - [Ageing-related genes](#ageing-related-genes)
  - [Drug side effects](#drug-side-effects)
- [System requirements](#system-requirements)
  - [Operating systems](#operating-systems)
  - [Software dependencies](#software-dependencies)
- [Installation guide](#installation-guide)
- [Instructions for use](#instructions-for-use)
  - [Predictive performance tests](#predictive-performance-tests)
  - [Model interpretability tests](#model-interpretability-tests)
  - [Reproduction](#reproduction)
- [License](#license)

## Overview
This repository contains the datasets and code related to the work reported in the paper referenced below.

M. R. H. Maia, A. Plastino, A. A. Freitas and J. P. Magalhães, "Interpretable Ensembles of Classifiers for Uncertain Data with Bioinformatics Applications", in *IEEE/ACM Transactions on Computational Biology and Bioinformatics*, 2022, DOI: [10.1109/TCBB.2022.3218588](https://doi.org/10.1109/TCBB.2022.3218588).

## Datasets
### Ageing-related genes
There are four datasets in this domain, corresponding to the [GenAge model organisms](https://genomics.senescence.info/genes/models.html):
- *C. elegans* ([`data/AG-Worm.csv`](data/AG-Worm.csv))
- *D. melanogaster* ([`data/AG-Fly.csv`](data/AG-Fly.csv))
- *M. musculus* ([`data/AG-Mouse.csv`](data/AG-Mouse.csv))
- *S. cerevisiae* ([`data/AG-Yeast.csv`](data/AG-Yeast.csv))

These datasets were generated in previous work. For details, please refer to the following paper:

M. R. H. Maia, A. Plastino and A. A. Freitas, "An Ensemble of Naive Bayes Classifiers for Uncertain Categorical Data", *2021 IEEE International Conference on Data Mining (ICDM)*, 2021, pp. 1222-1227, DOI: [10.1109/ICDM51629.2021.00148](https://doi.org/10.1109/ICDM51629.2021.00148).

They contain the following columns:
- `entrez` (Entrez gene ID).
- A variable number of feature columns (one for each protein) named with the corresponding STRING database IDs. Their values are the probabilities of the corresponding interactions (scaled in the interval [0, 1], where zeros represent missing values).
- `class` (gene's class: 0 = "anti-longevity", 1 = "pro-longevity").

### Drug side effects
The six datasets from this domain used in the reference paper are all encoded in a single file ([`data/SE.csv`](data/SE.csv)) as they all share the same features.

The file contains the following columns:
- `CID` (STITCH compound ID).
- 9096 feature columns (one for each protein) named `cpi_<STRING ID>`. Their values are the probabilities of the corresponding interactions (scaled in the interval [0, 1000], where zeros represent missing values).
- 1750 class columns (one for each side effect) named `se_<UMLS concept ID>` (0 = "no", 1 = "yes"). Note that not only the six side effects used in the paper are included in the file but all side effects from the [SIDER database](http://sideeffects.embl.de/).


## System requirements
### Operating systems
This software is compatible with Windows and Linux. It has been tested on Windows 10 x64.

### Software dependencies
#### Python environment
Python 3 is required. The code has been tested on Python 3.9.  
The following Python packages are required (as listed in [`requirements.txt`](requirements.txt)):
```
numpy~=1.19.5
pandas~=1.2.1
scikit-learn~=0.24.1
setuptools~=51.3.3
Cython~=0.29.21
joblib~=1.0.0
scipy~=1.6.0
```

#### External libraries
The [GNU Scientific Library (GSL)](https://www.gnu.org/software/gsl/) is required. The code has been tested with GSL 2.6.
The required GSL 2.6 headers and binaries for Windows x64 are available from this repository.

#### Compiler
A C compiler is required to build Python extension modules (see the [Cython documentation](https://cython.readthedocs.io/en/stable/src/quickstart/install.html)).  
The compiled extension modules compatible with Python 3.9 and Windows x64 are available from this repository.

## Installation guide

Clone the project from GitHub:
```
git clone https://github.com/marcelorhmaia/interpretable-ensembles-for-ucd
```

Build the extension modules (only required if not on Python 3.9/Windows x64):
1. If on Windows x64, unzip the [`external_lib/gsl_x64-windows.zip`](external_lib/gsl_x64-windows.zip) file. Otherwise, replace the GSL paths on lines 11 and 15 of file [`code/setup.py`](code/setup.py) with the corresponding paths on your system.
2. `cd code`
3. `python setup.py build_ext --inplace`

## Instructions for use

### Predictive performance tests

#### Usage
```
cd code
python eval.py <model> <dataset>
```

Replace `<model>` with one of the following: {`ENB-NV`, `ENB-NV+BB`, `ENB-NV+BRS`, `ENB-NV+BB+BRS`, `ENB-EV`, `ENB-EV+BB`, `ENB-EV+BRS`, `ENB-EV+BB+BRS`, `RF-DFE`, `RF-DFE+BB`, `RF-DFE+BS`, `RF-DFE+BB+BS`}  

Replace `<dataset>` with one of the following: {`AG-Worm`, `AG-Fly`, `AG-Mouse`, `AG-Yeast`, `SE-Nausea`, `SE-Headache`, `SE-Dermatitis`, `SE-Rash`, `SE-Vomiting`, `SE-Dizziness`}

#### Example
```
cd code
python eval.py ENB-NV AG-Worm
```

It should take less than a minute to run on a typical computer.  
The expected output is:
```
dataset = AG-Worm | model = ENB-NV
      precision(neg)  recall(neg)  f1-score(neg)  support(neg)  precision(pos)  recall(pos)  f1-score(pos)  support(pos)  accuracy  b-accuracy   roc-auc    g-mean
0           0.920000     0.403509       0.560976          57.0        0.346154     0.900000       0.500000          20.0  0.532468    0.651754  0.665789  0.602626
1           0.947368     0.352941       0.514286          51.0        0.431034     0.961538       0.595238          26.0  0.558442    0.657240  0.651584  0.582552
2           0.882353     0.306122       0.454545          49.0        0.433333     0.928571       0.590909          28.0  0.532468    0.617347  0.596210  0.533157
3           0.947368     0.333333       0.493151          54.0        0.368421     0.954545       0.531646          22.0  0.513158    0.643939  0.750000  0.564076
4           0.947368     0.346154       0.507042          52.0        0.403509     0.958333       0.567901          24.0  0.539474    0.652244  0.736378  0.575961
5           0.842105     0.340426       0.484848          47.0        0.456140     0.896552       0.604651          29.0  0.552632    0.618489  0.684519  0.552457
6           1.000000     0.404255       0.575758          47.0        0.508772     1.000000       0.674419          29.0  0.631579    0.702128  0.812913  0.635811
7           1.000000     0.320000       0.484848          50.0        0.433333     1.000000       0.604651          26.0  0.552632    0.660000  0.718462  0.565685
8           0.863636     0.380000       0.527778          50.0        0.425926     0.884615       0.575000          26.0  0.552632    0.632308  0.770000  0.579788
9           0.947368     0.367347       0.529412          49.0        0.456140     0.962963       0.619048          27.0  0.578947    0.665155  0.760393  0.594762
mean        0.929757     0.355409       0.514243          50.6        0.426276     0.944712       0.587472          25.7  0.554443    0.650060  0.714625  0.579447
```

### Model interpretability tests

#### Usage
```
cd code
python eval_top_features.py <model> <dataset>
```

Replace `<model>` with one of the following: {`ENB-EV+BRS`, `RF-DFE+BB+BS`}  

Replace `<dataset>` with one of the following: {`AG-Worm`, `AG-Fly`, `AG-Mouse`, `AG-Yeast`}

#### Example
```
cd code
python eval_top_features.py ENB-EV+BRS AG-Mouse
```

It should take less than a minute to run on a typical computer.  
The expected output is:
```
dataset = AG-Mouse | model = ENB-EV+BRS
top 10 features (conditional probabilities)
                          name  importance  rank
870   10090.ENSMUSP00000056668    0.005421   1.0
52    10090.ENSMUSP00000029175    0.004300   2.0
460   10090.ENSMUSP00000050683    0.003372   3.0
896   10090.ENSMUSP00000055308    0.003310   4.0
134   10090.ENSMUSP00000000369    0.002933   5.0
1679  10090.ENSMUSP00000101315    0.002699   6.0
1075  10090.ENSMUSP00000102538    0.002673   7.0
580   10090.ENSMUSP00000031697    0.002671   8.0
558   10090.ENSMUSP00000120152    0.002528   9.0
161   10090.ENSMUSP00000115578    0.002482  10.0

top 10 features (minimal sufficient sets)
                          name  importance  rank
870   10090.ENSMUSP00000056668    0.033143   1.0
52    10090.ENSMUSP00000029175    0.022175   2.0
460   10090.ENSMUSP00000050683    0.016528   3.0
896   10090.ENSMUSP00000055308    0.016059   4.0
790   10090.ENSMUSP00000101553    0.012588   5.0
194   10090.ENSMUSP00000099878    0.011042   6.0
1245  10090.ENSMUSP00000021090    0.010406   7.0
763   10090.ENSMUSP00000099621    0.010377   8.0
558   10090.ENSMUSP00000120152    0.010023   9.0
1075  10090.ENSMUSP00000102538    0.008750  10.0
```

### Reproduction

To reproduce all results from the paper, run the tests described above with all possible combinations of models and datasets.

Expected running time on a typical computer:
- Predictive performance tests
  - ENB-* models: less than 10 minutes
  - RF-* models: up to 3 days
- Model interpretability tests
  - ENB-EV+BRS: less than 10 minutes
  - RF-DFE+BB+BS: up to 4 hours

## License

This project is covered under the terms of the [GNU General Public License (GPL) version 3](LICENSE).
