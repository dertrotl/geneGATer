# geneGATer: Uncovering Cell-Cell Communication Genes from Spatial Transcriptomics Data

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/dertrotl/geneGATer/test.yaml?branch=main
[link-tests]: https://github.com/dertrotl/geneGATer/actions/workflows/test.yml
[badge-docs]: https://img.shields.io/readthedocs/genegater

geneGATer is a Python package designed to facilitate the identification and ranking of potential communication genes from spatial transcriptomics data. The package integrates an adapted metagene construction method by Hue et al. [2022] and a Graph Attention Network (GAT) to enable the exploration of cell-to-cell communication without relying on explicit cell-type annotations.

The package is the result of the authors master thesis which can be found [here][link-thesis]. By employing geneGATer, researchers can efficiently recreate the pipeline showcased in the associated thesis. The package also includes visualization tools to aid in the interpretation of the results, providing a comprehensive framework for studying intricate cellular communication networks.

## Getting started

Please refer to the [documentation][link-docs]. In particular, the

-   [API documentation][link-api].
-   [Tutorial notebook][link-tutorial].

## Installation

You need to have Python 3.9 or newer installed on your system. If you don't have
Python installed, we recommend installing [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).

There are several alternative options to install geneGATer:

<!--
1) Install the latest release of `geneGATer` from `PyPI <https://pypi.org/project/geneGATer/>`_:

```bash
pip install geneGATer
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/dertrotl/geneGATer.git@main
```

## Release notes

See the [changelog][changelog].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][scverse-discourse].
If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

> Analysis and Methods in Glioblastoma Spatial Transcriptomics, Benjamin Weinert, 2023.

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/dertrotl/geneGATer/issues
[changelog]: https://geneGATer.readthedocs.io/latest/changelog.html
[link-docs]: https://geneGATer.readthedocs.io
[link-api]: https://geneGATer.readthedocs.io/latest/api.html
[link-tutorial]: https://genegater.readthedocs.io/en/latest/notebooks/geneGATer_tutorial.html
[link-thesis]: https://drive.google.com/file/d/1lnRbx0mPHqJODJNX0HSQi0aJ33nlbPxb/view?usp=sharing
