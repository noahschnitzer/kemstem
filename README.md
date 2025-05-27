# kemstem
`kemstem` is a python package for quantification and visualization of atomic resolution STEM data developed in the [Kourkoutis Electron Microscopy group](https://kourkoutis.research.engineering.cornell.edu/) at Cornell University. It includes algorithms for:
- Mapping translational symmetry breaking distortions, such as picometer scale atomic displacements associated with charge order as well as chemical and vacancy orderings, with Fourier space damping of superlattice peaks. See: [`B. H. Savitzky, I. El Baggari, A.S. Admasu, J. Kim, S.W. Cheong, R. Hovden & L.F. Kourkoutis (2017). Nature communications, 8(1), 1883.`](https://doi.org/10.1038/s41467-017-02156-1)
- Mapping the amplitude and phase of periodic features to identify local orderings, defects, strains, and other nanoscale inhomogeneity with Fourier filtering and phase-lock in analysis. See: [`B.H. Goodge, I. El Baggari, S.S. Hong, Z. Wang, D.G. Schlom, H.Y. Hwang & L. F. Kourkoutis (2022). Microscopy and Microanalysis, 28(2), 404-411.`](https://doi.org/10.1017/S1431927622000125)
- Measuring inter-planar spacings and orientations with high real-space resolution and precision, as well as quantified uncertainty, with local wave fitting. See: [`M. A. Smeaton, I. El Baggari, D. M. Balazs, T. Hanrath & L. F. Kourkoutis (2021). ACS nano, 15(1), 719-726.`](https://doi.org/10.1021/acsnano.0c06990)

Detailed descriptions, example applications and validation of these techniques are provided in our preprint: [`N. Schnitzer, L. Bhatt, I. El Baggari, R. Hovden, B.H. Savitzky, M.A. Smeaton, B.H. Goodge (2025). arXiv:2504.01159.`](https://arxiv.org/abs/2504.01159) 

## Installation
``kemstem`` releases can be installed from PyPI with pip:
```shell
pip install kemstem
```

Or to install the development version, clone from github and install with pip:
```shell
git clone https://github.com/noahschnitzer/kemstem.git
cd kemstem
pip install -e . 
```

## Examples
Example jupyter notebooks with demonstration workflows can be found along with sample data under `example_notebooks`.

## Documentation
Package documentation is available at [https://kemstem.readthedocs.io/en/latest/](https://kemstem.readthedocs.io/en/latest/).

<img src="kem_logo.png" width="400">
