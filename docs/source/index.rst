.. kemstem documentation master file, created by
   sphinx-quickstart on Sun Feb 16 22:17:09 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to kemstem's documentation
==================================

About kemstem
-------------
``kemstem`` is a python package for quantification and visualization of atomic resolution STEM data developed in the Kourkoutis Electron Microscopy group at Cornell University, based on the techniques described in:

- `Savitzky, B. H., El Baggari, I., Admasu, A. S., Kim, J., Cheong, S. W., Hovden, R., & Kourkoutis, L. F. (2017). Bending and breaking of stripes in a charge ordered manganite. Nature communications, 8(1), 1883. <https://doi.org/10.1038/s41467-017-02156-1>`_
- `Smeaton, M. A., El Baggari, I., Balazs, D. M., Hanrath, T., & Kourkoutis, L. F. (2021). Mapping defect relaxation in quantum dot solids upon in situ heating. ACS nano, 15(1), 719-726. <https://doi.org/10.1021/acsnano.0c06990>`_
- `Goodge, B. H., El Baggari, I., Hong, S. S., Wang, Z., Schlom, D. G., Hwang, H. Y., & Kourkoutis, L. F. (2022). Disentangling coexisting structural order through phase lock-in analysis of atomic-resolution STEM data. Microscopy and Microanalysis, 28(2), 404-411. <https://doi.org/10.1017/S1431927622000125>`_

Installation
------------
``kemstem`` can be installed from PyPI with pip:

.. code-block:: bash

    $ pip install kemstem

Or to install the development version, clone from github and install with pip:

.. code-block:: bash

    $ git clone https://github.com/noahschnitzer/kemstem.git
    $ cd kemstem
    $ pip install -e .

Examples
--------
Example notebooks can be found `on the github repository <https://github.com/noahschnitzer/kemstem/tree/main/example_notebooks>`_.


Documentation contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   overview
   conventions
   api
