.. _overview:
Package Overview
================
The ``kemstem`` package is entirely composed of functions - for flexibility and interoperability no custom data structures are implemented, functions take and return numpy arrays. 

All functions of the package are callable from the top level module, but the code and documentation is organized into submodules. A full API can be found here: :ref:`api`

Atomic submodule
----------------
:ref:`atomic` includes functions for real space analysis including identifying and refining atomic column positions, and working with atomic "neighborhoods" to calculate correlation functions, separate sublattices and take intercolumn measurements.

Fourier submodule
-----------------
:ref:`fourier` includes functions for fourier analysis including fourier filtering and damping, and phase lock-in and wave fitting for strain measurements.

util submodule
--------------
:ref:`util` includes utility functions called in other functions but also generally useful in user code. Visualization functions are included under viz.


Extensions
----------
Non-core and experimental functionality, may have requirements exceeding those of the base package
