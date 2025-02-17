.. _conventions:

Conventions
===========
``kemstem`` functions accept and return numpy arrays without metadata encoding axis orders and other details. The following conventions must be followed for functions to provide meaningful results.

Points
------
Points (e.g. positions of atomic columns or FFT peaks) are assumed to be real-valued and in 2D space. Lists of points are numpy ndarrays with shape ``(n,2)`` where ``n`` is the number of points. The order of the coordinates is ``(y,x)`` (i.e. ``(row,column)``)to match the representation of images in numpy and matplotlib. The dtype can be floating point or integer.


In general, the size and order of point ndarrays is conserved, so output points of e.g. peak fitting can be assumed to correspond 1:1 with input points. NaN values are acceptable and will be excluded from analysis, non-meaningful or erroring points are set to NaN. 


From 1D arrays of x- and y-coordinates ``x`` and ``y``, a list of points ``p`` following this convention can be created with: ``np.stack((y,x),axis=1)``

Single points can be represented as just ``(y,x)`` without need to create a second dimension. 

Images
------
Images are generally assumed to be real valued 2D floating point numpy ndarrays, unless otherwise noted. Integer ndarrays should be cast to a floating point dtype. Some functions (e.g. atomic column identification and fitting) assume image intensity will be positive, and greater on atomic columns than over vacuum. For best compatibility, using ``kemstem.normalize`` to adjust the image minimum to 0 and maximum to 1 is recommended before analysis. 

Most analyses should work for non-square images but not all have been robustly tested, for best results crop images to square before analysis. 

Image patches
-------------
Image patches are often extracted for e.g. local fitting around a point of interest. As implemented in the ``get_patch`` function, patches must be square with an odd number of pixels. The patch center will be located at the pixel corresponding to the floor of the input position (i.e. rounded down to the nearest integer), with an equal number of pixels included in the patch on either side. 

Fourier transforms
------------------
Functions operating on Fourier transforms expect a complex valued 2D ndarray and generally assume the origin (i.e. 0-frequency) is in the center of the image - that is, that the image has been FFT shifted. ``prepare_fourier_pattern`` is a helper function that can calculate appropriatey processed transforms for both further processing and visualization (e.g. with log transform).
