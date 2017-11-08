# python_convolution
A Python module providing alternative 1D and 2D convolution and moving average functions to numpy/scipy's implementations, with control over maximum tolerable missing values in convolution window and better treatment of NaNs.

# Purpose of this module

The way that *numpy* and *scipy* 's convolution functions treat missing values:

* when missing is masked (data is `numpy.ma.core.MaskedArray`): I think the mask is just ignored.
* when missing is represented as `numpy.nan`: any overlaping between the kernel and any NaN will create an Nan in the result. So if the data is 101x101 by shape, and there is an NaN at the center (`data[50,50]`), the kernel is 5x5 by shape, then the result will have a hole of NaNs at the center, with a size of 5x5.

A relevant question on SO: https://stackoverflow.com/q/38318362/2005415

Depending on application, this might be the desired result.

But in other cases, you probably don't want to lose so much information just because of a single missing, particularly when doing a moving average. Perhaps <=50 % of missing is still tolerable. This is what this module is trying to provide: **1D and 2D convolution and running mean functions that allow user to have a control on how much missing is tolerable.**


# What's in this

## `convolve.py`

A Python module providing functions:

  1. `convolve1D()`: 1D convolution on n-d array.
  2. `convolve2D()`: 2D convolution on 2-d array.
  3. `runMean1D()`: 1D running mean on n-d array.
  4. `runMean2D()`: 2D running mean on 2-d array.
  
The 1D convolution functions call the Fortran module `conv1d` for the core computations, and the 2D functions the `conv2d` module.

## `conv1d.f90`, `conv2d.f90`

Fortran 90 codes for the core computation of convolution and running mean.

## `conv1d.pyf`, `conv2d.pyf`

Signature files used to compile Fortran `.90` codes to Python modules.


# Usage

To compile `conv1d.f90` and `conv2d.f90` using *f2py*:

```
f2py -c conv1d.pyf conv1d.f90
f2py -c conv2d.pyf conv2d.f90
```
Then in you Python script:

```
import conv1d
print conv1d.conv1d.__doc__
```

# Further notes on the treatment of edges

No padding, mirroring or reflecting is done at the edges. The kernel takes only data within range, and at the same time the counting of valid number of data points within kernel takes only data in range.

E.g. kernel is 5x5. 

* At data center, number of valid data within a kernel window is 25 (assuming no missing).
* At a corner, number of valid data is 9.
* At an edge, number of valid data is 15.

When missings are present, the percentage computation is done wrt these numbers:

* x/25 at center,
* x/9 at corner,
* x/15 at edge.

# Related things:

* *astropy*: A python package providing another implementation of 2D convolution, **I think** it interpolates the missings before convoling.
* https://github.com/Xunius/Random-Fortran-scripts: some other random Fortran codes I made.

