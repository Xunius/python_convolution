'''Functions to perform 1D or 2D convolution with control on maximum
allowable missing data percentage in convolution window.

Main functions:

    - convolve1D(): 1D convolution on nD array.
    - convolve2D(): 2D convolution on 2D array.
    - runMean1D(): 1D running mean using 1D convolution, on nD array.
    - runMean2D(): 2D running mean using 2D convolution, on 2D array.
    
Calls a fortran module for the core convolution computation:
    
    - conv1d.so: for 1D convolution and running mean.
    - conv2d.so: for 2D convolution and running mean.


Author: guangzhi XU (xugzhi1987@gmail.com; guangzhi.xu@outlook.com)
Update time: 2017-11-08 16:07:36.
'''


import numpy
from conv1d import conv1d
from conv2d import conv2d



#----Get mask for missing data (masked or nan)----
def getMissingMask(slab):
    '''Get a bindary denoting missing (masked or nan).

    <slab>: nd array, possibly contains masked values or nans.
    
    Return <mask>: nd bindary, 1s for missing, 0s otherwise.
    '''
    nan_mask=numpy.where(numpy.isnan(slab),1,0)

    if not hasattr(slab,'mask'):
        mask_mask=numpy.zeros(slab.shape)
    else:
        if slab.mask.size==1 and slab.mask==False:
            mask_mask=numpy.zeros(slab.shape)
        else:
            mask_mask=numpy.where(slab.mask,1,0)

    mask=numpy.where(mask_mask+nan_mask>0,1,0)

    return mask


#----------Preprocess data for 1D convolution----------
def preProcess1D(slab,axis=0,verbose=True):

    #--------------Get mask for missings--------------
    slabmask=getMissingMask(slab)
    slab=numpy.where(slabmask==0,slab,0)

    #-----------------Reorder variable-----------------
    shape=list(slab.shape)
    n=shape[axis]
    ngrid=slab.size/n
    order_var=range(numpy.ndim(slab))

    #----Put the specified axis to 1st in order------------
    if axis!=0:
        #----Switch order------
        order_var[axis]=0
        order_var[0]=axis
        if verbose:
            print '# <preProcess1D>: Re-order <var> to:',order_var

        slab=numpy.transpose(slab,order_var)
        slabmask=numpy.transpose(slabmask,order_var)

    shape_reordered=slab.shape

    #-------------------Tabulate var-------------------
    slab=numpy.reshape(slab,(n,ngrid))
    slabmask=numpy.reshape(slabmask,(n,ngrid))

    return slab,slabmask,order_var,shape_reordered



#----------Postprocess data after 1D convolution----------
def postProcess1D(slab,slabmask,axis,order_var,shape_reordered,verbose=True):

    #-------------------Reshape back-------------------
    slab=numpy.reshape(slab,shape_reordered)
    slabmask=numpy.reshape(slabmask,shape_reordered)

    #-------------------Reorder back-------------------
    if axis!=0:
        slab=numpy.transpose(slab,order_var)
        slabmask=numpy.transpose(slabmask,order_var)
        if verbose:
            print '# <postProcess1D>: Re-order <var> to:',order_var

    slab=numpy.ma.masked_where(slabmask==1,slab)

    return slab



#----------1D Convolution using Fortran ----------
def convolve1D(slab,kernel,axis=0,max_missing=0.5,verbose=True):
    '''1D convolution using Fortran.

    <slab>: nd array, with optional mask.
    <kernel>: 1d array, convolution kernel.
    <axis>: int, axis on <slab> to do convolution.
    <max_missing>: real, max tolerable percentage of missings within any
                   convolution window.
                   E.g. if <max_missing> is 0.5, when over 50% of values
                   within a given element are missing, the center will be
                   set as missing (<res>=0, <resmask>=1). If only 40% is
                   missing, center value will be computed using the remaining
                   60% data in the element.
                   NOTE that out-of-bound grids are not counted as missings,
                   i.e. the number of valid values at edges drops as the kernel
                   approaches the edge.

    Return <result>: nd array, <slab> convolved with <kernel> at axis <axis>.

    Author: guangzhi XU (xugzhi1987@gmail.com; guangzhi.xu@outlook.com)
    Update time: 2017-11-08 09:27:49.

    '''

    assert numpy.ndim(kernel)==1, "<kernel> needs to be 1D."
    assert axis==int(axis) and axis>=0 and axis<=numpy.ndim(slab)-1,\
            "<axis> needs to be an int within [0,%d]" %(numpy.ndim(slab)-1)

    kernelflag=numpy.where(kernel==0,0,1)
    slab,slabmask,order_var,shape_reordered=preProcess1D(slab,axis,verbose)
    
    result,result_mask=conv1d.convolve1d(slab,slabmask,
            kernel,kernelflag,max_missing)

    result=postProcess1D(result,result_mask,axis,order_var,shape_reordered,verbose)

    return result


#----------1D running meaning using Fortran ----------
def runMean1D(slab,kernel,axis=0,max_missing=0.5,verbose=True):
    '''1D moving average with valid values control

    <slab>: nd array, with optional mask, variable to do running mean.
    <kernel>: 1d array, filtering kernel.
    <axis>: int, axis on <slab> to do convolution.
    <max_missing>: real, max tolerable percentage of missings within any
                   convolution window.
                   E.g. if <max_missing> is 0.5, when over 50% of values
                   within a given element are missing, the center will be
                   set as missing (<res>=0, <resmask>=1). If only 40% is
                   missing, center value will be computed using the remaining
                   60% data in the element.
                   NOTE that out-of-bound grids are not counted as missings,
                   i.e. the number of valid values at edges drops as the kernel
                   approaches the edge.

    Return <result>: nd array, moving average done on <slab>, with same shape
                     as <slab>.

    Author: guangzhi XU (xugzhi1987@gmail.com; guangzhi.xu@outlook.com)
    Update time: 2017-11-08 14:06:15.
    '''

    assert numpy.ndim(kernel)==1, "<kernel> needs to be 1D."
    assert axis==int(axis) and axis>=0 and axis<=numpy.ndim(slab)-1,\
            "<axis> needs to be an int within [0,%d]" %(numpy.ndim(slab)-1)

    #-----------------Flip the kernel-----------------
    kernel=kernel[::-1]
    kernelflag=numpy.where(kernel==0,0,1)

    slab,slabmask,order_var,shape_reordered=preProcess1D(slab,axis,verbose)
    
    result,result_mask=conv1d.runmean1d(slab,slabmask,
            kernel,kernelflag,max_missing)

    result=postProcess1D(result,result_mask,axis,order_var,shape_reordered,verbose)

    return result


#----------2D Convolution using Fortran ----------
def convolve2D(slab,kernel,max_missing=0.5,verbose=True):
    '''2D convolution using Fortran.

    <slab>: 2d array, with optional mask.
    <kernel>: 2d array, convolution kernel.
    <max_missing>: real, max tolerable percentage of missings within any
                   convolution window.
                   E.g. if <max_missing> is 0.5, when over 50% of values
                   within a given element are missing, the center will be
                   set as missing (<res>=0, <resmask>=1). If only 40% is
                   missing, center value will be computed using the remaining
                   60% data in the element.
                   NOTE that out-of-bound grids are not counted as missings,
                   i.e. the number of valid values at edges drops as the kernel
                   approaches the edge.

    Return <result>: 2d array, convolution of <slab> with <kernel>.

    Author: guangzhi XU (xugzhi1987@gmail.com; guangzhi.xu@outlook.com)
    Update time: 2017-01-16 10:59:55.
    '''

    assert numpy.ndim(slab)==2, "<slab> needs to be 2D."
    assert numpy.ndim(kernel)==2, "<kernel> needs to be 2D."

    kernelflag=numpy.where(kernel==0,0,1)

    #--------------Get mask for missings--------------
    slabmask=getMissingMask(slab)

    # this is to set np.nan to a float, this won't affect the result as
    # masked values are not used in convolution. Otherwise, nans will
    # affect convolution in the same way as scipy.signal.convolve()
    # and the result will contain nans.
    slab=numpy.where(slabmask==1,0,slab)

    result,result_mask=conv2d.convolve2d(slab,slabmask,kernel,
            kernelflag,max_missing)
    result=numpy.ma.masked_where(result_mask==1,result)

    return result


#------------2D moving average wit valid values control------------
def runMean2D(slab,kernel,max_missing=0.5,verbose=True):
    '''2D moving average with valid values control

    <slab>: 2d array, with optional mask.
    <kernel>: 2d array, convolution kernel.
    <max_missing>: real, max tolerable percentage of missings within any
                   convolution window.
                   E.g. if <max_missing> is 0.5, when over 50% of values
                   within a given element are missing, the center will be
                   set as missing (<res>=0, <resmask>=1). If only 40% is
                   missing, center value will be computed using the remaining
                   60% data in the element.
                   NOTE that out-of-bound grids are not counted as missings,
                   i.e. the number of valid values at edges drops as the kernel
                   approaches the edge.

    Return <result>: 2d moving average.

    Author: guangzhi XU (xugzhi1987@gmail.com; guangzhi.xu@outlook.com)
    Update time: 2016-11-10 17:28:00.
    '''

    assert numpy.ndim(slab)==2, "<slab> needs to be 2D."
    assert numpy.ndim(kernel)==2, "<kernel> needs to be 2D."

    #-----------------Flip the kernel-----------------
    kernel=kernel[::-1,::-1]
    kernelflag=numpy.where(kernel==0,0,1)

    #--------------Get mask for missings--------------
    slabmask=getMissingMask(slab)
    slab=numpy.where(slabmask==1,0,slab)

    result,result_mask=conv2d.runmean2d(slab,slabmask,kernel,kernelflag,max_missing)
    result=numpy.ma.masked_where(result_mask==1,result)

    return result


