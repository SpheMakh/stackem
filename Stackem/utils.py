#Stackem tools

import math
import numpy
import pyfits
from astLib.astWCS import WCS
import logging
import scipy.ndimage as ndimage


def logger(level=0):
    logging.basicConfig()

    LOGL = {"0": "INFO",
            "1": "DEBUG",
            "2": "ERROR",
            "3": "CRITICAL"}

    log = logging.getLogger("Stackem")
    log.setLevel(eval("logging."+LOGL[str(level)]))

    return log


def fitsInd(hdr, ctype):
    return filter( lambda ind: hdr["ctype%d"%ind].startswith(ctype), 
            range(1, hdr['naxis']+1) )[0]


def loadFits(imagename):

    with pyfits.open(imagename) as hdu:
        data = hdu[0].data
        hdr = hdu[0].header

    wcs = WCS(hdr, mode="pyfits")

    return data, hdr, wcs


def subregion(data, centre, radius):

    lx, ly = data.shape
    xstart = centre[1] - radius
    xend = centre[1] + radius
    ystart = centre[0] - radius
    yend = centre[0] + radius

    imslice = [slice(None)] * len(data.shape)
    imslice = [slice(xstart, xend),
               slice(ystart, yend)]
                                       
    scube = data[imslice]

    return scube


def sum_region(data, radius):

    w = data.shape[0]
    rad = numpy.linspace(-w/2, w/2, w)
    rad =  numpy.sqrt(rad[numpy.newaxis,:]**2+rad[:,numpy.newaxis]**2)
    mask = rad<=radius

    return (data*mask).sum()


def negnoise(data):
    neg = data[data<0]
    return numpy.concatenate([neg, -neg]).std()


def gauss(x, a0, mu, sigma):
    return  a0*numpy.exp(-(x-mu)**2/(2*sigma**2))


def gauss2d(height, center_x, center_y, width_x, width_y, beta=0):
    """Returns a gaussian function with the given parameters"""

    width_x = float(width_x)
    width_y = float(width_y)
    fct_x = lambda x: (x-center_x)**2/(2*width_x**2)
    fct_y = lambda y: (y-center_y)**2/(2*width_y**2)
    fct_xy = lambda x,y: beta*(x-center_x)*(y-center_x)/(width_x*width_y)

    return lambda x, y: height * numpy.exp(-fct_x(x) - fct_xy(x, y)-fct_y(y))


def elliptical_mask(data, emaj, emin, pa):
        
    px, py = data.shape
    _max = numpy.where(data==data.max())
    ox = _max[0][0] - px/2
    oy = _max[1][0] - py/2

    xx, yy = numpy.ogrid[-px/2:px/2, -py/2:py/2]
    mask0 = ( ((xx-ox)/emaj)**2 + ((yy-oy)/emin)**2 <= 1 )*1
    rmask = ndimage.rotate(mask0, angle=pa, reshape=False)
    
    return rmask!=0 #[(pxn-px)/2:-(pxn-px)/2, (pyn-py)/2:-(pyn-py)/2]


def gauss_weights(data, sigma1, sigma2, mask=None):
    
    px, py = data.shape[-2:]

    x, y = [ numpy.linspace(-a/2, a/2, a) for a in [px, py] ]
    xx, yy = numpy.meshgrid(x, y)

    _max = numpy.where(data==data.max())
    ox = _max[0][0] - px/2
    oy = _max[1][0] - py/2

    mask = 1 if mask is None else mask
    weight = gauss2d(1, ox, oy, sigma1, sigma2)(xx, yy)*mask

    return (data*mask)*weight/weight.sum()
