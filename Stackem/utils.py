#Stackem tools

import math
import numpy
import pyfits
from astLib.astWCS import WCS
import logging


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
