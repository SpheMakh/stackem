## Image plane continuum stacking script
## Sphesihle Makhathini <sphemakh@gmail.com>

import math
import numpy
import sys
import pylab

from Stackem import utils

from multiprocessing import Process, Manager, Lock
import psutil
manager = Manager()


class load(object):

    def  __init__(self, imagename, catalogname, beam=0, width=None,
                  stokes_ind=0, delimiter=",", beam2pix=False, verbosity=0):
        """ Continuum stacking tool
            imagename: FITS image
            catalogname: List of RA, DEC values. Weights can be added as a 3rd column
            beam: PSF FWHM in degrees
            stokes_id: Index of stokes plane to stack (0,1,2,3) -> (I,Q,U,V)
        """

        self.imagename = imagename
        self.catalogname = catalogname
        self.catalog = numpy.loadtxt(catalogname, delimiter=delimiter)
        self.npos = len(self.catalog)

        if self.catalog.shape[-1]==2:
            self.catalog = numpy.append(self.catalog,
                            numpy.ones([self.npos, 1]), 1)

        self.weights = manager.Value("f",0)

        self.stokes_ind = stokes_ind

        self.log = utils.logger(verbosity)


        self.data, self.hdr, self.wcs = utils.loadFits(imagename)

        # Find restoring beam in FITS header if not specified
        if isinstance(beam, (float, int)):
            if beam==0:
                beam = None
            else:
                self.bmaj = self.bmin = beam
                self.bpa = 0
        elif isinstance(beam, (list, tuple)):
            self.bmaj, self.bmin, self.bpa = beam

        elif beam is None:
            try:
                self.bmaj = self.hdr["bmaj"]
                self.bmin = self.hdr["bmin"]
                self.bpa = self.hdr["bpa"]
            except KeyError: 
                self.log.critical("Beam not specified, and no beam information in FITS header")
        else:
            raise TypeError("Beam must be a list, tuple, int or float")

        self.bmajPix = int(self.bmaj/abs( self.wcs.getXPixelSizeDeg() ) )
        self.bminPix = int(self.bmin/abs( self.wcs.getXPixelSizeDeg() ) )

        self.beamPix = self.bmajPix

        self.ndim = self.hdr["naxis"]

        stokes = self.ndim - utils.fitsInd(self.hdr, "STOKES")
        freq = self.ndim - utils.fitsInd(self.hdr, "FREQ")

        imslice = [slice(None)]*self.ndim
        imslice[stokes] = stokes_ind
        self.data = self.data[imslice].sum(0)
        #raise SystemError((self.data.shape, imslice, freq))

        self.cell = self.wcs.getXPixelSizeDeg()

        self.width = self.beamPix*self.beam if beam else self.beamPix*10
        self.stamps = manager.list([])
        self.beam2pix = beam2pix
        self.track = manager.Value("d",0)
        self.lock = Lock()

    
    def stack(self):

        def worker(ra, dec, weight):

            rapix, decpix = self.wcs.wcs2pix(ra, dec)
            out = utils.subregion(self.data, [rapix, decpix], self.width/2)*weight
            self.lock.acquire()
            if out.shape == (self.width, self.width):
                self.weights.value += weight
                self.stamps.append(out)

            self.track.value += 1
            self.lock.release()

        procs = []
        range_ = range(10, 110, 10)
        print("Progress:"),
        for ra, dec, weight in self.catalog:
            proc = Process(target=worker, args=(ra, dec, weight))
            procs.append(proc)
            proc.start()

            nn = int(self.track.value/float(self.npos)*100)
            if nn in range_:
                print("..{:d}%".format(nn)),
                range_.remove(nn)

        for proc in procs:
            proc.join()

        print("..100%\n")

        stacked = numpy.array(self.stamps).sum(0)/self.weights.value

        return stacked


    def mc_noise_stack(self, imagename, runs=400, stacks=None):
    
        stacks = stacks or xrange(1000, 8000, 1000)

        # get field centre
        cx, cy = self.wcs.wcs2pix(*self.centre)
        self.track.value = 0
    
        def getpos(npos):
            ra = (numpy.random.random(npos) - 0.49)*self.fov/2 + cx
            dec = (numpy.random.random(npos) - 0.49)*self.fov/2 + cy
    
            return numpy.array([ra, dec]).T
    
        noise = manager.list([])
    
        procs = []
        def worker(npos):
            self.track.value += 1
            _noise = numpy.ones(runs)
            for i in range(runs):
                radec = getpos(npos)
                _noise[i] = self.stack()
            
            noise.append([ npos, _noise.std()])
    
        for npos in stacks:
            proc = Process(target=worker, args=(npos,))
            proc.start()
            procs.append(proc)
    
            nn = int(self.track.value/float(len(stacks))*100)
            if nn in range_:
                print("..{:d}%".format(nn)),
                range_.remove(nn)

        for proc in procs:
            proc.join()

        print("..100%\n")
    
        return noise
