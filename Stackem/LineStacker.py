#!/usr/bin/env python

###
# Image based spectral line stacking tool
# S. Makhathini <sphemakh@gmail.com>
###

import math
import numpy 
import pyfits
import math

from scipy.optimize import curve_fit
from multiprocessing import Process, Manager, Lock
from stackem import utils


class load(object):
    
    def __init__(self, imagename, catalogname, 
                 width=1, beam=None, delimiter=",", 
                 verbosity=0, beam2pix=False):

        self.log = utils.logger(verbosity)

        ncpu = psutil.cpu_count()
        self.log.info("Found {:d} CPUs".format(ncpu))

        self.log.info("Laoding Image data and catalog info")

        self.imagename = imagename
        self.catalogname = catalogname
        self.beam = beam
        self.delimiter = delimiter

        # Load data
        self.catalog = numpy.loadtxt(self.catalogname, delimiter=self.delimiter)
        self.nprofs = len(self.catalog)
        
        self.data, self.hdr, self.wcs = utils.loadFits(imagename)

        self.ndim = self.hdr["naxis"]
        self.centre = self.wcs.getCentreWCSCoords()

        self.log.info("Image Centre RA,DEC {:+.3g}, {:+.3g} Deg".format(*self.centre))
        
        cubeslice = [slice(None)]*self.ndim
        if self.ndim >3:
            stokes_ind = self.ndim - _ind(self.hdr, "STOKES")
            cubeslice[stokes_ind] = 0

        self.cube = data[cubeslice]

        self.profiles = Manager().list([])
        self.weights = Manager().Value("d", 0)

        ind = _ind(self.hdr, "FREQ")
        self.crpix = self.hdr["crpix%d"%ind]
        self.crval = self.hdr["crval%d"%ind]
        self.dfreq = self.hdr["cdelt%d"%ind]
        self.freq0 = self.crval + (self.crpix-1)*self.dfreq
        self.nchan = self.hdr["naxis%d"%ind]
        self.width = int(width*1e6/self.dfreq)

        # Find restoring beam in FITS header if not specified
        if self.beam is None:
            try:
                bmaj = self.hdr["bmaj"]
                bmin = self.hdr["bmin"]
            except KeyError: 
                self.log.critical("Beam not specified, and no beam information in FITS header")

            self.beam = math.sqrt(bmaj*bmin)

        self.beamPix = int(self.beam/abs( self.wcs.getXPixelSizeDeg() ) )
        self.beam2pix = beam2pix
    
        self.excluded = Manager().Value("d",0)
        self.track = Manager().Value("d",0)
        self.lock = Lock()


    def profile(self, radeg, decdeg, cfreq, weight, pid):

        
        rapix, decpix = self.wcs.wcs2pix(radeg, decdeg)

        cfreqPix = int((cfreq - self.freq0)/self.dfreq )

        zstart = cfreqPix - self.width/2
        zend = cfreqPix + self.width/2

        beamPix = self.beamPix
        #raise ValueError(beamPix)

        ystart, yend = (decpix-beamPix/2.), (decpix+beamPix/2.)
        xstart, xend = (rapix-beamPix/2.), (rapix+beamPix/2.)

        self.log.debug("Line profile {:.3f} {:.3f} {:d}-{:d}".format(rapix, decpix, zstart, zend))
        
        pcube = self.cube[zstart:zend, ystart:yend, xstart:xend]

        # Check if this profile is with stacking

        if pcube.shape != (self.width, beamPix, beamPix):
            padz, pady, padx = (0,0), (0,0), (0,0)
            diffx, diffy, diffz = 0, 0, 0

            if pcube.shape[0] != self.width:
                diffz = self.width - pcube.shape[0]
                if cfreqPix < self.cube.shape[0]/2:
                    padz = diffz, 0
                else:
                    padz = 0, diffz

            if pcube.shape[1] != beamPix:
                diffy = beamPix - pcube.shape[1]
                if ystart<0:
                    pady = diffy, 0
                else:
                    pady = 0, diffy
                
            if pcube.shape[2] != beamPix:
                diffx = beamPix - pcube.shape[2]
                if xstart<0:
                    padx = diffx, 0
                else:
                    padx = 0, diffx

            if diffz > self.width/2 or diffx > beamPix/2 or diffy > beamPix/2:
                self.log.debug("Skipping Profile {:d}, its close too an edge (s).".format(pid))
                self.excluded.value += 1
                return
            else:
                npad = padz, pady, padx
                self.log.debug("Profile {:d} is close an edge(s). Padding the exctracted cube by {:s} ".format(pid, repr(npad)))

                pcube = numpy.pad(pcube, pad_width=npad, mode="constant")
        else:
            self.log.debug("Extracting profile {:d}".format(pid))

        self.lock.acquire()
        self.weights.value += weight
        self.track.value += 1
        self.profiles.append(pcube*weight)
        self.lock.release()

        
    def stack(self):

        nprofs = len(self.catalog)
        
        self.log.info("Stacking {:d} line profiles".format(nprofs))

        # Run these jobs in parallel
        procs = []
        range_ = range(10, 110, 10)
        print("Progress:"),
        for i, (ra, dec, cfreq, w, _id) in enumerate(self.catalog):
            proc = Process(target=self.profile, args = (ra, dec, cfreq, w, i) )
            proc.start()
            procs.append(proc)

            nn = int(self.track.value/float(self.nprofs)*100)
            if nn in range_:
                print("..{:d}%".format(nn)),
                range_.remove(nn)
        
        for proc in procs:
            proc.join()

        print("..100%\n")

        self.log.info("Have stackem all")
        self.log.info("{:d} out of {:d} profiles were excluded because they \
were too close to an edge".format(self.excluded.value, nprofs))
        
        stack = numpy.sum(self.profiles, 0)
        # Create circular mask
        rad = numpy.linspace(-self.beamPix/2, self.beamPix/2, self.beamPix)
        rad =  numpy.sqrt(rad[numpy.newaxis, :]**2+rad[:, numpy.newaxis]**2)
        mask = rad<=self.beamPix/2

        pixels_per_beam = (6/math.pi)*(self.beamPix/2.)***2 if self.beam2pix else 1.0
        profile = (stack*mask).sum((1,2))/self.weights.value / pixels_per_beam

        return profile


    def fit_gaussian(self, profile):

        nn = len(profile)
        xx = range(nn)
        import scipy.stats as stats
        from scipy.optimize import leastsq

        sigma = 1 #stats.moment(profile, moment=1)
        mu = xx[nn/2]
        peak = profile.max()
        

        def res(p0, x, y):
            peak, mu, sigma = p0
            yf = utils.gauss(x, peak, mu, sigma)
            return y - yf
    
        params = leastsq(res, (peak, mu, sigma), args=(xx, profile))[0]
        
        return params