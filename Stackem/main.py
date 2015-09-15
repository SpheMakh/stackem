# Image plane stacking tools
# Sphesihle Makhathini <sphemakh@gmail.com>

import matplotlib
matplotlib.use('Agg')

import pylab

import math
import numpy
import pyfits
from Stackem import utils
import sys
from argparse import ArgumentParser
import astLib.astCoords as coords

matplotlib.rcParams.update({'font.size': 20})

def main():

    __version_info__ = (0,0,1)
    __version__ = ".".join( map(str,__version_info__) )

    for i, arg in enumerate(sys.argv):
        if (arg[0] == '-') and arg[1].isdigit(): sys.argv[i] = ' ' + arg

    parser = ArgumentParser(description='Image based stacking tool S Makhathini <sphemakh@gmail.com>')

    add = parser.add_argument
    add("-v","--version", action='version',version='{:s} version {:s}'.format(parser.prog, __version__))

    add("-i", "--image", 
            help="FITS image name")

    add("-c", "--catalog", metavar="CATALOG_NAME:DELIMITER",
            help="Catalog name. Default delimiter is a comma. Format: 'ra,dec,freq' ")

    add("-p", "--prefix", default="gota_stackem_all",
            help="Prefix for output products.")

    add("-w", "--width", type=float, default=0,
            help="For line stacking: Band width [MHz] to sample across frequency (for line stacking). Default is 1."
                 "For continuum stacking: Width (in beams) of subregion to stack. Default is 10 beams")

    add("-vbl", "--vebosity-level", dest="vbl", choices=["0", "1", "2", "3"], default="0",
            help="Verbosity level. 0-> INFO, 1-> DEBUG, 2-> ERROR, 3-> CRITICAL. Default is 0")

    add("-b", "--beam", metavar="BMIN[:BMIN:BPA]",
            help="PSF (a.k.a dirty beam) FWHM in degrees. No default")

    add("-b2p", "--beam2pix", action="store_true",
            help="Do Jy/beam to Jy/pixel conversion")

    add("-L", "--line", action="store_true",
            help="Do line stacking")

    add("-C", "--cont", action="store_true",
            help="Do continuum stacking")

    args = parser.parse_args()


    catalog_string = args.catalog.split(":")
    if len(catalog_string)>1:
        catalgname, delimiter = catalog_string
    else:
        catalogname, delimiter = catalog_string[0], ","

    if args.beam:
        beam = args.beam.split(":")
        if len(beam)==1:
            beam = float(beam[0])
        elif len(beam)==2:
            beam = map(float, beam) + [0]
        else:
            beam = map(float, beam)
    else:
        beam = None

    pylab.clf()

    prefix = args.prefix or "stackem_default"

    if args.line:
        pylab.figure(figsize=(15,10))
        from Stackem import LineStacker

        stack = LineStacker.load(args.image, catalogname, delimiter=delimiter,
                beam=beam, width=args.width, beam2pix=args.beam2pix,
                verbosity=args.vbl)

        stacked_line = stack.stack()*1e6 # convert to uJy
        peak, nu, sigma = gfit_params = stack.fit_gaussian(stacked_line)
        gfit = utils.gauss(range(stack.width), *gfit_params)

        freqs = (numpy.linspace(-stack.width/2, stack.width/2, stack.width)*stack.dfreq - stack.freq0)*1e-9 # convert to GHz

        # plot profile
        pylab.plot(freqs, stacked_line, "r.", label="stacked profile")
        pylab.plot(freqs, gfit, "k-", label="Gaussian fit")
        pylab.grid()
        pylab.xlim(freqs[0], freqs[-1])
        pylab.xlabel("Frequency [GHz]")
        pylab.ylabel("Flux density [mJy/{:s}]".format("beam" if args.beam2pix else "pixel"))
        pylab.legend(loc=1)
        pylab.savefig(prefix+"-line.png")
        pylab.clf()

        # save
        numpy.savetxt(prefix+"-line.txt", stacked_line, delimiter=",")

        with open(prefix+"-line_stats.txt", "w") as info:
            tot = sigma*peak*math.sqrt(2*math.pi)

            stack.log.info("Gaussian Fit parameters.")
            info.write("Gaussian Fit parameters\n")

            info.write("Peak Flux : {:.4g} uJy\n".format(peak))
            stack.log.info("Peak Flux : {:.4g} uJy".format(peak))

            info.write("Integrated Flux : {:.4g} uJy\n".format(tot))
            stack.log.info("Integrated Flux : {:.4g} uJy".format(tot))

            info.write("Profile width : {:.4g} MHz \n".format(sigma*stack.dfreq))
            stack.log.info("Profile width : {:.4g} kHz".format(sigma*stack.dfreq*1e-3))

        stack.log.info("Line stacking ran successfully. Find your outputs at {:s}-line*".format(prefix))

    if args.cont:
        pylab.figure(figsize=(20,20))
        from Stackem import ContStacker

        stack = ContStacker.load(args.image, catalogname, delimiter=delimiter, 
                beam=beam, beam2pix=args.beam2pix, width=args.width,
                verbosity=args.vbl)

        stacked = stack.stack()
        mask = utils.elliptical_mask(stacked, stack.bmajPix/2, stack.bminPix/2, stack.bpa)
        print(stack.bmajPix)
        pylab.imshow(mask)
        pylab.savefig("mask.png")
        pylab.imshow(stacked)
        pylab.savefig("image.png")
        pylab.clf()
        flux = (mask*stacked).sum()

        if args.beam2pix:
            pixels_per_beam = mask.sum()

        rms = utils.negnoise(stacked)

        with open(prefix+"-cont.txt", "w") as cont:
            stack.log.info("Stacked flux: {:.3g} +/- {:.3g} uJy/beam".format(flux*1e6, rms*1e6))
            stack.log.info("Stacked flux: {:.3g} +/- {:.3g} uJy/Pixel".format(flux/pixels_per_beam*1e6, rms/pixels_per_beam*1e6))
            cont.write("Stacked flux: {:.3g} +/- {:.3g} uJy/beam".format(flux*1e6, rms*1e6))
            cont.write("Stacked flux: {:.3g} +/- {:.3g} uJy/Pixel".format(flux/pixels_per_beam*1e6, rms/pixels_per_beam*1e6))

        # Plot stacked image, and cross-sections

        rotated = numpy.rot90(stacked.T)
        import matplotlib.gridspec as gridspec
        
        gs = gridspec.GridSpec(2, 2,
                               width_ratios = [4,1],
                               height_ratios = [1,4])

        gs.update(wspace=0.05, hspace=0.05)
        ax1 = pylab.subplot(gs[2])
        ax2 = pylab.subplot(gs[3])
        ax3 = pylab.subplot(gs[0])

        pylab.setp(ax2.get_yticklabels(), visible=False)
        pylab.setp(ax2.get_xticklabels(), rotation=90)
        pylab.setp(ax3.get_xticklabels(), visible=False)

        ax1.imshow(rotated)
        ra0, dec0 = stack.wcs.getCentreWCSCoords()
        ras = numpy.linspace(stack.width/2, -stack.width/2, stack.width)*stack.cell - ra0
        decs = numpy.linspace(-stack.width/2, stack.width/2, stack.width)*stack.cell - dec0

        ax1.set_xticklabels( map(lambda x: coords.decimal2hms(x, ":"), ras) )
        ax1.set_yticklabels( map(lambda x: coords.decimal2dms(x, ":"), decs) )
        ax1.set_xlabel("RA [hms]")
        ax1.set_ylabel("DEC [dms]")
        pylab.setp(ax1.get_xticklabels(), rotation=90)

        ax2.plot(rotated[stack.width/2,:]*1e6, range(stack.width))
        ax3.plot(rotated[:,stack.width/2]*1e6)
        ax2.set_xlabel("Flux density [uJy]")
        ax3.set_ylabel("Flux density [uJy]")
        pylab.savefig(prefix+"-cont.png")

        # save stacked image as fits
        hdr = stack.hdr
        hdr["crpix1"] = stack.width/2 
        hdr["crpix2"] = stack.width/2 
        pyfits.writeto(prefix+"-cont.fits", stacked, hdr, clobber=True)

        stack.log.info("Continuum stacking ran successfully. Find your outputs at {:s}-line*".format(prefix))
