#!/usr/bin/env python

import os
from setuptools import setup
import stackem

setup(name="stackem",
    version=stackem.__version__,
    description="Image plane stacking tools.",
    author="Sphesihle Makhathini",
    author_email="Sphesihle Makhathini <sphemakh@gmail.com>",
    url="https://github.com/sphemakh/Stackem",
    packages=["Stackem"],
    requires=["numpy","matplotlib","scipy", "astLib", "pyfits"],
    scripts=["Stackem/bin/" + i for i in os.listdir("Stackem/bin")], 
    licence="This program should come with the GNU General Public Licence. "\
            "If not, find it at http://www.gnu.org/licenses/old-licenses/gpl-2.0-standalone.html",
    classifiers=[],
     )
