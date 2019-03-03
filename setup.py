# Copyright (c) 2015-2019 Patricio Cubillos and contributors.
# pytips is open-source software under the MIT license (see LICENSE).

import os
import re
import sys
import setuptools
from numpy import get_include
from setuptools import setup, Extension

topdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(topdir + "/pytips")
import VERSION as ver

srcdir = topdir + '/src_c/'          # C-code source folder
incdir = topdir + '/src_c/include/'  # Include folder with header files

# Get all file from source dir:
files = os.listdir(srcdir)

# This will filter the results for just the c files:
files = list(filter(lambda x:     re.search('.+[.]c$',     x), files))
files = list(filter(lambda x: not re.search('[.#].+[.]c$', x), files))

inc = [get_include(), incdir]
eca = []
ela = []

extensions = []
for efile in files:
  e = Extension('pytips.lib.'+efile.rstrip('.c'),
                sources=["{:s}{:s}".format(srcdir, efile)],
                include_dirs=inc,
                extra_compile_args=eca,
                extra_link_args=ela)
  extensions.append(e)


setup(name         = "pytips",
      version      = '{:d}.{:d}.{:d}'.format(ver.pytips_VER,
                                             ver.pytips_MIN, ver.pytips_REV),
      author       = "Patricio Cubillos",
      author_email = "patricio.cubillos@oeaw.ac.at",
      url          = "https://github.com/pcubillos/pytips",
      packages     = setuptools.find_packages(),
      install_requires = ['numpy>=1.8.1'],
      license      = "MIT",
      description  = 'Partition-Function Calculator',
      include_dirs = inc,
      ext_modules  = extensions)
