# Copyright (c) 2015-2018 Patricio Cubillos and contributors.
# pytips is open-source software under the MIT license (see LICENSE).

from numpy import get_include
import os, re, sys
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
for i in range(len(files)):
  e = Extension(files[i].rstrip('.c'),
                sources=["{:s}{:s}".format(srcdir,files[i])],
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
      packages     = ["pytips"],
      license      = ["MIT"],
      description  = 'Partition-Function Calculator',
      include_dirs = inc,
      ext_modules  = extensions)
