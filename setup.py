from numpy import get_include
import os, re, sys
from distutils.core import setup, Extension

srcdir = 'src/'    # C-code source folder
incdir = 'include' # Include folder with header files
#libdir = 'lib/'    # Where the shared objects are put

files = os.listdir(srcdir)
# This will filter the results for just the c files:
files = filter(lambda x:     re.search('.+[.]c$',     x), files)
files = filter(lambda x: not re.search('[.#].+[.]c$', x), files)

inc = [get_include(), incdir]
eca = []  # ['-fopenmp']
ela = []  # ['-lgomp']

extensions = []
for i in range(len(files)):
  e = Extension(files[i].rstrip('.c'),
                sources=["{:s}{:s}".format(srcdir,files[i])],
                include_dirs=inc,
                extra_compile_args=eca,
                extra_link_args=ela)
  extensions.append(e)


setup(name          = "CTIPS",
      version       = '1.0.0',
      author        = "Patricio Cubillos",
      author_email  = "pcubillos@fulbrightmail.org",
      url           = "https://github.com/pcubillos/ctips",
      description   = 'Partition Function Calculator',
      ext_modules   = extensions)
