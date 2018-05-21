#!/usr/bin/env python

from distutils.core import setup
import codecs
import os

# Get the long description from the README file
here = os.path.abspath(os.path.dirname(__file__))
with codecs.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='cosmo',
      version='0.1', # major.minor[.patch[.sub]].
      description='Consensus Self-Organizing Models',
      long_description=long_description,
      author='Yuantao Fan',
      # author_email='',
      # maintainer='',
      # maintainer_email='',
      url='https://github.com/saeedghsh/cosmo',
      packages=['cosmo',],
      # keywords='',
      license='GPL'
     )
