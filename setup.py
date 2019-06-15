#!/usr/bin/env python

"""
Meshes: easy to use meshes generator that export to PLY and OBJ
"""

# from distutils.core import setup
from setuptools import setup

doc_lines = __doc__.split('\n')

setup(  
    name              = 'Meshes',
    description       = doc_lines[0],
    long_description  = '\n'.join(doc_lines[2:]),
    # install_requires  = ['numpy', 'triangle'],
    version           = '0.2',
    author            = 'Patricio Gonzalez Vivo',
    author_email      = 'patriciogonzalezvivo@gmail.com',
    packages          = [ 'Meshes' ],
)