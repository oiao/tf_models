from setuptools import setup, find_packages
import os
from os.path import join as opj

NAME = 'tf_models'
DESCR = ''
VERFILE = opj(NAME, '_version.py')
packages= find_packages()

if os.path.exists(VERFILE):
    with open(VERFILE) as f:
        exec(f.read())
else:
    __version__ = '0.0'

setup(
    name             = NAME,
    version          = __version__,
    author           = 'Leo Komissarov',
    url              = f'https://github.com/oiao/{NAME}',
    download_url     = f'https://github.com/oiao/{NAME}/archive/master.zip',
    description      = DESCR,
    classifiers      = [
            'Development Status :: 3 - Alpha',
            'Programming Language :: Python :: 3.6',
    ],
    python_requires  = '>=3.6',
    install_requires = ['numpy', 'tensorflow>=2', 'scipy', 'sklearn', 'matplotlib'],
    packages         = packages,
    package_dir      = {NAME : '.'},
    # package_data     = {NAME : ['tests/*']},
    # scripts          = [opj('scripts', NAME)],
)
