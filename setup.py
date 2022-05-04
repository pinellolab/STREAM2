import sys

if sys.version_info < (3, 6):
    sys.exit('simba requires Python >= 3.6')

from setuptools import setup, find_packages
from pathlib import Path
setup(
    name='stream2',
    version='0.1a',
    author='Huidong Chen',
    athor_email='hd7chen AT gmail DOT com',
    license='BSD',
    description="STREAM2: Fast and scalable trajectory analysis"
    "of single-cell omics data",
    long_description=Path('README.md').read_text('utf-8'),
    long_description_content_type="text/markdown",
    url='https://github.com/pinellolab/STREAM2',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        p.strip()
        for p in Path('requirements.txt').read_text('utf-8').splitlines()
    ],
)
