# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name="SpyDust",  # Required
    version="1.2",  # Required
    description="A code for modeling spinning dust radiation",  # Optional
    long_description=long_description,  # Optional
    long_description_content_type="text/markdown",  # Optional (see note above)
    url="https://github.com/zzhang0123/SpyDust",  # Optional
    author="Zheng Zhang",  # Optional
    author_email="zheng.zhang@manchester.ac.uk",  # Optional
    # For a list of valid classifiers, see https://pypi.org/classifiers/
    classifiers=[  # Optional
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    keywords="astrophysics, astronomy",  # Optional
    #package_dir={"": "SpyDust"},  # Optional
    packages=find_packages(include=["SpyDust", 
                                    "SpyDust.SPDUST_as_is",
                                    #"SpyDust.core", 
                                    #"SpyDust.main", 
                                    #"SpyDust.utils"
                                    ]),  # Required
    python_requires=">=3.7, <4",
    # This field lists other packages that your project depends on to run.
    # Any package you put here will be installed by pip when your project is
    # installed, so they must be valid existing projects.
    #
    # For an analysis of "install_requires" vs pip's requirements files see:
    # https://packaging.python.org/discussions/install-requires-vs-requirements/
    install_requires=[
        "numpy", 
        "scipy",
        "mpi4py",
        #"pandas" # not necessary for spinning dust spectra; 
                  # but if free-free emission is also desired (using free_free.py), then it is needed
        ],
    extras_require={},
    zip_safe=False, # don't compile the package into a zip file when installed
    # If there are data files included in your packages that need to be
    # installed, specify them here.
    package_data={'SpyDust': ['Data_Files/*']},
)

