"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding=("utf-8"))
print(long_description)
# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    # There are some restrictions on what makes a valid project name
    # specification here:
    # https://packaging.python.org/specifications/core-metadata/#name
    name="shockwavePropagation", 
    version="0.0.1",
    description="Software for simulating shockwave propagation including overpressure peak, positive and negative wave zone.",
    url="https://github.com/JGil-polsl/shockwavePropagation",
    author="Jaroslaw Gil",  
    author_email="jaroslaw.gil@polsl.pl",
    long_description=long_description,
    long_description_content_type='text/markdown',
    # Classifiers help users find your project by categorizing it.
    #
    # For a list of valid classifiers, see https://pypi.org/classifiers/
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # Pick your license as you wish
        "License :: OSI Approved :: GNU Affero General Public License v3",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate you support Python 3. These classifiers are *not*
        # checked by 'pip install'. See instead 'python_requires' below.
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
    ],
    # This field adds keywords for your project which will appear on the
    # project page. What does your project relate to?
    #
    # Note that this is a list of additional keywords, separated
    # by commas, to be used to assist searching for the distribution in a
    # larger catalog.
    keywords="shockwave propagation",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.7",
    
    install_requires=["numpy", "pandas", "matplotlib"], 
    
    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # `pip` to create the appropriate form of executable for the target
    # platform.
    #
    # For example, the following would provide a command called `sample` which
    # executes the function `main` from this package when invoked:
    entry_points={ 
        "console_scripts": [
            "clonaEvolution=__main__:main",
        ],
    },
    # List additional URLs that are relevant to your project as a dict. 
    #
    # This field corresponds to the "Project-URL" metadata fields:
    # https://packaging.python.org/specifications/core-metadata/#project-url-multiple-use
    #
    # Examples listed include a pattern for specifying where the package tracks
    # issues, where the source is hosted, where to say thanks to the package
    # maintainers, and where to support the project financially. The key is
    # what's used to render the link text on PyPI.
    project_urls={  # Optional
        "Bug Reports": "https://github.com/JGil-polsl/shockwavePropagation/issues",
        "Source": "https://github.com/JGil-polsl/shockwavePropagation",
    },
)
