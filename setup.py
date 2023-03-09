from setuptools import setup, find_packages
import codecs
import os
import pathlib
import pkg_resources



with pathlib.Path("/home/simon_g/MICCAI_SUBMISSION/TESTPYPI/requirements.txt").open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]

VERSION = '0.0.4'


# Setting up
setup(
    name="GENUINE",
    version=VERSION,
    author="",
    author_email="",
    packages=find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)