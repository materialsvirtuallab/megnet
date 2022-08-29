import os
import re

from setuptools import find_packages, setup

this_dir = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(this_dir, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


with open("megnet/__init__.py", encoding="utf-8") as fd:
    try:
        lines = ""
        for item in fd.readlines():
            item = item
            lines += item + "\n"
    except Exception as exc:
        raise Exception(f"Caught exception {exc}")


version = re.search('__version__ = "(.*)"', lines).group(1)


setup(
    name="megnet",
    version=version,
    description="MatErials Graph Networks for machine learning of molecules and crystals.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Chi Chen",
    author_email="chc273@eng.ucsd.edu",
    download_url="https://github.com/materialsvirtuallab/megnet",
    license="BSD",
    install_requires=["numpy", "scikit-learn", "pymatgen>=2019.10.4", "monty", "tqdm"],
    extras_require={
        "model_saving": ["h5py"],
        "molecules": ["openbabel", "rdkit"],
        "tensorflow": ["tensorflow>=2.1"],
        "tensorflow with gpu": ["tensorflow-gpu>=2.1"],
    },
    packages=find_packages(),
    package_data={
        "megnet": ["*.json", "*.md"],
    },
    include_package_data=True,
    keywords=["materials", "science", "machine", "learning", "deep", "graph", "networks", "neural"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    entry_points={
        "console_scripts": [
            "meg = megnet.cli.meg:main",
        ]
    },
)
