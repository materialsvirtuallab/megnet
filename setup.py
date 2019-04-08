from setuptools import setup
from setuptools import find_packages
from os.path import dirname, abspath, join
this_dir = abspath(dirname(__file__))

with open(join(this_dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='megnet',
    version='0.2.0',
    decription='MatErials Graph Networks for machine learning of molecules and crystals.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Chi Chen',
    author_email='chc273@eng.ucsd.edu',
    download_url='https://github.com/materialsvirtuallab/megnet',
    license='BSD',
    install_requires=['keras', 'numpy', 'tensorflow', "scikit-learn",
                      'pymatgen', 'monty'],
    extras_require={
        'model_saving': ['h5py'], },
    package=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
)
