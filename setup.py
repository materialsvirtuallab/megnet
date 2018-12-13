from setuptools import setup
from setuptools import find_packages

setup(
    name='megnet',
    version='0.0.1',
    decription='MatErials Graph Networks for machine learning of molecules and crystals.',
    author='Chi Chen',
    author_email='chc273@eng.ucsd.edu',
    download_url='https://github.com/materialsvirtuallab/megnet',
    license='BSD',
    install_requires=['keras', 'numpy', 'tensorflow', "sklearn", 'pymatgen'],
    extras_require={
        'model_saving': ['h5py'], },
    package_data={'megnet': ['README.md']},
    package=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
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
