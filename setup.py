from setuptools import setup
from setuptools import find_packages

setup(name='megnet',
      version='0.0.1',
      decription='MatErials Graph Networks for machine learning of molecules and crystals.',
      author='Chi Chen',
      author_email='chc273@eng.ucsd.edu',
      download_url='...',
      license='MIT',
      install_requires=['keras', 'numpy', 'tensorflow', "sklearn", 'pymatgen'],
      extras_require={
          'model_saving': ['json', 'h5py'],},
      package_data={'megnet':['README.md']},
      package=find_packages())
