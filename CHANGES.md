# v1.2.0
* Bug fix

# v1.1.9
* Add multi-fidelity model examples 
* Add sample weights for model training
* Add default optimizer gradient norm clip 

# v1.1.8
* Bug fix of megnet descriptors

# v1.1.7
* Update the model training mechanism and move Gaussian expansion to tensorflow, training speed up 100%

# v1.1.6
* minor fix to include more linear readout option
* add data type control
* refactor local_env

# v1.1.5
* Code refactor and reformat
* Fix tensorflow and numpy type compatibility issues

# v1.1.4
* Update to tensorflow.keras API instead of using keras

# v1.1.3
* Download mvl_models from figshare if not present

# v1.1.2
* Add mvl_models in wheel release file

# v1.1.0
* Bug fix and version correction

# v1.0.3
* Fix bug brought by migrating to tensorflow 2.0
* New elasticity models trained on 2019 MP data base
* Add meg command line tools 

# v1.0.2
* Add mypy typing hint for non-tensorflow codes
* Update keras to 2.3.1 to fix thread-safety issues

# v1.0.1
* New find_points_in_spheres algorithm in pymatgen for graph construction 

# v1.0.0
* Tensorflow 2.0 version.

# v0.3.4
* Change `convertor` to `converter` in all model APIs
* Improve `ReduceLRUponNan` callback function
* @WardLT major contributions to the `MolecularGraph` class
* Add serialization methods for `local_env` classes
* delete `data/mp.py`  

# v0.3.3
* GraphModel and MEGNetModel now supports a metadata tag, which is included in
  the JSON. (suggestion of @mhorton).
* Misc bug fixes for edge cases as well as improved error messages for
  mismatches in inputs.

# v0.3.2
* Implement the option for a scaler in models, which is used in efermi models at
  the moment but also can be helpful for extensive quantities.

# v0.3.1
* Minor fixes to setup.py and licenses.

# v0.3.0
* Proper fix to release on PyPi.

# v0.2.0
* Major refactoring to conform to OOP principles. Note that the
  changes are not backwards compatible, but many things are a lot
  simpler. We do not expect much disruption to existing users.
* Added pre-trained models developed in our work for users who
  wish to simply use them for prediction.
* Major improvements to README and documentation.

# v0.1.0

* Bug fix for dimension problem when only one atom in structure

# v0.0.1

* Initial release
