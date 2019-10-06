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
