# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Top-level module of TensorFlow. By convention, we refer to this module as
`tf` instead of `tensorflow`, following the common practice of importing
TensorFlow via the command `import tensorflow as tf`.

The primary function of this module is to import all of the public TensorFlow
interfaces into a single place. The interfaces themselves are located in
sub-modules, as described below.

Note that the file `__init__.py` in the TensorFlow source code tree is actually
only a placeholder to enable test cases to run. The TensorFlow build replaces
this file with a file generated from [`api_template.__init__.py`](https://www.github.com/tensorflow/tensorflow/blob/master/tensorflow/api_template.__init__.py)
"""

from __future__ import absolute_import as _absolute_import
from __future__ import division as _division
from __future__ import print_function as _print_function

import distutils as _distutils
import inspect as _inspect
import logging as _logging
import os as _os
import site as _site
import six as _six
import sys as _sys

from tensorflow.python.tools import module_util as _module_util
from tensorflow.python.util.lazy_loader import LazyLoader as _LazyLoader

# Make sure code inside the TensorFlow codebase can use tf2.enabled() at import.
_os.environ['TF2_BEHAVIOR'] = '1'
from tensorflow.python import tf2 as _tf2
_tf2.enable()

from . import __internal__
from . import __operators__
from . import audio
from . import autodiff
from . import autograph
from . import bitwise
from . import compat
from . import config
from . import data
from . import debugging
from . import distribute
from . import dtypes
from . import errors
from . import experimental
from . import feature_column
from . import graph_util
from . import image
from . import io
from . import linalg
from . import lite
from . import lookup
from . import math
from . import mixed_precision
from . import mlir
from . import nest
from . import nn
from . import profiler
from . import quantization
from . import queue
from . import ragged
from . import random
from . import raw_ops
from . import saved_model
from . import sets
from . import signal
from . import sparse
from . import strings
from . import summary
from . import sysconfig
from . import test
from . import tpu
from . import train
from . import types
from . import version
from . import xla
from tensorflow.python.data.ops.optional_ops import OptionalSpec
from tensorflow.python.eager.backprop import GradientTape
from tensorflow.python.eager.context import executing_eagerly
from tensorflow.python.eager.def_function import function
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.framework.device_spec import DeviceSpecV2 as DeviceSpec
from tensorflow.python.framework.dtypes import DType
from tensorflow.python.framework.dtypes import as_dtype
from tensorflow.python.framework.dtypes import bfloat16
from tensorflow.python.framework.dtypes import bool
from tensorflow.python.framework.dtypes import complex128
from tensorflow.python.framework.dtypes import complex64
from tensorflow.python.framework.dtypes import double
from tensorflow.python.framework.dtypes import float16
from tensorflow.python.framework.dtypes import float32
from tensorflow.python.framework.dtypes import float64
from tensorflow.python.framework.dtypes import half
from tensorflow.python.framework.dtypes import int16
from tensorflow.python.framework.dtypes import int32
from tensorflow.python.framework.dtypes import int64
from tensorflow.python.framework.dtypes import int8
from tensorflow.python.framework.dtypes import qint16
from tensorflow.python.framework.dtypes import qint32
from tensorflow.python.framework.dtypes import qint8
from tensorflow.python.framework.dtypes import quint16
from tensorflow.python.framework.dtypes import quint8
from tensorflow.python.framework.dtypes import resource
from tensorflow.python.framework.dtypes import string
from tensorflow.python.framework.dtypes import uint16
from tensorflow.python.framework.dtypes import uint32
from tensorflow.python.framework.dtypes import uint64
from tensorflow.python.framework.dtypes import uint8
from tensorflow.python.framework.dtypes import variant
from tensorflow.python.framework.importer import import_graph_def
from tensorflow.python.framework.indexed_slices import IndexedSlices
from tensorflow.python.framework.indexed_slices import IndexedSlicesSpec
from tensorflow.python.framework.load_library import load_library
from tensorflow.python.framework.load_library import load_op_library
from tensorflow.python.framework.ops import Graph
from tensorflow.python.framework.ops import Operation
from tensorflow.python.framework.ops import RegisterGradient
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.framework.ops import control_dependencies
from tensorflow.python.framework.ops import convert_to_tensor_v2_with_dispatch as convert_to_tensor
from tensorflow.python.framework.ops import device_v2 as device
from tensorflow.python.framework.ops import get_current_name_scope
from tensorflow.python.framework.ops import init_scope
from tensorflow.python.framework.ops import inside_function
from tensorflow.python.framework.ops import name_scope_v2 as name_scope
from tensorflow.python.framework.ops import no_gradient
from tensorflow.python.framework.sparse_tensor import SparseTensor
from tensorflow.python.framework.sparse_tensor import SparseTensorSpec
from tensorflow.python.framework.tensor_conversion_registry import register_tensor_conversion_function
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.framework.tensor_spec import TensorSpec
from tensorflow.python.framework.tensor_util import MakeNdarray as make_ndarray
from tensorflow.python.framework.tensor_util import constant_value as get_static_value
from tensorflow.python.framework.tensor_util import is_tf_type as is_tensor
from tensorflow.python.framework.tensor_util import make_tensor_proto
from tensorflow.python.framework.type_spec import TypeSpec
from tensorflow.python.framework.type_spec import type_spec_from_value
from tensorflow.python.framework.versions import COMPILER_VERSION as __compiler_version__
from tensorflow.python.framework.versions import CXX11_ABI_FLAG as __cxx11_abi_flag__
from tensorflow.python.framework.versions import GIT_VERSION as __git_version__
from tensorflow.python.framework.versions import MONOLITHIC_BUILD as __monolithic_build__
from tensorflow.python.framework.versions import VERSION as __version__
from tensorflow.python.module.module import Module
from tensorflow.python.ops.array_ops import batch_to_space_v2 as batch_to_space
from tensorflow.python.ops.array_ops import boolean_mask_v2 as boolean_mask
from tensorflow.python.ops.array_ops import broadcast_dynamic_shape
from tensorflow.python.ops.array_ops import broadcast_static_shape
from tensorflow.python.ops.array_ops import concat
from tensorflow.python.ops.array_ops import edit_distance
from tensorflow.python.ops.array_ops import expand_dims_v2 as expand_dims
from tensorflow.python.ops.array_ops import fill
from tensorflow.python.ops.array_ops import fingerprint
from tensorflow.python.ops.array_ops import gather_nd_v2 as gather_nd
from tensorflow.python.ops.array_ops import gather_v2 as gather
from tensorflow.python.ops.array_ops import guarantee_const
from tensorflow.python.ops.array_ops import identity
from tensorflow.python.ops.array_ops import meshgrid
from tensorflow.python.ops.array_ops import newaxis
from tensorflow.python.ops.array_ops import one_hot
from tensorflow.python.ops.array_ops import ones
from tensorflow.python.ops.array_ops import ones_like_v2 as ones_like
from tensorflow.python.ops.array_ops import pad_v2 as pad
from tensorflow.python.ops.array_ops import parallel_stack
from tensorflow.python.ops.array_ops import rank
from tensorflow.python.ops.array_ops import repeat
from tensorflow.python.ops.array_ops import required_space_to_batch_paddings
from tensorflow.python.ops.array_ops import reshape
from tensorflow.python.ops.array_ops import reverse_sequence_v2 as reverse_sequence
from tensorflow.python.ops.array_ops import searchsorted
from tensorflow.python.ops.array_ops import sequence_mask
from tensorflow.python.ops.array_ops import shape_n
from tensorflow.python.ops.array_ops import shape_v2 as shape
from tensorflow.python.ops.array_ops import size_v2 as size
from tensorflow.python.ops.array_ops import slice
from tensorflow.python.ops.array_ops import space_to_batch_v2 as space_to_batch
from tensorflow.python.ops.array_ops import split
from tensorflow.python.ops.array_ops import squeeze_v2 as squeeze
from tensorflow.python.ops.array_ops import stack
from tensorflow.python.ops.array_ops import stop_gradient
from tensorflow.python.ops.array_ops import strided_slice
from tensorflow.python.ops.array_ops import tensor_scatter_nd_update
from tensorflow.python.ops.array_ops import transpose_v2 as transpose
from tensorflow.python.ops.array_ops import unique
from tensorflow.python.ops.array_ops import unique_with_counts
from tensorflow.python.ops.array_ops import unstack
from tensorflow.python.ops.array_ops import where_v2 as where
from tensorflow.python.ops.array_ops import zeros
from tensorflow.python.ops.array_ops import zeros_like_v2 as zeros_like
from tensorflow.python.ops.batch_ops import batch_function as nondifferentiable_batch_function
from tensorflow.python.ops.check_ops import assert_equal_v2 as assert_equal
from tensorflow.python.ops.check_ops import assert_greater_v2 as assert_greater
from tensorflow.python.ops.check_ops import assert_less_v2 as assert_less
from tensorflow.python.ops.check_ops import assert_rank_v2 as assert_rank
from tensorflow.python.ops.check_ops import ensure_shape
from tensorflow.python.ops.clip_ops import clip_by_global_norm
from tensorflow.python.ops.clip_ops import clip_by_norm
from tensorflow.python.ops.clip_ops import clip_by_value
from tensorflow.python.ops.control_flow_ops import Assert
from tensorflow.python.ops.control_flow_ops import case_v2 as case
from tensorflow.python.ops.control_flow_ops import cond_for_tf_v2 as cond
from tensorflow.python.ops.control_flow_ops import group
from tensorflow.python.ops.control_flow_ops import switch_case
from tensorflow.python.ops.control_flow_ops import tuple_v2 as tuple
from tensorflow.python.ops.control_flow_ops import while_loop_v2 as while_loop
from tensorflow.python.ops.critical_section_ops import CriticalSection
from tensorflow.python.ops.custom_gradient import custom_gradient
from tensorflow.python.ops.custom_gradient import grad_pass_through
from tensorflow.python.ops.custom_gradient import recompute_grad
from tensorflow.python.ops.functional_ops import foldl_v2 as foldl
from tensorflow.python.ops.functional_ops import foldr_v2 as foldr
from tensorflow.python.ops.functional_ops import scan_v2 as scan
from tensorflow.python.ops.gen_array_ops import bitcast
from tensorflow.python.ops.gen_array_ops import broadcast_to
from tensorflow.python.ops.gen_array_ops import extract_volume_patches
from tensorflow.python.ops.gen_array_ops import identity_n
from tensorflow.python.ops.gen_array_ops import reverse_v2 as reverse
from tensorflow.python.ops.gen_array_ops import scatter_nd
from tensorflow.python.ops.gen_array_ops import space_to_batch_nd
from tensorflow.python.ops.gen_array_ops import tensor_scatter_add as tensor_scatter_nd_add
from tensorflow.python.ops.gen_array_ops import tensor_scatter_max as tensor_scatter_nd_max
from tensorflow.python.ops.gen_array_ops import tensor_scatter_min as tensor_scatter_nd_min
from tensorflow.python.ops.gen_array_ops import tensor_scatter_sub as tensor_scatter_nd_sub
from tensorflow.python.ops.gen_array_ops import tile
from tensorflow.python.ops.gen_array_ops import unravel_index
from tensorflow.python.ops.gen_control_flow_ops import no_op
from tensorflow.python.ops.gen_data_flow_ops import dynamic_partition
from tensorflow.python.ops.gen_data_flow_ops import dynamic_stitch
from tensorflow.python.ops.gen_linalg_ops import matrix_square_root
from tensorflow.python.ops.gen_logging_ops import timestamp
from tensorflow.python.ops.gen_math_ops import acosh
from tensorflow.python.ops.gen_math_ops import asin
from tensorflow.python.ops.gen_math_ops import asinh
from tensorflow.python.ops.gen_math_ops import atan
from tensorflow.python.ops.gen_math_ops import atan2
from tensorflow.python.ops.gen_math_ops import atanh
from tensorflow.python.ops.gen_math_ops import cos
from tensorflow.python.ops.gen_math_ops import cosh
from tensorflow.python.ops.gen_math_ops import greater
from tensorflow.python.ops.gen_math_ops import greater_equal
from tensorflow.python.ops.gen_math_ops import less
from tensorflow.python.ops.gen_math_ops import less_equal
from tensorflow.python.ops.gen_math_ops import logical_and
from tensorflow.python.ops.gen_math_ops import logical_not
from tensorflow.python.ops.gen_math_ops import logical_or
from tensorflow.python.ops.gen_math_ops import maximum
from tensorflow.python.ops.gen_math_ops import minimum
from tensorflow.python.ops.gen_math_ops import neg as negative
from tensorflow.python.ops.gen_math_ops import real_div as realdiv
from tensorflow.python.ops.gen_math_ops import sin
from tensorflow.python.ops.gen_math_ops import sinh
from tensorflow.python.ops.gen_math_ops import square
from tensorflow.python.ops.gen_math_ops import tan
from tensorflow.python.ops.gen_math_ops import tanh
from tensorflow.python.ops.gen_math_ops import truncate_div as truncatediv
from tensorflow.python.ops.gen_math_ops import truncate_mod as truncatemod
from tensorflow.python.ops.gen_string_ops import as_string
from tensorflow.python.ops.gradients_impl import HessiansV2 as hessians
from tensorflow.python.ops.gradients_impl import gradients_v2 as gradients
from tensorflow.python.ops.gradients_util import AggregationMethod
from tensorflow.python.ops.histogram_ops import histogram_fixed_width
from tensorflow.python.ops.histogram_ops import histogram_fixed_width_bins
from tensorflow.python.ops.init_ops_v2 import Constant as constant_initializer
from tensorflow.python.ops.init_ops_v2 import Ones as ones_initializer
from tensorflow.python.ops.init_ops_v2 import RandomNormal as random_normal_initializer
from tensorflow.python.ops.init_ops_v2 import RandomUniform as random_uniform_initializer
from tensorflow.python.ops.init_ops_v2 import Zeros as zeros_initializer
from tensorflow.python.ops.linalg_ops import eig
from tensorflow.python.ops.linalg_ops import eigvals
from tensorflow.python.ops.linalg_ops import eye
from tensorflow.python.ops.linalg_ops import norm_v2 as norm
from tensorflow.python.ops.logging_ops import print_v2 as print
from tensorflow.python.ops.manip_ops import roll
from tensorflow.python.ops.map_fn import map_fn_v2 as map_fn
from tensorflow.python.ops.math_ops import abs
from tensorflow.python.ops.math_ops import acos
from tensorflow.python.ops.math_ops import add
from tensorflow.python.ops.math_ops import add_n
from tensorflow.python.ops.math_ops import argmax_v2 as argmax
from tensorflow.python.ops.math_ops import argmin_v2 as argmin
from tensorflow.python.ops.math_ops import cast
from tensorflow.python.ops.math_ops import complex
from tensorflow.python.ops.math_ops import cumsum
from tensorflow.python.ops.math_ops import divide
from tensorflow.python.ops.math_ops import equal
from tensorflow.python.ops.math_ops import exp
from tensorflow.python.ops.math_ops import floor
from tensorflow.python.ops.math_ops import linspace_nd as linspace
from tensorflow.python.ops.math_ops import matmul
from tensorflow.python.ops.math_ops import multiply
from tensorflow.python.ops.math_ops import not_equal
from tensorflow.python.ops.math_ops import pow
from tensorflow.python.ops.math_ops import range
from tensorflow.python.ops.math_ops import reduce_all
from tensorflow.python.ops.math_ops import reduce_any
from tensorflow.python.ops.math_ops import reduce_logsumexp
from tensorflow.python.ops.math_ops import reduce_max
from tensorflow.python.ops.math_ops import reduce_mean
from tensorflow.python.ops.math_ops import reduce_min
from tensorflow.python.ops.math_ops import reduce_prod
from tensorflow.python.ops.math_ops import reduce_sum
from tensorflow.python.ops.math_ops import round
from tensorflow.python.ops.math_ops import saturate_cast
from tensorflow.python.ops.math_ops import scalar_mul_v2 as scalar_mul
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import sign
from tensorflow.python.ops.math_ops import sqrt
from tensorflow.python.ops.math_ops import subtract
from tensorflow.python.ops.math_ops import tensordot
from tensorflow.python.ops.math_ops import truediv
from tensorflow.python.ops.parallel_for.control_flow_ops import vectorized_map
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensorSpec
from tensorflow.python.ops.script_ops import eager_py_func as py_function
from tensorflow.python.ops.script_ops import numpy_function
from tensorflow.python.ops.sort_ops import argsort
from tensorflow.python.ops.sort_ops import sort
from tensorflow.python.ops.special_math_ops import einsum
from tensorflow.python.ops.tensor_array_ops import TensorArray
from tensorflow.python.ops.tensor_array_ops import TensorArraySpec
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.ops.variable_scope import variable_creator_scope
from tensorflow.python.ops.variables import Variable
from tensorflow.python.ops.variables import VariableAggregationV2 as VariableAggregation
from tensorflow.python.ops.variables import VariableSynchronization
from tensorflow.python.platform.tf_logging import get_logger

# WRAPPER_PLACEHOLDER

# Make sure directory containing top level submodules is in
# the __path__ so that "from tensorflow.foo import bar" works.
# We're using bitwise, but there's nothing special about that.
_API_MODULE = _sys.modules[__name__].bitwise
_tf_api_dir = _os.path.dirname(_os.path.dirname(_API_MODULE.__file__))
_current_module = _sys.modules[__name__]

if not hasattr(_current_module, '__path__'):
  __path__ = [_tf_api_dir]
elif _tf_api_dir not in __path__:
  __path__.append(_tf_api_dir)

# Hook external TensorFlow modules.
# Import compat before trying to import summary from tensorboard, so that
# reexport_tf_summary can get compat from sys.modules. Only needed if using
# lazy loading.
_current_module.compat.v2  # pylint: disable=pointless-statement
try:
  from tensorboard.summary._tf import summary
  _current_module.__path__ = (
      [_module_util.get_parent_dir(summary)] + _current_module.__path__)
  setattr(_current_module, "summary", summary)
except ImportError:
  _logging.warning(
      "Limited tf.summary API due to missing TensorBoard installation.")

# Load tensorflow-io-gcs-filesystem if enabled
# pylint: disable=g-import-not-at-top
if (_os.getenv('TF_USE_MODULAR_FILESYSTEM', '0') == 'true' or
    _os.getenv('TF_USE_MODULAR_FILESYSTEM', '0') == '1'):
  import tensorflow_io_gcs_filesystem as _tensorflow_io_gcs_filesystem
# pylint: enable=g-import-not-at-top

# Lazy-load estimator.
_estimator_module = "tensorflow_estimator.python.estimator.api._v2.estimator"
estimator = _LazyLoader("estimator", globals(), _estimator_module)
_module_dir = _module_util.get_parent_dir_for_name(_estimator_module)
if _module_dir:
  _current_module.__path__ = [_module_dir] + _current_module.__path__
setattr(_current_module, "estimator", estimator)

_keras_module = "keras.api._v2.keras"
keras = _LazyLoader("keras", globals(), _keras_module)
_module_dir = _module_util.get_parent_dir_for_name(_keras_module)
if _module_dir:
  _current_module.__path__ = [_module_dir] + _current_module.__path__
setattr(_current_module, "keras", keras)

# Explicitly import lazy-loaded modules to support autocompletion.
# pylint: disable=g-import-not-at-top
if not _six.PY2:
  import typing as _typing
  if _typing.TYPE_CHECKING:
    from tensorflow_estimator.python.estimator.api._v2 import estimator
# pylint: enable=g-import-not-at-top

# Enable TF2 behaviors
from tensorflow.python.compat import v2_compat as _compat  # pylint: disable=g-import-not-at-top
_compat.enable_v2_behavior()
_major_api_version = 2


# Load all plugin libraries from site-packages/tensorflow-plugins if we are
# running under pip.
# TODO(gunan): Enable setting an environment variable to define arbitrary plugin
# directories.
# TODO(gunan): Find a better location for this code snippet.
from tensorflow.python.framework import load_library as _ll
from tensorflow.python.lib.io import file_io as _fi

# Get sitepackages directories for the python installation.
_site_packages_dirs = []
if _site.ENABLE_USER_SITE and _site.USER_SITE is not None:
  _site_packages_dirs += [_site.USER_SITE]
_site_packages_dirs += [_p for _p in _sys.path if 'site-packages' in _p]
if 'getsitepackages' in dir(_site):
  _site_packages_dirs += _site.getsitepackages()

if 'sysconfig' in dir(_distutils):
  _site_packages_dirs += [_distutils.sysconfig.get_python_lib()]

_site_packages_dirs = list(set(_site_packages_dirs))

# Find the location of this exact file.
_current_file_location = _inspect.getfile(_inspect.currentframe())

def _running_from_pip_package():
  return any(
      _current_file_location.startswith(dir_) for dir_ in _site_packages_dirs)

if _running_from_pip_package():
  # TODO(gunan): Add sanity checks to loaded modules here.

  # Load first party dynamic kernels.
  _tf_dir = _os.path.dirname(_current_file_location)
  _kernel_dir = _os.path.join(_tf_dir, 'core', 'kernels')
  if _os.path.exists(_kernel_dir):
    _ll.load_library(_kernel_dir)

  # Load third party dynamic kernels.
  for _s in _site_packages_dirs:
    _plugin_dir = _os.path.join(_s, 'tensorflow-plugins')
    if _os.path.exists(_plugin_dir):
      _ll.load_library(_plugin_dir)
      # Load Pluggable Device Library
      _ll.load_pluggable_device_library(_plugin_dir)

# Add module aliases
if hasattr(_current_module, 'keras'):
  # It is possible that keras is a lazily loaded module, which might break when
  # actually trying to import it. Have a Try-Catch to make sure it doesn't break
  # when it doing some very initial loading, like tf.compat.v2, etc.
  try:
    _keras_package = "keras.api._v2.keras."
    losses = _LazyLoader("losses", globals(), _keras_package + "losses")
    metrics = _LazyLoader("metrics", globals(), _keras_package + "metrics")
    optimizers = _LazyLoader(
        "optimizers", globals(), _keras_package + "optimizers")
    initializers = _LazyLoader(
        "initializers", globals(), _keras_package + "initializers")
    setattr(_current_module, "losses", losses)
    setattr(_current_module, "metrics", metrics)
    setattr(_current_module, "optimizers", optimizers)
    setattr(_current_module, "initializers", initializers)
  except ImportError:
    pass

# Do an eager load for Keras' code so that any function/method that needs to
# happen at load time will trigger, eg registration of optimizers in the
# SavedModel registry.
# See b/196254385 for more details.
if hasattr(_current_module, "keras"):
  try:
    keras._load()
  except ImportError:
    pass

# pylint: enable=undefined-variable

# Delete modules that should be hidden from dir().
# Don't fail if these modules are not available.
# For e.g. this file will be originally placed under tensorflow/_api/v1 which
# does not have 'python', 'core' directories. Then, it will be copied
# to tensorflow/ which does have these two directories.
# pylint: disable=undefined-variable
try:
  del python
except NameError:
  pass
try:
  del core
except NameError:
  pass
try:
  del compiler
except NameError:
  pass


_names_with_underscore = ['__compiler_version__', '__cxx11_abi_flag__', '__git_version__', '__internal__', '__monolithic_build__', '__operators__', '__version__']
__all__ = [_s for _s in dir() if not _s.startswith('_')]
__all__.extend([_s for _s in _names_with_underscore])

