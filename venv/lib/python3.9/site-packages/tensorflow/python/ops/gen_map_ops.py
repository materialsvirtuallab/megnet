"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: map_ops.cc
"""

import collections

from tensorflow.python import pywrap_tfe as pywrap_tfe
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes

from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.deprecation import deprecated_endpoints
from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export

from typing import TypeVar

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('empty_tensor_map')
def empty_tensor_map(name=None):
  r"""Creates and returns an empty tensor map.

  handle: an empty tensor map

  Args:
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "EmptyTensorMap", name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_empty_tensor_map(
          (name,), None)
      if _result is not NotImplemented:
        return _result
      return empty_tensor_map_eager_fallback(
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            empty_tensor_map, (), dict(name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_empty_tensor_map(
        (name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "EmptyTensorMap", name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          empty_tensor_map, (), dict(name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "EmptyTensorMap", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

EmptyTensorMap = tf_export("raw_ops.EmptyTensorMap")(_ops.to_raw_op(empty_tensor_map))
_dispatcher_for_empty_tensor_map = empty_tensor_map._tf_type_based_dispatcher.Dispatch


def empty_tensor_map_eager_fallback(name, ctx):
  _inputs_flat = []
  _attrs = None
  _result = _execute.execute(b"EmptyTensorMap", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "EmptyTensorMap", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('tensor_map_erase')
def tensor_map_erase(input_handle, key, value_dtype, name=None):
  r"""Returns a tensor map with item from given key erased.

  input_handle: the original map
  output_handle: the map with value from given key removed
  key: the key of the value to be erased

  Args:
    input_handle: A `Tensor` of type `variant`.
    key: A `Tensor`.
    value_dtype: A `tf.DType`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorMapErase", name, input_handle, key, "value_dtype",
        value_dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_tensor_map_erase(
          (input_handle, key, value_dtype, name,), None)
      if _result is not NotImplemented:
        return _result
      return tensor_map_erase_eager_fallback(
          input_handle, key, value_dtype=value_dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            tensor_map_erase, (), dict(input_handle=input_handle, key=key,
                                       value_dtype=value_dtype, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_tensor_map_erase(
        (input_handle, key, value_dtype, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  value_dtype = _execute.make_type(value_dtype, "value_dtype")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorMapErase", input_handle=input_handle, key=key,
                          value_dtype=value_dtype, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          tensor_map_erase, (), dict(input_handle=input_handle, key=key,
                                     value_dtype=value_dtype, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("key_dtype", _op._get_attr_type("key_dtype"), "value_dtype",
              _op._get_attr_type("value_dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorMapErase", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorMapErase = tf_export("raw_ops.TensorMapErase")(_ops.to_raw_op(tensor_map_erase))
_dispatcher_for_tensor_map_erase = tensor_map_erase._tf_type_based_dispatcher.Dispatch


def tensor_map_erase_eager_fallback(input_handle, key, value_dtype, name, ctx):
  value_dtype = _execute.make_type(value_dtype, "value_dtype")
  _attr_key_dtype, (key,) = _execute.args_to_matching_eager([key], ctx, [])
  input_handle = _ops.convert_to_tensor(input_handle, _dtypes.variant)
  _inputs_flat = [input_handle, key]
  _attrs = ("key_dtype", _attr_key_dtype, "value_dtype", value_dtype)
  _result = _execute.execute(b"TensorMapErase", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorMapErase", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('tensor_map_has_key')
def tensor_map_has_key(input_handle, key, name=None):
  r"""Returns whether the given key exists in the map.

  input_handle: the input map
  key: the key to check
  has_key: whether the key is already in the map or not

  Args:
    input_handle: A `Tensor` of type `variant`.
    key: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorMapHasKey", name, input_handle, key)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_tensor_map_has_key(
          (input_handle, key, name,), None)
      if _result is not NotImplemented:
        return _result
      return tensor_map_has_key_eager_fallback(
          input_handle, key, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            tensor_map_has_key, (), dict(input_handle=input_handle, key=key,
                                         name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_tensor_map_has_key(
        (input_handle, key, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorMapHasKey", input_handle=input_handle, key=key, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          tensor_map_has_key, (), dict(input_handle=input_handle, key=key,
                                       name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("key_dtype", _op._get_attr_type("key_dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorMapHasKey", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorMapHasKey = tf_export("raw_ops.TensorMapHasKey")(_ops.to_raw_op(tensor_map_has_key))
_dispatcher_for_tensor_map_has_key = tensor_map_has_key._tf_type_based_dispatcher.Dispatch


def tensor_map_has_key_eager_fallback(input_handle, key, name, ctx):
  _attr_key_dtype, (key,) = _execute.args_to_matching_eager([key], ctx, [])
  input_handle = _ops.convert_to_tensor(input_handle, _dtypes.variant)
  _inputs_flat = [input_handle, key]
  _attrs = ("key_dtype", _attr_key_dtype)
  _result = _execute.execute(b"TensorMapHasKey", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorMapHasKey", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('tensor_map_insert')
def tensor_map_insert(input_handle, key, value, name=None):
  r"""Returns a map that is the 'input_handle' with the given key-value pair inserted.

  input_handle: the original map
  output_handle: the map with key and value inserted
  key: the key to be inserted
  value: the value to be inserted

  Args:
    input_handle: A `Tensor` of type `variant`.
    key: A `Tensor`.
    value: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorMapInsert", name, input_handle, key, value)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_tensor_map_insert(
          (input_handle, key, value, name,), None)
      if _result is not NotImplemented:
        return _result
      return tensor_map_insert_eager_fallback(
          input_handle, key, value, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            tensor_map_insert, (), dict(input_handle=input_handle, key=key,
                                        value=value, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_tensor_map_insert(
        (input_handle, key, value, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorMapInsert", input_handle=input_handle, key=key, value=value,
                           name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          tensor_map_insert, (), dict(input_handle=input_handle, key=key,
                                      value=value, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("key_dtype", _op._get_attr_type("key_dtype"), "value_dtype",
              _op._get_attr_type("value_dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorMapInsert", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorMapInsert = tf_export("raw_ops.TensorMapInsert")(_ops.to_raw_op(tensor_map_insert))
_dispatcher_for_tensor_map_insert = tensor_map_insert._tf_type_based_dispatcher.Dispatch


def tensor_map_insert_eager_fallback(input_handle, key, value, name, ctx):
  _attr_key_dtype, (key,) = _execute.args_to_matching_eager([key], ctx, [])
  _attr_value_dtype, (value,) = _execute.args_to_matching_eager([value], ctx, [])
  input_handle = _ops.convert_to_tensor(input_handle, _dtypes.variant)
  _inputs_flat = [input_handle, key, value]
  _attrs = ("key_dtype", _attr_key_dtype, "value_dtype", _attr_value_dtype)
  _result = _execute.execute(b"TensorMapInsert", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorMapInsert", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('tensor_map_lookup')
def tensor_map_lookup(input_handle, key, value_dtype, name=None):
  r"""Returns the value from a given key in a tensor map.

  input_handle: the input map
  key: the key to be looked up
  value: the value found from the given key

  Args:
    input_handle: A `Tensor` of type `variant`.
    key: A `Tensor`.
    value_dtype: A `tf.DType`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `value_dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorMapLookup", name, input_handle, key, "value_dtype",
        value_dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_tensor_map_lookup(
          (input_handle, key, value_dtype, name,), None)
      if _result is not NotImplemented:
        return _result
      return tensor_map_lookup_eager_fallback(
          input_handle, key, value_dtype=value_dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            tensor_map_lookup, (), dict(input_handle=input_handle, key=key,
                                        value_dtype=value_dtype, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_tensor_map_lookup(
        (input_handle, key, value_dtype, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  value_dtype = _execute.make_type(value_dtype, "value_dtype")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorMapLookup", input_handle=input_handle, key=key,
                           value_dtype=value_dtype, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          tensor_map_lookup, (), dict(input_handle=input_handle, key=key,
                                      value_dtype=value_dtype, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("key_dtype", _op._get_attr_type("key_dtype"), "value_dtype",
              _op._get_attr_type("value_dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorMapLookup", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorMapLookup = tf_export("raw_ops.TensorMapLookup")(_ops.to_raw_op(tensor_map_lookup))
_dispatcher_for_tensor_map_lookup = tensor_map_lookup._tf_type_based_dispatcher.Dispatch


def tensor_map_lookup_eager_fallback(input_handle, key, value_dtype, name, ctx):
  value_dtype = _execute.make_type(value_dtype, "value_dtype")
  _attr_key_dtype, (key,) = _execute.args_to_matching_eager([key], ctx, [])
  input_handle = _ops.convert_to_tensor(input_handle, _dtypes.variant)
  _inputs_flat = [input_handle, key]
  _attrs = ("key_dtype", _attr_key_dtype, "value_dtype", value_dtype)
  _result = _execute.execute(b"TensorMapLookup", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorMapLookup", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('tensor_map_size')
def tensor_map_size(input_handle, name=None):
  r"""Returns the number of tensors in the input tensor map.

  input_handle: the input map
  size: the number of tensors in the map

  Args:
    input_handle: A `Tensor` of type `variant`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorMapSize", name, input_handle)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_tensor_map_size(
          (input_handle, name,), None)
      if _result is not NotImplemented:
        return _result
      return tensor_map_size_eager_fallback(
          input_handle, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            tensor_map_size, (), dict(input_handle=input_handle, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_tensor_map_size(
        (input_handle, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorMapSize", input_handle=input_handle, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          tensor_map_size, (), dict(input_handle=input_handle, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorMapSize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorMapSize = tf_export("raw_ops.TensorMapSize")(_ops.to_raw_op(tensor_map_size))
_dispatcher_for_tensor_map_size = tensor_map_size._tf_type_based_dispatcher.Dispatch


def tensor_map_size_eager_fallback(input_handle, name, ctx):
  input_handle = _ops.convert_to_tensor(input_handle, _dtypes.variant)
  _inputs_flat = [input_handle]
  _attrs = None
  _result = _execute.execute(b"TensorMapSize", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorMapSize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('tensor_map_stack_keys')
def tensor_map_stack_keys(input_handle, key_dtype, name=None):
  r"""Returns a Tensor stack of all keys in a tensor map.

  input_handle: the input map
  keys: the returned Tensor of all keys in the map

  Args:
    input_handle: A `Tensor` of type `variant`.
    key_dtype: A `tf.DType`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `key_dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorMapStackKeys", name, input_handle, "key_dtype",
        key_dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_tensor_map_stack_keys(
          (input_handle, key_dtype, name,), None)
      if _result is not NotImplemented:
        return _result
      return tensor_map_stack_keys_eager_fallback(
          input_handle, key_dtype=key_dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            tensor_map_stack_keys, (), dict(input_handle=input_handle,
                                            key_dtype=key_dtype, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_tensor_map_stack_keys(
        (input_handle, key_dtype, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  key_dtype = _execute.make_type(key_dtype, "key_dtype")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorMapStackKeys", input_handle=input_handle, key_dtype=key_dtype,
                              name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          tensor_map_stack_keys, (), dict(input_handle=input_handle,
                                          key_dtype=key_dtype, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("key_dtype", _op._get_attr_type("key_dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorMapStackKeys", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorMapStackKeys = tf_export("raw_ops.TensorMapStackKeys")(_ops.to_raw_op(tensor_map_stack_keys))
_dispatcher_for_tensor_map_stack_keys = tensor_map_stack_keys._tf_type_based_dispatcher.Dispatch


def tensor_map_stack_keys_eager_fallback(input_handle, key_dtype, name, ctx):
  key_dtype = _execute.make_type(key_dtype, "key_dtype")
  input_handle = _ops.convert_to_tensor(input_handle, _dtypes.variant)
  _inputs_flat = [input_handle]
  _attrs = ("key_dtype", key_dtype)
  _result = _execute.execute(b"TensorMapStackKeys", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorMapStackKeys", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

