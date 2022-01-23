"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: gen_xla_ops.cc
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
@tf_export('xla_all_reduce')
def xla_all_reduce(input, group_assignment, reduce_op, name=None):
  r"""Wraps the XLA AllReduce operator

    documented at https://www.tensorflow.org/xla/operation_semantics#allreduce.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `int32`, `uint32`.
      Array or a non-empty tuple of arrays to reduce across replicas.
    group_assignment: A `Tensor` of type `int32`.
      Groups between which the reductions are performed.
    reduce_op: A `string` from: `"Min", "Max", "Mul", "Add", "Mean"`.
      Reduction computation.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaAllReduce", name, input, group_assignment, "reduce_op",
        reduce_op)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_all_reduce(
          (input, group_assignment, reduce_op, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_all_reduce_eager_fallback(
          input, group_assignment, reduce_op=reduce_op, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_all_reduce, (), dict(input=input,
                                     group_assignment=group_assignment,
                                     reduce_op=reduce_op, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_all_reduce(
        (input, group_assignment, reduce_op, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  reduce_op = _execute.make_str(reduce_op, "reduce_op")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaAllReduce", input=input, group_assignment=group_assignment,
                        reduce_op=reduce_op, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_all_reduce, (), dict(input=input,
                                   group_assignment=group_assignment,
                                   reduce_op=reduce_op, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "reduce_op",
              _op.get_attr("reduce_op"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaAllReduce", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaAllReduce = tf_export("raw_ops.XlaAllReduce")(_ops.to_raw_op(xla_all_reduce))
_dispatcher_for_xla_all_reduce = xla_all_reduce._tf_type_based_dispatcher.Dispatch


def xla_all_reduce_eager_fallback(input, group_assignment, reduce_op, name, ctx):
  reduce_op = _execute.make_str(reduce_op, "reduce_op")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.int32, _dtypes.uint32, ])
  group_assignment = _ops.convert_to_tensor(group_assignment, _dtypes.int32)
  _inputs_flat = [input, group_assignment]
  _attrs = ("T", _attr_T, "reduce_op", reduce_op)
  _result = _execute.execute(b"XlaAllReduce", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaAllReduce", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_XlaBroadcastHelperOutput = collections.namedtuple(
    "XlaBroadcastHelper",
    ["lhs_output", "rhs_output"])


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_broadcast_helper')
def xla_broadcast_helper(lhs, rhs, broadcast_dims, name=None):
  r"""Helper operator for performing XLA-style broadcasts

  Broadcasts `lhs` and `rhs` to the same rank, by adding size 1 dimensions to
  whichever of `lhs` and `rhs` has the lower rank, using XLA's broadcasting rules
  for binary operators.

  Args:
    lhs: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      the LHS input tensor
    rhs: A `Tensor`. Must have the same type as `lhs`. the RHS input tensor
    broadcast_dims: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      an XLA-style broadcast dimension specification
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (lhs_output, rhs_output).

    lhs_output: A `Tensor`. Has the same type as `lhs`. the broadcasted LHS tensor
    rhs_output: A `Tensor`. Has the same type as `lhs`. the broadcasted RHS tensor
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaBroadcastHelper", name, lhs, rhs, broadcast_dims)
      _result = _XlaBroadcastHelperOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_broadcast_helper(
          (lhs, rhs, broadcast_dims, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_broadcast_helper_eager_fallback(
          lhs, rhs, broadcast_dims, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_broadcast_helper, (), dict(lhs=lhs, rhs=rhs,
                                           broadcast_dims=broadcast_dims,
                                           name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_broadcast_helper(
        (lhs, rhs, broadcast_dims, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaBroadcastHelper", lhs=lhs, rhs=rhs, broadcast_dims=broadcast_dims,
                              name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_broadcast_helper, (), dict(lhs=lhs, rhs=rhs,
                                         broadcast_dims=broadcast_dims,
                                         name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaBroadcastHelper", _inputs_flat, _attrs, _result)
  _result = _XlaBroadcastHelperOutput._make(_result)
  return _result

XlaBroadcastHelper = tf_export("raw_ops.XlaBroadcastHelper")(_ops.to_raw_op(xla_broadcast_helper))
_dispatcher_for_xla_broadcast_helper = xla_broadcast_helper._tf_type_based_dispatcher.Dispatch


def xla_broadcast_helper_eager_fallback(lhs, rhs, broadcast_dims, name, ctx):
  _attr_T, _inputs_T = _execute.args_to_matching_eager([lhs, rhs], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (lhs, rhs) = _inputs_T
  _attr_Tindices, (broadcast_dims,) = _execute.args_to_matching_eager([broadcast_dims], ctx, [_dtypes.int32, _dtypes.int64, ])
  _inputs_flat = [lhs, rhs, broadcast_dims]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices)
  _result = _execute.execute(b"XlaBroadcastHelper", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaBroadcastHelper", _inputs_flat, _attrs, _result)
  _result = _XlaBroadcastHelperOutput._make(_result)
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_conv')
def xla_conv(lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation, feature_group_count, dimension_numbers, precision_config, name=None):
  r"""Wraps the XLA ConvGeneralDilated operator, documented at

   https://www.tensorflow.org/performance/xla/operation_semantics#conv_convolution
  .

  Args:
    lhs: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      the input tensor
    rhs: A `Tensor`. Must have the same type as `lhs`. the kernel tensor
    window_strides: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      the inter-window strides
    padding: A `Tensor`. Must have the same type as `window_strides`.
      the padding to apply at the start and end of each input dimensions
    lhs_dilation: A `Tensor`. Must have the same type as `window_strides`.
      dilation to apply between input elements
    rhs_dilation: A `Tensor`. Must have the same type as `window_strides`.
      dilation to apply between kernel elements
    feature_group_count: A `Tensor`. Must have the same type as `window_strides`.
      number of feature groups for grouped convolution.
    dimension_numbers: A `string`.
      a serialized xla::ConvolutionDimensionNumbers proto.
    precision_config: A `string`. a serialized xla::PrecisionConfig proto.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `lhs`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaConv", name, lhs, rhs, window_strides, padding,
        lhs_dilation, rhs_dilation, feature_group_count, "dimension_numbers",
        dimension_numbers, "precision_config", precision_config)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_conv(
          (lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation,
          feature_group_count, dimension_numbers, precision_config, name,),
          None)
      if _result is not NotImplemented:
        return _result
      return xla_conv_eager_fallback(
          lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation,
          feature_group_count, dimension_numbers=dimension_numbers,
          precision_config=precision_config, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_conv, (), dict(lhs=lhs, rhs=rhs,
                               window_strides=window_strides, padding=padding,
                               lhs_dilation=lhs_dilation,
                               rhs_dilation=rhs_dilation,
                               feature_group_count=feature_group_count,
                               dimension_numbers=dimension_numbers,
                               precision_config=precision_config, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_conv(
        (lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation,
        feature_group_count, dimension_numbers, precision_config, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  dimension_numbers = _execute.make_str(dimension_numbers, "dimension_numbers")
  precision_config = _execute.make_str(precision_config, "precision_config")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaConv", lhs=lhs, rhs=rhs, window_strides=window_strides,
                   padding=padding, lhs_dilation=lhs_dilation,
                   rhs_dilation=rhs_dilation,
                   feature_group_count=feature_group_count,
                   dimension_numbers=dimension_numbers,
                   precision_config=precision_config, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_conv, (), dict(lhs=lhs, rhs=rhs, window_strides=window_strides,
                             padding=padding, lhs_dilation=lhs_dilation,
                             rhs_dilation=rhs_dilation,
                             feature_group_count=feature_group_count,
                             dimension_numbers=dimension_numbers,
                             precision_config=precision_config, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"), "dimension_numbers",
              _op.get_attr("dimension_numbers"), "precision_config",
              _op.get_attr("precision_config"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaConv", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaConv = tf_export("raw_ops.XlaConv")(_ops.to_raw_op(xla_conv))
_dispatcher_for_xla_conv = xla_conv._tf_type_based_dispatcher.Dispatch


def xla_conv_eager_fallback(lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation, feature_group_count, dimension_numbers, precision_config, name, ctx):
  dimension_numbers = _execute.make_str(dimension_numbers, "dimension_numbers")
  precision_config = _execute.make_str(precision_config, "precision_config")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([lhs, rhs], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (lhs, rhs) = _inputs_T
  _attr_Tindices, _inputs_Tindices = _execute.args_to_matching_eager([window_strides, padding, lhs_dilation, rhs_dilation, feature_group_count], ctx, [_dtypes.int32, _dtypes.int64, ])
  (window_strides, padding, lhs_dilation, rhs_dilation, feature_group_count) = _inputs_Tindices
  _inputs_flat = [lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation, feature_group_count]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices, "dimension_numbers",
  dimension_numbers, "precision_config", precision_config)
  _result = _execute.execute(b"XlaConv", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaConv", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_conv_v2')
def xla_conv_v2(lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation, feature_group_count, dimension_numbers, precision_config, preferred_element_type, name=None):
  r"""Wraps the XLA ConvGeneralDilated operator, documented at

   https://www.tensorflow.org/performance/xla/operation_semantics#conv_convolution
  .

  Args:
    lhs: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      the input tensor
    rhs: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      the kernel tensor
    window_strides: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      the inter-window strides
    padding: A `Tensor`. Must have the same type as `window_strides`.
      the padding to apply at the start and end of each input dimensions
    lhs_dilation: A `Tensor`. Must have the same type as `window_strides`.
      dilation to apply between input elements
    rhs_dilation: A `Tensor`. Must have the same type as `window_strides`.
      dilation to apply between kernel elements
    feature_group_count: A `Tensor`. Must have the same type as `window_strides`.
      number of feature groups for grouped convolution.
    dimension_numbers: A `string`.
      a serialized xla::ConvolutionDimensionNumbers proto.
    precision_config: A `string`. a serialized xla::PrecisionConfig proto.
    preferred_element_type: A `tf.DType` from: `tf.float32, tf.float64, tf.int32, tf.uint8, tf.int16, tf.int8, tf.complex64, tf.int64, tf.qint8, tf.quint8, tf.qint32, tf.bfloat16, tf.uint16, tf.complex128, tf.half, tf.uint32, tf.uint64`.
      The type of the tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `preferred_element_type`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaConvV2", name, lhs, rhs, window_strides, padding,
        lhs_dilation, rhs_dilation, feature_group_count, "dimension_numbers",
        dimension_numbers, "precision_config", precision_config,
        "preferred_element_type", preferred_element_type)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_conv_v2(
          (lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation,
          feature_group_count, dimension_numbers, precision_config,
          preferred_element_type, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_conv_v2_eager_fallback(
          lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation,
          feature_group_count, dimension_numbers=dimension_numbers,
          precision_config=precision_config,
          preferred_element_type=preferred_element_type, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_conv_v2, (), dict(lhs=lhs, rhs=rhs,
                                  window_strides=window_strides,
                                  padding=padding, lhs_dilation=lhs_dilation,
                                  rhs_dilation=rhs_dilation,
                                  feature_group_count=feature_group_count,
                                  dimension_numbers=dimension_numbers,
                                  precision_config=precision_config,
                                  preferred_element_type=preferred_element_type,
                                  name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_conv_v2(
        (lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation,
        feature_group_count, dimension_numbers, precision_config,
        preferred_element_type, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  dimension_numbers = _execute.make_str(dimension_numbers, "dimension_numbers")
  precision_config = _execute.make_str(precision_config, "precision_config")
  preferred_element_type = _execute.make_type(preferred_element_type, "preferred_element_type")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaConvV2", lhs=lhs, rhs=rhs, window_strides=window_strides,
                     padding=padding, lhs_dilation=lhs_dilation,
                     rhs_dilation=rhs_dilation,
                     feature_group_count=feature_group_count,
                     dimension_numbers=dimension_numbers,
                     precision_config=precision_config,
                     preferred_element_type=preferred_element_type, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_conv_v2, (), dict(lhs=lhs, rhs=rhs,
                                window_strides=window_strides,
                                padding=padding, lhs_dilation=lhs_dilation,
                                rhs_dilation=rhs_dilation,
                                feature_group_count=feature_group_count,
                                dimension_numbers=dimension_numbers,
                                precision_config=precision_config,
                                preferred_element_type=preferred_element_type,
                                name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("LhsT", _op._get_attr_type("LhsT"), "RhsT",
              _op._get_attr_type("RhsT"), "Tindices",
              _op._get_attr_type("Tindices"), "dimension_numbers",
              _op.get_attr("dimension_numbers"), "precision_config",
              _op.get_attr("precision_config"), "preferred_element_type",
              _op._get_attr_type("preferred_element_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaConvV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaConvV2 = tf_export("raw_ops.XlaConvV2")(_ops.to_raw_op(xla_conv_v2))
_dispatcher_for_xla_conv_v2 = xla_conv_v2._tf_type_based_dispatcher.Dispatch


def xla_conv_v2_eager_fallback(lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation, feature_group_count, dimension_numbers, precision_config, preferred_element_type, name, ctx):
  dimension_numbers = _execute.make_str(dimension_numbers, "dimension_numbers")
  precision_config = _execute.make_str(precision_config, "precision_config")
  preferred_element_type = _execute.make_type(preferred_element_type, "preferred_element_type")
  _attr_LhsT, (lhs,) = _execute.args_to_matching_eager([lhs], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  _attr_RhsT, (rhs,) = _execute.args_to_matching_eager([rhs], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  _attr_Tindices, _inputs_Tindices = _execute.args_to_matching_eager([window_strides, padding, lhs_dilation, rhs_dilation, feature_group_count], ctx, [_dtypes.int32, _dtypes.int64, ])
  (window_strides, padding, lhs_dilation, rhs_dilation, feature_group_count) = _inputs_Tindices
  _inputs_flat = [lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation, feature_group_count]
  _attrs = ("LhsT", _attr_LhsT, "RhsT", _attr_RhsT, "Tindices",
  _attr_Tindices, "dimension_numbers", dimension_numbers, "precision_config",
  precision_config, "preferred_element_type", preferred_element_type)
  _result = _execute.execute(b"XlaConvV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaConvV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_dequantize')
def xla_dequantize(input, min_range, max_range, mode, transpose_output, name=None):
  r"""Takes the packed uint32 input and unpacks the input to uint8 to do

  Dequantization on device.

  Args:
    input: A `Tensor` of type `uint32`.
      Input tensors whose types is uint32, shape is [d0, ..., dn].
    min_range: A `float`.
      The minimum scalar value possibly produced for the input.
    max_range: A `float`.
      The maximum scalar value possibly produced for the input.
    mode: A `string`.
      String to determine the dequantize mode in {"MIN_COMBINED", "MIN_FIRST", "SCALED"}.
    transpose_output: A `bool`.
      Boolean to determine if output is transposed. transpose_output
      is faster when input is large and rank of input is higher than 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bfloat16`.
    Output tensors whose types is bloat16. If transpose_output is true,
    output shape is [dn * 4, dn-1, ..., d1, d0]. If transpose_output
    is false, output shape is [d0,..., dn * 4].
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaDequantize", name, input, "min_range", min_range,
        "max_range", max_range, "mode", mode, "transpose_output",
        transpose_output)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_dequantize(
          (input, min_range, max_range, mode, transpose_output, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_dequantize_eager_fallback(
          input, min_range=min_range, max_range=max_range, mode=mode,
          transpose_output=transpose_output, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_dequantize, (), dict(input=input, min_range=min_range,
                                     max_range=max_range, mode=mode,
                                     transpose_output=transpose_output,
                                     name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_dequantize(
        (input, min_range, max_range, mode, transpose_output, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  min_range = _execute.make_float(min_range, "min_range")
  max_range = _execute.make_float(max_range, "max_range")
  mode = _execute.make_str(mode, "mode")
  transpose_output = _execute.make_bool(transpose_output, "transpose_output")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaDequantize", input=input, min_range=min_range,
                         max_range=max_range, mode=mode,
                         transpose_output=transpose_output, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_dequantize, (), dict(input=input, min_range=min_range,
                                   max_range=max_range, mode=mode,
                                   transpose_output=transpose_output,
                                   name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("min_range", _op.get_attr("min_range"), "max_range",
              _op.get_attr("max_range"), "mode", _op.get_attr("mode"),
              "transpose_output", _op._get_attr_bool("transpose_output"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaDequantize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaDequantize = tf_export("raw_ops.XlaDequantize")(_ops.to_raw_op(xla_dequantize))
_dispatcher_for_xla_dequantize = xla_dequantize._tf_type_based_dispatcher.Dispatch


def xla_dequantize_eager_fallback(input, min_range, max_range, mode, transpose_output, name, ctx):
  min_range = _execute.make_float(min_range, "min_range")
  max_range = _execute.make_float(max_range, "max_range")
  mode = _execute.make_str(mode, "mode")
  transpose_output = _execute.make_bool(transpose_output, "transpose_output")
  input = _ops.convert_to_tensor(input, _dtypes.uint32)
  _inputs_flat = [input]
  _attrs = ("min_range", min_range, "max_range", max_range, "mode", mode,
  "transpose_output", transpose_output)
  _result = _execute.execute(b"XlaDequantize", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaDequantize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_dot')
def xla_dot(lhs, rhs, dimension_numbers, precision_config, name=None):
  r"""Wraps the XLA DotGeneral operator, documented at

   https://www.tensorflow.org/performance/xla/operation_semantics#dotgeneral
  .

  Args:
    lhs: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      the LHS tensor
    rhs: A `Tensor`. Must have the same type as `lhs`. the RHS tensor
    dimension_numbers: A `string`.
      a serialized xla::DotDimensionNumbers proto.
    precision_config: A `string`. a serialized xla::PrecisionConfig proto.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `lhs`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaDot", name, lhs, rhs, "dimension_numbers",
        dimension_numbers, "precision_config", precision_config)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_dot(
          (lhs, rhs, dimension_numbers, precision_config, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_dot_eager_fallback(
          lhs, rhs, dimension_numbers=dimension_numbers,
          precision_config=precision_config, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_dot, (), dict(lhs=lhs, rhs=rhs,
                              dimension_numbers=dimension_numbers,
                              precision_config=precision_config, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_dot(
        (lhs, rhs, dimension_numbers, precision_config, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  dimension_numbers = _execute.make_str(dimension_numbers, "dimension_numbers")
  precision_config = _execute.make_str(precision_config, "precision_config")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaDot", lhs=lhs, rhs=rhs, dimension_numbers=dimension_numbers,
                  precision_config=precision_config, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_dot, (), dict(lhs=lhs, rhs=rhs,
                            dimension_numbers=dimension_numbers,
                            precision_config=precision_config, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "dimension_numbers",
              _op.get_attr("dimension_numbers"), "precision_config",
              _op.get_attr("precision_config"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaDot", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaDot = tf_export("raw_ops.XlaDot")(_ops.to_raw_op(xla_dot))
_dispatcher_for_xla_dot = xla_dot._tf_type_based_dispatcher.Dispatch


def xla_dot_eager_fallback(lhs, rhs, dimension_numbers, precision_config, name, ctx):
  dimension_numbers = _execute.make_str(dimension_numbers, "dimension_numbers")
  precision_config = _execute.make_str(precision_config, "precision_config")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([lhs, rhs], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (lhs, rhs) = _inputs_T
  _inputs_flat = [lhs, rhs]
  _attrs = ("T", _attr_T, "dimension_numbers", dimension_numbers,
  "precision_config", precision_config)
  _result = _execute.execute(b"XlaDot", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaDot", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_dot_v2')
def xla_dot_v2(lhs, rhs, dimension_numbers, precision_config, preferred_element_type, name=None):
  r"""Wraps the XLA DotGeneral operator, documented at

   https://www.tensorflow.org/performance/xla/operation_semantics#dotgeneral
  .

  Args:
    lhs: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      the LHS tensor
    rhs: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      the RHS tensor
    dimension_numbers: A `string`.
      a serialized xla::DotDimensionNumbers proto.
    precision_config: A `string`. a serialized xla::PrecisionConfig proto.
    preferred_element_type: A `tf.DType` from: `tf.float32, tf.float64, tf.int32, tf.uint8, tf.int16, tf.int8, tf.complex64, tf.int64, tf.qint8, tf.quint8, tf.qint32, tf.bfloat16, tf.uint16, tf.complex128, tf.half, tf.uint32, tf.uint64`.
      The type of the tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `preferred_element_type`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaDotV2", name, lhs, rhs, "dimension_numbers",
        dimension_numbers, "precision_config", precision_config,
        "preferred_element_type", preferred_element_type)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_dot_v2(
          (lhs, rhs, dimension_numbers, precision_config,
          preferred_element_type, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_dot_v2_eager_fallback(
          lhs, rhs, dimension_numbers=dimension_numbers,
          precision_config=precision_config,
          preferred_element_type=preferred_element_type, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_dot_v2, (), dict(lhs=lhs, rhs=rhs,
                                 dimension_numbers=dimension_numbers,
                                 precision_config=precision_config,
                                 preferred_element_type=preferred_element_type,
                                 name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_dot_v2(
        (lhs, rhs, dimension_numbers, precision_config,
        preferred_element_type, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  dimension_numbers = _execute.make_str(dimension_numbers, "dimension_numbers")
  precision_config = _execute.make_str(precision_config, "precision_config")
  preferred_element_type = _execute.make_type(preferred_element_type, "preferred_element_type")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaDotV2", lhs=lhs, rhs=rhs, dimension_numbers=dimension_numbers,
                    precision_config=precision_config,
                    preferred_element_type=preferred_element_type, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_dot_v2, (), dict(lhs=lhs, rhs=rhs,
                               dimension_numbers=dimension_numbers,
                               precision_config=precision_config,
                               preferred_element_type=preferred_element_type,
                               name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("LhsT", _op._get_attr_type("LhsT"), "RhsT",
              _op._get_attr_type("RhsT"), "dimension_numbers",
              _op.get_attr("dimension_numbers"), "precision_config",
              _op.get_attr("precision_config"), "preferred_element_type",
              _op._get_attr_type("preferred_element_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaDotV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaDotV2 = tf_export("raw_ops.XlaDotV2")(_ops.to_raw_op(xla_dot_v2))
_dispatcher_for_xla_dot_v2 = xla_dot_v2._tf_type_based_dispatcher.Dispatch


def xla_dot_v2_eager_fallback(lhs, rhs, dimension_numbers, precision_config, preferred_element_type, name, ctx):
  dimension_numbers = _execute.make_str(dimension_numbers, "dimension_numbers")
  precision_config = _execute.make_str(precision_config, "precision_config")
  preferred_element_type = _execute.make_type(preferred_element_type, "preferred_element_type")
  _attr_LhsT, (lhs,) = _execute.args_to_matching_eager([lhs], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  _attr_RhsT, (rhs,) = _execute.args_to_matching_eager([rhs], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  _inputs_flat = [lhs, rhs]
  _attrs = ("LhsT", _attr_LhsT, "RhsT", _attr_RhsT, "dimension_numbers",
  dimension_numbers, "precision_config", precision_config,
  "preferred_element_type", preferred_element_type)
  _result = _execute.execute(b"XlaDotV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaDotV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_dynamic_slice')
def xla_dynamic_slice(input, start_indices, size_indices, name=None):
  r"""Wraps the XLA DynamicSlice operator, documented at

   https://www.tensorflow.org/performance/xla/operation_semantics#dynamicslice
  .

  DynamicSlice extracts a sub-array from the input array at dynamic
  start_indices. The size of the slice in each dimension is passed in
  size_indices, which specify the end point of exclusive slice intervals in each
  dimension -- [start, start + size). The shape of start_indices must have rank 1,
  with dimension size equal to the rank of operand.

  Args:
    input: A `Tensor`. A `Tensor` of type T.
    start_indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      List of N integers containing the slice size for each
      dimension. Each value must be strictly greater than zero, and start + size
      must be less than or equal to the size of the dimension to avoid
      implementation defined behavior.
    size_indices: A `Tensor`. Must have the same type as `start_indices`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaDynamicSlice", name, input, start_indices, size_indices)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_dynamic_slice(
          (input, start_indices, size_indices, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_dynamic_slice_eager_fallback(
          input, start_indices, size_indices, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_dynamic_slice, (), dict(input=input,
                                        start_indices=start_indices,
                                        size_indices=size_indices, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_dynamic_slice(
        (input, start_indices, size_indices, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaDynamicSlice", input=input, start_indices=start_indices,
                           size_indices=size_indices, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_dynamic_slice, (), dict(input=input,
                                      start_indices=start_indices,
                                      size_indices=size_indices, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaDynamicSlice", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaDynamicSlice = tf_export("raw_ops.XlaDynamicSlice")(_ops.to_raw_op(xla_dynamic_slice))
_dispatcher_for_xla_dynamic_slice = xla_dynamic_slice._tf_type_based_dispatcher.Dispatch


def xla_dynamic_slice_eager_fallback(input, start_indices, size_indices, name, ctx):
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _attr_Tindices, _inputs_Tindices = _execute.args_to_matching_eager([start_indices, size_indices], ctx, [_dtypes.int32, _dtypes.int64, ])
  (start_indices, size_indices) = _inputs_Tindices
  _inputs_flat = [input, start_indices, size_indices]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices)
  _result = _execute.execute(b"XlaDynamicSlice", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaDynamicSlice", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_dynamic_update_slice')
def xla_dynamic_update_slice(input, update, indices, name=None):
  r"""Wraps the XLA DynamicUpdateSlice operator, documented at

   https://www.tensorflow.org/performance/xla/operation_semantics#dynamicupdateslice
  .

  XlaDynamicUpdateSlice generates a result which is the value of the `input`
  operand, with a slice update overwritten at `indices`. The shape of `update`
  determines the shape of the sub-array of the result which is updated. The shape
  of indices must be rank == 1, with dimension size equal to the rank of `input`.

  Handling of out-of-bounds slice indices is implementation-defined.

  Args:
    input: A `Tensor`. A `Tensor` of type T.
    update: A `Tensor`. Must have the same type as `input`.
      A `Tensor` of type T. Same rank as `input`.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into `input`. Must have length equal to the rank of
      `input`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. A `Tensor` of type T.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaDynamicUpdateSlice", name, input, update, indices)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_dynamic_update_slice(
          (input, update, indices, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_dynamic_update_slice_eager_fallback(
          input, update, indices, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_dynamic_update_slice, (), dict(input=input, update=update,
                                               indices=indices, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_dynamic_update_slice(
        (input, update, indices, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaDynamicUpdateSlice", input=input, update=update, indices=indices,
                                 name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_dynamic_update_slice, (), dict(input=input, update=update,
                                             indices=indices, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaDynamicUpdateSlice", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaDynamicUpdateSlice = tf_export("raw_ops.XlaDynamicUpdateSlice")(_ops.to_raw_op(xla_dynamic_update_slice))
_dispatcher_for_xla_dynamic_update_slice = xla_dynamic_update_slice._tf_type_based_dispatcher.Dispatch


def xla_dynamic_update_slice_eager_fallback(input, update, indices, name, ctx):
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input, update], ctx, [])
  (input, update) = _inputs_T
  _attr_Tindices, (indices,) = _execute.args_to_matching_eager([indices], ctx, [_dtypes.int32, _dtypes.int64, ])
  _inputs_flat = [input, update, indices]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices)
  _result = _execute.execute(b"XlaDynamicUpdateSlice", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaDynamicUpdateSlice", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_einsum')
def xla_einsum(a, b, equation, name=None):
  r"""An op which supports basic einsum op with 2 inputs and 1 output.

  This op has better TPU performance since it doesn't have explicitly reshape and
  transpose operations as tf.einsum does.

  Args:
    a: A `Tensor`. Must be one of the following types: `complex64`, `bfloat16`, `float32`.
    b: A `Tensor`. Must have the same type as `a`.
    equation: A `string`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaEinsum", name, a, b, "equation", equation)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_einsum(
          (a, b, equation, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_einsum_eager_fallback(
          a, b, equation=equation, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_einsum, (), dict(a=a, b=b, equation=equation, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_einsum(
        (a, b, equation, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  equation = _execute.make_str(equation, "equation")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaEinsum", a=a, b=b, equation=equation, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_einsum, (), dict(a=a, b=b, equation=equation, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("equation", _op.get_attr("equation"), "T",
              _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaEinsum", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaEinsum = tf_export("raw_ops.XlaEinsum")(_ops.to_raw_op(xla_einsum))
_dispatcher_for_xla_einsum = xla_einsum._tf_type_based_dispatcher.Dispatch


def xla_einsum_eager_fallback(a, b, equation, name, ctx):
  equation = _execute.make_str(equation, "equation")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([a, b], ctx, [_dtypes.complex64, _dtypes.bfloat16, _dtypes.float32, ])
  (a, b) = _inputs_T
  _inputs_flat = [a, b]
  _attrs = ("equation", equation, "T", _attr_T)
  _result = _execute.execute(b"XlaEinsum", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaEinsum", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_gather')
def xla_gather(operand, start_indices, slice_sizes, dimension_numbers, indices_are_sorted, name=None):
  r"""Wraps the XLA Gather operator documented at

    https://www.tensorflow.org/xla/operation_semantics#gather

  Args:
    operand: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`, `bool`.
      The array we're gathering from.
    start_indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Array containing the starting indices of the slices we gather.
    slice_sizes: A `Tensor`. Must have the same type as `start_indices`.
      slice_sizes[i] is the bounds for the slice on dimension i.
    dimension_numbers: A `string`.
      A serialized xla::GatherDimensionNumbers proto.
    indices_are_sorted: A `bool`.
      Boolean indicating if the indices are sorted.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `operand`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaGather", name, operand, start_indices, slice_sizes,
        "dimension_numbers", dimension_numbers, "indices_are_sorted",
        indices_are_sorted)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_gather(
          (operand, start_indices, slice_sizes, dimension_numbers,
          indices_are_sorted, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_gather_eager_fallback(
          operand, start_indices, slice_sizes,
          dimension_numbers=dimension_numbers,
          indices_are_sorted=indices_are_sorted, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_gather, (), dict(operand=operand, start_indices=start_indices,
                                 slice_sizes=slice_sizes,
                                 dimension_numbers=dimension_numbers,
                                 indices_are_sorted=indices_are_sorted,
                                 name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_gather(
        (operand, start_indices, slice_sizes, dimension_numbers,
        indices_are_sorted, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  dimension_numbers = _execute.make_str(dimension_numbers, "dimension_numbers")
  indices_are_sorted = _execute.make_bool(indices_are_sorted, "indices_are_sorted")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaGather", operand=operand, start_indices=start_indices,
                     slice_sizes=slice_sizes,
                     dimension_numbers=dimension_numbers,
                     indices_are_sorted=indices_are_sorted, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_gather, (), dict(operand=operand, start_indices=start_indices,
                               slice_sizes=slice_sizes,
                               dimension_numbers=dimension_numbers,
                               indices_are_sorted=indices_are_sorted,
                               name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dimension_numbers", _op.get_attr("dimension_numbers"),
              "indices_are_sorted", _op._get_attr_bool("indices_are_sorted"),
              "T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaGather", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaGather = tf_export("raw_ops.XlaGather")(_ops.to_raw_op(xla_gather))
_dispatcher_for_xla_gather = xla_gather._tf_type_based_dispatcher.Dispatch


def xla_gather_eager_fallback(operand, start_indices, slice_sizes, dimension_numbers, indices_are_sorted, name, ctx):
  dimension_numbers = _execute.make_str(dimension_numbers, "dimension_numbers")
  indices_are_sorted = _execute.make_bool(indices_are_sorted, "indices_are_sorted")
  _attr_T, (operand,) = _execute.args_to_matching_eager([operand], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, _dtypes.bool, ])
  _attr_Tindices, _inputs_Tindices = _execute.args_to_matching_eager([start_indices, slice_sizes], ctx, [_dtypes.int32, _dtypes.int64, ])
  (start_indices, slice_sizes) = _inputs_Tindices
  _inputs_flat = [operand, start_indices, slice_sizes]
  _attrs = ("dimension_numbers", dimension_numbers, "indices_are_sorted",
  indices_are_sorted, "T", _attr_T, "Tindices", _attr_Tindices)
  _result = _execute.execute(b"XlaGather", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaGather", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_if')
def xla_if(cond, inputs, then_branch, else_branch, Tout, name=None):
  r"""output = cond ? then_branch(inputs) : else_branch(inputs).

  Args:
    cond: A `Tensor`. A boolean scalar.
    inputs: A list of `Tensor` objects. A list of input tensors.
    then_branch: A function decorated with @Defun.
      A function takes 'inputs' and returns a list of tensors,
      whose types are the same as what else_branch returns.
    else_branch: A function decorated with @Defun.
      A function takes 'inputs' and returns a list of tensors.
      whose types are the same as what then_branch returns.
    Tout: A list of `tf.DTypes`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `Tout`.
    A list of tensors returned by either then_branch(inputs) or
    else_branch(inputs). The input shapes of the then_branch and
    else_branch must match.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaIf", name, cond, inputs, "then_branch", then_branch,
        "else_branch", else_branch, "Tout", Tout)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_if(
          (cond, inputs, then_branch, else_branch, Tout, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_if_eager_fallback(
          cond, inputs, then_branch=then_branch, else_branch=else_branch,
          Tout=Tout, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_if, (), dict(cond=cond, inputs=inputs,
                             then_branch=then_branch, else_branch=else_branch,
                             Tout=Tout, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_if(
        (cond, inputs, then_branch, else_branch, Tout, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(Tout, (list, tuple)):
    raise TypeError(
        "Expected list for 'Tout' argument to "
        "'xla_if' Op, not %r." % Tout)
  Tout = [_execute.make_type(_t, "Tout") for _t in Tout]
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaIf", cond=cond, inputs=inputs, then_branch=then_branch,
                 else_branch=else_branch, Tout=Tout, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_if, (), dict(cond=cond, inputs=inputs, then_branch=then_branch,
                           else_branch=else_branch, Tout=Tout, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if not _result:
    return _op
  if _execute.must_record_gradient():
    _attrs = ("Tcond", _op._get_attr_type("Tcond"), "then_branch",
              _op.get_attr("then_branch"), "else_branch",
              _op.get_attr("else_branch"), "Tin", _op.get_attr("Tin"), "Tout",
              _op.get_attr("Tout"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaIf", _inputs_flat, _attrs, _result)
  return _result

XlaIf = tf_export("raw_ops.XlaIf")(_ops.to_raw_op(xla_if))
_dispatcher_for_xla_if = xla_if._tf_type_based_dispatcher.Dispatch


def xla_if_eager_fallback(cond, inputs, then_branch, else_branch, Tout, name, ctx):
  if not isinstance(Tout, (list, tuple)):
    raise TypeError(
        "Expected list for 'Tout' argument to "
        "'xla_if' Op, not %r." % Tout)
  Tout = [_execute.make_type(_t, "Tout") for _t in Tout]
  _attr_Tcond, (cond,) = _execute.args_to_matching_eager([cond], ctx, [])
  _attr_Tin, inputs = _execute.convert_to_mixed_eager_tensors(inputs, ctx)
  _inputs_flat = [cond] + list(inputs)
  _attrs = ("Tcond", _attr_Tcond, "then_branch", then_branch, "else_branch",
  else_branch, "Tin", _attr_Tin, "Tout", Tout)
  _result = _execute.execute(b"XlaIf", len(Tout), inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaIf", _inputs_flat, _attrs, _result)
  return _result

_XlaKeyValueSortOutput = collections.namedtuple(
    "XlaKeyValueSort",
    ["sorted_keys", "sorted_values"])


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_key_value_sort')
def xla_key_value_sort(keys, values, name=None):
  r"""Wraps the XLA Sort operator, documented at

   https://www.tensorflow.org/performance/xla/operation_semantics#sort
  .

  Sorts a tensor. Currently only sorts in ascending order are supported.

  Args:
    keys: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      A `Tensor` of type K.
    values: A `Tensor`. A `Tensor` of type V.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (sorted_keys, sorted_values).

    sorted_keys: A `Tensor`. Has the same type as `keys`. A `Tensor` of type K.
    sorted_values: A `Tensor`. Has the same type as `values`. A `Tensor` of type V.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaKeyValueSort", name, keys, values)
      _result = _XlaKeyValueSortOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_key_value_sort(
          (keys, values, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_key_value_sort_eager_fallback(
          keys, values, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_key_value_sort, (), dict(keys=keys, values=values, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_key_value_sort(
        (keys, values, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaKeyValueSort", keys=keys, values=values, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_key_value_sort, (), dict(keys=keys, values=values, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("K", _op._get_attr_type("K"), "V", _op._get_attr_type("V"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaKeyValueSort", _inputs_flat, _attrs, _result)
  _result = _XlaKeyValueSortOutput._make(_result)
  return _result

XlaKeyValueSort = tf_export("raw_ops.XlaKeyValueSort")(_ops.to_raw_op(xla_key_value_sort))
_dispatcher_for_xla_key_value_sort = xla_key_value_sort._tf_type_based_dispatcher.Dispatch


def xla_key_value_sort_eager_fallback(keys, values, name, ctx):
  _attr_K, (keys,) = _execute.args_to_matching_eager([keys], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  _attr_V, (values,) = _execute.args_to_matching_eager([values], ctx, [])
  _inputs_flat = [keys, values]
  _attrs = ("K", _attr_K, "V", _attr_V)
  _result = _execute.execute(b"XlaKeyValueSort", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaKeyValueSort", _inputs_flat, _attrs, _result)
  _result = _XlaKeyValueSortOutput._make(_result)
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_pad')
def xla_pad(input, padding_value, padding_low, padding_high, padding_interior, name=None):
  r"""Wraps the XLA Pad operator, documented at

   https://www.tensorflow.org/performance/xla/operation_semantics#pad
  .

  Args:
    input: A `Tensor`. A `Tensor` of type T.
    padding_value: A `Tensor`. Must have the same type as `input`.
      A scalar `Tensor` of type T.
    padding_low: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      the padding to apply at the start of each input dimensions. Must
      be a compile-time constant 1D tensor of length equal to rank of input.
    padding_high: A `Tensor`. Must have the same type as `padding_low`.
      the padding to apply at the end of each input dimension. Must
      be a compile-time constant 1D tensor of length equal to rank of input.
    padding_interior: A `Tensor`. Must have the same type as `padding_low`.
      the padding to apply between each input element. Must
      be a compile-time constant 1D tensor of length equal to rank of input,
      containing only non-negative values.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. A `Tensor` of type T.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaPad", name, input, padding_value, padding_low, padding_high,
        padding_interior)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_pad(
          (input, padding_value, padding_low, padding_high, padding_interior,
          name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_pad_eager_fallback(
          input, padding_value, padding_low, padding_high, padding_interior,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_pad, (), dict(input=input, padding_value=padding_value,
                              padding_low=padding_low,
                              padding_high=padding_high,
                              padding_interior=padding_interior, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_pad(
        (input, padding_value, padding_low, padding_high, padding_interior,
        name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaPad", input=input, padding_value=padding_value,
                  padding_low=padding_low, padding_high=padding_high,
                  padding_interior=padding_interior, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_pad, (), dict(input=input, padding_value=padding_value,
                            padding_low=padding_low,
                            padding_high=padding_high,
                            padding_interior=padding_interior, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaPad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaPad = tf_export("raw_ops.XlaPad")(_ops.to_raw_op(xla_pad))
_dispatcher_for_xla_pad = xla_pad._tf_type_based_dispatcher.Dispatch


def xla_pad_eager_fallback(input, padding_value, padding_low, padding_high, padding_interior, name, ctx):
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input, padding_value], ctx, [])
  (input, padding_value) = _inputs_T
  _attr_Tindices, _inputs_Tindices = _execute.args_to_matching_eager([padding_low, padding_high, padding_interior], ctx, [_dtypes.int32, _dtypes.int64, ])
  (padding_low, padding_high, padding_interior) = _inputs_Tindices
  _inputs_flat = [input, padding_value, padding_low, padding_high, padding_interior]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices)
  _result = _execute.execute(b"XlaPad", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaPad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_recv')
def xla_recv(dtype, tensor_name, shape, name=None):
  r"""Receives the named tensor from another XLA computation. Wraps the XLA Recv

  operator documented at
   https://www.tensorflow.org/performance/xla/operation_semantics#recv .

  Args:
    dtype: A `tf.DType`. The type of the tensor.
    tensor_name: A `string`. A string key that identifies the channel.
    shape: A `tf.TensorShape` or list of `ints`. The shape of the tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`. The tensor to receive.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaRecv", name, "dtype", dtype, "tensor_name", tensor_name,
        "shape", shape)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_recv(
          (dtype, tensor_name, shape, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_recv_eager_fallback(
          dtype=dtype, tensor_name=tensor_name, shape=shape, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_recv, (), dict(dtype=dtype, tensor_name=tensor_name,
                               shape=shape, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_recv(
        (dtype, tensor_name, shape, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  dtype = _execute.make_type(dtype, "dtype")
  tensor_name = _execute.make_str(tensor_name, "tensor_name")
  shape = _execute.make_shape(shape, "shape")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaRecv", dtype=dtype, tensor_name=tensor_name, shape=shape,
                   name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_recv, (), dict(dtype=dtype, tensor_name=tensor_name,
                             shape=shape, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "tensor_name",
              _op.get_attr("tensor_name"), "shape", _op.get_attr("shape"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaRecv", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaRecv = tf_export("raw_ops.XlaRecv")(_ops.to_raw_op(xla_recv))
_dispatcher_for_xla_recv = xla_recv._tf_type_based_dispatcher.Dispatch


def xla_recv_eager_fallback(dtype, tensor_name, shape, name, ctx):
  dtype = _execute.make_type(dtype, "dtype")
  tensor_name = _execute.make_str(tensor_name, "tensor_name")
  shape = _execute.make_shape(shape, "shape")
  _inputs_flat = []
  _attrs = ("dtype", dtype, "tensor_name", tensor_name, "shape", shape)
  _result = _execute.execute(b"XlaRecv", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaRecv", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_reduce')
def xla_reduce(input, init_value, dimensions_to_reduce, reducer, name=None):
  r"""Wraps the XLA Reduce operator, documented at

   https://www.tensorflow.org/performance/xla/operation_semantics#reduce .

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`, `bool`.
      the input tensor
    init_value: A `Tensor`. Must have the same type as `input`.
      a scalar representing the initial value for the reduction
    dimensions_to_reduce: A list of `ints`.
      dimension numbers over which to reduce
    reducer: A function decorated with @Defun. a reducer function to apply
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaReduce", name, input, init_value, "dimensions_to_reduce",
        dimensions_to_reduce, "reducer", reducer)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_reduce(
          (input, init_value, dimensions_to_reduce, reducer, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_reduce_eager_fallback(
          input, init_value, dimensions_to_reduce=dimensions_to_reduce,
          reducer=reducer, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_reduce, (), dict(input=input, init_value=init_value,
                                 dimensions_to_reduce=dimensions_to_reduce,
                                 reducer=reducer, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_reduce(
        (input, init_value, dimensions_to_reduce, reducer, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(dimensions_to_reduce, (list, tuple)):
    raise TypeError(
        "Expected list for 'dimensions_to_reduce' argument to "
        "'xla_reduce' Op, not %r." % dimensions_to_reduce)
  dimensions_to_reduce = [_execute.make_int(_i, "dimensions_to_reduce") for _i in dimensions_to_reduce]
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaReduce", input=input, init_value=init_value,
                     dimensions_to_reduce=dimensions_to_reduce,
                     reducer=reducer, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_reduce, (), dict(input=input, init_value=init_value,
                               dimensions_to_reduce=dimensions_to_reduce,
                               reducer=reducer, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "dimensions_to_reduce",
              _op.get_attr("dimensions_to_reduce"), "reducer",
              _op.get_attr("reducer"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaReduce", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaReduce = tf_export("raw_ops.XlaReduce")(_ops.to_raw_op(xla_reduce))
_dispatcher_for_xla_reduce = xla_reduce._tf_type_based_dispatcher.Dispatch


def xla_reduce_eager_fallback(input, init_value, dimensions_to_reduce, reducer, name, ctx):
  if not isinstance(dimensions_to_reduce, (list, tuple)):
    raise TypeError(
        "Expected list for 'dimensions_to_reduce' argument to "
        "'xla_reduce' Op, not %r." % dimensions_to_reduce)
  dimensions_to_reduce = [_execute.make_int(_i, "dimensions_to_reduce") for _i in dimensions_to_reduce]
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input, init_value], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, _dtypes.bool, ])
  (input, init_value) = _inputs_T
  _inputs_flat = [input, init_value]
  _attrs = ("T", _attr_T, "dimensions_to_reduce", dimensions_to_reduce,
  "reducer", reducer)
  _result = _execute.execute(b"XlaReduce", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaReduce", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_reduce_scatter')
def xla_reduce_scatter(input, group_assignment, scatter_dimension, reduce_op, name=None):
  r"""Wraps the XLA ReduceScatter operator

    documented at https://www.tensorflow.org/xla/operation_semantics#reducescatter.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `int32`, `uint32`.
      Array or a non-empty tuple of arrays to reduce across replicas.
    group_assignment: A `Tensor` of type `int32`.
      Groups between which the reductions are performed.
    scatter_dimension: A `Tensor` of type `int32`. Dimension to scatter.
    reduce_op: A `string` from: `"Min", "Max", "Mul", "Add", "Mean"`.
      Reduction computation.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaReduceScatter", name, input, group_assignment,
        scatter_dimension, "reduce_op", reduce_op)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_reduce_scatter(
          (input, group_assignment, scatter_dimension, reduce_op, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_reduce_scatter_eager_fallback(
          input, group_assignment, scatter_dimension, reduce_op=reduce_op,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_reduce_scatter, (), dict(input=input,
                                         group_assignment=group_assignment,
                                         scatter_dimension=scatter_dimension,
                                         reduce_op=reduce_op, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_reduce_scatter(
        (input, group_assignment, scatter_dimension, reduce_op, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  reduce_op = _execute.make_str(reduce_op, "reduce_op")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaReduceScatter", input=input, group_assignment=group_assignment,
                            scatter_dimension=scatter_dimension,
                            reduce_op=reduce_op, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_reduce_scatter, (), dict(input=input,
                                       group_assignment=group_assignment,
                                       scatter_dimension=scatter_dimension,
                                       reduce_op=reduce_op, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "reduce_op",
              _op.get_attr("reduce_op"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaReduceScatter", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaReduceScatter = tf_export("raw_ops.XlaReduceScatter")(_ops.to_raw_op(xla_reduce_scatter))
_dispatcher_for_xla_reduce_scatter = xla_reduce_scatter._tf_type_based_dispatcher.Dispatch


def xla_reduce_scatter_eager_fallback(input, group_assignment, scatter_dimension, reduce_op, name, ctx):
  reduce_op = _execute.make_str(reduce_op, "reduce_op")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.int32, _dtypes.uint32, ])
  group_assignment = _ops.convert_to_tensor(group_assignment, _dtypes.int32)
  scatter_dimension = _ops.convert_to_tensor(scatter_dimension, _dtypes.int32)
  _inputs_flat = [input, group_assignment, scatter_dimension]
  _attrs = ("T", _attr_T, "reduce_op", reduce_op)
  _result = _execute.execute(b"XlaReduceScatter", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaReduceScatter", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_reduce_window')
def xla_reduce_window(input, init_value, window_dimensions, window_strides, base_dilations, window_dilations, padding, computation, name=None):
  r"""Wraps the XLA ReduceWindow operator, documented at

   https://www.tensorflow.org/performance/xla/operation_semantics#reducewindow .

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`, `bool`.
      the input tensor
    init_value: A `Tensor`. Must have the same type as `input`.
      a scalar representing the initial value for the reduction
    window_dimensions: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      the shape of the window
    window_strides: A `Tensor`. Must have the same type as `window_dimensions`.
      the inter-window strides
    base_dilations: A `Tensor`. Must have the same type as `window_dimensions`.
    window_dilations: A `Tensor`. Must have the same type as `window_dimensions`.
    padding: A `Tensor`. Must have the same type as `window_dimensions`.
      the padding to apply at the start and end of each input dimensions
    computation: A function decorated with @Defun. a reducer function to apply
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaReduceWindow", name, input, init_value, window_dimensions,
        window_strides, base_dilations, window_dilations, padding,
        "computation", computation)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_reduce_window(
          (input, init_value, window_dimensions, window_strides,
          base_dilations, window_dilations, padding, computation, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_reduce_window_eager_fallback(
          input, init_value, window_dimensions, window_strides,
          base_dilations, window_dilations, padding, computation=computation,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_reduce_window, (), dict(input=input, init_value=init_value,
                                        window_dimensions=window_dimensions,
                                        window_strides=window_strides,
                                        base_dilations=base_dilations,
                                        window_dilations=window_dilations,
                                        padding=padding,
                                        computation=computation, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_reduce_window(
        (input, init_value, window_dimensions, window_strides, base_dilations,
        window_dilations, padding, computation, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaReduceWindow", input=input, init_value=init_value,
                           window_dimensions=window_dimensions,
                           window_strides=window_strides,
                           base_dilations=base_dilations,
                           window_dilations=window_dilations, padding=padding,
                           computation=computation, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_reduce_window, (), dict(input=input, init_value=init_value,
                                      window_dimensions=window_dimensions,
                                      window_strides=window_strides,
                                      base_dilations=base_dilations,
                                      window_dilations=window_dilations,
                                      padding=padding,
                                      computation=computation, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"), "computation",
              _op.get_attr("computation"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaReduceWindow", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaReduceWindow = tf_export("raw_ops.XlaReduceWindow")(_ops.to_raw_op(xla_reduce_window))
_dispatcher_for_xla_reduce_window = xla_reduce_window._tf_type_based_dispatcher.Dispatch


def xla_reduce_window_eager_fallback(input, init_value, window_dimensions, window_strides, base_dilations, window_dilations, padding, computation, name, ctx):
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input, init_value], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, _dtypes.bool, ])
  (input, init_value) = _inputs_T
  _attr_Tindices, _inputs_Tindices = _execute.args_to_matching_eager([window_dimensions, window_strides, base_dilations, window_dilations, padding], ctx, [_dtypes.int32, _dtypes.int64, ])
  (window_dimensions, window_strides, base_dilations, window_dilations, padding) = _inputs_Tindices
  _inputs_flat = [input, init_value, window_dimensions, window_strides, base_dilations, window_dilations, padding]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices, "computation",
  computation)
  _result = _execute.execute(b"XlaReduceWindow", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaReduceWindow", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_remove_dynamic_dimension_size')
def xla_remove_dynamic_dimension_size(input, dim_index, name=None):
  r"""Inverse of XlaSetDynamicDimensionSize.

  Make an xla bounded dynamic dimension into a static dimension. The bound of the
  size of dimension `dim_index` becomes the static dimension size.

  Args:
    input: A `Tensor`.
    dim_index: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaRemoveDynamicDimensionSize", name, input, dim_index)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_remove_dynamic_dimension_size(
          (input, dim_index, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_remove_dynamic_dimension_size_eager_fallback(
          input, dim_index, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_remove_dynamic_dimension_size, (), dict(input=input,
                                                        dim_index=dim_index,
                                                        name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_remove_dynamic_dimension_size(
        (input, dim_index, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaRemoveDynamicDimensionSize", input=input, dim_index=dim_index,
                                         name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_remove_dynamic_dimension_size, (), dict(input=input,
                                                      dim_index=dim_index,
                                                      name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaRemoveDynamicDimensionSize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaRemoveDynamicDimensionSize = tf_export("raw_ops.XlaRemoveDynamicDimensionSize")(_ops.to_raw_op(xla_remove_dynamic_dimension_size))
_dispatcher_for_xla_remove_dynamic_dimension_size = xla_remove_dynamic_dimension_size._tf_type_based_dispatcher.Dispatch


def xla_remove_dynamic_dimension_size_eager_fallback(input, dim_index, name, ctx):
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  dim_index = _ops.convert_to_tensor(dim_index, _dtypes.int32)
  _inputs_flat = [input, dim_index]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"XlaRemoveDynamicDimensionSize", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaRemoveDynamicDimensionSize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_replica_id')
def xla_replica_id(name=None):
  r"""Replica ID.

  Args:
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaReplicaId", name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_replica_id(
          (name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_replica_id_eager_fallback(
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_replica_id, (), dict(name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_replica_id(
        (name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaReplicaId", name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_replica_id, (), dict(name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaReplicaId", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaReplicaId = tf_export("raw_ops.XlaReplicaId")(_ops.to_raw_op(xla_replica_id))
_dispatcher_for_xla_replica_id = xla_replica_id._tf_type_based_dispatcher.Dispatch


def xla_replica_id_eager_fallback(name, ctx):
  _inputs_flat = []
  _attrs = None
  _result = _execute.execute(b"XlaReplicaId", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaReplicaId", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_XlaRngBitGeneratorOutput = collections.namedtuple(
    "XlaRngBitGenerator",
    ["output_key", "output"])


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_rng_bit_generator')
def xla_rng_bit_generator(algorithm, initial_state, shape, dtype=_dtypes.uint64, name=None):
  r"""Stateless PRNG bit generator.

  Wraps the XLA RngBitGenerator operator, documented at
   https://www.tensorflow.org/performance/xla/operation_semantics#rngbitgenerator.

  Args:
    algorithm: A `Tensor` of type `int32`. The PRNG algorithm to use, one of
      tf.random.Algorithm.{PHILOX, THREEFRY, AUTO_SELECT}.
    initial_state: A `Tensor` of type `uint64`.
      Initial state for the PRNG algorithm. For THREEFRY, it should be
      a u64[2] and for PHILOX a u64[3].
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The output shape of the generated data.
    dtype: An optional `tf.DType` from: `tf.int32, tf.int64, tf.uint32, tf.uint64`. Defaults to `tf.uint64`.
      The type of the tensor.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_key, output).

    output_key: A `Tensor` of type `uint64`.
    output: A `Tensor` of type `dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaRngBitGenerator", name, algorithm, initial_state, shape,
        "dtype", dtype)
      _result = _XlaRngBitGeneratorOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_rng_bit_generator(
          (algorithm, initial_state, shape, dtype, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_rng_bit_generator_eager_fallback(
          algorithm, initial_state, shape, dtype=dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_rng_bit_generator, (), dict(algorithm=algorithm,
                                            initial_state=initial_state,
                                            shape=shape, dtype=dtype,
                                            name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_rng_bit_generator(
        (algorithm, initial_state, shape, dtype, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if dtype is None:
    dtype = _dtypes.uint64
  dtype = _execute.make_type(dtype, "dtype")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaRngBitGenerator", algorithm=algorithm,
                              initial_state=initial_state, shape=shape,
                              dtype=dtype, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_rng_bit_generator, (), dict(algorithm=algorithm,
                                          initial_state=initial_state,
                                          shape=shape, dtype=dtype, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "Tshape",
              _op._get_attr_type("Tshape"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaRngBitGenerator", _inputs_flat, _attrs, _result)
  _result = _XlaRngBitGeneratorOutput._make(_result)
  return _result

XlaRngBitGenerator = tf_export("raw_ops.XlaRngBitGenerator")(_ops.to_raw_op(xla_rng_bit_generator))
_dispatcher_for_xla_rng_bit_generator = xla_rng_bit_generator._tf_type_based_dispatcher.Dispatch


def xla_rng_bit_generator_eager_fallback(algorithm, initial_state, shape, dtype, name, ctx):
  if dtype is None:
    dtype = _dtypes.uint64
  dtype = _execute.make_type(dtype, "dtype")
  _attr_Tshape, (shape,) = _execute.args_to_matching_eager([shape], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  algorithm = _ops.convert_to_tensor(algorithm, _dtypes.int32)
  initial_state = _ops.convert_to_tensor(initial_state, _dtypes.uint64)
  _inputs_flat = [algorithm, initial_state, shape]
  _attrs = ("dtype", dtype, "Tshape", _attr_Tshape)
  _result = _execute.execute(b"XlaRngBitGenerator", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaRngBitGenerator", _inputs_flat, _attrs, _result)
  _result = _XlaRngBitGeneratorOutput._make(_result)
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_scatter')
def xla_scatter(operand, scatter_indices, updates, update_computation, dimension_numbers, indices_are_sorted, name=None):
  r"""Wraps the XLA Scatter operator documented at

    https://www.tensorflow.org/xla/operation_semantics#scatter.

  Args:
    operand: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`, `bool`.
      Array to be scattered into.
    scatter_indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Array containing the starting indices of the slices that must
      be scattered to.
    updates: A `Tensor`. Must have the same type as `operand`.
      Array containing the values that must be used for scattering.
    update_computation: A function decorated with @Defun.
      Computation to be used for combining the existing values in
      the input array and the updates during scatter.
    dimension_numbers: A `string`.
      A serialized xla::ScatterDimensionNumbers proto.
    indices_are_sorted: A `bool`.
      Boolean indicating if the indices are sorted.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `operand`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaScatter", name, operand, scatter_indices, updates,
        "update_computation", update_computation, "dimension_numbers",
        dimension_numbers, "indices_are_sorted", indices_are_sorted)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_scatter(
          (operand, scatter_indices, updates, update_computation,
          dimension_numbers, indices_are_sorted, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_scatter_eager_fallback(
          operand, scatter_indices, updates,
          update_computation=update_computation,
          dimension_numbers=dimension_numbers,
          indices_are_sorted=indices_are_sorted, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_scatter, (), dict(operand=operand,
                                  scatter_indices=scatter_indices,
                                  updates=updates,
                                  update_computation=update_computation,
                                  dimension_numbers=dimension_numbers,
                                  indices_are_sorted=indices_are_sorted,
                                  name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_scatter(
        (operand, scatter_indices, updates, update_computation,
        dimension_numbers, indices_are_sorted, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  dimension_numbers = _execute.make_str(dimension_numbers, "dimension_numbers")
  indices_are_sorted = _execute.make_bool(indices_are_sorted, "indices_are_sorted")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaScatter", operand=operand, scatter_indices=scatter_indices,
                      updates=updates, update_computation=update_computation,
                      dimension_numbers=dimension_numbers,
                      indices_are_sorted=indices_are_sorted, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_scatter, (), dict(operand=operand,
                                scatter_indices=scatter_indices,
                                updates=updates,
                                update_computation=update_computation,
                                dimension_numbers=dimension_numbers,
                                indices_are_sorted=indices_are_sorted,
                                name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("update_computation", _op.get_attr("update_computation"),
              "dimension_numbers", _op.get_attr("dimension_numbers"),
              "indices_are_sorted", _op._get_attr_bool("indices_are_sorted"),
              "T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaScatter", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaScatter = tf_export("raw_ops.XlaScatter")(_ops.to_raw_op(xla_scatter))
_dispatcher_for_xla_scatter = xla_scatter._tf_type_based_dispatcher.Dispatch


def xla_scatter_eager_fallback(operand, scatter_indices, updates, update_computation, dimension_numbers, indices_are_sorted, name, ctx):
  dimension_numbers = _execute.make_str(dimension_numbers, "dimension_numbers")
  indices_are_sorted = _execute.make_bool(indices_are_sorted, "indices_are_sorted")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([operand, updates], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, _dtypes.bool, ])
  (operand, updates) = _inputs_T
  _attr_Tindices, (scatter_indices,) = _execute.args_to_matching_eager([scatter_indices], ctx, [_dtypes.int32, _dtypes.int64, ])
  _inputs_flat = [operand, scatter_indices, updates]
  _attrs = ("update_computation", update_computation, "dimension_numbers",
  dimension_numbers, "indices_are_sorted", indices_are_sorted, "T", _attr_T,
  "Tindices", _attr_Tindices)
  _result = _execute.execute(b"XlaScatter", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaScatter", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_select_and_scatter')
def xla_select_and_scatter(operand, window_dimensions, window_strides, padding, source, init_value, select, scatter, name=None):
  r"""Wraps the XLA SelectAndScatter operator, documented at

   https://www.tensorflow.org/performance/xla/operation_semantics#selectandscatter
  .

  Args:
    operand: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      the input tensor
    window_dimensions: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      the shape of the window
    window_strides: A `Tensor`. Must have the same type as `window_dimensions`.
      the inter-window strides
    padding: A `Tensor`. Must have the same type as `window_dimensions`.
      the padding to apply at the start and end of each input dimensions
    source: A `Tensor`. Must have the same type as `operand`.
      a tensor of values to scatter
    init_value: A `Tensor`. Must have the same type as `operand`.
      a scalar representing the initial value for the output tensor
    select: A function decorated with @Defun. a selection function to apply
    scatter: A function decorated with @Defun. a scatter function to apply
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `operand`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaSelectAndScatter", name, operand, window_dimensions,
        window_strides, padding, source, init_value, "select", select,
        "scatter", scatter)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_select_and_scatter(
          (operand, window_dimensions, window_strides, padding, source,
          init_value, select, scatter, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_select_and_scatter_eager_fallback(
          operand, window_dimensions, window_strides, padding, source,
          init_value, select=select, scatter=scatter, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_select_and_scatter, (), dict(operand=operand,
                                             window_dimensions=window_dimensions,
                                             window_strides=window_strides,
                                             padding=padding, source=source,
                                             init_value=init_value,
                                             select=select, scatter=scatter,
                                             name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_select_and_scatter(
        (operand, window_dimensions, window_strides, padding, source,
        init_value, select, scatter, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaSelectAndScatter", operand=operand,
                               window_dimensions=window_dimensions,
                               window_strides=window_strides, padding=padding,
                               source=source, init_value=init_value,
                               select=select, scatter=scatter, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_select_and_scatter, (), dict(operand=operand,
                                           window_dimensions=window_dimensions,
                                           window_strides=window_strides,
                                           padding=padding, source=source,
                                           init_value=init_value,
                                           select=select, scatter=scatter,
                                           name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"), "select",
              _op.get_attr("select"), "scatter", _op.get_attr("scatter"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaSelectAndScatter", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaSelectAndScatter = tf_export("raw_ops.XlaSelectAndScatter")(_ops.to_raw_op(xla_select_and_scatter))
_dispatcher_for_xla_select_and_scatter = xla_select_and_scatter._tf_type_based_dispatcher.Dispatch


def xla_select_and_scatter_eager_fallback(operand, window_dimensions, window_strides, padding, source, init_value, select, scatter, name, ctx):
  _attr_T, _inputs_T = _execute.args_to_matching_eager([operand, source, init_value], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (operand, source, init_value) = _inputs_T
  _attr_Tindices, _inputs_Tindices = _execute.args_to_matching_eager([window_dimensions, window_strides, padding], ctx, [_dtypes.int32, _dtypes.int64, ])
  (window_dimensions, window_strides, padding) = _inputs_Tindices
  _inputs_flat = [operand, window_dimensions, window_strides, padding, source, init_value]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices, "select", select,
  "scatter", scatter)
  _result = _execute.execute(b"XlaSelectAndScatter", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaSelectAndScatter", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_XlaSelfAdjointEigOutput = collections.namedtuple(
    "XlaSelfAdjointEig",
    ["w", "v"])


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_self_adjoint_eig')
def xla_self_adjoint_eig(a, lower, max_iter, epsilon, name=None):
  r"""Computes the eigen decomposition of a batch of self-adjoint matrices

  (Note: Only real inputs are supported).

  Computes the eigenvalues and eigenvectors of the innermost N-by-N matrices in
  tensor such that tensor[...,:,:] * v[..., :,i] = e[..., i] * v[...,:,i], for
  i=0...N-1.

  Args:
    a: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      the input tensor.
    lower: A `bool`.
      a boolean specifies whether the calculation is done with the lower
      triangular part or the upper triangular part.
    max_iter: An `int`.
      maximum number of sweep update, i.e., the whole lower triangular
      part or upper triangular part based on parameter lower. Heuristically, it has
      been argued that approximately logN sweeps are needed in practice (Ref: Golub &
      van Loan "Matrix Computation").
    epsilon: A `float`. the tolerance ratio.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (w, v).

    w: A `Tensor`. Has the same type as `a`. The eigenvalues in ascending order, each repeated according to its
      multiplicity.
    v: A `Tensor`. Has the same type as `a`. The column v[..., :, i] is the normalized eigenvector corresponding to the
      eigenvalue w[..., i].
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaSelfAdjointEig", name, a, "lower", lower, "max_iter",
        max_iter, "epsilon", epsilon)
      _result = _XlaSelfAdjointEigOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_self_adjoint_eig(
          (a, lower, max_iter, epsilon, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_self_adjoint_eig_eager_fallback(
          a, lower=lower, max_iter=max_iter, epsilon=epsilon, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_self_adjoint_eig, (), dict(a=a, lower=lower,
                                           max_iter=max_iter, epsilon=epsilon,
                                           name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_self_adjoint_eig(
        (a, lower, max_iter, epsilon, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  lower = _execute.make_bool(lower, "lower")
  max_iter = _execute.make_int(max_iter, "max_iter")
  epsilon = _execute.make_float(epsilon, "epsilon")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaSelfAdjointEig", a=a, lower=lower, max_iter=max_iter,
                             epsilon=epsilon, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_self_adjoint_eig, (), dict(a=a, lower=lower, max_iter=max_iter,
                                         epsilon=epsilon, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("lower", _op._get_attr_bool("lower"), "max_iter",
              _op._get_attr_int("max_iter"), "epsilon",
              _op.get_attr("epsilon"), "T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaSelfAdjointEig", _inputs_flat, _attrs, _result)
  _result = _XlaSelfAdjointEigOutput._make(_result)
  return _result

XlaSelfAdjointEig = tf_export("raw_ops.XlaSelfAdjointEig")(_ops.to_raw_op(xla_self_adjoint_eig))
_dispatcher_for_xla_self_adjoint_eig = xla_self_adjoint_eig._tf_type_based_dispatcher.Dispatch


def xla_self_adjoint_eig_eager_fallback(a, lower, max_iter, epsilon, name, ctx):
  lower = _execute.make_bool(lower, "lower")
  max_iter = _execute.make_int(max_iter, "max_iter")
  epsilon = _execute.make_float(epsilon, "epsilon")
  _attr_T, (a,) = _execute.args_to_matching_eager([a], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  _inputs_flat = [a]
  _attrs = ("lower", lower, "max_iter", max_iter, "epsilon", epsilon, "T",
  _attr_T)
  _result = _execute.execute(b"XlaSelfAdjointEig", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaSelfAdjointEig", _inputs_flat, _attrs, _result)
  _result = _XlaSelfAdjointEigOutput._make(_result)
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_send')
def xla_send(tensor, tensor_name, name=None):
  r"""Sends the named tensor to another XLA computation. Wraps the XLA Send operator

  documented at
   https://www.tensorflow.org/performance/xla/operation_semantics#send .

  Args:
    tensor: A `Tensor`. The tensor to send.
    tensor_name: A `string`. A string key that identifies the channel.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaSend", name, tensor, "tensor_name", tensor_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_send(
          (tensor, tensor_name, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_send_eager_fallback(
          tensor, tensor_name=tensor_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_send, (), dict(tensor=tensor, tensor_name=tensor_name,
                               name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_send(
        (tensor, tensor_name, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  tensor_name = _execute.make_str(tensor_name, "tensor_name")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaSend", tensor=tensor, tensor_name=tensor_name, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_send, (), dict(tensor=tensor, tensor_name=tensor_name,
                             name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
XlaSend = tf_export("raw_ops.XlaSend")(_ops.to_raw_op(xla_send))
_dispatcher_for_xla_send = xla_send._tf_type_based_dispatcher.Dispatch


def xla_send_eager_fallback(tensor, tensor_name, name, ctx):
  tensor_name = _execute.make_str(tensor_name, "tensor_name")
  _attr_T, (tensor,) = _execute.args_to_matching_eager([tensor], ctx, [])
  _inputs_flat = [tensor]
  _attrs = ("T", _attr_T, "tensor_name", tensor_name)
  _result = _execute.execute(b"XlaSend", 0, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_set_bound')
def xla_set_bound(input, bound, name=None):
  r"""Set a bound for the given input value as a hint to Xla compiler,

          returns the same value.

  Args:
    input: A `Tensor` of type `int32`.
    bound: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaSetBound", name, input, bound)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_set_bound(
          (input, bound, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_set_bound_eager_fallback(
          input, bound, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_set_bound, (), dict(input=input, bound=bound, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_set_bound(
        (input, bound, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaSetBound", input=input, bound=bound, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_set_bound, (), dict(input=input, bound=bound, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaSetBound", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaSetBound = tf_export("raw_ops.XlaSetBound")(_ops.to_raw_op(xla_set_bound))
_dispatcher_for_xla_set_bound = xla_set_bound._tf_type_based_dispatcher.Dispatch


def xla_set_bound_eager_fallback(input, bound, name, ctx):
  input = _ops.convert_to_tensor(input, _dtypes.int32)
  bound = _ops.convert_to_tensor(bound, _dtypes.int32)
  _inputs_flat = [input, bound]
  _attrs = None
  _result = _execute.execute(b"XlaSetBound", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaSetBound", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_set_dynamic_dimension_size')
def xla_set_dynamic_dimension_size(input, dim_index, size, name=None):
  r"""Make a static dimension into a xla bounded dynamic dimension.

          The current static dimension size will become the bound and the second
          operand becomes the dynamic size of the dimension.

  Args:
    input: A `Tensor`.
    dim_index: A `Tensor` of type `int32`.
    size: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaSetDynamicDimensionSize", name, input, dim_index, size)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_set_dynamic_dimension_size(
          (input, dim_index, size, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_set_dynamic_dimension_size_eager_fallback(
          input, dim_index, size, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_set_dynamic_dimension_size, (), dict(input=input,
                                                     dim_index=dim_index,
                                                     size=size, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_set_dynamic_dimension_size(
        (input, dim_index, size, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaSetDynamicDimensionSize", input=input, dim_index=dim_index,
                                      size=size, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_set_dynamic_dimension_size, (), dict(input=input,
                                                   dim_index=dim_index,
                                                   size=size, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaSetDynamicDimensionSize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaSetDynamicDimensionSize = tf_export("raw_ops.XlaSetDynamicDimensionSize")(_ops.to_raw_op(xla_set_dynamic_dimension_size))
_dispatcher_for_xla_set_dynamic_dimension_size = xla_set_dynamic_dimension_size._tf_type_based_dispatcher.Dispatch


def xla_set_dynamic_dimension_size_eager_fallback(input, dim_index, size, name, ctx):
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  dim_index = _ops.convert_to_tensor(dim_index, _dtypes.int32)
  size = _ops.convert_to_tensor(size, _dtypes.int32)
  _inputs_flat = [input, dim_index, size]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"XlaSetDynamicDimensionSize", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaSetDynamicDimensionSize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_sharding')
def xla_sharding(input, sharding="", unspecified_dims=[], name=None):
  r"""An op which shards the input based on the given sharding attribute. It can

  selectively annotate a subset of tensor dimensions by skipping unspecified_dims,
  and the sharding annotation should be replicated in those dims.

  Args:
    input: A `Tensor`.
    sharding: An optional `string`. Defaults to `""`.
    unspecified_dims: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaSharding", name, input, "sharding", sharding,
        "unspecified_dims", unspecified_dims)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_sharding(
          (input, sharding, unspecified_dims, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_sharding_eager_fallback(
          input, sharding=sharding, unspecified_dims=unspecified_dims,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_sharding, (), dict(input=input, sharding=sharding,
                                   unspecified_dims=unspecified_dims,
                                   name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_sharding(
        (input, sharding, unspecified_dims, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if sharding is None:
    sharding = ""
  sharding = _execute.make_str(sharding, "sharding")
  if unspecified_dims is None:
    unspecified_dims = []
  if not isinstance(unspecified_dims, (list, tuple)):
    raise TypeError(
        "Expected list for 'unspecified_dims' argument to "
        "'xla_sharding' Op, not %r." % unspecified_dims)
  unspecified_dims = [_execute.make_int(_i, "unspecified_dims") for _i in unspecified_dims]
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaSharding", input=input, sharding=sharding,
                       unspecified_dims=unspecified_dims, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_sharding, (), dict(input=input, sharding=sharding,
                                 unspecified_dims=unspecified_dims, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "sharding",
              _op.get_attr("sharding"), "unspecified_dims",
              _op.get_attr("unspecified_dims"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaSharding", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaSharding = tf_export("raw_ops.XlaSharding")(_ops.to_raw_op(xla_sharding))
_dispatcher_for_xla_sharding = xla_sharding._tf_type_based_dispatcher.Dispatch


def xla_sharding_eager_fallback(input, sharding, unspecified_dims, name, ctx):
  if sharding is None:
    sharding = ""
  sharding = _execute.make_str(sharding, "sharding")
  if unspecified_dims is None:
    unspecified_dims = []
  if not isinstance(unspecified_dims, (list, tuple)):
    raise TypeError(
        "Expected list for 'unspecified_dims' argument to "
        "'xla_sharding' Op, not %r." % unspecified_dims)
  unspecified_dims = [_execute.make_int(_i, "unspecified_dims") for _i in unspecified_dims]
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _inputs_flat = [input]
  _attrs = ("T", _attr_T, "sharding", sharding, "unspecified_dims",
  unspecified_dims)
  _result = _execute.execute(b"XlaSharding", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaSharding", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_sort')
def xla_sort(input, name=None):
  r"""Wraps the XLA Sort operator, documented at

   https://www.tensorflow.org/performance/xla/operation_semantics#sort
  .

  Sorts a tensor. Currently only sorts in ascending order are supported.

  Args:
    input: A `Tensor`. A `Tensor` of type T.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. A `Tensor` of type T.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaSort", name, input)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_sort(
          (input, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_sort_eager_fallback(
          input, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_sort, (), dict(input=input, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_sort(
        (input, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaSort", input=input, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_sort, (), dict(input=input, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaSort", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaSort = tf_export("raw_ops.XlaSort")(_ops.to_raw_op(xla_sort))
_dispatcher_for_xla_sort = xla_sort._tf_type_based_dispatcher.Dispatch


def xla_sort_eager_fallback(input, name, ctx):
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _inputs_flat = [input]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"XlaSort", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaSort", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_spmd_full_to_shard_shape')
def xla_spmd_full_to_shard_shape(input, manual_sharding, dim=-1, unspecified_dims=[], name=None):
  r"""An op used by XLA SPMD partitioner to switch from automatic partitioning to

  manual partitioning. It annotates the input (full-shape, to be automatically
  partitioned) with the same sharding used by manual partitioning, and outputs a
  shard-shaped tensor to be consumed by later manually-partitioned ops. If the
  shape is not evenly partitionable, the padding region will be masked with 0s.
  The conversion can happen partially in subgroups, by specifying the dim
  attribute, where only that dim will be converted.

  Args:
    input: A `Tensor`.
    manual_sharding: A `string`.
    dim: An optional `int`. Defaults to `-1`.
    unspecified_dims: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaSpmdFullToShardShape", name, input, "manual_sharding",
        manual_sharding, "dim", dim, "unspecified_dims", unspecified_dims)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_spmd_full_to_shard_shape(
          (input, manual_sharding, dim, unspecified_dims, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_spmd_full_to_shard_shape_eager_fallback(
          input, manual_sharding=manual_sharding, dim=dim,
          unspecified_dims=unspecified_dims, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_spmd_full_to_shard_shape, (), dict(input=input,
                                                   manual_sharding=manual_sharding,
                                                   dim=dim,
                                                   unspecified_dims=unspecified_dims,
                                                   name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_spmd_full_to_shard_shape(
        (input, manual_sharding, dim, unspecified_dims, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  manual_sharding = _execute.make_str(manual_sharding, "manual_sharding")
  if dim is None:
    dim = -1
  dim = _execute.make_int(dim, "dim")
  if unspecified_dims is None:
    unspecified_dims = []
  if not isinstance(unspecified_dims, (list, tuple)):
    raise TypeError(
        "Expected list for 'unspecified_dims' argument to "
        "'xla_spmd_full_to_shard_shape' Op, not %r." % unspecified_dims)
  unspecified_dims = [_execute.make_int(_i, "unspecified_dims") for _i in unspecified_dims]
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaSpmdFullToShardShape", input=input,
                                   manual_sharding=manual_sharding, dim=dim,
                                   unspecified_dims=unspecified_dims,
                                   name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_spmd_full_to_shard_shape, (), dict(input=input,
                                                 manual_sharding=manual_sharding,
                                                 dim=dim,
                                                 unspecified_dims=unspecified_dims,
                                                 name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "manual_sharding",
              _op.get_attr("manual_sharding"), "dim",
              _op._get_attr_int("dim"), "unspecified_dims",
              _op.get_attr("unspecified_dims"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaSpmdFullToShardShape", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaSpmdFullToShardShape = tf_export("raw_ops.XlaSpmdFullToShardShape")(_ops.to_raw_op(xla_spmd_full_to_shard_shape))
_dispatcher_for_xla_spmd_full_to_shard_shape = xla_spmd_full_to_shard_shape._tf_type_based_dispatcher.Dispatch


def xla_spmd_full_to_shard_shape_eager_fallback(input, manual_sharding, dim, unspecified_dims, name, ctx):
  manual_sharding = _execute.make_str(manual_sharding, "manual_sharding")
  if dim is None:
    dim = -1
  dim = _execute.make_int(dim, "dim")
  if unspecified_dims is None:
    unspecified_dims = []
  if not isinstance(unspecified_dims, (list, tuple)):
    raise TypeError(
        "Expected list for 'unspecified_dims' argument to "
        "'xla_spmd_full_to_shard_shape' Op, not %r." % unspecified_dims)
  unspecified_dims = [_execute.make_int(_i, "unspecified_dims") for _i in unspecified_dims]
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _inputs_flat = [input]
  _attrs = ("T", _attr_T, "manual_sharding", manual_sharding, "dim", dim,
  "unspecified_dims", unspecified_dims)
  _result = _execute.execute(b"XlaSpmdFullToShardShape", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaSpmdFullToShardShape", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_spmd_shard_to_full_shape')
def xla_spmd_shard_to_full_shape(input, manual_sharding, full_shape, dim=-1, unspecified_dims=[], name=None):
  r"""An op used by XLA SPMD partitioner to switch from manual partitioning to

  automatic partitioning. It converts the shard-shaped, manually partitioned input
  into full-shaped tensor to be partitioned automatically with the same sharding
  used by manual partitioning. The conversion can happen partially in subgroups,
  by specifying the dim attribute, where only that dim will be converted.

  Args:
    input: A `Tensor`.
    manual_sharding: A `string`.
    full_shape: A `tf.TensorShape` or list of `ints`.
    dim: An optional `int`. Defaults to `-1`.
    unspecified_dims: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaSpmdShardToFullShape", name, input, "manual_sharding",
        manual_sharding, "full_shape", full_shape, "dim", dim,
        "unspecified_dims", unspecified_dims)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_spmd_shard_to_full_shape(
          (input, manual_sharding, full_shape, dim, unspecified_dims, name,),
          None)
      if _result is not NotImplemented:
        return _result
      return xla_spmd_shard_to_full_shape_eager_fallback(
          input, manual_sharding=manual_sharding, full_shape=full_shape,
          dim=dim, unspecified_dims=unspecified_dims, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_spmd_shard_to_full_shape, (), dict(input=input,
                                                   manual_sharding=manual_sharding,
                                                   full_shape=full_shape,
                                                   dim=dim,
                                                   unspecified_dims=unspecified_dims,
                                                   name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_spmd_shard_to_full_shape(
        (input, manual_sharding, full_shape, dim, unspecified_dims, name,),
        None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  manual_sharding = _execute.make_str(manual_sharding, "manual_sharding")
  full_shape = _execute.make_shape(full_shape, "full_shape")
  if dim is None:
    dim = -1
  dim = _execute.make_int(dim, "dim")
  if unspecified_dims is None:
    unspecified_dims = []
  if not isinstance(unspecified_dims, (list, tuple)):
    raise TypeError(
        "Expected list for 'unspecified_dims' argument to "
        "'xla_spmd_shard_to_full_shape' Op, not %r." % unspecified_dims)
  unspecified_dims = [_execute.make_int(_i, "unspecified_dims") for _i in unspecified_dims]
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaSpmdShardToFullShape", input=input,
                                   manual_sharding=manual_sharding,
                                   full_shape=full_shape, dim=dim,
                                   unspecified_dims=unspecified_dims,
                                   name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_spmd_shard_to_full_shape, (), dict(input=input,
                                                 manual_sharding=manual_sharding,
                                                 full_shape=full_shape,
                                                 dim=dim,
                                                 unspecified_dims=unspecified_dims,
                                                 name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "manual_sharding",
              _op.get_attr("manual_sharding"), "full_shape",
              _op.get_attr("full_shape"), "dim", _op._get_attr_int("dim"),
              "unspecified_dims", _op.get_attr("unspecified_dims"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaSpmdShardToFullShape", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaSpmdShardToFullShape = tf_export("raw_ops.XlaSpmdShardToFullShape")(_ops.to_raw_op(xla_spmd_shard_to_full_shape))
_dispatcher_for_xla_spmd_shard_to_full_shape = xla_spmd_shard_to_full_shape._tf_type_based_dispatcher.Dispatch


def xla_spmd_shard_to_full_shape_eager_fallback(input, manual_sharding, full_shape, dim, unspecified_dims, name, ctx):
  manual_sharding = _execute.make_str(manual_sharding, "manual_sharding")
  full_shape = _execute.make_shape(full_shape, "full_shape")
  if dim is None:
    dim = -1
  dim = _execute.make_int(dim, "dim")
  if unspecified_dims is None:
    unspecified_dims = []
  if not isinstance(unspecified_dims, (list, tuple)):
    raise TypeError(
        "Expected list for 'unspecified_dims' argument to "
        "'xla_spmd_shard_to_full_shape' Op, not %r." % unspecified_dims)
  unspecified_dims = [_execute.make_int(_i, "unspecified_dims") for _i in unspecified_dims]
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _inputs_flat = [input]
  _attrs = ("T", _attr_T, "manual_sharding", manual_sharding, "full_shape",
  full_shape, "dim", dim, "unspecified_dims", unspecified_dims)
  _result = _execute.execute(b"XlaSpmdShardToFullShape", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaSpmdShardToFullShape", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_XlaSvdOutput = collections.namedtuple(
    "XlaSvd",
    ["s", "u", "v"])


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_svd')
def xla_svd(a, max_iter, epsilon, precision_config, name=None):
  r"""Computes the eigen decomposition of a batch of self-adjoint matrices

  (Note: Only real inputs are supported).

  Computes the eigenvalues and eigenvectors of the innermost M-by-N matrices in
  tensor such that tensor[...,:,:] = u[..., :, :] * Diag(s[..., :]) * Transpose(v[...,:,:]).

  Args:
    a: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      the input tensor.
    max_iter: An `int`.
      maximum number of sweep update, i.e., the whole lower triangular
      part or upper triangular part based on parameter lower. Heuristically, it has
      been argued that approximately log(min (M, N)) sweeps are needed in practice
      (Ref: Golub & van Loan "Matrix Computation").
    epsilon: A `float`. the tolerance ratio.
    precision_config: A `string`. a serialized xla::PrecisionConfig proto.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (s, u, v).

    s: A `Tensor`. Has the same type as `a`. Singular values. The values are sorted in reverse order of magnitude, so
      s[..., 0] is the largest value, s[..., 1] is the second largest, etc.
    u: A `Tensor`. Has the same type as `a`. Left singular vectors.
    v: A `Tensor`. Has the same type as `a`. Right singular vectors.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaSvd", name, a, "max_iter", max_iter, "epsilon", epsilon,
        "precision_config", precision_config)
      _result = _XlaSvdOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_svd(
          (a, max_iter, epsilon, precision_config, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_svd_eager_fallback(
          a, max_iter=max_iter, epsilon=epsilon,
          precision_config=precision_config, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_svd, (), dict(a=a, max_iter=max_iter, epsilon=epsilon,
                              precision_config=precision_config, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_svd(
        (a, max_iter, epsilon, precision_config, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  max_iter = _execute.make_int(max_iter, "max_iter")
  epsilon = _execute.make_float(epsilon, "epsilon")
  precision_config = _execute.make_str(precision_config, "precision_config")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaSvd", a=a, max_iter=max_iter, epsilon=epsilon,
                  precision_config=precision_config, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_svd, (), dict(a=a, max_iter=max_iter, epsilon=epsilon,
                            precision_config=precision_config, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("max_iter", _op._get_attr_int("max_iter"), "epsilon",
              _op.get_attr("epsilon"), "precision_config",
              _op.get_attr("precision_config"), "T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaSvd", _inputs_flat, _attrs, _result)
  _result = _XlaSvdOutput._make(_result)
  return _result

XlaSvd = tf_export("raw_ops.XlaSvd")(_ops.to_raw_op(xla_svd))
_dispatcher_for_xla_svd = xla_svd._tf_type_based_dispatcher.Dispatch


def xla_svd_eager_fallback(a, max_iter, epsilon, precision_config, name, ctx):
  max_iter = _execute.make_int(max_iter, "max_iter")
  epsilon = _execute.make_float(epsilon, "epsilon")
  precision_config = _execute.make_str(precision_config, "precision_config")
  _attr_T, (a,) = _execute.args_to_matching_eager([a], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  _inputs_flat = [a]
  _attrs = ("max_iter", max_iter, "epsilon", epsilon, "precision_config",
  precision_config, "T", _attr_T)
  _result = _execute.execute(b"XlaSvd", 3, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaSvd", _inputs_flat, _attrs, _result)
  _result = _XlaSvdOutput._make(_result)
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_variadic_reduce')
def xla_variadic_reduce(input, init_value, dimensions_to_reduce, reducer, name=None):
  r"""Wraps the variadic XLA Reduce operator.

  Semantics are documented at
   https://www.tensorflow.org/performance/xla/operation_semantics#variadic_reduce.

  This version is limited to operands of the same dtype.
  XlaVariadicReduceV2 is a version that supports heterogeneous operands.

  Args:
    input: A list of at least 1 `Tensor` objects with the same type in: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`, `bool`.
      the input tensor(s)
    init_value: A list with the same length as `input` of `Tensor` objects with the same type as `input`.
      scalar initial value(s) for the reduction
    dimensions_to_reduce: A list of `ints`.
      dimension numbers over which to reduce
    reducer: A function decorated with @Defun. a reducer function to apply
    name: A name for the operation (optional).

  Returns:
    A list with the same length as `input` of `Tensor` objects with the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaVariadicReduce", name, input, init_value,
        "dimensions_to_reduce", dimensions_to_reduce, "reducer", reducer)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_variadic_reduce(
          (input, init_value, dimensions_to_reduce, reducer, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_variadic_reduce_eager_fallback(
          input, init_value, dimensions_to_reduce=dimensions_to_reduce,
          reducer=reducer, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_variadic_reduce, (), dict(input=input, init_value=init_value,
                                          dimensions_to_reduce=dimensions_to_reduce,
                                          reducer=reducer, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_variadic_reduce(
        (input, init_value, dimensions_to_reduce, reducer, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(input, (list, tuple)):
    raise TypeError(
        "Expected list for 'input' argument to "
        "'xla_variadic_reduce' Op, not %r." % input)
  _attr_N = len(input)
  if not isinstance(init_value, (list, tuple)):
    raise TypeError(
        "Expected list for 'init_value' argument to "
        "'xla_variadic_reduce' Op, not %r." % init_value)
  if len(init_value) != _attr_N:
    raise ValueError(
        "List argument 'init_value' to 'xla_variadic_reduce' Op with length %d "
        "must match length %d of argument 'input'." %
        (len(init_value), _attr_N))
  if not isinstance(dimensions_to_reduce, (list, tuple)):
    raise TypeError(
        "Expected list for 'dimensions_to_reduce' argument to "
        "'xla_variadic_reduce' Op, not %r." % dimensions_to_reduce)
  dimensions_to_reduce = [_execute.make_int(_i, "dimensions_to_reduce") for _i in dimensions_to_reduce]
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaVariadicReduce", input=input, init_value=init_value,
                             dimensions_to_reduce=dimensions_to_reduce,
                             reducer=reducer, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_variadic_reduce, (), dict(input=input, init_value=init_value,
                                        dimensions_to_reduce=dimensions_to_reduce,
                                        reducer=reducer, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("N", _op._get_attr_int("N"), "T", _op._get_attr_type("T"),
              "dimensions_to_reduce", _op.get_attr("dimensions_to_reduce"),
              "reducer", _op.get_attr("reducer"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaVariadicReduce", _inputs_flat, _attrs, _result)
  return _result

XlaVariadicReduce = tf_export("raw_ops.XlaVariadicReduce")(_ops.to_raw_op(xla_variadic_reduce))
_dispatcher_for_xla_variadic_reduce = xla_variadic_reduce._tf_type_based_dispatcher.Dispatch


def xla_variadic_reduce_eager_fallback(input, init_value, dimensions_to_reduce, reducer, name, ctx):
  if not isinstance(input, (list, tuple)):
    raise TypeError(
        "Expected list for 'input' argument to "
        "'xla_variadic_reduce' Op, not %r." % input)
  _attr_N = len(input)
  if not isinstance(init_value, (list, tuple)):
    raise TypeError(
        "Expected list for 'init_value' argument to "
        "'xla_variadic_reduce' Op, not %r." % init_value)
  if len(init_value) != _attr_N:
    raise ValueError(
        "List argument 'init_value' to 'xla_variadic_reduce' Op with length %d "
        "must match length %d of argument 'input'." %
        (len(init_value), _attr_N))
  if not isinstance(dimensions_to_reduce, (list, tuple)):
    raise TypeError(
        "Expected list for 'dimensions_to_reduce' argument to "
        "'xla_variadic_reduce' Op, not %r." % dimensions_to_reduce)
  dimensions_to_reduce = [_execute.make_int(_i, "dimensions_to_reduce") for _i in dimensions_to_reduce]
  _attr_T, _inputs_T = _execute.args_to_matching_eager(list(input) + list(init_value), ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, _dtypes.bool, ])
  _inputs_T = [_inputs_T[:_attr_N]] + _inputs_T[_attr_N:]
  _inputs_T = _inputs_T[:1] + [_inputs_T[1:]]
  (input, init_value) = _inputs_T
  _inputs_flat = list(input) + list(init_value)
  _attrs = ("N", _attr_N, "T", _attr_T, "dimensions_to_reduce",
  dimensions_to_reduce, "reducer", reducer)
  _result = _execute.execute(b"XlaVariadicReduce", _attr_N,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaVariadicReduce", _inputs_flat, _attrs, _result)
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_variadic_reduce_v2')
def xla_variadic_reduce_v2(inputs, init_values, dimensions_to_reduce, reducer, name=None):
  r"""Wraps the variadic XLA Reduce operator.

  Semantics are documented at
   https://www.tensorflow.org/performance/xla/operation_semantics#variadic_reduce.

  This is an expanded version of XlaVariadicReduce, with support for
  operands of different dtypes, and improved shape inference.

  Args:
    inputs: A list of `Tensor` objects. the input tensor(s)
    init_values: A list of `Tensor` objects. Must have the same type as `inputs`.
      scalar initial value(s) for the reduction
    dimensions_to_reduce: A list of `ints`.
      dimension numbers over which to reduce
    reducer: A function decorated with @Defun. a reducer function to apply
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects. Has the same type as `inputs`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaVariadicReduceV2", name, inputs, init_values,
        "dimensions_to_reduce", dimensions_to_reduce, "reducer", reducer)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_variadic_reduce_v2(
          (inputs, init_values, dimensions_to_reduce, reducer, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_variadic_reduce_v2_eager_fallback(
          inputs, init_values, dimensions_to_reduce=dimensions_to_reduce,
          reducer=reducer, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_variadic_reduce_v2, (), dict(inputs=inputs,
                                             init_values=init_values,
                                             dimensions_to_reduce=dimensions_to_reduce,
                                             reducer=reducer, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_variadic_reduce_v2(
        (inputs, init_values, dimensions_to_reduce, reducer, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(dimensions_to_reduce, (list, tuple)):
    raise TypeError(
        "Expected list for 'dimensions_to_reduce' argument to "
        "'xla_variadic_reduce_v2' Op, not %r." % dimensions_to_reduce)
  dimensions_to_reduce = [_execute.make_int(_i, "dimensions_to_reduce") for _i in dimensions_to_reduce]
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaVariadicReduceV2", inputs=inputs, init_values=init_values,
                               dimensions_to_reduce=dimensions_to_reduce,
                               reducer=reducer, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_variadic_reduce_v2, (), dict(inputs=inputs,
                                           init_values=init_values,
                                           dimensions_to_reduce=dimensions_to_reduce,
                                           reducer=reducer, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op.get_attr("T"), "dimensions_to_reduce",
              _op.get_attr("dimensions_to_reduce"), "reducer",
              _op.get_attr("reducer"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaVariadicReduceV2", _inputs_flat, _attrs, _result)
  return _result

XlaVariadicReduceV2 = tf_export("raw_ops.XlaVariadicReduceV2")(_ops.to_raw_op(xla_variadic_reduce_v2))
_dispatcher_for_xla_variadic_reduce_v2 = xla_variadic_reduce_v2._tf_type_based_dispatcher.Dispatch


def xla_variadic_reduce_v2_eager_fallback(inputs, init_values, dimensions_to_reduce, reducer, name, ctx):
  if not isinstance(dimensions_to_reduce, (list, tuple)):
    raise TypeError(
        "Expected list for 'dimensions_to_reduce' argument to "
        "'xla_variadic_reduce_v2' Op, not %r." % dimensions_to_reduce)
  dimensions_to_reduce = [_execute.make_int(_i, "dimensions_to_reduce") for _i in dimensions_to_reduce]
  _attr_T, (inputs, init_values) = _execute.args_to_mixed_eager_tensors((inputs, init_values), ctx)
  _inputs_flat = list(inputs) + list(init_values)
  _attrs = ("T", _attr_T, "dimensions_to_reduce", dimensions_to_reduce,
  "reducer", reducer)
  _result = _execute.execute(b"XlaVariadicReduceV2", len(inputs),
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaVariadicReduceV2", _inputs_flat, _attrs, _result)
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_variadic_sort')
def xla_variadic_sort(inputs, dimension, comparator, is_stable, name=None):
  r"""Wraps the XLA Sort operator, documented at

   https://www.tensorflow.org/performance/xla/operation_semantics#sort
  .

  Sorts one or more tensors, with support for custom comparator, dimension, and
  is_stable attributes.

  Args:
    inputs: A list of `Tensor` objects.
      A list of `Tensor` of identical shape but possibly different types.
    dimension: A `Tensor` of type `int32`.
      The dimension along which to sort. Must be a compile-time constant.
    comparator: A function decorated with @Defun.
      A comparator function to apply to 2*N scalars and returning a
      boolean. N is the number of sort inputs. If you want to sort in ascending
      order then the comparator should perform a less-than comparison.
    is_stable: A `bool`. Whether to use stable sort.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects. Has the same type as `inputs`.
    A list of `Tensor` of same shape and types as the `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaVariadicSort", name, inputs, dimension, "comparator",
        comparator, "is_stable", is_stable)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_variadic_sort(
          (inputs, dimension, comparator, is_stable, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_variadic_sort_eager_fallback(
          inputs, dimension, comparator=comparator, is_stable=is_stable,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_variadic_sort, (), dict(inputs=inputs, dimension=dimension,
                                        comparator=comparator,
                                        is_stable=is_stable, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_variadic_sort(
        (inputs, dimension, comparator, is_stable, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  is_stable = _execute.make_bool(is_stable, "is_stable")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaVariadicSort", inputs=inputs, dimension=dimension,
                           comparator=comparator, is_stable=is_stable,
                           name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_variadic_sort, (), dict(inputs=inputs, dimension=dimension,
                                      comparator=comparator,
                                      is_stable=is_stable, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op.get_attr("T"), "comparator",
              _op.get_attr("comparator"), "is_stable",
              _op._get_attr_bool("is_stable"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaVariadicSort", _inputs_flat, _attrs, _result)
  return _result

XlaVariadicSort = tf_export("raw_ops.XlaVariadicSort")(_ops.to_raw_op(xla_variadic_sort))
_dispatcher_for_xla_variadic_sort = xla_variadic_sort._tf_type_based_dispatcher.Dispatch


def xla_variadic_sort_eager_fallback(inputs, dimension, comparator, is_stable, name, ctx):
  is_stable = _execute.make_bool(is_stable, "is_stable")
  _attr_T, inputs = _execute.convert_to_mixed_eager_tensors(inputs, ctx)
  dimension = _ops.convert_to_tensor(dimension, _dtypes.int32)
  _inputs_flat = list(inputs) + [dimension]
  _attrs = ("T", _attr_T, "comparator", comparator, "is_stable", is_stable)
  _result = _execute.execute(b"XlaVariadicSort", len(inputs),
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaVariadicSort", _inputs_flat, _attrs, _result)
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_while')
def xla_while(input, cond, body, name=None):
  r"""output = input; While (Cond(output)) { output = Body(output) }

  Args:
    input: A list of `Tensor` objects.
      A list of input tensors whose types are T.
    cond: A function decorated with @Defun.
      A function takes 'input' and returns a tensor.  If the tensor is
      a scalar of non-boolean, the scalar is converted to a boolean
      according to the following rule: if the scalar is a numerical
      value, non-zero means True and zero means False; if the scalar is
      a string, non-empty means True and empty means False. If the
      tensor is not a scalar, non-emptiness means True and False
      otherwise.
    body: A function decorated with @Defun.
      A function that takes a list of tensors and returns another
      list of tensors. Both lists have the same types as specified by T.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects. Has the same type as `input`.
    A list of output tensors whose types are T.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaWhile", name, input, "cond", cond, "body", body)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_while(
          (input, cond, body, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_while_eager_fallback(
          input, cond=cond, body=body, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_while, (), dict(input=input, cond=cond, body=body, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_while(
        (input, cond, body, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaWhile", input=input, cond=cond, body=body, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_while, (), dict(input=input, cond=cond, body=body, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if not _result:
    return _op
  if _execute.must_record_gradient():
    _attrs = ("T", _op.get_attr("T"), "cond", _op.get_attr("cond"), "body",
              _op.get_attr("body"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaWhile", _inputs_flat, _attrs, _result)
  return _result

XlaWhile = tf_export("raw_ops.XlaWhile")(_ops.to_raw_op(xla_while))
_dispatcher_for_xla_while = xla_while._tf_type_based_dispatcher.Dispatch


def xla_while_eager_fallback(input, cond, body, name, ctx):
  _attr_T, input = _execute.convert_to_mixed_eager_tensors(input, ctx)
  _inputs_flat = list(input)
  _attrs = ("T", _attr_T, "cond", cond, "body", body)
  _result = _execute.execute(b"XlaWhile", len(input), inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaWhile", _inputs_flat, _attrs, _result)
  return _result

