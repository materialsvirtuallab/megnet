"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: decode_proto_ops.cc
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
_DecodeProtoV2Output = collections.namedtuple(
    "DecodeProtoV2",
    ["sizes", "values"])


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('io.decode_proto')
def decode_proto_v2(bytes, message_type, field_names, output_types, descriptor_source="local://", message_format="binary", sanitize=False, name=None):
  r"""The op extracts fields from a serialized protocol buffers message into tensors.

  The `decode_proto` op extracts fields from a serialized protocol buffers
  message into tensors.  The fields in `field_names` are decoded and converted
  to the corresponding `output_types` if possible.

  A `message_type` name must be provided to give context for the field names.
  The actual message descriptor can be looked up either in the linked-in
  descriptor pool or a filename provided by the caller using the
  `descriptor_source` attribute.

  Each output tensor is a dense tensor. This means that it is padded to hold
  the largest number of repeated elements seen in the input minibatch. (The
  shape is also padded by one to prevent zero-sized dimensions). The actual
  repeat counts for each example in the minibatch can be found in the `sizes`
  output. In many cases the output of `decode_proto` is fed immediately into
  tf.squeeze if missing values are not a concern. When using tf.squeeze, always
  pass the squeeze dimension explicitly to avoid surprises.

  For the most part, the mapping between Proto field types and TensorFlow dtypes
  is straightforward. However, there are a few special cases:

  - A proto field that contains a submessage or group can only be converted
  to `DT_STRING` (the serialized submessage). This is to reduce the complexity
  of the API. The resulting string can be used as input to another instance of
  the decode_proto op.

  - TensorFlow lacks support for unsigned integers. The ops represent uint64
  types as a `DT_INT64` with the same twos-complement bit pattern (the obvious
  way). Unsigned int32 values can be represented exactly by specifying type
  `DT_INT64`, or using twos-complement if the caller specifies `DT_INT32` in
  the `output_types` attribute.

  Both binary and text proto serializations are supported, and can be
  chosen using the `format` attribute.

  The `descriptor_source` attribute selects the source of protocol
  descriptors to consult when looking up `message_type`. This may be:

  - An empty string  or "local://", in which case protocol descriptors are
  created for C++ (not Python) proto definitions linked to the binary.

  - A file, in which case protocol descriptors are created from the file,
  which is expected to contain a `FileDescriptorSet` serialized as a string.
  NOTE: You can build a `descriptor_source` file using the `--descriptor_set_out`
  and `--include_imports` options to the protocol compiler `protoc`.

  - A "bytes://<bytes>", in which protocol descriptors are created from `<bytes>`,
  which is expected to be a `FileDescriptorSet` serialized as a string.

  Args:
    bytes: A `Tensor` of type `string`.
      Tensor of serialized protos with shape `batch_shape`.
    message_type: A `string`. Name of the proto message type to decode.
    field_names: A list of `strings`.
      List of strings containing proto field names. An extension field can be decoded
      by using its full name, e.g. EXT_PACKAGE.EXT_FIELD_NAME.
    output_types: A list of `tf.DTypes`.
      List of TF types to use for the respective field in field_names.
    descriptor_source: An optional `string`. Defaults to `"local://"`.
      Either the special value `local://` or a path to a file containing
      a serialized `FileDescriptorSet`.
    message_format: An optional `string`. Defaults to `"binary"`.
      Either `binary` or `text`.
    sanitize: An optional `bool`. Defaults to `False`.
      Whether to sanitize the result or not.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (sizes, values).

    sizes: A `Tensor` of type `int32`.
    values: A list of `Tensor` objects of type `output_types`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DecodeProtoV2", name, bytes, "message_type", message_type,
        "field_names", field_names, "output_types", output_types,
        "descriptor_source", descriptor_source, "message_format",
        message_format, "sanitize", sanitize)
      _result = _DecodeProtoV2Output._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_decode_proto_v2(
          (bytes, message_type, field_names, output_types, descriptor_source,
          message_format, sanitize, name,), None)
      if _result is not NotImplemented:
        return _result
      return decode_proto_v2_eager_fallback(
          bytes, message_type=message_type, field_names=field_names,
          output_types=output_types, descriptor_source=descriptor_source,
          message_format=message_format, sanitize=sanitize, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            decode_proto_v2, (), dict(bytes=bytes, message_type=message_type,
                                      field_names=field_names,
                                      output_types=output_types,
                                      descriptor_source=descriptor_source,
                                      message_format=message_format,
                                      sanitize=sanitize, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_decode_proto_v2(
        (bytes, message_type, field_names, output_types, descriptor_source,
        message_format, sanitize, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  message_type = _execute.make_str(message_type, "message_type")
  if not isinstance(field_names, (list, tuple)):
    raise TypeError(
        "Expected list for 'field_names' argument to "
        "'decode_proto_v2' Op, not %r." % field_names)
  field_names = [_execute.make_str(_s, "field_names") for _s in field_names]
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'decode_proto_v2' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if descriptor_source is None:
    descriptor_source = "local://"
  descriptor_source = _execute.make_str(descriptor_source, "descriptor_source")
  if message_format is None:
    message_format = "binary"
  message_format = _execute.make_str(message_format, "message_format")
  if sanitize is None:
    sanitize = False
  sanitize = _execute.make_bool(sanitize, "sanitize")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DecodeProtoV2", bytes=bytes, message_type=message_type,
                         field_names=field_names, output_types=output_types,
                         descriptor_source=descriptor_source,
                         message_format=message_format, sanitize=sanitize,
                         name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          decode_proto_v2, (), dict(bytes=bytes, message_type=message_type,
                                    field_names=field_names,
                                    output_types=output_types,
                                    descriptor_source=descriptor_source,
                                    message_format=message_format,
                                    sanitize=sanitize, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("message_type", _op.get_attr("message_type"), "field_names",
              _op.get_attr("field_names"), "output_types",
              _op.get_attr("output_types"), "descriptor_source",
              _op.get_attr("descriptor_source"), "message_format",
              _op.get_attr("message_format"), "sanitize",
              _op._get_attr_bool("sanitize"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DecodeProtoV2", _inputs_flat, _attrs, _result)
  _result = _result[:1] + [_result[1:]]
  _result = _DecodeProtoV2Output._make(_result)
  return _result

DecodeProtoV2 = tf_export("raw_ops.DecodeProtoV2")(_ops.to_raw_op(decode_proto_v2))
_dispatcher_for_decode_proto_v2 = decode_proto_v2._tf_type_based_dispatcher.Dispatch


def decode_proto_v2_eager_fallback(bytes, message_type, field_names, output_types, descriptor_source, message_format, sanitize, name, ctx):
  message_type = _execute.make_str(message_type, "message_type")
  if not isinstance(field_names, (list, tuple)):
    raise TypeError(
        "Expected list for 'field_names' argument to "
        "'decode_proto_v2' Op, not %r." % field_names)
  field_names = [_execute.make_str(_s, "field_names") for _s in field_names]
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'decode_proto_v2' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if descriptor_source is None:
    descriptor_source = "local://"
  descriptor_source = _execute.make_str(descriptor_source, "descriptor_source")
  if message_format is None:
    message_format = "binary"
  message_format = _execute.make_str(message_format, "message_format")
  if sanitize is None:
    sanitize = False
  sanitize = _execute.make_bool(sanitize, "sanitize")
  bytes = _ops.convert_to_tensor(bytes, _dtypes.string)
  _inputs_flat = [bytes]
  _attrs = ("message_type", message_type, "field_names", field_names,
  "output_types", output_types, "descriptor_source", descriptor_source,
  "message_format", message_format, "sanitize", sanitize)
  _result = _execute.execute(b"DecodeProtoV2", len(output_types) + 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DecodeProtoV2", _inputs_flat, _attrs, _result)
  _result = _result[:1] + [_result[1:]]
  _result = _DecodeProtoV2Output._make(_result)
  return _result

