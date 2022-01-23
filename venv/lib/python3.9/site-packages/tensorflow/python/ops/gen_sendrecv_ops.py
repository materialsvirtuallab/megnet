"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: sendrecv_ops.cc
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

def recv(tensor_type, tensor_name, send_device, send_device_incarnation, recv_device, client_terminated=False, name=None):
  r"""Receives the named tensor from send_device on recv_device.

  Args:
    tensor_type: A `tf.DType`.
    tensor_name: A `string`. The name of the tensor to receive.
    send_device: A `string`. The name of the device sending the tensor.
    send_device_incarnation: An `int`. The current incarnation of send_device.
    recv_device: A `string`. The name of the device receiving the tensor.
    client_terminated: An optional `bool`. Defaults to `False`.
      If set to true, this indicates that the node was added
      to the graph as a result of a client-side feed or fetch of Tensor data,
      in which case the corresponding send or recv is expected to be managed
      locally by the caller.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `tensor_type`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Recv", name, "tensor_type", tensor_type, "tensor_name",
        tensor_name, "send_device", send_device, "send_device_incarnation",
        send_device_incarnation, "recv_device", recv_device,
        "client_terminated", client_terminated)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return recv_eager_fallback(
          tensor_type=tensor_type, tensor_name=tensor_name,
          send_device=send_device,
          send_device_incarnation=send_device_incarnation,
          recv_device=recv_device, client_terminated=client_terminated,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  tensor_type = _execute.make_type(tensor_type, "tensor_type")
  tensor_name = _execute.make_str(tensor_name, "tensor_name")
  send_device = _execute.make_str(send_device, "send_device")
  send_device_incarnation = _execute.make_int(send_device_incarnation, "send_device_incarnation")
  recv_device = _execute.make_str(recv_device, "recv_device")
  if client_terminated is None:
    client_terminated = False
  client_terminated = _execute.make_bool(client_terminated, "client_terminated")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Recv", tensor_type=tensor_type, tensor_name=tensor_name,
                send_device=send_device,
                send_device_incarnation=send_device_incarnation,
                recv_device=recv_device, client_terminated=client_terminated,
                name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("tensor_type", _op._get_attr_type("tensor_type"), "tensor_name",
              _op.get_attr("tensor_name"), "send_device",
              _op.get_attr("send_device"), "send_device_incarnation",
              _op._get_attr_int("send_device_incarnation"), "recv_device",
              _op.get_attr("recv_device"), "client_terminated",
              _op._get_attr_bool("client_terminated"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Recv", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Recv = tf_export("raw_ops.Recv")(_ops.to_raw_op(recv))


def recv_eager_fallback(tensor_type, tensor_name, send_device, send_device_incarnation, recv_device, client_terminated, name, ctx):
  tensor_type = _execute.make_type(tensor_type, "tensor_type")
  tensor_name = _execute.make_str(tensor_name, "tensor_name")
  send_device = _execute.make_str(send_device, "send_device")
  send_device_incarnation = _execute.make_int(send_device_incarnation, "send_device_incarnation")
  recv_device = _execute.make_str(recv_device, "recv_device")
  if client_terminated is None:
    client_terminated = False
  client_terminated = _execute.make_bool(client_terminated, "client_terminated")
  _inputs_flat = []
  _attrs = ("tensor_type", tensor_type, "tensor_name", tensor_name,
  "send_device", send_device, "send_device_incarnation",
  send_device_incarnation, "recv_device", recv_device, "client_terminated",
  client_terminated)
  _result = _execute.execute(b"Recv", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Recv", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def send(tensor, tensor_name, send_device, send_device_incarnation, recv_device, client_terminated=False, name=None):
  r"""Sends the named tensor from send_device to recv_device.

  Args:
    tensor: A `Tensor`. The tensor to send.
    tensor_name: A `string`. The name of the tensor to send.
    send_device: A `string`. The name of the device sending the tensor.
    send_device_incarnation: An `int`. The current incarnation of send_device.
    recv_device: A `string`. The name of the device receiving the tensor.
    client_terminated: An optional `bool`. Defaults to `False`.
      If set to true, this indicates that the node was added
      to the graph as a result of a client-side feed or fetch of Tensor data,
      in which case the corresponding send or recv is expected to be managed
      locally by the caller.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Send", name, tensor, "tensor_name", tensor_name, "send_device",
        send_device, "send_device_incarnation", send_device_incarnation,
        "recv_device", recv_device, "client_terminated", client_terminated)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return send_eager_fallback(
          tensor, tensor_name=tensor_name, send_device=send_device,
          send_device_incarnation=send_device_incarnation,
          recv_device=recv_device, client_terminated=client_terminated,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  tensor_name = _execute.make_str(tensor_name, "tensor_name")
  send_device = _execute.make_str(send_device, "send_device")
  send_device_incarnation = _execute.make_int(send_device_incarnation, "send_device_incarnation")
  recv_device = _execute.make_str(recv_device, "recv_device")
  if client_terminated is None:
    client_terminated = False
  client_terminated = _execute.make_bool(client_terminated, "client_terminated")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Send", tensor=tensor, tensor_name=tensor_name,
                send_device=send_device,
                send_device_incarnation=send_device_incarnation,
                recv_device=recv_device, client_terminated=client_terminated,
                name=name)
  return _op
Send = tf_export("raw_ops.Send")(_ops.to_raw_op(send))


def send_eager_fallback(tensor, tensor_name, send_device, send_device_incarnation, recv_device, client_terminated, name, ctx):
  tensor_name = _execute.make_str(tensor_name, "tensor_name")
  send_device = _execute.make_str(send_device, "send_device")
  send_device_incarnation = _execute.make_int(send_device_incarnation, "send_device_incarnation")
  recv_device = _execute.make_str(recv_device, "recv_device")
  if client_terminated is None:
    client_terminated = False
  client_terminated = _execute.make_bool(client_terminated, "client_terminated")
  _attr_T, (tensor,) = _execute.args_to_matching_eager([tensor], ctx, [])
  _inputs_flat = [tensor]
  _attrs = ("T", _attr_T, "tensor_name", tensor_name, "send_device",
  send_device, "send_device_incarnation", send_device_incarnation,
  "recv_device", recv_device, "client_terminated", client_terminated)
  _result = _execute.execute(b"Send", 0, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  _result = None
  return _result

