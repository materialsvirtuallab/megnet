// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_LOOKUP_OPS_INTERNAL_H_
#define TENSORFLOW_CC_OPS_LOOKUP_OPS_INTERNAL_H_

// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace ops {
namespace internal {
// NOTE: This namespace has internal TensorFlow details that
// are not part of TensorFlow's public API.

/// @defgroup lookup_ops_internal Lookup Ops Internal
/// @{

/// Creates a uninitialized anonymous hash table.
///
/// This op creates a new anonymous hash table (as a resource) everytime
/// it is executed, with the specified dtype of its keys and values,
/// returning the resource handle.  Before using the table you will have
/// to initialize it.  After initialization the table will be
/// immutable. The table is anonymous in the sense that it can only be
/// accessed by the returned resource handle (e.g. it cannot be looked up
/// by a name in a resource manager). The table will be automatically
/// deleted when all resource handles pointing to it are gone.
///
/// Args:
/// * scope: A Scope object
/// * key_dtype: Type of the table keys.
/// * value_dtype: Type of the table values.
///
/// Returns:
/// * `Output`: The resource handle to the newly created hash-table resource.
class AnonymousHashTable {
 public:
  AnonymousHashTable(const ::tensorflow::Scope& scope, DataType key_dtype,
                   DataType value_dtype);
  operator ::tensorflow::Output() const { return table_handle; }
  operator ::tensorflow::Input() const { return table_handle; }
  ::tensorflow::Node* node() const { return table_handle.node(); }

  Operation operation;
  ::tensorflow::Output table_handle;
};

/// Removes keys and its associated values from a table.
///
/// The tensor `keys` must of the same type as the keys of the table. Keys not
/// already in the table are silently ignored.
///
/// Args:
/// * scope: A Scope object
/// * table_handle: Handle to the table.
/// * keys: Any shape.  Keys of the elements to remove.
///
/// Returns:
/// * the created `Operation`
class LookupTableRemove {
 public:
  LookupTableRemove(const ::tensorflow::Scope& scope, ::tensorflow::Input
                  table_handle, ::tensorflow::Input keys);
  operator ::tensorflow::Operation() const { return operation; }

  Operation operation;
};

}  // namespace internal
}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_LOOKUP_OPS_INTERNAL_H_
