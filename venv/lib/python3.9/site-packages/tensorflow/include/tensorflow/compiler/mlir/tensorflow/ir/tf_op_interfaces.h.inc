/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

class FoldOperandsTransposeInterface;
namespace detail {
struct FoldOperandsTransposeInterfaceInterfaceTraits {
  struct Concept {
    SmallVector<unsigned, 4> (*GetLayoutDependentArgs)(const Concept *impl, ::mlir::Operation *);
    SmallVector<unsigned, 4> (*GetLayoutDependentResults)(const Concept *impl, ::mlir::Operation *);
    LogicalResult (*FoldOperandsPermutation)(const Concept *impl, ::mlir::Operation *, ArrayRef<int64_t>);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = FoldOperandsTransposeInterface;
    Model() : Concept{GetLayoutDependentArgs, GetLayoutDependentResults, FoldOperandsPermutation} {}

    static inline SmallVector<unsigned, 4> GetLayoutDependentArgs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<unsigned, 4> GetLayoutDependentResults(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline LogicalResult FoldOperandsPermutation(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ArrayRef<int64_t> permutation);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = FoldOperandsTransposeInterface;
    FallbackModel() : Concept{GetLayoutDependentArgs, GetLayoutDependentResults, FoldOperandsPermutation} {}

    static inline SmallVector<unsigned, 4> GetLayoutDependentArgs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<unsigned, 4> GetLayoutDependentResults(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline LogicalResult FoldOperandsPermutation(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ArrayRef<int64_t> permutation);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
  };
};template <typename ConcreteOp>
struct FoldOperandsTransposeInterfaceTrait;

} // end namespace detail
class FoldOperandsTransposeInterface : public ::mlir::OpInterface<FoldOperandsTransposeInterface, detail::FoldOperandsTransposeInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<FoldOperandsTransposeInterface, detail::FoldOperandsTransposeInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::FoldOperandsTransposeInterfaceTrait<ConcreteOp> {};
  SmallVector<unsigned, 4> GetLayoutDependentArgs();
  SmallVector<unsigned, 4> GetLayoutDependentResults();
  LogicalResult FoldOperandsPermutation(ArrayRef<int64_t> permutation);
};
namespace detail {
  template <typename ConcreteOp>
  struct FoldOperandsTransposeInterfaceTrait : public ::mlir::OpInterface<FoldOperandsTransposeInterface, detail::FoldOperandsTransposeInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    static ::mlir::LogicalResult verifyTrait(::mlir::Operation *op) {
      return VerifyFoldOperandsTransposeInterface(op);
    }
  };
}// namespace detail
template<typename ConcreteOp>
SmallVector<unsigned, 4> detail::FoldOperandsTransposeInterfaceInterfaceTraits::Model<ConcreteOp>::GetLayoutDependentArgs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).GetLayoutDependentArgs();
}
template<typename ConcreteOp>
SmallVector<unsigned, 4> detail::FoldOperandsTransposeInterfaceInterfaceTraits::Model<ConcreteOp>::GetLayoutDependentResults(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).GetLayoutDependentResults();
}
template<typename ConcreteOp>
LogicalResult detail::FoldOperandsTransposeInterfaceInterfaceTraits::Model<ConcreteOp>::FoldOperandsPermutation(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ArrayRef<int64_t> permutation) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).FoldOperandsPermutation(permutation);
}
template<typename ConcreteOp>
SmallVector<unsigned, 4> detail::FoldOperandsTransposeInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::GetLayoutDependentArgs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->GetLayoutDependentArgs(tablegen_opaque_val);
}
template<typename ConcreteOp>
SmallVector<unsigned, 4> detail::FoldOperandsTransposeInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::GetLayoutDependentResults(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->GetLayoutDependentResults(tablegen_opaque_val);
}
template<typename ConcreteOp>
LogicalResult detail::FoldOperandsTransposeInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::FoldOperandsPermutation(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ArrayRef<int64_t> permutation) {
  return static_cast<const ConcreteOp *>(impl)->FoldOperandsPermutation(tablegen_opaque_val, permutation);
}
class LayoutSensitiveInterface;
namespace detail {
struct LayoutSensitiveInterfaceInterfaceTraits {
  struct Concept {
    StringRef (*data_format)(const Concept *impl, ::mlir::Operation *);
    SmallVector<unsigned, 4> (*GetLayoutDependentArgs)(const Concept *impl, ::mlir::Operation *);
    SmallVector<unsigned, 4> (*GetLayoutDependentResults)(const Concept *impl, ::mlir::Operation *);
    StringRef (*GetOptimalLayout)(const Concept *impl, ::mlir::Operation *, const RuntimeDevices&);
    LogicalResult (*UpdateDataFormat)(const Concept *impl, ::mlir::Operation *, StringRef);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = LayoutSensitiveInterface;
    Model() : Concept{data_format, GetLayoutDependentArgs, GetLayoutDependentResults, GetOptimalLayout, UpdateDataFormat} {}

    static inline StringRef data_format(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<unsigned, 4> GetLayoutDependentArgs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<unsigned, 4> GetLayoutDependentResults(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline StringRef GetOptimalLayout(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const RuntimeDevices& devices);
    static inline LogicalResult UpdateDataFormat(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, StringRef data_format);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = LayoutSensitiveInterface;
    FallbackModel() : Concept{data_format, GetLayoutDependentArgs, GetLayoutDependentResults, GetOptimalLayout, UpdateDataFormat} {}

    static inline StringRef data_format(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<unsigned, 4> GetLayoutDependentArgs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<unsigned, 4> GetLayoutDependentResults(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline StringRef GetOptimalLayout(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const RuntimeDevices& devices);
    static inline LogicalResult UpdateDataFormat(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, StringRef data_format);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
  };
};template <typename ConcreteOp>
struct LayoutSensitiveInterfaceTrait;

} // end namespace detail
class LayoutSensitiveInterface : public ::mlir::OpInterface<LayoutSensitiveInterface, detail::LayoutSensitiveInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<LayoutSensitiveInterface, detail::LayoutSensitiveInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::LayoutSensitiveInterfaceTrait<ConcreteOp> {};
  StringRef data_format();
  SmallVector<unsigned, 4> GetLayoutDependentArgs();
  SmallVector<unsigned, 4> GetLayoutDependentResults();
  StringRef GetOptimalLayout(const RuntimeDevices& devices);
  LogicalResult UpdateDataFormat(StringRef data_format);
};
namespace detail {
  template <typename ConcreteOp>
  struct LayoutSensitiveInterfaceTrait : public ::mlir::OpInterface<LayoutSensitiveInterface, detail::LayoutSensitiveInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    static ::mlir::LogicalResult verifyTrait(::mlir::Operation *op) {
      return VerifyLayoutSensitiveInterface(op);
    }
  };
}// namespace detail
template<typename ConcreteOp>
StringRef detail::LayoutSensitiveInterfaceInterfaceTraits::Model<ConcreteOp>::data_format(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).data_format();
}
template<typename ConcreteOp>
SmallVector<unsigned, 4> detail::LayoutSensitiveInterfaceInterfaceTraits::Model<ConcreteOp>::GetLayoutDependentArgs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).GetLayoutDependentArgs();
}
template<typename ConcreteOp>
SmallVector<unsigned, 4> detail::LayoutSensitiveInterfaceInterfaceTraits::Model<ConcreteOp>::GetLayoutDependentResults(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).GetLayoutDependentResults();
}
template<typename ConcreteOp>
StringRef detail::LayoutSensitiveInterfaceInterfaceTraits::Model<ConcreteOp>::GetOptimalLayout(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const RuntimeDevices& devices) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).GetOptimalLayout(devices);
}
template<typename ConcreteOp>
LogicalResult detail::LayoutSensitiveInterfaceInterfaceTraits::Model<ConcreteOp>::UpdateDataFormat(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, StringRef data_format) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).UpdateDataFormat(data_format);
}
template<typename ConcreteOp>
StringRef detail::LayoutSensitiveInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::data_format(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->data_format(tablegen_opaque_val);
}
template<typename ConcreteOp>
SmallVector<unsigned, 4> detail::LayoutSensitiveInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::GetLayoutDependentArgs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->GetLayoutDependentArgs(tablegen_opaque_val);
}
template<typename ConcreteOp>
SmallVector<unsigned, 4> detail::LayoutSensitiveInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::GetLayoutDependentResults(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->GetLayoutDependentResults(tablegen_opaque_val);
}
template<typename ConcreteOp>
StringRef detail::LayoutSensitiveInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::GetOptimalLayout(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const RuntimeDevices& devices) {
  return static_cast<const ConcreteOp *>(impl)->GetOptimalLayout(tablegen_opaque_val, devices);
}
template<typename ConcreteOp>
LogicalResult detail::LayoutSensitiveInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::UpdateDataFormat(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, StringRef data_format) {
  return static_cast<const ConcreteOp *>(impl)->UpdateDataFormat(tablegen_opaque_val, data_format);
}
class ResourceHandleAllocatorInterface;
namespace detail {
struct ResourceHandleAllocatorInterfaceInterfaceTraits {
  struct Concept {
    llvm::SmallVector<ResourceHandleValueAndId, 4> (*GetResourceHandleValueAndIdList)(const Concept *impl, ::mlir::Operation *, llvm::SmallDenseMap<ResourceHandle, int64_t>&, int64_t&);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ResourceHandleAllocatorInterface;
    Model() : Concept{GetResourceHandleValueAndIdList} {}

    static inline llvm::SmallVector<ResourceHandleValueAndId, 4> GetResourceHandleValueAndIdList(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, llvm::SmallDenseMap<ResourceHandle, int64_t>& resource_handle_id_map, int64_t& next_id);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ResourceHandleAllocatorInterface;
    FallbackModel() : Concept{GetResourceHandleValueAndIdList} {}

    static inline llvm::SmallVector<ResourceHandleValueAndId, 4> GetResourceHandleValueAndIdList(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, llvm::SmallDenseMap<ResourceHandle, int64_t>& resource_handle_id_map, int64_t& next_id);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
  };
};template <typename ConcreteOp>
struct ResourceHandleAllocatorInterfaceTrait;

} // end namespace detail
class ResourceHandleAllocatorInterface : public ::mlir::OpInterface<ResourceHandleAllocatorInterface, detail::ResourceHandleAllocatorInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<ResourceHandleAllocatorInterface, detail::ResourceHandleAllocatorInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::ResourceHandleAllocatorInterfaceTrait<ConcreteOp> {};
  llvm::SmallVector<ResourceHandleValueAndId, 4> GetResourceHandleValueAndIdList(llvm::SmallDenseMap<ResourceHandle, int64_t>& resource_handle_id_map, int64_t& next_id);
};
namespace detail {
  template <typename ConcreteOp>
  struct ResourceHandleAllocatorInterfaceTrait : public ::mlir::OpInterface<ResourceHandleAllocatorInterface, detail::ResourceHandleAllocatorInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
}// namespace detail
template<typename ConcreteOp>
llvm::SmallVector<ResourceHandleValueAndId, 4> detail::ResourceHandleAllocatorInterfaceInterfaceTraits::Model<ConcreteOp>::GetResourceHandleValueAndIdList(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, llvm::SmallDenseMap<ResourceHandle, int64_t>& resource_handle_id_map, int64_t& next_id) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).GetResourceHandleValueAndIdList(resource_handle_id_map, next_id);
}
template<typename ConcreteOp>
llvm::SmallVector<ResourceHandleValueAndId, 4> detail::ResourceHandleAllocatorInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::GetResourceHandleValueAndIdList(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, llvm::SmallDenseMap<ResourceHandle, int64_t>& resource_handle_id_map, int64_t& next_id) {
  return static_cast<const ConcreteOp *>(impl)->GetResourceHandleValueAndIdList(tablegen_opaque_val, resource_handle_id_map, next_id);
}
