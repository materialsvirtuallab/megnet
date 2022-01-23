/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class TilingInterface;
namespace detail {
struct TilingInterfaceInterfaceTraits {
  struct Concept {
    SmallVector<Value> (*getDestinationOperands)(const Concept *impl, ::mlir::Operation *, OpBuilder &);
    SmallVector<StringRef> (*getLoopIteratorTypes)(const Concept *impl, ::mlir::Operation *);
    SmallVector<Range> (*getLoopBounds)(const Concept *impl, ::mlir::Operation *, OpBuilder &);
    Operation *(*getTiledImplementation)(const Concept *impl, ::mlir::Operation *, OpBuilder &, ValueRange , ArrayRef<OpFoldResult> , ArrayRef<OpFoldResult> );
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::TilingInterface;
    Model() : Concept{getDestinationOperands, getLoopIteratorTypes, getLoopBounds, getTiledImplementation} {}

    static inline SmallVector<Value> getDestinationOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b);
    static inline SmallVector<StringRef> getLoopIteratorTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<Range> getLoopBounds(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b);
    static inline Operation *getTiledImplementation(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, ValueRange  dest, ArrayRef<OpFoldResult>  offsets, ArrayRef<OpFoldResult>  sizes);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::TilingInterface;
    FallbackModel() : Concept{getDestinationOperands, getLoopIteratorTypes, getLoopBounds, getTiledImplementation} {}

    static inline SmallVector<Value> getDestinationOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b);
    static inline SmallVector<StringRef> getLoopIteratorTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<Range> getLoopBounds(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b);
    static inline Operation *getTiledImplementation(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, ValueRange  dest, ArrayRef<OpFoldResult>  offsets, ArrayRef<OpFoldResult>  sizes);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    SmallVector<Value> getDestinationOperands(::mlir::Operation *tablegen_opaque_val, OpBuilder &b) const;
    Operation *getTiledImplementation(::mlir::Operation *tablegen_opaque_val, OpBuilder &b, ValueRange dest, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes) const;
  };
};template <typename ConcreteOp>
struct TilingInterfaceTrait;

} // end namespace detail
class TilingInterface : public ::mlir::OpInterface<TilingInterface, detail::TilingInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<TilingInterface, detail::TilingInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::TilingInterfaceTrait<ConcreteOp> {};
  SmallVector<Value> getDestinationOperands(OpBuilder & b);
  SmallVector<StringRef> getLoopIteratorTypes();
  SmallVector<Range> getLoopBounds(OpBuilder & b);
  Operation *getTiledImplementation(OpBuilder & b, ValueRange  dest, ArrayRef<OpFoldResult>  offsets, ArrayRef<OpFoldResult>  sizes);
};
namespace detail {
  template <typename ConcreteOp>
  struct TilingInterfaceTrait : public ::mlir::OpInterface<TilingInterface, detail::TilingInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    SmallVector<Value> getDestinationOperands(OpBuilder & b) {
      return ValueRange{};
    }
    Operation *getTiledImplementation(OpBuilder & b, ValueRange  dest, ArrayRef<OpFoldResult>  offsets, ArrayRef<OpFoldResult>  sizes) {
      return nullptr;
    }
  };
}// namespace detail
template<typename ConcreteOp>
SmallVector<Value> detail::TilingInterfaceInterfaceTraits::Model<ConcreteOp>::getDestinationOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getDestinationOperands(b);
}
template<typename ConcreteOp>
SmallVector<StringRef> detail::TilingInterfaceInterfaceTraits::Model<ConcreteOp>::getLoopIteratorTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getLoopIteratorTypes();
}
template<typename ConcreteOp>
SmallVector<Range> detail::TilingInterfaceInterfaceTraits::Model<ConcreteOp>::getLoopBounds(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getLoopBounds(b);
}
template<typename ConcreteOp>
Operation *detail::TilingInterfaceInterfaceTraits::Model<ConcreteOp>::getTiledImplementation(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, ValueRange  dest, ArrayRef<OpFoldResult>  offsets, ArrayRef<OpFoldResult>  sizes) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getTiledImplementation(b, dest, offsets, sizes);
}
template<typename ConcreteOp>
SmallVector<Value> detail::TilingInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getDestinationOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b) {
  return static_cast<const ConcreteOp *>(impl)->getDestinationOperands(tablegen_opaque_val, b);
}
template<typename ConcreteOp>
SmallVector<StringRef> detail::TilingInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getLoopIteratorTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getLoopIteratorTypes(tablegen_opaque_val);
}
template<typename ConcreteOp>
SmallVector<Range> detail::TilingInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getLoopBounds(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b) {
  return static_cast<const ConcreteOp *>(impl)->getLoopBounds(tablegen_opaque_val, b);
}
template<typename ConcreteOp>
Operation *detail::TilingInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getTiledImplementation(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, ValueRange  dest, ArrayRef<OpFoldResult>  offsets, ArrayRef<OpFoldResult>  sizes) {
  return static_cast<const ConcreteOp *>(impl)->getTiledImplementation(tablegen_opaque_val, b, dest, offsets, sizes);
}
template<typename ConcreteModel, typename ConcreteOp>
SmallVector<Value> detail::TilingInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getDestinationOperands(::mlir::Operation *tablegen_opaque_val, OpBuilder &b) const {
return ValueRange{};
}
template<typename ConcreteModel, typename ConcreteOp>
Operation *detail::TilingInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getTiledImplementation(::mlir::Operation *tablegen_opaque_val, OpBuilder &b, ValueRange dest, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes) const {
return nullptr;
}
} // namespace mlir
