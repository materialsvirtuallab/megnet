/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace LLVM {
class FastmathFlagsInterface;
namespace detail {
struct FastmathFlagsInterfaceInterfaceTraits {
  struct Concept {
    ::mlir::LLVM::FastmathFlags (*fastmathFlags)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::LLVM::FastmathFlagsInterface;
    Model() : Concept{fastmathFlags} {}

    static inline ::mlir::LLVM::FastmathFlags fastmathFlags(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::LLVM::FastmathFlagsInterface;
    FallbackModel() : Concept{fastmathFlags} {}

    static inline ::mlir::LLVM::FastmathFlags fastmathFlags(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
  };
};template <typename ConcreteOp>
struct FastmathFlagsInterfaceTrait;

} // end namespace detail
class FastmathFlagsInterface : public ::mlir::OpInterface<FastmathFlagsInterface, detail::FastmathFlagsInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<FastmathFlagsInterface, detail::FastmathFlagsInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::FastmathFlagsInterfaceTrait<ConcreteOp> {};
  ::mlir::LLVM::FastmathFlags fastmathFlags();
};
namespace detail {
  template <typename ConcreteOp>
  struct FastmathFlagsInterfaceTrait : public ::mlir::OpInterface<FastmathFlagsInterface, detail::FastmathFlagsInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
}// namespace detail
template<typename ConcreteOp>
::mlir::LLVM::FastmathFlags detail::FastmathFlagsInterfaceInterfaceTraits::Model<ConcreteOp>::fastmathFlags(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).fastmathFlags();
}
template<typename ConcreteOp>
::mlir::LLVM::FastmathFlags detail::FastmathFlagsInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::fastmathFlags(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->fastmathFlags(tablegen_opaque_val);
}
} // namespace LLVM
} // namespace mlir
