/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace lmhlo {
class LmhloOp;
namespace detail {
struct LmhloOpInterfaceTraits {
  struct Concept {
    Value (*getResultBuffer)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::lmhlo::LmhloOp;
    Model() : Concept{getResultBuffer} {}

    static inline Value getResultBuffer(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::lmhlo::LmhloOp;
    FallbackModel() : Concept{getResultBuffer} {}

    static inline Value getResultBuffer(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    Value getResultBuffer(::mlir::Operation *tablegen_opaque_val) const;
  };
};template <typename ConcreteOp>
struct LmhloOpTrait;

} // end namespace detail
class LmhloOp : public ::mlir::OpInterface<LmhloOp, detail::LmhloOpInterfaceTraits> {
public:
  using ::mlir::OpInterface<LmhloOp, detail::LmhloOpInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::LmhloOpTrait<ConcreteOp> {};
  Value getResultBuffer();
};
namespace detail {
  template <typename ConcreteOp>
  struct LmhloOpTrait : public ::mlir::OpInterface<LmhloOp, detail::LmhloOpInterfaceTraits>::Trait<ConcreteOp> {
    Value getResultBuffer() {
      /// By default, the result buffer is the last operand 
        unsigned num_operands = this->getOperation()->getNumOperands();
        if (num_operands > 1) {
          return this->getOperation()->getOperand(num_operands - 1);
        }
        return nullptr;
    }
  };
}// namespace detail
template<typename ConcreteOp>
Value detail::LmhloOpInterfaceTraits::Model<ConcreteOp>::getResultBuffer(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getResultBuffer();
}
template<typename ConcreteOp>
Value detail::LmhloOpInterfaceTraits::FallbackModel<ConcreteOp>::getResultBuffer(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getResultBuffer(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteOp>
Value detail::LmhloOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getResultBuffer(::mlir::Operation *tablegen_opaque_val) const {
/// By default, the result buffer is the last operand 
        unsigned num_operands = this->getOperation()->getNumOperands();
        if (num_operands > 1) {
          return this->getOperation()->getOperand(num_operands - 1);
        }
        return nullptr;
}
} // namespace lmhlo
} // namespace mlir
