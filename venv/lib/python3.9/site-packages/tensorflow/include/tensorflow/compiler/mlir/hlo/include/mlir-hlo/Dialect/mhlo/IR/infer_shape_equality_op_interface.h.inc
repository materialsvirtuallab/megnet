/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

class InferShapeEqualityOpInterface;
namespace detail {
struct InferShapeEqualityOpInterfaceInterfaceTraits {
  struct Concept {
    bool (*inferInputsShapeEquality)(const Concept *impl, ::mlir::Operation *, int, int);
    bool (*inferOutputsShapeEquality)(const Concept *impl, ::mlir::Operation *, int, int);
    bool (*inferInputOutputShapeEquality)(const Concept *impl, ::mlir::Operation *, int, int);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = InferShapeEqualityOpInterface;
    Model() : Concept{inferInputsShapeEquality, inferOutputsShapeEquality, inferInputOutputShapeEquality} {}

    static inline bool inferInputsShapeEquality(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, int lhs, int rhs);
    static inline bool inferOutputsShapeEquality(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, int lhs, int rhs);
    static inline bool inferInputOutputShapeEquality(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, int input, int output);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = InferShapeEqualityOpInterface;
    FallbackModel() : Concept{inferInputsShapeEquality, inferOutputsShapeEquality, inferInputOutputShapeEquality} {}

    static inline bool inferInputsShapeEquality(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, int lhs, int rhs);
    static inline bool inferOutputsShapeEquality(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, int lhs, int rhs);
    static inline bool inferInputOutputShapeEquality(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, int input, int output);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    bool inferInputsShapeEquality(::mlir::Operation *tablegen_opaque_val, int lhs, int rhs) const;
    bool inferOutputsShapeEquality(::mlir::Operation *tablegen_opaque_val, int lhs, int rhs) const;
    bool inferInputOutputShapeEquality(::mlir::Operation *tablegen_opaque_val, int input, int output) const;
  };
};template <typename ConcreteOp>
struct InferShapeEqualityOpInterfaceTrait;

} // end namespace detail
class InferShapeEqualityOpInterface : public ::mlir::OpInterface<InferShapeEqualityOpInterface, detail::InferShapeEqualityOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<InferShapeEqualityOpInterface, detail::InferShapeEqualityOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::InferShapeEqualityOpInterfaceTrait<ConcreteOp> {};
  bool inferInputsShapeEquality(int lhs, int rhs);
  bool inferOutputsShapeEquality(int lhs, int rhs);
  bool inferInputOutputShapeEquality(int input, int output);

    // Returns whether the given values have the same static shape.
    static bool inferShapeEquality(Value first, Value second) {
      // If both lhs and rhs have static shapes, check them directly.
      auto first_ty = first.getType().dyn_cast<RankedTensorType>();
      auto second_ty = second.getType().dyn_cast<RankedTensorType>();
      if (!first_ty || !first_ty.hasStaticShape() ||
          !second_ty || !second_ty.hasStaticShape() ||
          first_ty.getRank() != second_ty.getRank()) {
        return false;
      }
      return first_ty.getShape() == second_ty.getShape();
    }
  
};
namespace detail {
  template <typename ConcreteOp>
  struct InferShapeEqualityOpInterfaceTrait : public ::mlir::OpInterface<InferShapeEqualityOpInterface, detail::InferShapeEqualityOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    bool inferInputsShapeEquality(int lhs, int rhs) {
      /// Return whether two inputs have the same shape.
        Operation *op = this->getOperation();
        assert(lhs >= 0 && rhs >= 0);
        if (lhs == rhs) return true;
        return InferShapeEqualityOpInterface::inferShapeEquality(op->getOperand(lhs), op->getOperand(rhs));
    }
    bool inferOutputsShapeEquality(int lhs, int rhs) {
      /// Return whether two outputs have the same shape.
        Operation *op = this->getOperation();
        assert(lhs >= 0 && rhs >= 0);
        if (lhs == rhs) return true;
        return InferShapeEqualityOpInterface::inferShapeEquality(op->getResult(lhs), op->getResult(rhs));
    }
    bool inferInputOutputShapeEquality(int input, int output) {
      /// Return whether the input and the output have the same shape.
        Operation *op = this->getOperation();
        assert(input >= 0 && output >= 0);
        return InferShapeEqualityOpInterface::inferShapeEquality(op->getOperand(input), op->getResult(output));
    }
  };
}// namespace detail
template<typename ConcreteOp>
bool detail::InferShapeEqualityOpInterfaceInterfaceTraits::Model<ConcreteOp>::inferInputsShapeEquality(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, int lhs, int rhs) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).inferInputsShapeEquality(lhs, rhs);
}
template<typename ConcreteOp>
bool detail::InferShapeEqualityOpInterfaceInterfaceTraits::Model<ConcreteOp>::inferOutputsShapeEquality(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, int lhs, int rhs) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).inferOutputsShapeEquality(lhs, rhs);
}
template<typename ConcreteOp>
bool detail::InferShapeEqualityOpInterfaceInterfaceTraits::Model<ConcreteOp>::inferInputOutputShapeEquality(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, int input, int output) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).inferInputOutputShapeEquality(input, output);
}
template<typename ConcreteOp>
bool detail::InferShapeEqualityOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::inferInputsShapeEquality(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, int lhs, int rhs) {
  return static_cast<const ConcreteOp *>(impl)->inferInputsShapeEquality(tablegen_opaque_val, lhs, rhs);
}
template<typename ConcreteOp>
bool detail::InferShapeEqualityOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::inferOutputsShapeEquality(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, int lhs, int rhs) {
  return static_cast<const ConcreteOp *>(impl)->inferOutputsShapeEquality(tablegen_opaque_val, lhs, rhs);
}
template<typename ConcreteOp>
bool detail::InferShapeEqualityOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::inferInputOutputShapeEquality(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, int input, int output) {
  return static_cast<const ConcreteOp *>(impl)->inferInputOutputShapeEquality(tablegen_opaque_val, input, output);
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::InferShapeEqualityOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::inferInputsShapeEquality(::mlir::Operation *tablegen_opaque_val, int lhs, int rhs) const {
/// Return whether two inputs have the same shape.
        Operation *op = this->getOperation();
        assert(lhs >= 0 && rhs >= 0);
        if (lhs == rhs) return true;
        return InferShapeEqualityOpInterface::inferShapeEquality(op->getOperand(lhs), op->getOperand(rhs));
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::InferShapeEqualityOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::inferOutputsShapeEquality(::mlir::Operation *tablegen_opaque_val, int lhs, int rhs) const {
/// Return whether two outputs have the same shape.
        Operation *op = this->getOperation();
        assert(lhs >= 0 && rhs >= 0);
        if (lhs == rhs) return true;
        return InferShapeEqualityOpInterface::inferShapeEquality(op->getResult(lhs), op->getResult(rhs));
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::InferShapeEqualityOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::inferInputOutputShapeEquality(::mlir::Operation *tablegen_opaque_val, int input, int output) const {
/// Return whether the input and the output have the same shape.
        Operation *op = this->getOperation();
        assert(input >= 0 && output >= 0);
        return InferShapeEqualityOpInterface::inferShapeEquality(op->getOperand(input), op->getResult(output));
}
