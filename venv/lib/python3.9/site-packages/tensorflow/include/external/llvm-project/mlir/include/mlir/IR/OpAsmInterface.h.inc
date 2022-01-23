/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class OpAsmOpInterface;
namespace detail {
struct OpAsmOpInterfaceInterfaceTraits {
  struct Concept {
    void (*getAsmResultNames)(const Concept *impl, ::mlir::Operation *, ::mlir::OpAsmSetValueNameFn);
    StringRef (*getDefaultDialect)();
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::OpAsmOpInterface;
    Model() : Concept{getAsmResultNames, getDefaultDialect} {}

    static inline void getAsmResultNames(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpAsmSetValueNameFn setNameFn);
    static inline StringRef getDefaultDialect();
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::OpAsmOpInterface;
    FallbackModel() : Concept{getAsmResultNames, getDefaultDialect} {}

    static inline void getAsmResultNames(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpAsmSetValueNameFn setNameFn);
    static inline StringRef getDefaultDialect();
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    void getAsmResultNames(::mlir::Operation *tablegen_opaque_val, ::mlir::OpAsmSetValueNameFn setNameFn) const;
    static StringRef getDefaultDialect();
  };
};template <typename ConcreteOp>
struct OpAsmOpInterfaceTrait;

} // end namespace detail
class OpAsmOpInterface : public ::mlir::OpInterface<OpAsmOpInterface, detail::OpAsmOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<OpAsmOpInterface, detail::OpAsmOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::OpAsmOpInterfaceTrait<ConcreteOp> {};
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  StringRef getDefaultDialect();
};
namespace detail {
  template <typename ConcreteOp>
  struct OpAsmOpInterfaceTrait : public ::mlir::OpInterface<OpAsmOpInterface, detail::OpAsmOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
      ;
    }
    static StringRef getDefaultDialect() {
      return "";
    }
  };
}// namespace detail
template<typename ConcreteOp>
void detail::OpAsmOpInterfaceInterfaceTraits::Model<ConcreteOp>::getAsmResultNames(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpAsmSetValueNameFn setNameFn) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAsmResultNames(setNameFn);
}
template<typename ConcreteOp>
StringRef detail::OpAsmOpInterfaceInterfaceTraits::Model<ConcreteOp>::getDefaultDialect() {
  return ConcreteOp::getDefaultDialect();
}
template<typename ConcreteOp>
void detail::OpAsmOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getAsmResultNames(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpAsmSetValueNameFn setNameFn) {
  return static_cast<const ConcreteOp *>(impl)->getAsmResultNames(tablegen_opaque_val, setNameFn);
}
template<typename ConcreteOp>
StringRef detail::OpAsmOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getDefaultDialect() {
  return ConcreteOp::getDefaultDialect();
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::OpAsmOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getAsmResultNames(::mlir::Operation *tablegen_opaque_val, ::mlir::OpAsmSetValueNameFn setNameFn) const {
;
}
template<typename ConcreteModel, typename ConcreteOp>
StringRef detail::OpAsmOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getDefaultDialect() {
return "";
}
} // namespace mlir
