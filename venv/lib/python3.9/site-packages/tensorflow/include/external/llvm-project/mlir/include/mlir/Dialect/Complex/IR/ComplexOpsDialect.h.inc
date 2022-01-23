/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace complex {

class ComplexDialect : public ::mlir::Dialect {
  explicit ComplexDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context,
      ::mlir::TypeID::get<ComplexDialect>()) {
    
    initialize();
  }

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~ComplexDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("complex");
  }
};
} // namespace complex
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::complex::ComplexDialect)
