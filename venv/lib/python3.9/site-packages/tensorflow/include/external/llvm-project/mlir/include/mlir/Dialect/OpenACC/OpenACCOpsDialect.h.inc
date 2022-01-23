/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace acc {

class OpenACCDialect : public ::mlir::Dialect {
  explicit OpenACCDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context,
      ::mlir::TypeID::get<OpenACCDialect>()) {
    
    initialize();
  }

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~OpenACCDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("acc");
  }
};
} // namespace acc
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::OpenACCDialect)
