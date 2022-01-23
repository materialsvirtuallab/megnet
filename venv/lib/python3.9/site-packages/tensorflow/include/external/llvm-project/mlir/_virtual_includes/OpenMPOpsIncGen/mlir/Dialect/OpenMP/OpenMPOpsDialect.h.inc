/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace omp {

class OpenMPDialect : public ::mlir::Dialect {
  explicit OpenMPDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context,
      ::mlir::TypeID::get<OpenMPDialect>()) {
    
    getContext()->getOrLoadDialect<::mlir::LLVM::LLVMDialect>();

    initialize();
  }

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~OpenMPDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("omp");
  }
};
} // namespace omp
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::OpenMPDialect)
