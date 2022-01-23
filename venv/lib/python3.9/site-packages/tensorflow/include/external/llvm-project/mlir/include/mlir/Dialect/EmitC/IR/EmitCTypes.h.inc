/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES


namespace mlir {
class DialectAsmParser;
class DialectAsmPrinter;
} // namespace mlir
namespace mlir {
namespace emitc {
  class OpaqueType;

  namespace detail {
    struct OpaqueTypeStorage;
  } // end namespace detail
  class OpaqueType : public ::mlir::Type::TypeBase<OpaqueType, ::mlir::Type,
                                         detail::OpaqueTypeStorage> {
  public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

    static OpaqueType get(::mlir::MLIRContext *context, ::llvm::StringRef value);
    static constexpr ::llvm::StringLiteral getMnemonic() {
      return ::llvm::StringLiteral("opaque");
    }

    static ::mlir::Type parse(::mlir::MLIRContext *context,
                             ::mlir::DialectAsmParser &parser);
    void print(::mlir::DialectAsmPrinter &printer) const;
    ::llvm::StringRef getValue() const;
  };
} // namespace emitc
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::emitc::OpaqueType)

#endif  // GET_TYPEDEF_CLASSES

