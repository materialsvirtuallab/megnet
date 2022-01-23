/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES


namespace mlir {
class DialectAsmParser;
class DialectAsmPrinter;
} // namespace mlir
namespace mlir {
namespace emitc {
  class OpaqueAttr;

  namespace detail {
    struct OpaqueAttrStorage;
  } // end namespace detail
  class OpaqueAttr : public ::mlir::Attribute::AttrBase<OpaqueAttr, ::mlir::Attribute,
                                         detail::OpaqueAttrStorage> {
  public:
    /// Inherit some necessary constructors from 'AttrBase'.
    using Base::Base;

    static OpaqueAttr get(::mlir::MLIRContext *context, ::llvm::StringRef value);
    static constexpr ::llvm::StringLiteral getMnemonic() {
      return ::llvm::StringLiteral("opaque");
    }

    static ::mlir::Attribute parse(::mlir::MLIRContext *context,
                             ::mlir::DialectAsmParser &parser, ::mlir::Type type);
    void print(::mlir::DialectAsmPrinter &printer) const;
    ::llvm::StringRef getValue() const;
  };
} // namespace emitc
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::emitc::OpaqueAttr)

#endif  // GET_ATTRDEF_CLASSES

