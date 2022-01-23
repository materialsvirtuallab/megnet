/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::emitc::OpaqueAttr

#endif  // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES


static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::MLIRContext *context,
                                      ::mlir::DialectAsmParser &parser,
                                      ::llvm::StringRef mnemonic, ::mlir::Type type,
                                      ::mlir::Attribute &value) {
  if (mnemonic == ::mlir::emitc::OpaqueAttr::getMnemonic()) { 
    value = ::mlir::emitc::OpaqueAttr::parse(context, parser, type);
    return ::mlir::success(!!value);
  }
  return {};
}


static ::mlir::LogicalResult generatedAttributePrinter(
                         ::mlir::Attribute def, ::mlir::DialectAsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::mlir::LogicalResult>(def)
    .Case<::mlir::emitc::OpaqueAttr>([&](::mlir::emitc::OpaqueAttr t) {
      t.print(printer);
      return ::mlir::success();
    })
    .Default([](::mlir::Attribute) { return ::mlir::failure(); });
}

namespace mlir {
namespace emitc {

namespace detail {
  struct OpaqueAttrStorage : public ::mlir::AttributeStorage {
    OpaqueAttrStorage (::llvm::StringRef value)
      : ::mlir::AttributeStorage(), value(value) { }

    /// The hash key is a tuple of the parameter types.
    using KeyTy = std::tuple<::llvm::StringRef>;
  bool operator==(const KeyTy &tblgenKey) const {
    if (!(value == std::get<0>(tblgenKey)))
      return false;
    return true;
  }
  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
      return ::llvm::hash_combine(std::get<0>(tblgenKey));
    }

    /// Define a construction method for creating a new instance of this
    /// storage.
    static OpaqueAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator,
                          const KeyTy &tblgenKey) {
      auto value = std::get<0>(tblgenKey);
      value = allocator.copyInto(value);

      return new (allocator.allocate<OpaqueAttrStorage>())
          OpaqueAttrStorage(value);
    }
      ::llvm::StringRef value;
  };
} // namespace detail
OpaqueAttr OpaqueAttr::get(::mlir::MLIRContext *context, ::llvm::StringRef value) {
  return Base::get(context, value);
}
::llvm::StringRef OpaqueAttr::getValue() const { return getImpl()->value; }
} // namespace emitc
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::emitc::OpaqueAttr)

#endif  // GET_ATTRDEF_CLASSES

