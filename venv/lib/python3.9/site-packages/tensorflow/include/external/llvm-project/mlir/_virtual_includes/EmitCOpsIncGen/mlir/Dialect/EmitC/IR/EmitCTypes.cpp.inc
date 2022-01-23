/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_LIST
#undef GET_TYPEDEF_LIST

::mlir::emitc::OpaqueType

#endif  // GET_TYPEDEF_LIST

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES


static ::mlir::OptionalParseResult generatedTypeParser(::mlir::MLIRContext *context,
                                      ::mlir::DialectAsmParser &parser,
                                      ::llvm::StringRef mnemonic,
                                      ::mlir::Type &value) {
  if (mnemonic == ::mlir::emitc::OpaqueType::getMnemonic()) { 
    value = ::mlir::emitc::OpaqueType::parse(context, parser);
    return ::mlir::success(!!value);
  }
  return {};
}


static ::mlir::LogicalResult generatedTypePrinter(
                         ::mlir::Type def, ::mlir::DialectAsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Type, ::mlir::LogicalResult>(def)
    .Case<::mlir::emitc::OpaqueType>([&](::mlir::emitc::OpaqueType t) {
      t.print(printer);
      return ::mlir::success();
    })
    .Default([](::mlir::Type) { return ::mlir::failure(); });
}

namespace mlir {
namespace emitc {

namespace detail {
  struct OpaqueTypeStorage : public ::mlir::TypeStorage {
    OpaqueTypeStorage (::llvm::StringRef value)
      : value(value) { }

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
    static OpaqueTypeStorage *construct(::mlir::TypeStorageAllocator &allocator,
                          const KeyTy &tblgenKey) {
      auto value = std::get<0>(tblgenKey);
      value = allocator.copyInto(value);

      return new (allocator.allocate<OpaqueTypeStorage>())
          OpaqueTypeStorage(value);
    }
      ::llvm::StringRef value;
  };
} // namespace detail
OpaqueType OpaqueType::get(::mlir::MLIRContext *context, ::llvm::StringRef value) {
  return Base::get(context, value);
}
::llvm::StringRef OpaqueType::getValue() const { return getImpl()->value; }
} // namespace emitc
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::emitc::OpaqueType)

#endif  // GET_TYPEDEF_CLASSES

