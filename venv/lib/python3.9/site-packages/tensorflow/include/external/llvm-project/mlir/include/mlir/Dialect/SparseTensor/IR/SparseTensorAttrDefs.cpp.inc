/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::sparse_tensor::SparseTensorEncodingAttr

#endif  // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES


static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::MLIRContext *context,
                                      ::mlir::DialectAsmParser &parser,
                                      ::llvm::StringRef mnemonic, ::mlir::Type type,
                                      ::mlir::Attribute &value) {
  if (mnemonic == ::mlir::sparse_tensor::SparseTensorEncodingAttr::getMnemonic()) { 
    value = ::mlir::sparse_tensor::SparseTensorEncodingAttr::parse(context, parser, type);
    return ::mlir::success(!!value);
  }
  return {};
}


static ::mlir::LogicalResult generatedAttributePrinter(
                         ::mlir::Attribute def, ::mlir::DialectAsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::mlir::LogicalResult>(def)
    .Case<::mlir::sparse_tensor::SparseTensorEncodingAttr>([&](::mlir::sparse_tensor::SparseTensorEncodingAttr t) {
      t.print(printer);
      return ::mlir::success();
    })
    .Default([](::mlir::Attribute) { return ::mlir::failure(); });
}

namespace mlir {
namespace sparse_tensor {

namespace detail {
  struct SparseTensorEncodingAttrStorage : public ::mlir::AttributeStorage {
    SparseTensorEncodingAttrStorage (::llvm::ArrayRef<SparseTensorEncodingAttr::DimLevelType> dimLevelType, AffineMap dimOrdering, unsigned pointerBitWidth, unsigned indexBitWidth)
      : ::mlir::AttributeStorage(), dimLevelType(dimLevelType), dimOrdering(dimOrdering), pointerBitWidth(pointerBitWidth), indexBitWidth(indexBitWidth) { }

    /// The hash key is a tuple of the parameter types.
    using KeyTy = std::tuple<::llvm::ArrayRef<SparseTensorEncodingAttr::DimLevelType>, AffineMap, unsigned, unsigned>;
  bool operator==(const KeyTy &tblgenKey) const {
    if (!(dimLevelType == std::get<0>(tblgenKey)))
      return false;
    if (!(dimOrdering == std::get<1>(tblgenKey)))
      return false;
    if (!(pointerBitWidth == std::get<2>(tblgenKey)))
      return false;
    if (!(indexBitWidth == std::get<3>(tblgenKey)))
      return false;
    return true;
  }
  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
      return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey));
    }

    /// Define a construction method for creating a new instance of this
    /// storage.
    static SparseTensorEncodingAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator,
                          const KeyTy &tblgenKey) {
      auto dimLevelType = std::get<0>(tblgenKey);
      auto dimOrdering = std::get<1>(tblgenKey);
      auto pointerBitWidth = std::get<2>(tblgenKey);
      auto indexBitWidth = std::get<3>(tblgenKey);
      dimLevelType = allocator.copyInto(dimLevelType);

      return new (allocator.allocate<SparseTensorEncodingAttrStorage>())
          SparseTensorEncodingAttrStorage(dimLevelType, dimOrdering, pointerBitWidth, indexBitWidth);
    }
      ::llvm::ArrayRef<SparseTensorEncodingAttr::DimLevelType> dimLevelType;
      AffineMap dimOrdering;
      unsigned pointerBitWidth;
      unsigned indexBitWidth;
  };
} // namespace detail
SparseTensorEncodingAttr SparseTensorEncodingAttr::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<SparseTensorEncodingAttr::DimLevelType> dimLevelType, AffineMap dimOrdering, unsigned pointerBitWidth, unsigned indexBitWidth) {
  return Base::get(context, dimLevelType, dimOrdering, pointerBitWidth, indexBitWidth);
}
SparseTensorEncodingAttr SparseTensorEncodingAttr::getChecked(llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::llvm::ArrayRef<SparseTensorEncodingAttr::DimLevelType> dimLevelType, AffineMap dimOrdering, unsigned pointerBitWidth, unsigned indexBitWidth) {
  return Base::getChecked(emitError, context, dimLevelType, dimOrdering, pointerBitWidth, indexBitWidth);
}
::llvm::ArrayRef<SparseTensorEncodingAttr::DimLevelType> SparseTensorEncodingAttr::getDimLevelType() const { return getImpl()->dimLevelType; }
AffineMap SparseTensorEncodingAttr::getDimOrdering() const { return getImpl()->dimOrdering; }
unsigned SparseTensorEncodingAttr::getPointerBitWidth() const { return getImpl()->pointerBitWidth; }
unsigned SparseTensorEncodingAttr::getIndexBitWidth() const { return getImpl()->indexBitWidth; }
} // namespace sparse_tensor
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::SparseTensorEncodingAttr)

#endif  // GET_ATTRDEF_CLASSES

