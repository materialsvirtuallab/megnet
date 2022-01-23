/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_LIST
#undef GET_TYPEDEF_LIST

::mlir::pdl::AttributeType,
::mlir::pdl::OperationType,
::mlir::pdl::RangeType,
::mlir::pdl::TypeType,
::mlir::pdl::ValueType

#endif  // GET_TYPEDEF_LIST

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES


static ::mlir::OptionalParseResult generatedTypeParser(::mlir::MLIRContext *context,
                                      ::mlir::DialectAsmParser &parser,
                                      ::llvm::StringRef mnemonic,
                                      ::mlir::Type &value) {
  if (mnemonic == ::mlir::pdl::AttributeType::getMnemonic()) { 
    value = ::mlir::pdl::AttributeType::get(context);
    return ::mlir::success(!!value);
  }
  if (mnemonic == ::mlir::pdl::OperationType::getMnemonic()) { 
    value = ::mlir::pdl::OperationType::get(context);
    return ::mlir::success(!!value);
  }
  if (mnemonic == ::mlir::pdl::RangeType::getMnemonic()) { 
    value = ::mlir::pdl::RangeType::parse(context, parser);
    return ::mlir::success(!!value);
  }
  if (mnemonic == ::mlir::pdl::TypeType::getMnemonic()) { 
    value = ::mlir::pdl::TypeType::get(context);
    return ::mlir::success(!!value);
  }
  if (mnemonic == ::mlir::pdl::ValueType::getMnemonic()) { 
    value = ::mlir::pdl::ValueType::get(context);
    return ::mlir::success(!!value);
  }
  return {};
}


static ::mlir::LogicalResult generatedTypePrinter(
                         ::mlir::Type def, ::mlir::DialectAsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Type, ::mlir::LogicalResult>(def)
    .Case<::mlir::pdl::AttributeType>([&](::mlir::pdl::AttributeType t) {
      printer << ::mlir::pdl::AttributeType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::pdl::OperationType>([&](::mlir::pdl::OperationType t) {
      printer << ::mlir::pdl::OperationType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::pdl::RangeType>([&](::mlir::pdl::RangeType t) {
      t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::pdl::TypeType>([&](::mlir::pdl::TypeType t) {
      printer << ::mlir::pdl::TypeType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::pdl::ValueType>([&](::mlir::pdl::ValueType t) {
      printer << ::mlir::pdl::ValueType::getMnemonic();
      return ::mlir::success();
    })
    .Default([](::mlir::Type) { return ::mlir::failure(); });
}

namespace mlir {
namespace pdl {
} // namespace pdl
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl::AttributeType)
namespace mlir {
namespace pdl {
} // namespace pdl
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl::OperationType)
namespace mlir {
namespace pdl {

namespace detail {
  struct RangeTypeStorage : public ::mlir::TypeStorage {
    RangeTypeStorage (Type elementType)
      : elementType(elementType) { }

    /// The hash key is a tuple of the parameter types.
    using KeyTy = std::tuple<Type>;
  bool operator==(const KeyTy &tblgenKey) const {
    if (!(elementType == std::get<0>(tblgenKey)))
      return false;
    return true;
  }
  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
      return ::llvm::hash_combine(std::get<0>(tblgenKey));
    }

    /// Define a construction method for creating a new instance of this
    /// storage.
    static RangeTypeStorage *construct(::mlir::TypeStorageAllocator &allocator,
                          const KeyTy &tblgenKey) {
      auto elementType = std::get<0>(tblgenKey);

      return new (allocator.allocate<RangeTypeStorage>())
          RangeTypeStorage(elementType);
    }
      Type elementType;
  };
} // namespace detail
RangeType RangeType::get(Type elementType) {
  
      return Base::get(elementType.getContext(), elementType);
    ;
}
RangeType RangeType::getChecked(llvm::function_ref<::mlir::InFlightDiagnostic()> emitErrorFn, Type elementType) {
  
      return Base::getChecked(emitErrorFn, elementType.getContext(), elementType);
    ;
}
Type RangeType::getElementType() const { return getImpl()->elementType; }
} // namespace pdl
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl::RangeType)
namespace mlir {
namespace pdl {
} // namespace pdl
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl::TypeType)
namespace mlir {
namespace pdl {
} // namespace pdl
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl::ValueType)

#endif  // GET_TYPEDEF_CLASSES

