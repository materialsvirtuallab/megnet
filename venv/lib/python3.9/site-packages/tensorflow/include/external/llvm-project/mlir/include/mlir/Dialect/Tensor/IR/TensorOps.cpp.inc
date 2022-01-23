/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::tensor::CastOp,
::mlir::tensor::DimOp,
::mlir::tensor::ExtractOp,
::mlir::tensor::ExtractSliceOp,
::mlir::tensor::FromElementsOp,
::mlir::tensor::GenerateOp,
::mlir::tensor::InsertOp,
::mlir::tensor::InsertSliceOp,
::mlir::tensor::ReshapeOp,
::mlir::tensor::YieldOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace tensor {
static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TensorOps0(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isa<::mlir::TensorType>())) && ((true)))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be tensor of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TensorOps1(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isa<::mlir::IndexType>()))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be index, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TensorOps3(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((true))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be any type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TensorOps4(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((((type.isa<::mlir::TensorType>())) && ((type.cast<::mlir::ShapedType>().hasRank()))) && ((true)))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be ranked tensor of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TensorOps6(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((((type.isa<::mlir::TensorType>())) && ((true))) && (((type.cast<::mlir::ShapedType>().hasRank())) && ((type.cast<::mlir::ShapedType>().getRank()
                           == 1))))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be 1D tensor of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TensorOps7(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((((type.isa<::mlir::TensorType>())) && (((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((type.cast<::mlir::ShapedType>().getElementType().isa<::mlir::IndexType>())))) && (((type.cast<::mlir::ShapedType>().hasRank())) && ((type.cast<::mlir::ShapedType>().getRank()
                           == 1))))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be 1D tensor of signless integer or index values, but got " << type;
  }
  return ::mlir::success();
}

} // namespace tensor
} // namespace mlir
namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::CastOp definitions
//===----------------------------------------------------------------------===//

CastOpAdaptor::CastOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

CastOpAdaptor::CastOpAdaptor(CastOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange CastOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CastOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CastOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CastOpAdaptor::source() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr CastOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult CastOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> CastOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CastOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CastOp::source() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange CastOp::sourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CastOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CastOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CastOp::dest() {
  return *getODSResults(0).begin();
}

void CastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dest, ::mlir::Value source) {
  odsState.addOperands(source);
  odsState.addTypes(dest);
}

void CastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source) {
  odsState.addOperands(source);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CastOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CastOp::verify() {
  if (::mlir::failed(CastOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}





::mlir::ParseResult CastOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType sourceRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> sourceOperands(sourceRawOperands);  ::llvm::SMLoc sourceOperandsLoc;
  (void)sourceOperandsLoc;
  ::mlir::Type sourceRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> sourceTypes(sourceRawTypes);
  ::mlir::Type destRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> destTypes(destRawTypes);

  sourceOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sourceRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(sourceRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseType(destRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(destTypes);
  if (parser.resolveOperands(sourceOperands, sourceTypes, sourceOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CastOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << source();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(source().getType());
  p << ' ' << "to";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(dest().getType());
}

void CastOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tensor
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::tensor::CastOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::DimOp definitions
//===----------------------------------------------------------------------===//

DimOpAdaptor::DimOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

DimOpAdaptor::DimOpAdaptor(DimOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange DimOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> DimOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange DimOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value DimOpAdaptor::source() {
  return *getODSOperands(0).begin();
}

::mlir::Value DimOpAdaptor::index() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr DimOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult DimOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> DimOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range DimOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value DimOp::source() {
  return *getODSOperands(0).begin();
}

::mlir::Value DimOp::index() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange DimOp::sourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange DimOp::indexMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> DimOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range DimOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value DimOp::result() {
  return *getODSResults(0).begin();
}



void DimOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::Value index) {
  odsState.addOperands(source);
  odsState.addOperands(index);
  odsState.addTypes(result);
}

void DimOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value source, ::mlir::Value index) {
  odsState.addOperands(source);
  odsState.addOperands(index);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(DimOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void DimOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value index) {
  odsState.addOperands(source);
  odsState.addOperands(index);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void DimOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void DimOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

    ::mlir::SmallVector<::mlir::Type, 2> inferredReturnTypes;
    if (::mlir::succeeded(DimOp::inferReturnTypes(odsBuilder.getContext(),
                  odsState.location, operands,
                  odsState.attributes.getDictionary(odsState.getContext()),
                  /*regions=*/{}, inferredReturnTypes))) {  assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
      odsState.addTypes(inferredReturnTypes);
    } else
      ::llvm::report_fatal_error("Failed to infer result type(s).");
}

::mlir::LogicalResult DimOp::verify() {
  if (::mlir::failed(DimOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::verify(*this);
}





::mlir::LogicalResult DimOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  inferredReturnTypes[0] = odsBuilder.getIndexType();
  return ::mlir::success();
}

::mlir::ParseResult DimOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType sourceRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> sourceOperands(sourceRawOperands);  ::llvm::SMLoc sourceOperandsLoc;
  (void)sourceOperandsLoc;
  ::mlir::OpAsmParser::OperandType indexRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> indexOperands(indexRawOperands);  ::llvm::SMLoc indexOperandsLoc;
  (void)indexOperandsLoc;
  ::mlir::Type sourceRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> sourceTypes(sourceRawTypes);
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();

  sourceOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sourceRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  indexOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(indexRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(sourceRawTypes[0]))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(sourceOperands, sourceTypes, sourceOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(indexOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void DimOp::print(::mlir::OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ';
  p << source();
  p << ",";
  p << ' ';
  p << index();
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(source().getType());
}

void DimOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tensor
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::tensor::DimOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::ExtractOp definitions
//===----------------------------------------------------------------------===//

ExtractOpAdaptor::ExtractOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ExtractOpAdaptor::ExtractOpAdaptor(ExtractOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ExtractOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ExtractOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange ExtractOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExtractOpAdaptor::tensor() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange ExtractOpAdaptor::indices() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr ExtractOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ExtractOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> ExtractOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range ExtractOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExtractOp::tensor() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range ExtractOp::indices() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange ExtractOp::tensorMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ExtractOp::indicesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ExtractOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ExtractOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExtractOp::result() {
  return *getODSResults(0).begin();
}

void ExtractOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value tensor, ValueRange indices ) {
      auto resType = tensor.getType().cast<ShapedType>().getElementType();
      build(odsBuilder, odsState, resType, tensor, indices);
    
}

void ExtractOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value tensor, ::mlir::ValueRange indices) {
  odsState.addOperands(tensor);
  odsState.addOperands(indices);
  odsState.addTypes(result);
}

void ExtractOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value tensor, ::mlir::ValueRange indices) {
  odsState.addOperands(tensor);
  odsState.addOperands(indices);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ExtractOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ExtractOp::verify() {
  if (::mlir::failed(ExtractOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps3(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  if (!((std::equal_to<>()((*this->getODSOperands(0).begin()).getType().cast<ShapedType>().getElementType(), (*this->getODSResults(0).begin()).getType()))))
    return emitOpError("failed to verify that result type matches element type of tensor");
  return ::verify(*this);
}



::mlir::ParseResult ExtractOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType tensorRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> tensorOperands(tensorRawOperands);  ::llvm::SMLoc tensorOperandsLoc;
  (void)tensorOperandsLoc;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> indicesOperands;
  ::llvm::SMLoc indicesOperandsLoc;
  (void)indicesOperandsLoc;
  ::mlir::Type tensorRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> tensorTypes(tensorRawTypes);

  tensorOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(tensorRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  indicesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(indicesOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(tensorRawTypes[0]))
    return ::mlir::failure();
  for (::mlir::Type type : tensorTypes) {
    (void)type;
    if (!(((type.isa<::mlir::TensorType>())) && ((true)))) {
      return parser.emitError(parser.getNameLoc()) << "'tensor' must be tensor of any type values, but got " << type;
    }
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(tensorTypes[0].cast<ShapedType>().getElementType());
  if (parser.resolveOperands(tensorOperands, tensorTypes, tensorOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(indicesOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ExtractOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << tensor();
  p << "[";
  p << indices();
  p << "]";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(tensor().getType());
}

void ExtractOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tensor
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::tensor::ExtractOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::ExtractSliceOp definitions
//===----------------------------------------------------------------------===//

ExtractSliceOpAdaptor::ExtractSliceOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ExtractSliceOpAdaptor::ExtractSliceOpAdaptor(ExtractSliceOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ExtractSliceOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ExtractSliceOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::ValueRange ExtractSliceOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExtractSliceOpAdaptor::source() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange ExtractSliceOpAdaptor::offsets() {
  return getODSOperands(1);
}

::mlir::ValueRange ExtractSliceOpAdaptor::sizes() {
  return getODSOperands(2);
}

::mlir::ValueRange ExtractSliceOpAdaptor::strides() {
  return getODSOperands(3);
}

::mlir::DictionaryAttr ExtractSliceOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr ExtractSliceOpAdaptor::static_offsets() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("static_offsets").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr ExtractSliceOpAdaptor::static_sizes() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("static_sizes").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr ExtractSliceOpAdaptor::static_strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("static_strides").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::LogicalResult ExtractSliceOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 4)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 4 elements, but got ") << numElements;
  }
    {
  auto tblgen_static_offsets = odsAttrs.get("static_offsets");
  if (!tblgen_static_offsets) return emitError(loc, "'tensor.extract_slice' op ""requires attribute 'static_offsets'");
    if (!(((tblgen_static_offsets.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_static_offsets.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); })))) return emitError(loc, "'tensor.extract_slice' op ""attribute 'static_offsets' failed to satisfy constraint: 64-bit integer array attribute");
  }
  {
  auto tblgen_static_sizes = odsAttrs.get("static_sizes");
  if (!tblgen_static_sizes) return emitError(loc, "'tensor.extract_slice' op ""requires attribute 'static_sizes'");
    if (!(((tblgen_static_sizes.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_static_sizes.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); })))) return emitError(loc, "'tensor.extract_slice' op ""attribute 'static_sizes' failed to satisfy constraint: 64-bit integer array attribute");
  }
  {
  auto tblgen_static_strides = odsAttrs.get("static_strides");
  if (!tblgen_static_strides) return emitError(loc, "'tensor.extract_slice' op ""requires attribute 'static_strides'");
    if (!(((tblgen_static_strides.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_static_strides.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); })))) return emitError(loc, "'tensor.extract_slice' op ""attribute 'static_strides' failed to satisfy constraint: 64-bit integer array attribute");
  }
  return ::mlir::success();
}

























std::pair<unsigned, unsigned> ExtractSliceOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range ExtractSliceOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExtractSliceOp::source() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range ExtractSliceOp::offsets() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range ExtractSliceOp::sizes() {
  return getODSOperands(2);
}

::mlir::Operation::operand_range ExtractSliceOp::strides() {
  return getODSOperands(3);
}

::mlir::MutableOperandRange ExtractSliceOp::sourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ExtractSliceOp::offsetsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ExtractSliceOp::sizesMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ExtractSliceOp::stridesMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> ExtractSliceOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ExtractSliceOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExtractSliceOp::result() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr ExtractSliceOp::static_offsetsAttr() {
  return (*this)->getAttr(static_offsetsAttrName()).template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr ExtractSliceOp::static_offsets() {
  auto attr = static_offsetsAttr();
  return attr;
}

::mlir::ArrayAttr ExtractSliceOp::static_sizesAttr() {
  return (*this)->getAttr(static_sizesAttrName()).template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr ExtractSliceOp::static_sizes() {
  auto attr = static_sizesAttr();
  return attr;
}

::mlir::ArrayAttr ExtractSliceOp::static_stridesAttr() {
  return (*this)->getAttr(static_stridesAttrName()).template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr ExtractSliceOp::static_strides() {
  auto attr = static_stridesAttr();
  return attr;
}

void ExtractSliceOp::static_offsetsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(static_offsetsAttrName(), attr);
}

void ExtractSliceOp::static_sizesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(static_sizesAttrName(), attr);
}

void ExtractSliceOp::static_stridesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(static_stridesAttrName(), attr);
}









void ExtractSliceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::ArrayAttr static_offsets, ::mlir::ArrayAttr static_sizes, ::mlir::ArrayAttr static_strides) {
  odsState.addOperands(source);
  odsState.addOperands(offsets);
  odsState.addOperands(sizes);
  odsState.addOperands(strides);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({1, static_cast<int32_t>(offsets.size()), static_cast<int32_t>(sizes.size()), static_cast<int32_t>(strides.size())}));
  odsState.addAttribute(static_offsetsAttrName(odsState.name), static_offsets);
  odsState.addAttribute(static_sizesAttrName(odsState.name), static_sizes);
  odsState.addAttribute(static_stridesAttrName(odsState.name), static_strides);
  odsState.addTypes(result);
}

void ExtractSliceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::ArrayAttr static_offsets, ::mlir::ArrayAttr static_sizes, ::mlir::ArrayAttr static_strides) {
  odsState.addOperands(source);
  odsState.addOperands(offsets);
  odsState.addOperands(sizes);
  odsState.addOperands(strides);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({1, static_cast<int32_t>(offsets.size()), static_cast<int32_t>(sizes.size()), static_cast<int32_t>(strides.size())}));
  odsState.addAttribute(static_offsetsAttrName(odsState.name), static_offsets);
  odsState.addAttribute(static_sizesAttrName(odsState.name), static_sizes);
  odsState.addAttribute(static_stridesAttrName(odsState.name), static_strides);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ExtractSliceOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ExtractSliceOp::verify() {
  if (::mlir::failed(ExtractSliceOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps4(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup3 = getODSOperands(3);
    for (::mlir::Value v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps4(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::verify(*this);
}





::mlir::ParseResult ExtractSliceOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType sourceRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> sourceOperands(sourceRawOperands);  ::llvm::SMLoc sourceOperandsLoc;
  (void)sourceOperandsLoc;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> offsetsOperands;
  ::llvm::SMLoc offsetsOperandsLoc;
  (void)offsetsOperandsLoc;
  ::mlir::ArrayAttr static_offsetsAttr;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> sizesOperands;
  ::llvm::SMLoc sizesOperandsLoc;
  (void)sizesOperandsLoc;
  ::mlir::ArrayAttr static_sizesAttr;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> stridesOperands;
  ::llvm::SMLoc stridesOperandsLoc;
  (void)stridesOperandsLoc;
  ::mlir::ArrayAttr static_stridesAttr;
  ::mlir::Type sourceRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> sourceTypes(sourceRawTypes);
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  sourceOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sourceRawOperands[0]))
    return ::mlir::failure();
  {
    offsetsOperandsLoc = parser.getCurrentLocation();
    if (parseOperandsOrIntegersOffsetsOrStridesList(parser, offsetsOperands, static_offsetsAttr))
      return ::mlir::failure();
    result.addAttribute("static_offsets", static_offsetsAttr);
  }
  {
    sizesOperandsLoc = parser.getCurrentLocation();
    if (parseOperandsOrIntegersSizesList(parser, sizesOperands, static_sizesAttr))
      return ::mlir::failure();
    result.addAttribute("static_sizes", static_sizesAttr);
  }
  {
    stridesOperandsLoc = parser.getCurrentLocation();
    if (parseOperandsOrIntegersOffsetsOrStridesList(parser, stridesOperands, static_stridesAttr))
      return ::mlir::failure();
    result.addAttribute("static_strides", static_stridesAttr);
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(sourceRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseType(resultRawTypes[0]))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(resultTypes);
  if (parser.resolveOperands(sourceOperands, sourceTypes, sourceOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(offsetsOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(sizesOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(stridesOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getI32VectorAttr({1, static_cast<int32_t>(offsetsOperands.size()), static_cast<int32_t>(sizesOperands.size()), static_cast<int32_t>(stridesOperands.size())}));
  return ::mlir::success();
}

void ExtractSliceOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << source();
  printOperandsOrIntegersOffsetsOrStridesList(p, *this, offsets(), static_offsetsAttr());
  p << ' ';
  printOperandsOrIntegersSizesList(p, *this, sizes(), static_sizesAttr());
  p << ' ';
  printOperandsOrIntegersOffsetsOrStridesList(p, *this, strides(), static_stridesAttr());
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", "static_offsets", "static_sizes", "static_strides"});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(source().getType());
  p << ' ' << "to";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(result().getType());
}

void ExtractSliceOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tensor
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::tensor::ExtractSliceOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::FromElementsOp definitions
//===----------------------------------------------------------------------===//

FromElementsOpAdaptor::FromElementsOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

FromElementsOpAdaptor::FromElementsOpAdaptor(FromElementsOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange FromElementsOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FromElementsOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange FromElementsOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange FromElementsOpAdaptor::elements() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr FromElementsOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult FromElementsOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> FromElementsOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range FromElementsOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range FromElementsOp::elements() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange FromElementsOp::elementsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> FromElementsOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FromElementsOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FromElementsOp::result() {
  return *getODSResults(0).begin();
}





::mlir::LogicalResult FromElementsOp::verify() {
  if (::mlir::failed(FromElementsOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps6(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  if (!((std::equal_to<>()(SmallVector<Type, 2>((*this->getODSResults(0).begin()).getType().cast<ShapedType>().getDimSize(0), (*this->getODSResults(0).begin()).getType().cast<ShapedType>().getElementType()), this->getODSOperands(0).getType()))))
    return emitOpError("failed to verify that operand types match result element type");
  return ::mlir::success();
}





::mlir::ParseResult FromElementsOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> elementsOperands;
  ::llvm::SMLoc elementsOperandsLoc;
  (void)elementsOperandsLoc;
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  elementsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(elementsOperands))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resultRawTypes[0]))
    return ::mlir::failure();
  for (::mlir::Type type : resultTypes) {
    (void)type;
    if (!((((type.isa<::mlir::TensorType>())) && ((true))) && (((type.cast<::mlir::ShapedType>().hasRank())) && ((type.cast<::mlir::ShapedType>().getRank()
                         == 1))))) {
      return parser.emitError(parser.getNameLoc()) << "'result' must be 1D tensor of any type values, but got " << type;
    }
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(elementsOperands, SmallVector<Type, 2>(resultTypes[0].cast<ShapedType>().getDimSize(0), resultTypes[0].cast<ShapedType>().getElementType()), elementsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void FromElementsOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << elements();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(result().getType());
}

void FromElementsOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tensor
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::tensor::FromElementsOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::GenerateOp definitions
//===----------------------------------------------------------------------===//

GenerateOpAdaptor::GenerateOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

GenerateOpAdaptor::GenerateOpAdaptor(GenerateOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange GenerateOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GenerateOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange GenerateOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange GenerateOpAdaptor::dynamicExtents() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr GenerateOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange GenerateOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &GenerateOpAdaptor::body() {
  return *odsRegions[0];
}

::mlir::LogicalResult GenerateOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> GenerateOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range GenerateOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range GenerateOp::dynamicExtents() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange GenerateOp::dynamicExtentsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> GenerateOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GenerateOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GenerateOp::result() {
  return *getODSResults(0).begin();
}

::mlir::Region &GenerateOp::body() {
  return (*this)->getRegion(0);
}



void GenerateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::ValueRange dynamicExtents) {
  odsState.addOperands(dynamicExtents);
  (void)odsState.addRegion();
  odsState.addTypes(result);
}

void GenerateOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GenerateOp::verify() {
  if (::mlir::failed(GenerateOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps4(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((::llvm::hasNItems(region, 1)))) {
        return emitOpError("region #") << index << " ('body') failed to verify constraint: region with 1 blocks";
      }
      ++index;
    }
  }
  return ::verify(*this);
}



::mlir::ParseResult GenerateOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> dynamicExtentsOperands;
  ::llvm::SMLoc dynamicExtentsOperandsLoc;
  (void)dynamicExtentsOperandsLoc;
  std::unique_ptr<::mlir::Region> bodyRegion = std::make_unique<::mlir::Region>();
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  dynamicExtentsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(dynamicExtentsOperands))
    return ::mlir::failure();

  if (parser.parseRegion(*bodyRegion))
    return ::mlir::failure();

  ensureTerminator(*bodyRegion, parser.getBuilder(), result.location);
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resultRawTypes[0]))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(resultTypes);
  if (parser.resolveOperands(dynamicExtentsOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  result.addRegion(std::move(bodyRegion));
  return ::mlir::success();
}

void GenerateOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << dynamicExtents();
  p << ' ';

  {
    bool printTerminator = true;
    if (auto *term = body().empty() ? nullptr : body().begin()->getTerminator()) {
      printTerminator = !term->getAttrDictionary().empty() ||
                        term->getNumOperands() != 0 ||
                        term->getNumResults() != 0;
    }
    p.printRegion(body(), /*printEntryBlockArgs=*/true,
                  /*printBlockTerminators=*/printTerminator);
  }
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(result().getType());
}

} // namespace tensor
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::tensor::GenerateOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::InsertOp definitions
//===----------------------------------------------------------------------===//

InsertOpAdaptor::InsertOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

InsertOpAdaptor::InsertOpAdaptor(InsertOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange InsertOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> InsertOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange InsertOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value InsertOpAdaptor::scalar() {
  return *getODSOperands(0).begin();
}

::mlir::Value InsertOpAdaptor::dest() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange InsertOpAdaptor::indices() {
  return getODSOperands(2);
}

::mlir::DictionaryAttr InsertOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult InsertOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> InsertOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range InsertOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value InsertOp::scalar() {
  return *getODSOperands(0).begin();
}

::mlir::Value InsertOp::dest() {
  return *getODSOperands(1).begin();
}

::mlir::Operation::operand_range InsertOp::indices() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange InsertOp::scalarMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange InsertOp::destMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange InsertOp::indicesMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> InsertOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range InsertOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value InsertOp::result() {
  return *getODSResults(0).begin();
}

void InsertOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value scalar, Value dest, ValueRange indices ) {
      auto resType = dest.getType();
      build(odsBuilder, odsState, resType, scalar, dest, indices);
    
}

void InsertOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value scalar, ::mlir::Value dest, ::mlir::ValueRange indices) {
  odsState.addOperands(scalar);
  odsState.addOperands(dest);
  odsState.addOperands(indices);
  odsState.addTypes(result);
}

void InsertOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value scalar, ::mlir::Value dest, ::mlir::ValueRange indices) {
  odsState.addOperands(scalar);
  odsState.addOperands(dest);
  odsState.addOperands(indices);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void InsertOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult InsertOp::verify() {
  if (::mlir::failed(InsertOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  if (!((std::equal_to<>()((*this->getODSOperands(1).begin()).getType().cast<ShapedType>(), (*this->getODSResults(0).begin()).getType()))))
    return emitOpError("failed to verify that result type matches type of dest");
  if (!((std::equal_to<>()((*this->getODSOperands(1).begin()).getType().cast<ShapedType>().getElementType(), (*this->getODSOperands(0).begin()).getType()))))
    return emitOpError("failed to verify that scalar type matches element type of dest");
  return ::verify(*this);
}



::mlir::ParseResult InsertOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType scalarRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> scalarOperands(scalarRawOperands);  ::llvm::SMLoc scalarOperandsLoc;
  (void)scalarOperandsLoc;
  ::mlir::OpAsmParser::OperandType destRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> destOperands(destRawOperands);  ::llvm::SMLoc destOperandsLoc;
  (void)destOperandsLoc;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> indicesOperands;
  ::llvm::SMLoc indicesOperandsLoc;
  (void)indicesOperandsLoc;
  ::mlir::Type destRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> destTypes(destRawTypes);

  scalarOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(scalarRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("into"))
    return ::mlir::failure();

  destOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(destRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  indicesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(indicesOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(destRawTypes[0]))
    return ::mlir::failure();
  for (::mlir::Type type : destTypes) {
    (void)type;
    if (!(((type.isa<::mlir::TensorType>())) && ((true)))) {
      return parser.emitError(parser.getNameLoc()) << "'dest' must be tensor of any type values, but got " << type;
    }
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(destTypes[0].cast<ShapedType>());
  if (parser.resolveOperands(scalarOperands, destTypes[0].cast<ShapedType>().getElementType(), scalarOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(destOperands, destTypes, destOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(indicesOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void InsertOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << scalar();
  p << ' ' << "into";
  p << ' ';
  p << dest();
  p << "[";
  p << indices();
  p << "]";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(dest().getType());
}

void InsertOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tensor
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::tensor::InsertOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::InsertSliceOp definitions
//===----------------------------------------------------------------------===//

InsertSliceOpAdaptor::InsertSliceOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

InsertSliceOpAdaptor::InsertSliceOpAdaptor(InsertSliceOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange InsertSliceOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> InsertSliceOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::ValueRange InsertSliceOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value InsertSliceOpAdaptor::source() {
  return *getODSOperands(0).begin();
}

::mlir::Value InsertSliceOpAdaptor::dest() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange InsertSliceOpAdaptor::offsets() {
  return getODSOperands(2);
}

::mlir::ValueRange InsertSliceOpAdaptor::sizes() {
  return getODSOperands(3);
}

::mlir::ValueRange InsertSliceOpAdaptor::strides() {
  return getODSOperands(4);
}

::mlir::DictionaryAttr InsertSliceOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr InsertSliceOpAdaptor::static_offsets() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("static_offsets").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr InsertSliceOpAdaptor::static_sizes() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("static_sizes").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr InsertSliceOpAdaptor::static_strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("static_strides").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::LogicalResult InsertSliceOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 5)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 5 elements, but got ") << numElements;
  }
    {
  auto tblgen_static_offsets = odsAttrs.get("static_offsets");
  if (!tblgen_static_offsets) return emitError(loc, "'tensor.insert_slice' op ""requires attribute 'static_offsets'");
    if (!(((tblgen_static_offsets.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_static_offsets.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); })))) return emitError(loc, "'tensor.insert_slice' op ""attribute 'static_offsets' failed to satisfy constraint: 64-bit integer array attribute");
  }
  {
  auto tblgen_static_sizes = odsAttrs.get("static_sizes");
  if (!tblgen_static_sizes) return emitError(loc, "'tensor.insert_slice' op ""requires attribute 'static_sizes'");
    if (!(((tblgen_static_sizes.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_static_sizes.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); })))) return emitError(loc, "'tensor.insert_slice' op ""attribute 'static_sizes' failed to satisfy constraint: 64-bit integer array attribute");
  }
  {
  auto tblgen_static_strides = odsAttrs.get("static_strides");
  if (!tblgen_static_strides) return emitError(loc, "'tensor.insert_slice' op ""requires attribute 'static_strides'");
    if (!(((tblgen_static_strides.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_static_strides.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); })))) return emitError(loc, "'tensor.insert_slice' op ""attribute 'static_strides' failed to satisfy constraint: 64-bit integer array attribute");
  }
  return ::mlir::success();
}

























std::pair<unsigned, unsigned> InsertSliceOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range InsertSliceOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value InsertSliceOp::source() {
  return *getODSOperands(0).begin();
}

::mlir::Value InsertSliceOp::dest() {
  return *getODSOperands(1).begin();
}

::mlir::Operation::operand_range InsertSliceOp::offsets() {
  return getODSOperands(2);
}

::mlir::Operation::operand_range InsertSliceOp::sizes() {
  return getODSOperands(3);
}

::mlir::Operation::operand_range InsertSliceOp::strides() {
  return getODSOperands(4);
}

::mlir::MutableOperandRange InsertSliceOp::sourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange InsertSliceOp::destMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange InsertSliceOp::offsetsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange InsertSliceOp::sizesMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange InsertSliceOp::stridesMutable() {
  auto range = getODSOperandIndexAndLength(4);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(4u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> InsertSliceOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range InsertSliceOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value InsertSliceOp::result() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr InsertSliceOp::static_offsetsAttr() {
  return (*this)->getAttr(static_offsetsAttrName()).template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr InsertSliceOp::static_offsets() {
  auto attr = static_offsetsAttr();
  return attr;
}

::mlir::ArrayAttr InsertSliceOp::static_sizesAttr() {
  return (*this)->getAttr(static_sizesAttrName()).template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr InsertSliceOp::static_sizes() {
  auto attr = static_sizesAttr();
  return attr;
}

::mlir::ArrayAttr InsertSliceOp::static_stridesAttr() {
  return (*this)->getAttr(static_stridesAttrName()).template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr InsertSliceOp::static_strides() {
  auto attr = static_stridesAttr();
  return attr;
}

void InsertSliceOp::static_offsetsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(static_offsetsAttrName(), attr);
}

void InsertSliceOp::static_sizesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(static_sizesAttrName(), attr);
}

void InsertSliceOp::static_stridesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(static_stridesAttrName(), attr);
}





void InsertSliceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::Value dest, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::ArrayAttr static_offsets, ::mlir::ArrayAttr static_sizes, ::mlir::ArrayAttr static_strides) {
  odsState.addOperands(source);
  odsState.addOperands(dest);
  odsState.addOperands(offsets);
  odsState.addOperands(sizes);
  odsState.addOperands(strides);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({1, 1, static_cast<int32_t>(offsets.size()), static_cast<int32_t>(sizes.size()), static_cast<int32_t>(strides.size())}));
  odsState.addAttribute(static_offsetsAttrName(odsState.name), static_offsets);
  odsState.addAttribute(static_sizesAttrName(odsState.name), static_sizes);
  odsState.addAttribute(static_stridesAttrName(odsState.name), static_strides);
  odsState.addTypes(result);
}

void InsertSliceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value dest, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::ArrayAttr static_offsets, ::mlir::ArrayAttr static_sizes, ::mlir::ArrayAttr static_strides) {
  odsState.addOperands(source);
  odsState.addOperands(dest);
  odsState.addOperands(offsets);
  odsState.addOperands(sizes);
  odsState.addOperands(strides);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({1, 1, static_cast<int32_t>(offsets.size()), static_cast<int32_t>(sizes.size()), static_cast<int32_t>(strides.size())}));
  odsState.addAttribute(static_offsetsAttrName(odsState.name), static_offsets);
  odsState.addAttribute(static_sizesAttrName(odsState.name), static_sizes);
  odsState.addAttribute(static_stridesAttrName(odsState.name), static_strides);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void InsertSliceOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult InsertSliceOp::verify() {
  if (::mlir::failed(InsertSliceOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps4(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps4(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup3 = getODSOperands(3);
    for (::mlir::Value v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup4 = getODSOperands(4);
    for (::mlir::Value v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps4(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  if (!((std::equal_to<>()((*this->getODSOperands(1).begin()).getType(), (*this->getODSResults(0).begin()).getType()))))
    return emitOpError("failed to verify that expected result type to match dest type");
  return ::mlir::success();
}







::mlir::ParseResult InsertSliceOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType sourceRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> sourceOperands(sourceRawOperands);  ::llvm::SMLoc sourceOperandsLoc;
  (void)sourceOperandsLoc;
  ::mlir::OpAsmParser::OperandType destRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> destOperands(destRawOperands);  ::llvm::SMLoc destOperandsLoc;
  (void)destOperandsLoc;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> offsetsOperands;
  ::llvm::SMLoc offsetsOperandsLoc;
  (void)offsetsOperandsLoc;
  ::mlir::ArrayAttr static_offsetsAttr;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> sizesOperands;
  ::llvm::SMLoc sizesOperandsLoc;
  (void)sizesOperandsLoc;
  ::mlir::ArrayAttr static_sizesAttr;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> stridesOperands;
  ::llvm::SMLoc stridesOperandsLoc;
  (void)stridesOperandsLoc;
  ::mlir::ArrayAttr static_stridesAttr;
  ::mlir::Type sourceRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> sourceTypes(sourceRawTypes);
  ::mlir::Type destRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> destTypes(destRawTypes);

  sourceOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sourceRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("into"))
    return ::mlir::failure();

  destOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(destRawOperands[0]))
    return ::mlir::failure();
  {
    offsetsOperandsLoc = parser.getCurrentLocation();
    if (parseOperandsOrIntegersOffsetsOrStridesList(parser, offsetsOperands, static_offsetsAttr))
      return ::mlir::failure();
    result.addAttribute("static_offsets", static_offsetsAttr);
  }
  {
    sizesOperandsLoc = parser.getCurrentLocation();
    if (parseOperandsOrIntegersSizesList(parser, sizesOperands, static_sizesAttr))
      return ::mlir::failure();
    result.addAttribute("static_sizes", static_sizesAttr);
  }
  {
    stridesOperandsLoc = parser.getCurrentLocation();
    if (parseOperandsOrIntegersOffsetsOrStridesList(parser, stridesOperands, static_stridesAttr))
      return ::mlir::failure();
    result.addAttribute("static_strides", static_stridesAttr);
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(sourceRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("into"))
    return ::mlir::failure();

  if (parser.parseType(destRawTypes[0]))
    return ::mlir::failure();
  for (::mlir::Type type : destTypes) {
    (void)type;
    if (!((((type.isa<::mlir::TensorType>())) && ((type.cast<::mlir::ShapedType>().hasRank()))) && ((true)))) {
      return parser.emitError(parser.getNameLoc()) << "'dest' must be ranked tensor of any type values, but got " << type;
    }
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(destTypes[0]);
  if (parser.resolveOperands(sourceOperands, sourceTypes, sourceOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(destOperands, destTypes, destOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(offsetsOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(sizesOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(stridesOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getI32VectorAttr({1, 1, static_cast<int32_t>(offsetsOperands.size()), static_cast<int32_t>(sizesOperands.size()), static_cast<int32_t>(stridesOperands.size())}));
  return ::mlir::success();
}

void InsertSliceOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << source();
  p << ' ' << "into";
  p << ' ';
  p << dest();
  printOperandsOrIntegersOffsetsOrStridesList(p, *this, offsets(), static_offsetsAttr());
  p << ' ';
  printOperandsOrIntegersSizesList(p, *this, sizes(), static_sizesAttr());
  p << ' ';
  printOperandsOrIntegersOffsetsOrStridesList(p, *this, strides(), static_stridesAttr());
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", "static_offsets", "static_sizes", "static_strides"});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(source().getType());
  p << ' ' << "into";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(dest().getType());
}

void InsertSliceOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tensor
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::tensor::InsertSliceOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::ReshapeOp definitions
//===----------------------------------------------------------------------===//

ReshapeOpAdaptor::ReshapeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ReshapeOpAdaptor::ReshapeOpAdaptor(ReshapeOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ReshapeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ReshapeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ReshapeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReshapeOpAdaptor::source() {
  return *getODSOperands(0).begin();
}

::mlir::Value ReshapeOpAdaptor::shape() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr ReshapeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ReshapeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> ReshapeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ReshapeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReshapeOp::source() {
  return *getODSOperands(0).begin();
}

::mlir::Value ReshapeOp::shape() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange ReshapeOp::sourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ReshapeOp::shapeMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ReshapeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReshapeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReshapeOp::result() {
  return *getODSResults(0).begin();
}

void ReshapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TensorType resultType, Value operand, Value shape) {
       odsState.addOperands(operand);
       odsState.addOperands(shape);
       odsState.addTypes(resultType);
     
}

void ReshapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::Value shape) {
  odsState.addOperands(source);
  odsState.addOperands(shape);
  odsState.addTypes(result);
}

void ReshapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value shape) {
  odsState.addOperands(source);
  odsState.addOperands(shape);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReshapeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReshapeOp::verify() {
  if (::mlir::failed(ReshapeOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps7(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::verify(*this);
}

::mlir::ParseResult ReshapeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType sourceRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> sourceOperands(sourceRawOperands);  ::llvm::SMLoc sourceOperandsLoc;
  (void)sourceOperandsLoc;
  ::mlir::OpAsmParser::OperandType shapeRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> shapeOperands(shapeRawOperands);  ::llvm::SMLoc shapeOperandsLoc;
  (void)shapeOperandsLoc;
  ::llvm::ArrayRef<::mlir::Type> allOperandTypes;
  ::llvm::ArrayRef<::mlir::Type> allResultTypes;

  sourceOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sourceRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  shapeOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(shapeRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType allOperand__allResult_functionType;
  if (parser.parseType(allOperand__allResult_functionType))
    return ::mlir::failure();
  allOperandTypes = allOperand__allResult_functionType.getInputs();
  allResultTypes = allOperand__allResult_functionType.getResults();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(::llvm::concat<const ::mlir::OpAsmParser::OperandType>(sourceOperands, shapeOperands), allOperandTypes, parser.getNameLoc(), result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ReshapeOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << source();
  p << "(";
  p << shape();
  p << ")";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p.printFunctionalType(getOperation()->getOperandTypes(), getOperation()->getResultTypes());
}

void ReshapeOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tensor
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::tensor::ReshapeOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::YieldOp definitions
//===----------------------------------------------------------------------===//

YieldOpAdaptor::YieldOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

YieldOpAdaptor::YieldOpAdaptor(YieldOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange YieldOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> YieldOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange YieldOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value YieldOpAdaptor::value() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr YieldOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult YieldOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> YieldOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range YieldOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value YieldOp::value() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange YieldOp::valueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> YieldOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range YieldOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void YieldOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
 /* nothing to do */ 
}

void YieldOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value) {
  odsState.addOperands(value);
}

void YieldOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value) {
  odsState.addOperands(value);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void YieldOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult YieldOp::verify() {
  if (::mlir::failed(YieldOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::ParseResult YieldOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(valueRawTypes[0]))
    return ::mlir::failure();
  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void YieldOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << value();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(value().getType());
}

void YieldOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace tensor
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::tensor::YieldOp)


#endif  // GET_OP_CLASSES

