/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::arm_sve::ScalableAddFOp,
::mlir::arm_sve::ScalableAddIOp,
::mlir::arm_sve::ScalableCmpFOp,
::mlir::arm_sve::ScalableCmpIOp,
::mlir::arm_sve::ScalableDivFOp,
::mlir::arm_sve::ScalableLoadOp,
::mlir::arm_sve::ScalableMaskedAddFIntrOp,
::mlir::arm_sve::ScalableMaskedAddFOp,
::mlir::arm_sve::ScalableMaskedAddIIntrOp,
::mlir::arm_sve::ScalableMaskedAddIOp,
::mlir::arm_sve::ScalableMaskedDivFIntrOp,
::mlir::arm_sve::ScalableMaskedDivFOp,
::mlir::arm_sve::ScalableMaskedMulFIntrOp,
::mlir::arm_sve::ScalableMaskedMulFOp,
::mlir::arm_sve::ScalableMaskedMulIIntrOp,
::mlir::arm_sve::ScalableMaskedMulIOp,
::mlir::arm_sve::ScalableMaskedSDivIIntrOp,
::mlir::arm_sve::ScalableMaskedSDivIOp,
::mlir::arm_sve::ScalableMaskedSubFIntrOp,
::mlir::arm_sve::ScalableMaskedSubFOp,
::mlir::arm_sve::ScalableMaskedSubIIntrOp,
::mlir::arm_sve::ScalableMaskedSubIOp,
::mlir::arm_sve::ScalableMaskedUDivIIntrOp,
::mlir::arm_sve::ScalableMaskedUDivIOp,
::mlir::arm_sve::ScalableMulFOp,
::mlir::arm_sve::ScalableMulIOp,
::mlir::arm_sve::ScalableSDivIOp,
::mlir::arm_sve::ScalableStoreOp,
::mlir::arm_sve::ScalableSubFOp,
::mlir::arm_sve::ScalableSubIOp,
::mlir::arm_sve::ScalableUDivIOp,
::mlir::arm_sve::SdotIntrOp,
::mlir::arm_sve::SdotOp,
::mlir::arm_sve::SmmlaIntrOp,
::mlir::arm_sve::SmmlaOp,
::mlir::arm_sve::UdotIntrOp,
::mlir::arm_sve::UdotOp,
::mlir::arm_sve::UmmlaIntrOp,
::mlir::arm_sve::UmmlaOp,
::mlir::arm_sve::VectorScaleIntrOp,
::mlir::arm_sve::VectorScaleOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace arm_sve {
static ::mlir::LogicalResult __mlir_ods_local_type_constraint_ArmSVE0(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isa<::mlir::arm_sve::ScalableVectorType>())) && ((type.cast<::mlir::arm_sve::ScalableVectorType>().getElementType().isa<::mlir::FloatType>())))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be scalable vector of floating-point values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_ArmSVE1(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isa<::mlir::arm_sve::ScalableVectorType>())) && (((type.cast<::mlir::arm_sve::ScalableVectorType>().getElementType().isSignlessInteger(8))) || ((type.cast<::mlir::arm_sve::ScalableVectorType>().getElementType().isSignlessInteger(16))) || ((type.cast<::mlir::arm_sve::ScalableVectorType>().getElementType().isSignlessInteger(32))) || ((type.cast<::mlir::arm_sve::ScalableVectorType>().getElementType().isSignlessInteger(64)))))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be scalable vector of 8-bit signless integer or 16-bit signless integer or 32-bit signless integer or 64-bit signless integer values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_ArmSVE2(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isa<::mlir::arm_sve::ScalableVectorType>())) && ((type.cast<::mlir::arm_sve::ScalableVectorType>().getElementType().isSignlessInteger(1))))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be scalable vector of 1-bit signless integer values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_ArmSVE3(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isa<::mlir::MemRefType>())) && ((true)))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be memref of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_ArmSVE4(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isa<::mlir::IndexType>()))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be index, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_ArmSVE5(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isa<::mlir::arm_sve::ScalableVectorType>())) && ((true)))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be scalable vector of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_ArmSVE6(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isa<::mlir::LLVM::LLVMScalableVectorType>()))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be LLVM dialect scalable vector type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_ArmSVE7(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((::mlir::LLVM::isCompatibleType(type)))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be LLVM dialect-compatible type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_ArmSVE8(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((((type.isa<::mlir::arm_sve::ScalableVectorType>())) && (((type.cast<::mlir::arm_sve::ScalableVectorType>().getElementType().isSignlessInteger(32))) || ((type.cast<::mlir::arm_sve::ScalableVectorType>().getElementType().isSignlessInteger(64))))) && (((type.isa<::mlir::arm_sve::ScalableVectorType>())) && (((type.cast<::mlir::arm_sve::ScalableVectorType>().getNumElements() == 4)) || ((type.cast<::mlir::arm_sve::ScalableVectorType>().getNumElements() == 2)))))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be scalable vector of 32-bit signless integer or 64-bit signless integer values of length 4/2, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_ArmSVE9(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((((type.isa<::mlir::arm_sve::ScalableVectorType>())) && (((type.cast<::mlir::arm_sve::ScalableVectorType>().getElementType().isSignlessInteger(8))) || ((type.cast<::mlir::arm_sve::ScalableVectorType>().getElementType().isSignlessInteger(16))))) && (((type.isa<::mlir::arm_sve::ScalableVectorType>())) && (((type.cast<::mlir::arm_sve::ScalableVectorType>().getNumElements() == 16)) || ((type.cast<::mlir::arm_sve::ScalableVectorType>().getNumElements() == 8)))))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be scalable vector of 8-bit signless integer or 16-bit signless integer values of length 16/8, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_ArmSVE10(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((((type.isa<::mlir::arm_sve::ScalableVectorType>())) && ((type.cast<::mlir::arm_sve::ScalableVectorType>().getElementType().isSignlessInteger(32)))) && (((type.isa<::mlir::arm_sve::ScalableVectorType>())) && ((type.cast<::mlir::arm_sve::ScalableVectorType>().getNumElements() == 4))))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be scalable vector of 32-bit signless integer values of length 4, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_ArmSVE11(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((((type.isa<::mlir::arm_sve::ScalableVectorType>())) && ((type.cast<::mlir::arm_sve::ScalableVectorType>().getElementType().isSignlessInteger(8)))) && (((type.isa<::mlir::arm_sve::ScalableVectorType>())) && ((type.cast<::mlir::arm_sve::ScalableVectorType>().getNumElements() == 16))))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be scalable vector of 8-bit signless integer values of length 16, but got " << type;
  }
  return ::mlir::success();
}

} // namespace arm_sve
} // namespace mlir
namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableAddFOp definitions
//===----------------------------------------------------------------------===//

ScalableAddFOpAdaptor::ScalableAddFOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ScalableAddFOpAdaptor::ScalableAddFOpAdaptor(ScalableAddFOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ScalableAddFOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ScalableAddFOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ScalableAddFOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableAddFOpAdaptor::src1() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScalableAddFOpAdaptor::src2() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr ScalableAddFOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ScalableAddFOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> ScalableAddFOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ScalableAddFOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableAddFOp::src1() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScalableAddFOp::src2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange ScalableAddFOp::src1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScalableAddFOp::src2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ScalableAddFOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ScalableAddFOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableAddFOp::dst() {
  return *getODSResults(0).begin();
}

void ScalableAddFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dst, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  odsState.addTypes(dst);
}

void ScalableAddFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ScalableAddFOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ScalableAddFOp::verify() {
  if (::mlir::failed(ScalableAddFOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(0).begin()).getType(), (*this->getODSOperands(1).begin()).getType(), (*this->getODSResults(0).begin()).getType()})))))
    return emitOpError("failed to verify that all of {src1, src2, dst} have same type");
  return ::mlir::success();
}

::mlir::ParseResult ScalableAddFOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType src1RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> src1Operands(src1RawOperands);  ::llvm::SMLoc src1OperandsLoc;
  (void)src1OperandsLoc;
  ::mlir::OpAsmParser::OperandType src2RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> src2Operands(src2RawOperands);  ::llvm::SMLoc src2OperandsLoc;
  (void)src2OperandsLoc;
  ::mlir::Type src1RawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> src1Types(src1RawTypes);

  src1OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src1RawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  src2OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src2RawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(src1RawTypes[0]))
    return ::mlir::failure();
  result.addTypes(src1Types);
  if (parser.resolveOperands(src1Operands, src1Types, src1OperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(src2Operands, src1Types[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ScalableAddFOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << src1();
  p << ",";
  p << ' ';
  p << src2();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(src1().getType());
}

} // namespace arm_sve
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableAddFOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableAddIOp definitions
//===----------------------------------------------------------------------===//

ScalableAddIOpAdaptor::ScalableAddIOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ScalableAddIOpAdaptor::ScalableAddIOpAdaptor(ScalableAddIOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ScalableAddIOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ScalableAddIOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ScalableAddIOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableAddIOpAdaptor::src1() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScalableAddIOpAdaptor::src2() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr ScalableAddIOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ScalableAddIOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> ScalableAddIOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ScalableAddIOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableAddIOp::src1() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScalableAddIOp::src2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange ScalableAddIOp::src1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScalableAddIOp::src2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ScalableAddIOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ScalableAddIOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableAddIOp::dst() {
  return *getODSResults(0).begin();
}

void ScalableAddIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dst, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  odsState.addTypes(dst);
}

void ScalableAddIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ScalableAddIOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ScalableAddIOp::verify() {
  if (::mlir::failed(ScalableAddIOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(0).begin()).getType(), (*this->getODSOperands(1).begin()).getType(), (*this->getODSResults(0).begin()).getType()})))))
    return emitOpError("failed to verify that all of {src1, src2, dst} have same type");
  return ::mlir::success();
}

::mlir::ParseResult ScalableAddIOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType src1RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> src1Operands(src1RawOperands);  ::llvm::SMLoc src1OperandsLoc;
  (void)src1OperandsLoc;
  ::mlir::OpAsmParser::OperandType src2RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> src2Operands(src2RawOperands);  ::llvm::SMLoc src2OperandsLoc;
  (void)src2OperandsLoc;
  ::mlir::Type src1RawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> src1Types(src1RawTypes);

  src1OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src1RawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  src2OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src2RawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(src1RawTypes[0]))
    return ::mlir::failure();
  result.addTypes(src1Types);
  if (parser.resolveOperands(src1Operands, src1Types, src1OperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(src2Operands, src1Types[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ScalableAddIOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << src1();
  p << ",";
  p << ' ';
  p << src2();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(src1().getType());
}

} // namespace arm_sve
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableAddIOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableCmpFOp definitions
//===----------------------------------------------------------------------===//

ScalableCmpFOpAdaptor::ScalableCmpFOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ScalableCmpFOpAdaptor::ScalableCmpFOpAdaptor(ScalableCmpFOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ScalableCmpFOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ScalableCmpFOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ScalableCmpFOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableCmpFOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScalableCmpFOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr ScalableCmpFOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::CmpFPredicateAttr ScalableCmpFOpAdaptor::predicate() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::CmpFPredicateAttr attr = odsAttrs.get("predicate").cast<::mlir::CmpFPredicateAttr>();
  return attr;
}

::mlir::LogicalResult ScalableCmpFOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_predicate = odsAttrs.get("predicate");
  if (!tblgen_predicate) return emitError(loc, "'arm_sve.cmpf' op ""requires attribute 'predicate'");
    if (!((tblgen_predicate.isa<::mlir::CmpFPredicateAttr>()))) return emitError(loc, "'arm_sve.cmpf' op ""attribute 'predicate' failed to satisfy constraint: allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15");
  }
  return ::mlir::success();
}













std::pair<unsigned, unsigned> ScalableCmpFOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ScalableCmpFOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableCmpFOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScalableCmpFOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange ScalableCmpFOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScalableCmpFOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ScalableCmpFOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ScalableCmpFOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableCmpFOp::result() {
  return *getODSResults(0).begin();
}

::mlir::CmpFPredicateAttr ScalableCmpFOp::predicateAttr() {
  return (*this)->getAttr(predicateAttrName()).template cast<::mlir::CmpFPredicateAttr>();
}

::mlir::CmpFPredicate ScalableCmpFOp::predicate() {
  auto attr = predicateAttr();
  return attr.getValue();
}

void ScalableCmpFOp::predicateAttr(::mlir::CmpFPredicateAttr attr) {
  (*this)->setAttr(predicateAttrName(), attr);
}

void ScalableCmpFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, CmpFPredicate predicate, Value lhs, Value rhs) {
      buildScalableCmpFOp(odsBuilder, odsState, predicate, lhs, rhs);
    
}

void ScalableCmpFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::CmpFPredicateAttr predicate, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(predicateAttrName(odsState.name), predicate);
  odsState.addTypes(result);
}

void ScalableCmpFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::CmpFPredicateAttr predicate, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(predicateAttrName(odsState.name), predicate);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ScalableCmpFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::CmpFPredicate predicate, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(predicateAttrName(odsState.name), ::mlir::CmpFPredicateAttr::get(odsBuilder.getContext(), predicate));
  odsState.addTypes(result);
}

void ScalableCmpFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::CmpFPredicate predicate, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(predicateAttrName(odsState.name), ::mlir::CmpFPredicateAttr::get(odsBuilder.getContext(), predicate));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ScalableCmpFOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ScalableCmpFOp::verify() {
  if (::mlir::failed(ScalableCmpFOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  if (!((std::equal_to<>()(getI1SameShape((*this->getODSOperands(0).begin()).getType()), (*this->getODSResults(0).begin()).getType()))))
    return emitOpError("failed to verify that result type has i1 element type and same shape as operands");
  return success();
}

::mlir::ParseResult ScalableCmpFOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::CmpFPredicateAttr predicateAttr;
  ::mlir::OpAsmParser::OperandType lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::OperandType rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);

  {
    ::llvm::StringRef attrStr;
    ::mlir::NamedAttrList attrStorage;
    auto loc = parser.getCurrentLocation();
    if (parser.parseOptionalKeyword(&attrStr, {"false","oeq","ogt","oge","olt","ole","one","ord","ueq","ugt","uge","ult","ule","une","uno","true"})) {
      ::mlir::StringAttr attrVal;
      ::mlir::OptionalParseResult parseResult =
        parser.parseOptionalAttribute(attrVal,
                                      parser.getBuilder().getNoneType(),
                                      "predicate", attrStorage);
      if (parseResult.hasValue()) {
        if (failed(*parseResult))
          return ::mlir::failure();
        attrStr = attrVal.getValue();
      } else {
        return parser.emitError(loc, "expected string or keyword containing one of the following enum values for attribute 'predicate' [false, oeq, ogt, oge, olt, ole, one, ord, ueq, ugt, uge, ult, ule, une, uno, true]");
      }
    }
    if (!attrStr.empty()) {
      auto attrOptional = ::mlir::symbolizeCmpFPredicate(attrStr);
      if (!attrOptional)
        return parser.emitError(loc, "invalid ")
               << "predicate attribute specification: \"" << attrStr << '"';;

      predicateAttr = ::mlir::CmpFPredicateAttr::get(parser.getBuilder().getContext(), attrOptional.getValue());
      result.addAttribute("predicate", predicateAttr);
    }
  }
  if (parser.parseComma())
    return ::mlir::failure();

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(lhsRawTypes[0]))
    return ::mlir::failure();
  for (::mlir::Type type : lhsTypes) {
    (void)type;
    if (!(((type.isa<::mlir::arm_sve::ScalableVectorType>())) && ((type.cast<::mlir::arm_sve::ScalableVectorType>().getElementType().isa<::mlir::FloatType>())))) {
      return parser.emitError(parser.getNameLoc()) << "'lhs' must be scalable vector of floating-point values, but got " << type;
    }
  }
  result.addTypes(getI1SameShape(lhsTypes[0]));
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, lhsTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ScalableCmpFOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';

  {
    auto caseValue = predicate();
    auto caseValueStr = stringifyCmpFPredicate(caseValue);
    p << caseValueStr;
  }
  p << ",";
  p << ' ';
  p << lhs();
  p << ",";
  p << ' ';
  p << rhs();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"predicate"});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(lhs().getType());
}

void ScalableCmpFOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace arm_sve
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableCmpFOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableCmpIOp definitions
//===----------------------------------------------------------------------===//

ScalableCmpIOpAdaptor::ScalableCmpIOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ScalableCmpIOpAdaptor::ScalableCmpIOpAdaptor(ScalableCmpIOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ScalableCmpIOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ScalableCmpIOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ScalableCmpIOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableCmpIOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScalableCmpIOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr ScalableCmpIOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::CmpIPredicateAttr ScalableCmpIOpAdaptor::predicate() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::CmpIPredicateAttr attr = odsAttrs.get("predicate").cast<::mlir::CmpIPredicateAttr>();
  return attr;
}

::mlir::LogicalResult ScalableCmpIOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_predicate = odsAttrs.get("predicate");
  if (!tblgen_predicate) return emitError(loc, "'arm_sve.cmpi' op ""requires attribute 'predicate'");
    if (!((tblgen_predicate.isa<::mlir::CmpIPredicateAttr>()))) return emitError(loc, "'arm_sve.cmpi' op ""attribute 'predicate' failed to satisfy constraint: allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9");
  }
  return ::mlir::success();
}













std::pair<unsigned, unsigned> ScalableCmpIOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ScalableCmpIOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableCmpIOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScalableCmpIOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange ScalableCmpIOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScalableCmpIOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ScalableCmpIOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ScalableCmpIOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableCmpIOp::result() {
  return *getODSResults(0).begin();
}

::mlir::CmpIPredicateAttr ScalableCmpIOp::predicateAttr() {
  return (*this)->getAttr(predicateAttrName()).template cast<::mlir::CmpIPredicateAttr>();
}

::mlir::CmpIPredicate ScalableCmpIOp::predicate() {
  auto attr = predicateAttr();
  return attr.getValue();
}

void ScalableCmpIOp::predicateAttr(::mlir::CmpIPredicateAttr attr) {
  (*this)->setAttr(predicateAttrName(), attr);
}

void ScalableCmpIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, CmpIPredicate predicate, Value lhs, Value rhs) {
      buildScalableCmpIOp(odsBuilder, odsState, predicate, lhs, rhs);
    
}

void ScalableCmpIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::CmpIPredicateAttr predicate, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(predicateAttrName(odsState.name), predicate);
  odsState.addTypes(result);
}

void ScalableCmpIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::CmpIPredicateAttr predicate, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(predicateAttrName(odsState.name), predicate);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ScalableCmpIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::CmpIPredicate predicate, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(predicateAttrName(odsState.name), ::mlir::CmpIPredicateAttr::get(odsBuilder.getContext(), predicate));
  odsState.addTypes(result);
}

void ScalableCmpIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::CmpIPredicate predicate, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(predicateAttrName(odsState.name), ::mlir::CmpIPredicateAttr::get(odsBuilder.getContext(), predicate));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ScalableCmpIOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ScalableCmpIOp::verify() {
  if (::mlir::failed(ScalableCmpIOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  if (!((std::equal_to<>()(getI1SameShape((*this->getODSOperands(0).begin()).getType()), (*this->getODSResults(0).begin()).getType()))))
    return emitOpError("failed to verify that result type has i1 element type and same shape as operands");
  return success();
}

::mlir::ParseResult ScalableCmpIOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::CmpIPredicateAttr predicateAttr;
  ::mlir::OpAsmParser::OperandType lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::OperandType rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);

  {
    ::llvm::StringRef attrStr;
    ::mlir::NamedAttrList attrStorage;
    auto loc = parser.getCurrentLocation();
    if (parser.parseOptionalKeyword(&attrStr, {"eq","ne","slt","sle","sgt","sge","ult","ule","ugt","uge"})) {
      ::mlir::StringAttr attrVal;
      ::mlir::OptionalParseResult parseResult =
        parser.parseOptionalAttribute(attrVal,
                                      parser.getBuilder().getNoneType(),
                                      "predicate", attrStorage);
      if (parseResult.hasValue()) {
        if (failed(*parseResult))
          return ::mlir::failure();
        attrStr = attrVal.getValue();
      } else {
        return parser.emitError(loc, "expected string or keyword containing one of the following enum values for attribute 'predicate' [eq, ne, slt, sle, sgt, sge, ult, ule, ugt, uge]");
      }
    }
    if (!attrStr.empty()) {
      auto attrOptional = ::mlir::symbolizeCmpIPredicate(attrStr);
      if (!attrOptional)
        return parser.emitError(loc, "invalid ")
               << "predicate attribute specification: \"" << attrStr << '"';;

      predicateAttr = ::mlir::CmpIPredicateAttr::get(parser.getBuilder().getContext(), attrOptional.getValue());
      result.addAttribute("predicate", predicateAttr);
    }
  }
  if (parser.parseComma())
    return ::mlir::failure();

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(lhsRawTypes[0]))
    return ::mlir::failure();
  for (::mlir::Type type : lhsTypes) {
    (void)type;
    if (!(((type.isa<::mlir::arm_sve::ScalableVectorType>())) && (((type.cast<::mlir::arm_sve::ScalableVectorType>().getElementType().isSignlessInteger(8))) || ((type.cast<::mlir::arm_sve::ScalableVectorType>().getElementType().isSignlessInteger(16))) || ((type.cast<::mlir::arm_sve::ScalableVectorType>().getElementType().isSignlessInteger(32))) || ((type.cast<::mlir::arm_sve::ScalableVectorType>().getElementType().isSignlessInteger(64)))))) {
      return parser.emitError(parser.getNameLoc()) << "'lhs' must be scalable vector of 8-bit signless integer or 16-bit signless integer or 32-bit signless integer or 64-bit signless integer values, but got " << type;
    }
  }
  result.addTypes(getI1SameShape(lhsTypes[0]));
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, lhsTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ScalableCmpIOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';

  {
    auto caseValue = predicate();
    auto caseValueStr = stringifyCmpIPredicate(caseValue);
    p << caseValueStr;
  }
  p << ",";
  p << ' ';
  p << lhs();
  p << ",";
  p << ' ';
  p << rhs();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"predicate"});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(lhs().getType());
}

void ScalableCmpIOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace arm_sve
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableCmpIOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableDivFOp definitions
//===----------------------------------------------------------------------===//

ScalableDivFOpAdaptor::ScalableDivFOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ScalableDivFOpAdaptor::ScalableDivFOpAdaptor(ScalableDivFOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ScalableDivFOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ScalableDivFOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ScalableDivFOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableDivFOpAdaptor::src1() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScalableDivFOpAdaptor::src2() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr ScalableDivFOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ScalableDivFOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> ScalableDivFOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ScalableDivFOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableDivFOp::src1() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScalableDivFOp::src2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange ScalableDivFOp::src1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScalableDivFOp::src2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ScalableDivFOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ScalableDivFOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableDivFOp::dst() {
  return *getODSResults(0).begin();
}

void ScalableDivFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dst, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  odsState.addTypes(dst);
}

void ScalableDivFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ScalableDivFOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ScalableDivFOp::verify() {
  if (::mlir::failed(ScalableDivFOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(0).begin()).getType(), (*this->getODSOperands(1).begin()).getType(), (*this->getODSResults(0).begin()).getType()})))))
    return emitOpError("failed to verify that all of {src1, src2, dst} have same type");
  return ::mlir::success();
}

::mlir::ParseResult ScalableDivFOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType src1RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> src1Operands(src1RawOperands);  ::llvm::SMLoc src1OperandsLoc;
  (void)src1OperandsLoc;
  ::mlir::OpAsmParser::OperandType src2RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> src2Operands(src2RawOperands);  ::llvm::SMLoc src2OperandsLoc;
  (void)src2OperandsLoc;
  ::mlir::Type src1RawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> src1Types(src1RawTypes);

  src1OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src1RawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  src2OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src2RawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(src1RawTypes[0]))
    return ::mlir::failure();
  result.addTypes(src1Types);
  if (parser.resolveOperands(src1Operands, src1Types, src1OperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(src2Operands, src1Types[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ScalableDivFOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << src1();
  p << ",";
  p << ' ';
  p << src2();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(src1().getType());
}

} // namespace arm_sve
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableDivFOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableLoadOp definitions
//===----------------------------------------------------------------------===//

ScalableLoadOpAdaptor::ScalableLoadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ScalableLoadOpAdaptor::ScalableLoadOpAdaptor(ScalableLoadOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ScalableLoadOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ScalableLoadOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ScalableLoadOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableLoadOpAdaptor::base() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScalableLoadOpAdaptor::index() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr ScalableLoadOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ScalableLoadOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> ScalableLoadOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ScalableLoadOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableLoadOp::base() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScalableLoadOp::index() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange ScalableLoadOp::baseMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScalableLoadOp::indexMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ScalableLoadOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ScalableLoadOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableLoadOp::result() {
  return *getODSResults(0).begin();
}

void ScalableLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value base, ::mlir::Value index) {
  odsState.addOperands(base);
  odsState.addOperands(index);
  odsState.addTypes(result);
}

void ScalableLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value base, ::mlir::Value index) {
  odsState.addOperands(base);
  odsState.addOperands(index);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ScalableLoadOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ScalableLoadOp::verify() {
  if (::mlir::failed(ScalableLoadOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE4(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE5(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult ScalableLoadOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType baseRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> baseOperands(baseRawOperands);  ::llvm::SMLoc baseOperandsLoc;
  (void)baseOperandsLoc;
  ::mlir::OpAsmParser::OperandType indexRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> indexOperands(indexRawOperands);  ::llvm::SMLoc indexOperandsLoc;
  (void)indexOperandsLoc;
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);
  ::mlir::Type baseRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> baseTypes(baseRawTypes);

  baseOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(baseRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  indexOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(indexRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resultRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("from"))
    return ::mlir::failure();

  if (parser.parseType(baseRawTypes[0]))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(resultTypes);
  if (parser.resolveOperands(baseOperands, baseTypes, baseOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(indexOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ScalableLoadOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << base();
  p << "[";
  p << index();
  p << "]";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(result().getType());
  p << ' ' << "from";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(base().getType());
}

void ScalableLoadOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(::mlir::MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace arm_sve
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableLoadOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableMaskedAddFIntrOp definitions
//===----------------------------------------------------------------------===//

ScalableMaskedAddFIntrOpAdaptor::ScalableMaskedAddFIntrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ScalableMaskedAddFIntrOpAdaptor::ScalableMaskedAddFIntrOpAdaptor(ScalableMaskedAddFIntrOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ScalableMaskedAddFIntrOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ScalableMaskedAddFIntrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ScalableMaskedAddFIntrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr ScalableMaskedAddFIntrOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ScalableMaskedAddFIntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> ScalableMaskedAddFIntrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ScalableMaskedAddFIntrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> ScalableMaskedAddFIntrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ScalableMaskedAddFIntrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableMaskedAddFIntrOp::res() {
  return *getODSResults(0).begin();
}

void ScalableMaskedAddFIntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addOperands(odsArg_2);
  odsState.addTypes(res);
}

void ScalableMaskedAddFIntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addOperands(odsArg_2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ScalableMaskedAddFIntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ScalableMaskedAddFIntrOp::verify() {
  if (::mlir::failed(ScalableMaskedAddFIntrOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE7(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

} // namespace arm_sve
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableMaskedAddFIntrOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableMaskedAddFOp definitions
//===----------------------------------------------------------------------===//

ScalableMaskedAddFOpAdaptor::ScalableMaskedAddFOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ScalableMaskedAddFOpAdaptor::ScalableMaskedAddFOpAdaptor(ScalableMaskedAddFOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ScalableMaskedAddFOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ScalableMaskedAddFOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ScalableMaskedAddFOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableMaskedAddFOpAdaptor::mask() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScalableMaskedAddFOpAdaptor::src1() {
  return *getODSOperands(1).begin();
}

::mlir::Value ScalableMaskedAddFOpAdaptor::src2() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr ScalableMaskedAddFOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ScalableMaskedAddFOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> ScalableMaskedAddFOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ScalableMaskedAddFOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableMaskedAddFOp::mask() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScalableMaskedAddFOp::src1() {
  return *getODSOperands(1).begin();
}

::mlir::Value ScalableMaskedAddFOp::src2() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange ScalableMaskedAddFOp::maskMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScalableMaskedAddFOp::src1Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScalableMaskedAddFOp::src2Mutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ScalableMaskedAddFOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ScalableMaskedAddFOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableMaskedAddFOp::res() {
  return *getODSResults(0).begin();
}

void ScalableMaskedAddFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value mask, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(mask);
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  odsState.addTypes(res);
}

void ScalableMaskedAddFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value mask, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(mask);
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ScalableMaskedAddFOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ScalableMaskedAddFOp::verify() {
  if (::mlir::failed(ScalableMaskedAddFOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(1).begin()).getType(), (*this->getODSOperands(2).begin()).getType(), (*this->getODSResults(0).begin()).getType()})))))
    return emitOpError("failed to verify that all of {src1, src2, res} have same type");
  if (!((std::equal_to<>()(getI1SameShape((*this->getODSOperands(1).begin()).getType()), (*this->getODSOperands(0).begin()).getType()))))
    return emitOpError("failed to verify that mask has i1 element type and same shape as operands");
  return ::mlir::success();
}

::mlir::ParseResult ScalableMaskedAddFOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType maskRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> maskOperands(maskRawOperands);  ::llvm::SMLoc maskOperandsLoc;
  (void)maskOperandsLoc;
  ::mlir::OpAsmParser::OperandType src1RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> src1Operands(src1RawOperands);  ::llvm::SMLoc src1OperandsLoc;
  (void)src1OperandsLoc;
  ::mlir::OpAsmParser::OperandType src2RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> src2Operands(src2RawOperands);  ::llvm::SMLoc src2OperandsLoc;
  (void)src2OperandsLoc;
  ::mlir::Type maskRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> maskTypes(maskRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  maskOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(maskRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  src1OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src1RawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  src2OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src2RawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(maskRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resTypes);
  if (parser.resolveOperands(maskOperands, maskTypes, maskOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(src1Operands, resTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(src2Operands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ScalableMaskedAddFOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << mask();
  p << ",";
  p << ' ';
  p << src1();
  p << ",";
  p << ' ';
  p << src2();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(mask().getType());
  p << ",";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

} // namespace arm_sve
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableMaskedAddFOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableMaskedAddIIntrOp definitions
//===----------------------------------------------------------------------===//

ScalableMaskedAddIIntrOpAdaptor::ScalableMaskedAddIIntrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ScalableMaskedAddIIntrOpAdaptor::ScalableMaskedAddIIntrOpAdaptor(ScalableMaskedAddIIntrOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ScalableMaskedAddIIntrOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ScalableMaskedAddIIntrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ScalableMaskedAddIIntrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr ScalableMaskedAddIIntrOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ScalableMaskedAddIIntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> ScalableMaskedAddIIntrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ScalableMaskedAddIIntrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> ScalableMaskedAddIIntrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ScalableMaskedAddIIntrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableMaskedAddIIntrOp::res() {
  return *getODSResults(0).begin();
}

void ScalableMaskedAddIIntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addOperands(odsArg_2);
  odsState.addTypes(res);
}

void ScalableMaskedAddIIntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addOperands(odsArg_2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ScalableMaskedAddIIntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ScalableMaskedAddIIntrOp::verify() {
  if (::mlir::failed(ScalableMaskedAddIIntrOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE7(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

} // namespace arm_sve
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableMaskedAddIIntrOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableMaskedAddIOp definitions
//===----------------------------------------------------------------------===//

ScalableMaskedAddIOpAdaptor::ScalableMaskedAddIOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ScalableMaskedAddIOpAdaptor::ScalableMaskedAddIOpAdaptor(ScalableMaskedAddIOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ScalableMaskedAddIOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ScalableMaskedAddIOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ScalableMaskedAddIOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableMaskedAddIOpAdaptor::mask() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScalableMaskedAddIOpAdaptor::src1() {
  return *getODSOperands(1).begin();
}

::mlir::Value ScalableMaskedAddIOpAdaptor::src2() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr ScalableMaskedAddIOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ScalableMaskedAddIOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> ScalableMaskedAddIOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ScalableMaskedAddIOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableMaskedAddIOp::mask() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScalableMaskedAddIOp::src1() {
  return *getODSOperands(1).begin();
}

::mlir::Value ScalableMaskedAddIOp::src2() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange ScalableMaskedAddIOp::maskMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScalableMaskedAddIOp::src1Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScalableMaskedAddIOp::src2Mutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ScalableMaskedAddIOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ScalableMaskedAddIOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableMaskedAddIOp::res() {
  return *getODSResults(0).begin();
}

void ScalableMaskedAddIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value mask, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(mask);
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  odsState.addTypes(res);
}

void ScalableMaskedAddIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value mask, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(mask);
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ScalableMaskedAddIOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ScalableMaskedAddIOp::verify() {
  if (::mlir::failed(ScalableMaskedAddIOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(1).begin()).getType(), (*this->getODSOperands(2).begin()).getType(), (*this->getODSResults(0).begin()).getType()})))))
    return emitOpError("failed to verify that all of {src1, src2, res} have same type");
  if (!((std::equal_to<>()(getI1SameShape((*this->getODSOperands(1).begin()).getType()), (*this->getODSOperands(0).begin()).getType()))))
    return emitOpError("failed to verify that mask has i1 element type and same shape as operands");
  return ::mlir::success();
}

::mlir::ParseResult ScalableMaskedAddIOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType maskRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> maskOperands(maskRawOperands);  ::llvm::SMLoc maskOperandsLoc;
  (void)maskOperandsLoc;
  ::mlir::OpAsmParser::OperandType src1RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> src1Operands(src1RawOperands);  ::llvm::SMLoc src1OperandsLoc;
  (void)src1OperandsLoc;
  ::mlir::OpAsmParser::OperandType src2RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> src2Operands(src2RawOperands);  ::llvm::SMLoc src2OperandsLoc;
  (void)src2OperandsLoc;
  ::mlir::Type maskRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> maskTypes(maskRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  maskOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(maskRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  src1OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src1RawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  src2OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src2RawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(maskRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resTypes);
  if (parser.resolveOperands(maskOperands, maskTypes, maskOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(src1Operands, resTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(src2Operands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ScalableMaskedAddIOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << mask();
  p << ",";
  p << ' ';
  p << src1();
  p << ",";
  p << ' ';
  p << src2();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(mask().getType());
  p << ",";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

} // namespace arm_sve
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableMaskedAddIOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableMaskedDivFIntrOp definitions
//===----------------------------------------------------------------------===//

ScalableMaskedDivFIntrOpAdaptor::ScalableMaskedDivFIntrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ScalableMaskedDivFIntrOpAdaptor::ScalableMaskedDivFIntrOpAdaptor(ScalableMaskedDivFIntrOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ScalableMaskedDivFIntrOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ScalableMaskedDivFIntrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ScalableMaskedDivFIntrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr ScalableMaskedDivFIntrOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ScalableMaskedDivFIntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> ScalableMaskedDivFIntrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ScalableMaskedDivFIntrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> ScalableMaskedDivFIntrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ScalableMaskedDivFIntrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableMaskedDivFIntrOp::res() {
  return *getODSResults(0).begin();
}

void ScalableMaskedDivFIntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addOperands(odsArg_2);
  odsState.addTypes(res);
}

void ScalableMaskedDivFIntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addOperands(odsArg_2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ScalableMaskedDivFIntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ScalableMaskedDivFIntrOp::verify() {
  if (::mlir::failed(ScalableMaskedDivFIntrOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE7(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

} // namespace arm_sve
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableMaskedDivFIntrOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableMaskedDivFOp definitions
//===----------------------------------------------------------------------===//

ScalableMaskedDivFOpAdaptor::ScalableMaskedDivFOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ScalableMaskedDivFOpAdaptor::ScalableMaskedDivFOpAdaptor(ScalableMaskedDivFOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ScalableMaskedDivFOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ScalableMaskedDivFOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ScalableMaskedDivFOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableMaskedDivFOpAdaptor::mask() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScalableMaskedDivFOpAdaptor::src1() {
  return *getODSOperands(1).begin();
}

::mlir::Value ScalableMaskedDivFOpAdaptor::src2() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr ScalableMaskedDivFOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ScalableMaskedDivFOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> ScalableMaskedDivFOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ScalableMaskedDivFOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableMaskedDivFOp::mask() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScalableMaskedDivFOp::src1() {
  return *getODSOperands(1).begin();
}

::mlir::Value ScalableMaskedDivFOp::src2() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange ScalableMaskedDivFOp::maskMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScalableMaskedDivFOp::src1Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScalableMaskedDivFOp::src2Mutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ScalableMaskedDivFOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ScalableMaskedDivFOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableMaskedDivFOp::res() {
  return *getODSResults(0).begin();
}

void ScalableMaskedDivFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value mask, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(mask);
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  odsState.addTypes(res);
}

void ScalableMaskedDivFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value mask, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(mask);
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ScalableMaskedDivFOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ScalableMaskedDivFOp::verify() {
  if (::mlir::failed(ScalableMaskedDivFOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(1).begin()).getType(), (*this->getODSOperands(2).begin()).getType(), (*this->getODSResults(0).begin()).getType()})))))
    return emitOpError("failed to verify that all of {src1, src2, res} have same type");
  if (!((std::equal_to<>()(getI1SameShape((*this->getODSOperands(1).begin()).getType()), (*this->getODSOperands(0).begin()).getType()))))
    return emitOpError("failed to verify that mask has i1 element type and same shape as operands");
  return ::mlir::success();
}

::mlir::ParseResult ScalableMaskedDivFOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType maskRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> maskOperands(maskRawOperands);  ::llvm::SMLoc maskOperandsLoc;
  (void)maskOperandsLoc;
  ::mlir::OpAsmParser::OperandType src1RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> src1Operands(src1RawOperands);  ::llvm::SMLoc src1OperandsLoc;
  (void)src1OperandsLoc;
  ::mlir::OpAsmParser::OperandType src2RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> src2Operands(src2RawOperands);  ::llvm::SMLoc src2OperandsLoc;
  (void)src2OperandsLoc;
  ::mlir::Type maskRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> maskTypes(maskRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  maskOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(maskRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  src1OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src1RawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  src2OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src2RawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(maskRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resTypes);
  if (parser.resolveOperands(maskOperands, maskTypes, maskOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(src1Operands, resTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(src2Operands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ScalableMaskedDivFOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << mask();
  p << ",";
  p << ' ';
  p << src1();
  p << ",";
  p << ' ';
  p << src2();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(mask().getType());
  p << ",";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

} // namespace arm_sve
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableMaskedDivFOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableMaskedMulFIntrOp definitions
//===----------------------------------------------------------------------===//

ScalableMaskedMulFIntrOpAdaptor::ScalableMaskedMulFIntrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ScalableMaskedMulFIntrOpAdaptor::ScalableMaskedMulFIntrOpAdaptor(ScalableMaskedMulFIntrOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ScalableMaskedMulFIntrOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ScalableMaskedMulFIntrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ScalableMaskedMulFIntrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr ScalableMaskedMulFIntrOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ScalableMaskedMulFIntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> ScalableMaskedMulFIntrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ScalableMaskedMulFIntrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> ScalableMaskedMulFIntrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ScalableMaskedMulFIntrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableMaskedMulFIntrOp::res() {
  return *getODSResults(0).begin();
}

void ScalableMaskedMulFIntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addOperands(odsArg_2);
  odsState.addTypes(res);
}

void ScalableMaskedMulFIntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addOperands(odsArg_2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ScalableMaskedMulFIntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ScalableMaskedMulFIntrOp::verify() {
  if (::mlir::failed(ScalableMaskedMulFIntrOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE7(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

} // namespace arm_sve
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableMaskedMulFIntrOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableMaskedMulFOp definitions
//===----------------------------------------------------------------------===//

ScalableMaskedMulFOpAdaptor::ScalableMaskedMulFOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ScalableMaskedMulFOpAdaptor::ScalableMaskedMulFOpAdaptor(ScalableMaskedMulFOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ScalableMaskedMulFOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ScalableMaskedMulFOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ScalableMaskedMulFOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableMaskedMulFOpAdaptor::mask() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScalableMaskedMulFOpAdaptor::src1() {
  return *getODSOperands(1).begin();
}

::mlir::Value ScalableMaskedMulFOpAdaptor::src2() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr ScalableMaskedMulFOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ScalableMaskedMulFOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> ScalableMaskedMulFOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ScalableMaskedMulFOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableMaskedMulFOp::mask() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScalableMaskedMulFOp::src1() {
  return *getODSOperands(1).begin();
}

::mlir::Value ScalableMaskedMulFOp::src2() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange ScalableMaskedMulFOp::maskMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScalableMaskedMulFOp::src1Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScalableMaskedMulFOp::src2Mutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ScalableMaskedMulFOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ScalableMaskedMulFOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableMaskedMulFOp::res() {
  return *getODSResults(0).begin();
}

void ScalableMaskedMulFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value mask, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(mask);
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  odsState.addTypes(res);
}

void ScalableMaskedMulFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value mask, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(mask);
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ScalableMaskedMulFOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ScalableMaskedMulFOp::verify() {
  if (::mlir::failed(ScalableMaskedMulFOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(1).begin()).getType(), (*this->getODSOperands(2).begin()).getType(), (*this->getODSResults(0).begin()).getType()})))))
    return emitOpError("failed to verify that all of {src1, src2, res} have same type");
  if (!((std::equal_to<>()(getI1SameShape((*this->getODSOperands(1).begin()).getType()), (*this->getODSOperands(0).begin()).getType()))))
    return emitOpError("failed to verify that mask has i1 element type and same shape as operands");
  return ::mlir::success();
}

::mlir::ParseResult ScalableMaskedMulFOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType maskRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> maskOperands(maskRawOperands);  ::llvm::SMLoc maskOperandsLoc;
  (void)maskOperandsLoc;
  ::mlir::OpAsmParser::OperandType src1RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> src1Operands(src1RawOperands);  ::llvm::SMLoc src1OperandsLoc;
  (void)src1OperandsLoc;
  ::mlir::OpAsmParser::OperandType src2RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> src2Operands(src2RawOperands);  ::llvm::SMLoc src2OperandsLoc;
  (void)src2OperandsLoc;
  ::mlir::Type maskRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> maskTypes(maskRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  maskOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(maskRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  src1OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src1RawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  src2OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src2RawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(maskRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resTypes);
  if (parser.resolveOperands(maskOperands, maskTypes, maskOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(src1Operands, resTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(src2Operands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ScalableMaskedMulFOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << mask();
  p << ",";
  p << ' ';
  p << src1();
  p << ",";
  p << ' ';
  p << src2();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(mask().getType());
  p << ",";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

} // namespace arm_sve
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableMaskedMulFOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableMaskedMulIIntrOp definitions
//===----------------------------------------------------------------------===//

ScalableMaskedMulIIntrOpAdaptor::ScalableMaskedMulIIntrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ScalableMaskedMulIIntrOpAdaptor::ScalableMaskedMulIIntrOpAdaptor(ScalableMaskedMulIIntrOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ScalableMaskedMulIIntrOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ScalableMaskedMulIIntrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ScalableMaskedMulIIntrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr ScalableMaskedMulIIntrOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ScalableMaskedMulIIntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> ScalableMaskedMulIIntrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ScalableMaskedMulIIntrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> ScalableMaskedMulIIntrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ScalableMaskedMulIIntrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableMaskedMulIIntrOp::res() {
  return *getODSResults(0).begin();
}

void ScalableMaskedMulIIntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addOperands(odsArg_2);
  odsState.addTypes(res);
}

void ScalableMaskedMulIIntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addOperands(odsArg_2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ScalableMaskedMulIIntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ScalableMaskedMulIIntrOp::verify() {
  if (::mlir::failed(ScalableMaskedMulIIntrOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE7(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

} // namespace arm_sve
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableMaskedMulIIntrOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableMaskedMulIOp definitions
//===----------------------------------------------------------------------===//

ScalableMaskedMulIOpAdaptor::ScalableMaskedMulIOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ScalableMaskedMulIOpAdaptor::ScalableMaskedMulIOpAdaptor(ScalableMaskedMulIOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ScalableMaskedMulIOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ScalableMaskedMulIOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ScalableMaskedMulIOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableMaskedMulIOpAdaptor::mask() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScalableMaskedMulIOpAdaptor::src1() {
  return *getODSOperands(1).begin();
}

::mlir::Value ScalableMaskedMulIOpAdaptor::src2() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr ScalableMaskedMulIOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ScalableMaskedMulIOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> ScalableMaskedMulIOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ScalableMaskedMulIOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableMaskedMulIOp::mask() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScalableMaskedMulIOp::src1() {
  return *getODSOperands(1).begin();
}

::mlir::Value ScalableMaskedMulIOp::src2() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange ScalableMaskedMulIOp::maskMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScalableMaskedMulIOp::src1Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScalableMaskedMulIOp::src2Mutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ScalableMaskedMulIOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ScalableMaskedMulIOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableMaskedMulIOp::res() {
  return *getODSResults(0).begin();
}

void ScalableMaskedMulIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value mask, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(mask);
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  odsState.addTypes(res);
}

void ScalableMaskedMulIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value mask, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(mask);
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ScalableMaskedMulIOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ScalableMaskedMulIOp::verify() {
  if (::mlir::failed(ScalableMaskedMulIOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(1).begin()).getType(), (*this->getODSOperands(2).begin()).getType(), (*this->getODSResults(0).begin()).getType()})))))
    return emitOpError("failed to verify that all of {src1, src2, res} have same type");
  if (!((std::equal_to<>()(getI1SameShape((*this->getODSOperands(1).begin()).getType()), (*this->getODSOperands(0).begin()).getType()))))
    return emitOpError("failed to verify that mask has i1 element type and same shape as operands");
  return ::mlir::success();
}

::mlir::ParseResult ScalableMaskedMulIOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType maskRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> maskOperands(maskRawOperands);  ::llvm::SMLoc maskOperandsLoc;
  (void)maskOperandsLoc;
  ::mlir::OpAsmParser::OperandType src1RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> src1Operands(src1RawOperands);  ::llvm::SMLoc src1OperandsLoc;
  (void)src1OperandsLoc;
  ::mlir::OpAsmParser::OperandType src2RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> src2Operands(src2RawOperands);  ::llvm::SMLoc src2OperandsLoc;
  (void)src2OperandsLoc;
  ::mlir::Type maskRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> maskTypes(maskRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  maskOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(maskRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  src1OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src1RawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  src2OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src2RawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(maskRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resTypes);
  if (parser.resolveOperands(maskOperands, maskTypes, maskOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(src1Operands, resTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(src2Operands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ScalableMaskedMulIOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << mask();
  p << ",";
  p << ' ';
  p << src1();
  p << ",";
  p << ' ';
  p << src2();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(mask().getType());
  p << ",";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

} // namespace arm_sve
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableMaskedMulIOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableMaskedSDivIIntrOp definitions
//===----------------------------------------------------------------------===//

ScalableMaskedSDivIIntrOpAdaptor::ScalableMaskedSDivIIntrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ScalableMaskedSDivIIntrOpAdaptor::ScalableMaskedSDivIIntrOpAdaptor(ScalableMaskedSDivIIntrOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ScalableMaskedSDivIIntrOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ScalableMaskedSDivIIntrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ScalableMaskedSDivIIntrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr ScalableMaskedSDivIIntrOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ScalableMaskedSDivIIntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> ScalableMaskedSDivIIntrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ScalableMaskedSDivIIntrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> ScalableMaskedSDivIIntrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ScalableMaskedSDivIIntrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableMaskedSDivIIntrOp::res() {
  return *getODSResults(0).begin();
}

void ScalableMaskedSDivIIntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addOperands(odsArg_2);
  odsState.addTypes(res);
}

void ScalableMaskedSDivIIntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addOperands(odsArg_2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ScalableMaskedSDivIIntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ScalableMaskedSDivIIntrOp::verify() {
  if (::mlir::failed(ScalableMaskedSDivIIntrOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE7(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

} // namespace arm_sve
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableMaskedSDivIIntrOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableMaskedSDivIOp definitions
//===----------------------------------------------------------------------===//

ScalableMaskedSDivIOpAdaptor::ScalableMaskedSDivIOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ScalableMaskedSDivIOpAdaptor::ScalableMaskedSDivIOpAdaptor(ScalableMaskedSDivIOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ScalableMaskedSDivIOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ScalableMaskedSDivIOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ScalableMaskedSDivIOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableMaskedSDivIOpAdaptor::mask() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScalableMaskedSDivIOpAdaptor::src1() {
  return *getODSOperands(1).begin();
}

::mlir::Value ScalableMaskedSDivIOpAdaptor::src2() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr ScalableMaskedSDivIOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ScalableMaskedSDivIOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> ScalableMaskedSDivIOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ScalableMaskedSDivIOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableMaskedSDivIOp::mask() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScalableMaskedSDivIOp::src1() {
  return *getODSOperands(1).begin();
}

::mlir::Value ScalableMaskedSDivIOp::src2() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange ScalableMaskedSDivIOp::maskMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScalableMaskedSDivIOp::src1Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScalableMaskedSDivIOp::src2Mutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ScalableMaskedSDivIOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ScalableMaskedSDivIOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableMaskedSDivIOp::res() {
  return *getODSResults(0).begin();
}

void ScalableMaskedSDivIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value mask, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(mask);
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  odsState.addTypes(res);
}

void ScalableMaskedSDivIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value mask, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(mask);
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ScalableMaskedSDivIOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ScalableMaskedSDivIOp::verify() {
  if (::mlir::failed(ScalableMaskedSDivIOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(1).begin()).getType(), (*this->getODSOperands(2).begin()).getType(), (*this->getODSResults(0).begin()).getType()})))))
    return emitOpError("failed to verify that all of {src1, src2, res} have same type");
  if (!((std::equal_to<>()(getI1SameShape((*this->getODSOperands(1).begin()).getType()), (*this->getODSOperands(0).begin()).getType()))))
    return emitOpError("failed to verify that mask has i1 element type and same shape as operands");
  return ::mlir::success();
}

::mlir::ParseResult ScalableMaskedSDivIOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType maskRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> maskOperands(maskRawOperands);  ::llvm::SMLoc maskOperandsLoc;
  (void)maskOperandsLoc;
  ::mlir::OpAsmParser::OperandType src1RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> src1Operands(src1RawOperands);  ::llvm::SMLoc src1OperandsLoc;
  (void)src1OperandsLoc;
  ::mlir::OpAsmParser::OperandType src2RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> src2Operands(src2RawOperands);  ::llvm::SMLoc src2OperandsLoc;
  (void)src2OperandsLoc;
  ::mlir::Type maskRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> maskTypes(maskRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  maskOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(maskRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  src1OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src1RawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  src2OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src2RawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(maskRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resTypes);
  if (parser.resolveOperands(maskOperands, maskTypes, maskOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(src1Operands, resTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(src2Operands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ScalableMaskedSDivIOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << mask();
  p << ",";
  p << ' ';
  p << src1();
  p << ",";
  p << ' ';
  p << src2();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(mask().getType());
  p << ",";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

} // namespace arm_sve
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableMaskedSDivIOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableMaskedSubFIntrOp definitions
//===----------------------------------------------------------------------===//

ScalableMaskedSubFIntrOpAdaptor::ScalableMaskedSubFIntrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ScalableMaskedSubFIntrOpAdaptor::ScalableMaskedSubFIntrOpAdaptor(ScalableMaskedSubFIntrOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ScalableMaskedSubFIntrOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ScalableMaskedSubFIntrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ScalableMaskedSubFIntrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr ScalableMaskedSubFIntrOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ScalableMaskedSubFIntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> ScalableMaskedSubFIntrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ScalableMaskedSubFIntrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> ScalableMaskedSubFIntrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ScalableMaskedSubFIntrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableMaskedSubFIntrOp::res() {
  return *getODSResults(0).begin();
}

void ScalableMaskedSubFIntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addOperands(odsArg_2);
  odsState.addTypes(res);
}

void ScalableMaskedSubFIntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addOperands(odsArg_2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ScalableMaskedSubFIntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ScalableMaskedSubFIntrOp::verify() {
  if (::mlir::failed(ScalableMaskedSubFIntrOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE7(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

} // namespace arm_sve
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableMaskedSubFIntrOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableMaskedSubFOp definitions
//===----------------------------------------------------------------------===//

ScalableMaskedSubFOpAdaptor::ScalableMaskedSubFOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ScalableMaskedSubFOpAdaptor::ScalableMaskedSubFOpAdaptor(ScalableMaskedSubFOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ScalableMaskedSubFOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ScalableMaskedSubFOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ScalableMaskedSubFOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableMaskedSubFOpAdaptor::mask() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScalableMaskedSubFOpAdaptor::src1() {
  return *getODSOperands(1).begin();
}

::mlir::Value ScalableMaskedSubFOpAdaptor::src2() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr ScalableMaskedSubFOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ScalableMaskedSubFOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> ScalableMaskedSubFOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ScalableMaskedSubFOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableMaskedSubFOp::mask() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScalableMaskedSubFOp::src1() {
  return *getODSOperands(1).begin();
}

::mlir::Value ScalableMaskedSubFOp::src2() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange ScalableMaskedSubFOp::maskMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScalableMaskedSubFOp::src1Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScalableMaskedSubFOp::src2Mutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ScalableMaskedSubFOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ScalableMaskedSubFOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableMaskedSubFOp::res() {
  return *getODSResults(0).begin();
}

void ScalableMaskedSubFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value mask, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(mask);
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  odsState.addTypes(res);
}

void ScalableMaskedSubFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value mask, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(mask);
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ScalableMaskedSubFOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ScalableMaskedSubFOp::verify() {
  if (::mlir::failed(ScalableMaskedSubFOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(1).begin()).getType(), (*this->getODSOperands(2).begin()).getType(), (*this->getODSResults(0).begin()).getType()})))))
    return emitOpError("failed to verify that all of {src1, src2, res} have same type");
  if (!((std::equal_to<>()(getI1SameShape((*this->getODSOperands(1).begin()).getType()), (*this->getODSOperands(0).begin()).getType()))))
    return emitOpError("failed to verify that mask has i1 element type and same shape as operands");
  return ::mlir::success();
}

::mlir::ParseResult ScalableMaskedSubFOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType maskRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> maskOperands(maskRawOperands);  ::llvm::SMLoc maskOperandsLoc;
  (void)maskOperandsLoc;
  ::mlir::OpAsmParser::OperandType src1RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> src1Operands(src1RawOperands);  ::llvm::SMLoc src1OperandsLoc;
  (void)src1OperandsLoc;
  ::mlir::OpAsmParser::OperandType src2RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> src2Operands(src2RawOperands);  ::llvm::SMLoc src2OperandsLoc;
  (void)src2OperandsLoc;
  ::mlir::Type maskRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> maskTypes(maskRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  maskOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(maskRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  src1OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src1RawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  src2OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src2RawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(maskRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resTypes);
  if (parser.resolveOperands(maskOperands, maskTypes, maskOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(src1Operands, resTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(src2Operands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ScalableMaskedSubFOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << mask();
  p << ",";
  p << ' ';
  p << src1();
  p << ",";
  p << ' ';
  p << src2();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(mask().getType());
  p << ",";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

} // namespace arm_sve
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableMaskedSubFOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableMaskedSubIIntrOp definitions
//===----------------------------------------------------------------------===//

ScalableMaskedSubIIntrOpAdaptor::ScalableMaskedSubIIntrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ScalableMaskedSubIIntrOpAdaptor::ScalableMaskedSubIIntrOpAdaptor(ScalableMaskedSubIIntrOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ScalableMaskedSubIIntrOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ScalableMaskedSubIIntrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ScalableMaskedSubIIntrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr ScalableMaskedSubIIntrOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ScalableMaskedSubIIntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> ScalableMaskedSubIIntrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ScalableMaskedSubIIntrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> ScalableMaskedSubIIntrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ScalableMaskedSubIIntrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableMaskedSubIIntrOp::res() {
  return *getODSResults(0).begin();
}

void ScalableMaskedSubIIntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addOperands(odsArg_2);
  odsState.addTypes(res);
}

void ScalableMaskedSubIIntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addOperands(odsArg_2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ScalableMaskedSubIIntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ScalableMaskedSubIIntrOp::verify() {
  if (::mlir::failed(ScalableMaskedSubIIntrOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE7(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

} // namespace arm_sve
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableMaskedSubIIntrOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableMaskedSubIOp definitions
//===----------------------------------------------------------------------===//

ScalableMaskedSubIOpAdaptor::ScalableMaskedSubIOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ScalableMaskedSubIOpAdaptor::ScalableMaskedSubIOpAdaptor(ScalableMaskedSubIOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ScalableMaskedSubIOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ScalableMaskedSubIOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ScalableMaskedSubIOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableMaskedSubIOpAdaptor::mask() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScalableMaskedSubIOpAdaptor::src1() {
  return *getODSOperands(1).begin();
}

::mlir::Value ScalableMaskedSubIOpAdaptor::src2() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr ScalableMaskedSubIOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ScalableMaskedSubIOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> ScalableMaskedSubIOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ScalableMaskedSubIOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableMaskedSubIOp::mask() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScalableMaskedSubIOp::src1() {
  return *getODSOperands(1).begin();
}

::mlir::Value ScalableMaskedSubIOp::src2() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange ScalableMaskedSubIOp::maskMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScalableMaskedSubIOp::src1Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScalableMaskedSubIOp::src2Mutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ScalableMaskedSubIOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ScalableMaskedSubIOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableMaskedSubIOp::res() {
  return *getODSResults(0).begin();
}

void ScalableMaskedSubIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value mask, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(mask);
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  odsState.addTypes(res);
}

void ScalableMaskedSubIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value mask, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(mask);
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ScalableMaskedSubIOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ScalableMaskedSubIOp::verify() {
  if (::mlir::failed(ScalableMaskedSubIOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(1).begin()).getType(), (*this->getODSOperands(2).begin()).getType(), (*this->getODSResults(0).begin()).getType()})))))
    return emitOpError("failed to verify that all of {src1, src2, res} have same type");
  if (!((std::equal_to<>()(getI1SameShape((*this->getODSOperands(1).begin()).getType()), (*this->getODSOperands(0).begin()).getType()))))
    return emitOpError("failed to verify that mask has i1 element type and same shape as operands");
  return ::mlir::success();
}

::mlir::ParseResult ScalableMaskedSubIOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType maskRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> maskOperands(maskRawOperands);  ::llvm::SMLoc maskOperandsLoc;
  (void)maskOperandsLoc;
  ::mlir::OpAsmParser::OperandType src1RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> src1Operands(src1RawOperands);  ::llvm::SMLoc src1OperandsLoc;
  (void)src1OperandsLoc;
  ::mlir::OpAsmParser::OperandType src2RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> src2Operands(src2RawOperands);  ::llvm::SMLoc src2OperandsLoc;
  (void)src2OperandsLoc;
  ::mlir::Type maskRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> maskTypes(maskRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  maskOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(maskRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  src1OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src1RawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  src2OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src2RawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(maskRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resTypes);
  if (parser.resolveOperands(maskOperands, maskTypes, maskOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(src1Operands, resTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(src2Operands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ScalableMaskedSubIOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << mask();
  p << ",";
  p << ' ';
  p << src1();
  p << ",";
  p << ' ';
  p << src2();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(mask().getType());
  p << ",";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

} // namespace arm_sve
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableMaskedSubIOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableMaskedUDivIIntrOp definitions
//===----------------------------------------------------------------------===//

ScalableMaskedUDivIIntrOpAdaptor::ScalableMaskedUDivIIntrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ScalableMaskedUDivIIntrOpAdaptor::ScalableMaskedUDivIIntrOpAdaptor(ScalableMaskedUDivIIntrOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ScalableMaskedUDivIIntrOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ScalableMaskedUDivIIntrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ScalableMaskedUDivIIntrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr ScalableMaskedUDivIIntrOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ScalableMaskedUDivIIntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> ScalableMaskedUDivIIntrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ScalableMaskedUDivIIntrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> ScalableMaskedUDivIIntrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ScalableMaskedUDivIIntrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableMaskedUDivIIntrOp::res() {
  return *getODSResults(0).begin();
}

void ScalableMaskedUDivIIntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addOperands(odsArg_2);
  odsState.addTypes(res);
}

void ScalableMaskedUDivIIntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addOperands(odsArg_2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ScalableMaskedUDivIIntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ScalableMaskedUDivIIntrOp::verify() {
  if (::mlir::failed(ScalableMaskedUDivIIntrOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE7(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

} // namespace arm_sve
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableMaskedUDivIIntrOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableMaskedUDivIOp definitions
//===----------------------------------------------------------------------===//

ScalableMaskedUDivIOpAdaptor::ScalableMaskedUDivIOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ScalableMaskedUDivIOpAdaptor::ScalableMaskedUDivIOpAdaptor(ScalableMaskedUDivIOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ScalableMaskedUDivIOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ScalableMaskedUDivIOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ScalableMaskedUDivIOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableMaskedUDivIOpAdaptor::mask() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScalableMaskedUDivIOpAdaptor::src1() {
  return *getODSOperands(1).begin();
}

::mlir::Value ScalableMaskedUDivIOpAdaptor::src2() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr ScalableMaskedUDivIOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ScalableMaskedUDivIOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> ScalableMaskedUDivIOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ScalableMaskedUDivIOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableMaskedUDivIOp::mask() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScalableMaskedUDivIOp::src1() {
  return *getODSOperands(1).begin();
}

::mlir::Value ScalableMaskedUDivIOp::src2() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange ScalableMaskedUDivIOp::maskMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScalableMaskedUDivIOp::src1Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScalableMaskedUDivIOp::src2Mutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ScalableMaskedUDivIOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ScalableMaskedUDivIOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableMaskedUDivIOp::res() {
  return *getODSResults(0).begin();
}

void ScalableMaskedUDivIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value mask, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(mask);
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  odsState.addTypes(res);
}

void ScalableMaskedUDivIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value mask, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(mask);
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ScalableMaskedUDivIOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ScalableMaskedUDivIOp::verify() {
  if (::mlir::failed(ScalableMaskedUDivIOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(1).begin()).getType(), (*this->getODSOperands(2).begin()).getType(), (*this->getODSResults(0).begin()).getType()})))))
    return emitOpError("failed to verify that all of {src1, src2, res} have same type");
  if (!((std::equal_to<>()(getI1SameShape((*this->getODSOperands(1).begin()).getType()), (*this->getODSOperands(0).begin()).getType()))))
    return emitOpError("failed to verify that mask has i1 element type and same shape as operands");
  return ::mlir::success();
}

::mlir::ParseResult ScalableMaskedUDivIOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType maskRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> maskOperands(maskRawOperands);  ::llvm::SMLoc maskOperandsLoc;
  (void)maskOperandsLoc;
  ::mlir::OpAsmParser::OperandType src1RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> src1Operands(src1RawOperands);  ::llvm::SMLoc src1OperandsLoc;
  (void)src1OperandsLoc;
  ::mlir::OpAsmParser::OperandType src2RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> src2Operands(src2RawOperands);  ::llvm::SMLoc src2OperandsLoc;
  (void)src2OperandsLoc;
  ::mlir::Type maskRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> maskTypes(maskRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  maskOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(maskRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  src1OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src1RawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  src2OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src2RawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(maskRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resTypes);
  if (parser.resolveOperands(maskOperands, maskTypes, maskOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(src1Operands, resTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(src2Operands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ScalableMaskedUDivIOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << mask();
  p << ",";
  p << ' ';
  p << src1();
  p << ",";
  p << ' ';
  p << src2();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(mask().getType());
  p << ",";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

} // namespace arm_sve
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableMaskedUDivIOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableMulFOp definitions
//===----------------------------------------------------------------------===//

ScalableMulFOpAdaptor::ScalableMulFOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ScalableMulFOpAdaptor::ScalableMulFOpAdaptor(ScalableMulFOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ScalableMulFOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ScalableMulFOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ScalableMulFOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableMulFOpAdaptor::src1() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScalableMulFOpAdaptor::src2() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr ScalableMulFOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ScalableMulFOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> ScalableMulFOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ScalableMulFOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableMulFOp::src1() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScalableMulFOp::src2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange ScalableMulFOp::src1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScalableMulFOp::src2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ScalableMulFOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ScalableMulFOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableMulFOp::dst() {
  return *getODSResults(0).begin();
}

void ScalableMulFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dst, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  odsState.addTypes(dst);
}

void ScalableMulFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ScalableMulFOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ScalableMulFOp::verify() {
  if (::mlir::failed(ScalableMulFOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(0).begin()).getType(), (*this->getODSOperands(1).begin()).getType(), (*this->getODSResults(0).begin()).getType()})))))
    return emitOpError("failed to verify that all of {src1, src2, dst} have same type");
  return ::mlir::success();
}

::mlir::ParseResult ScalableMulFOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType src1RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> src1Operands(src1RawOperands);  ::llvm::SMLoc src1OperandsLoc;
  (void)src1OperandsLoc;
  ::mlir::OpAsmParser::OperandType src2RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> src2Operands(src2RawOperands);  ::llvm::SMLoc src2OperandsLoc;
  (void)src2OperandsLoc;
  ::mlir::Type src1RawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> src1Types(src1RawTypes);

  src1OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src1RawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  src2OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src2RawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(src1RawTypes[0]))
    return ::mlir::failure();
  result.addTypes(src1Types);
  if (parser.resolveOperands(src1Operands, src1Types, src1OperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(src2Operands, src1Types[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ScalableMulFOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << src1();
  p << ",";
  p << ' ';
  p << src2();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(src1().getType());
}

} // namespace arm_sve
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableMulFOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableMulIOp definitions
//===----------------------------------------------------------------------===//

ScalableMulIOpAdaptor::ScalableMulIOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ScalableMulIOpAdaptor::ScalableMulIOpAdaptor(ScalableMulIOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ScalableMulIOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ScalableMulIOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ScalableMulIOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableMulIOpAdaptor::src1() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScalableMulIOpAdaptor::src2() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr ScalableMulIOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ScalableMulIOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> ScalableMulIOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ScalableMulIOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableMulIOp::src1() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScalableMulIOp::src2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange ScalableMulIOp::src1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScalableMulIOp::src2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ScalableMulIOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ScalableMulIOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableMulIOp::dst() {
  return *getODSResults(0).begin();
}

void ScalableMulIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dst, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  odsState.addTypes(dst);
}

void ScalableMulIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ScalableMulIOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ScalableMulIOp::verify() {
  if (::mlir::failed(ScalableMulIOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(0).begin()).getType(), (*this->getODSOperands(1).begin()).getType(), (*this->getODSResults(0).begin()).getType()})))))
    return emitOpError("failed to verify that all of {src1, src2, dst} have same type");
  return ::mlir::success();
}

::mlir::ParseResult ScalableMulIOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType src1RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> src1Operands(src1RawOperands);  ::llvm::SMLoc src1OperandsLoc;
  (void)src1OperandsLoc;
  ::mlir::OpAsmParser::OperandType src2RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> src2Operands(src2RawOperands);  ::llvm::SMLoc src2OperandsLoc;
  (void)src2OperandsLoc;
  ::mlir::Type src1RawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> src1Types(src1RawTypes);

  src1OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src1RawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  src2OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src2RawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(src1RawTypes[0]))
    return ::mlir::failure();
  result.addTypes(src1Types);
  if (parser.resolveOperands(src1Operands, src1Types, src1OperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(src2Operands, src1Types[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ScalableMulIOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << src1();
  p << ",";
  p << ' ';
  p << src2();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(src1().getType());
}

} // namespace arm_sve
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableMulIOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableSDivIOp definitions
//===----------------------------------------------------------------------===//

ScalableSDivIOpAdaptor::ScalableSDivIOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ScalableSDivIOpAdaptor::ScalableSDivIOpAdaptor(ScalableSDivIOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ScalableSDivIOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ScalableSDivIOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ScalableSDivIOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableSDivIOpAdaptor::src1() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScalableSDivIOpAdaptor::src2() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr ScalableSDivIOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ScalableSDivIOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> ScalableSDivIOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ScalableSDivIOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableSDivIOp::src1() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScalableSDivIOp::src2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange ScalableSDivIOp::src1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScalableSDivIOp::src2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ScalableSDivIOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ScalableSDivIOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableSDivIOp::dst() {
  return *getODSResults(0).begin();
}

void ScalableSDivIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dst, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  odsState.addTypes(dst);
}

void ScalableSDivIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ScalableSDivIOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ScalableSDivIOp::verify() {
  if (::mlir::failed(ScalableSDivIOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(0).begin()).getType(), (*this->getODSOperands(1).begin()).getType(), (*this->getODSResults(0).begin()).getType()})))))
    return emitOpError("failed to verify that all of {src1, src2, dst} have same type");
  return ::mlir::success();
}

::mlir::ParseResult ScalableSDivIOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType src1RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> src1Operands(src1RawOperands);  ::llvm::SMLoc src1OperandsLoc;
  (void)src1OperandsLoc;
  ::mlir::OpAsmParser::OperandType src2RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> src2Operands(src2RawOperands);  ::llvm::SMLoc src2OperandsLoc;
  (void)src2OperandsLoc;
  ::mlir::Type src1RawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> src1Types(src1RawTypes);

  src1OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src1RawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  src2OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src2RawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(src1RawTypes[0]))
    return ::mlir::failure();
  result.addTypes(src1Types);
  if (parser.resolveOperands(src1Operands, src1Types, src1OperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(src2Operands, src1Types[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ScalableSDivIOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << src1();
  p << ",";
  p << ' ';
  p << src2();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(src1().getType());
}

} // namespace arm_sve
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableSDivIOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableStoreOp definitions
//===----------------------------------------------------------------------===//

ScalableStoreOpAdaptor::ScalableStoreOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ScalableStoreOpAdaptor::ScalableStoreOpAdaptor(ScalableStoreOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ScalableStoreOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ScalableStoreOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ScalableStoreOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableStoreOpAdaptor::base() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScalableStoreOpAdaptor::index() {
  return *getODSOperands(1).begin();
}

::mlir::Value ScalableStoreOpAdaptor::value() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr ScalableStoreOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ScalableStoreOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> ScalableStoreOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ScalableStoreOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableStoreOp::base() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScalableStoreOp::index() {
  return *getODSOperands(1).begin();
}

::mlir::Value ScalableStoreOp::value() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange ScalableStoreOp::baseMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScalableStoreOp::indexMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScalableStoreOp::valueMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ScalableStoreOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ScalableStoreOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void ScalableStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value base, ::mlir::Value index, ::mlir::Value value) {
  odsState.addOperands(base);
  odsState.addOperands(index);
  odsState.addOperands(value);
}

void ScalableStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value base, ::mlir::Value index, ::mlir::Value value) {
  odsState.addOperands(base);
  odsState.addOperands(index);
  odsState.addOperands(value);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ScalableStoreOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ScalableStoreOp::verify() {
  if (::mlir::failed(ScalableStoreOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE4(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE5(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::ParseResult ScalableStoreOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::OpAsmParser::OperandType baseRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> baseOperands(baseRawOperands);  ::llvm::SMLoc baseOperandsLoc;
  (void)baseOperandsLoc;
  ::mlir::OpAsmParser::OperandType indexRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> indexOperands(indexRawOperands);  ::llvm::SMLoc indexOperandsLoc;
  (void)indexOperandsLoc;
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);
  ::mlir::Type baseRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> baseTypes(baseRawTypes);

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  baseOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(baseRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  indexOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(indexRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(valueRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseType(baseRawTypes[0]))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  if (parser.resolveOperands(baseOperands, baseTypes, baseOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(indexOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ScalableStoreOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << value();
  p << ",";
  p << ' ';
  p << base();
  p << "[";
  p << index();
  p << "]";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(value().getType());
  p << ' ' << "to";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(base().getType());
}

void ScalableStoreOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(::mlir::MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace arm_sve
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableStoreOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableSubFOp definitions
//===----------------------------------------------------------------------===//

ScalableSubFOpAdaptor::ScalableSubFOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ScalableSubFOpAdaptor::ScalableSubFOpAdaptor(ScalableSubFOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ScalableSubFOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ScalableSubFOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ScalableSubFOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableSubFOpAdaptor::src1() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScalableSubFOpAdaptor::src2() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr ScalableSubFOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ScalableSubFOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> ScalableSubFOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ScalableSubFOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableSubFOp::src1() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScalableSubFOp::src2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange ScalableSubFOp::src1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScalableSubFOp::src2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ScalableSubFOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ScalableSubFOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableSubFOp::dst() {
  return *getODSResults(0).begin();
}

void ScalableSubFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dst, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  odsState.addTypes(dst);
}

void ScalableSubFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ScalableSubFOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ScalableSubFOp::verify() {
  if (::mlir::failed(ScalableSubFOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(0).begin()).getType(), (*this->getODSOperands(1).begin()).getType(), (*this->getODSResults(0).begin()).getType()})))))
    return emitOpError("failed to verify that all of {src1, src2, dst} have same type");
  return ::mlir::success();
}

::mlir::ParseResult ScalableSubFOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType src1RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> src1Operands(src1RawOperands);  ::llvm::SMLoc src1OperandsLoc;
  (void)src1OperandsLoc;
  ::mlir::OpAsmParser::OperandType src2RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> src2Operands(src2RawOperands);  ::llvm::SMLoc src2OperandsLoc;
  (void)src2OperandsLoc;
  ::mlir::Type src1RawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> src1Types(src1RawTypes);

  src1OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src1RawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  src2OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src2RawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(src1RawTypes[0]))
    return ::mlir::failure();
  result.addTypes(src1Types);
  if (parser.resolveOperands(src1Operands, src1Types, src1OperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(src2Operands, src1Types[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ScalableSubFOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << src1();
  p << ",";
  p << ' ';
  p << src2();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(src1().getType());
}

} // namespace arm_sve
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableSubFOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableSubIOp definitions
//===----------------------------------------------------------------------===//

ScalableSubIOpAdaptor::ScalableSubIOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ScalableSubIOpAdaptor::ScalableSubIOpAdaptor(ScalableSubIOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ScalableSubIOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ScalableSubIOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ScalableSubIOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableSubIOpAdaptor::src1() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScalableSubIOpAdaptor::src2() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr ScalableSubIOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ScalableSubIOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> ScalableSubIOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ScalableSubIOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableSubIOp::src1() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScalableSubIOp::src2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange ScalableSubIOp::src1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScalableSubIOp::src2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ScalableSubIOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ScalableSubIOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableSubIOp::dst() {
  return *getODSResults(0).begin();
}

void ScalableSubIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dst, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  odsState.addTypes(dst);
}

void ScalableSubIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ScalableSubIOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ScalableSubIOp::verify() {
  if (::mlir::failed(ScalableSubIOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(0).begin()).getType(), (*this->getODSOperands(1).begin()).getType(), (*this->getODSResults(0).begin()).getType()})))))
    return emitOpError("failed to verify that all of {src1, src2, dst} have same type");
  return ::mlir::success();
}

::mlir::ParseResult ScalableSubIOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType src1RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> src1Operands(src1RawOperands);  ::llvm::SMLoc src1OperandsLoc;
  (void)src1OperandsLoc;
  ::mlir::OpAsmParser::OperandType src2RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> src2Operands(src2RawOperands);  ::llvm::SMLoc src2OperandsLoc;
  (void)src2OperandsLoc;
  ::mlir::Type src1RawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> src1Types(src1RawTypes);

  src1OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src1RawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  src2OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src2RawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(src1RawTypes[0]))
    return ::mlir::failure();
  result.addTypes(src1Types);
  if (parser.resolveOperands(src1Operands, src1Types, src1OperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(src2Operands, src1Types[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ScalableSubIOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << src1();
  p << ",";
  p << ' ';
  p << src2();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(src1().getType());
}

} // namespace arm_sve
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableSubIOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableUDivIOp definitions
//===----------------------------------------------------------------------===//

ScalableUDivIOpAdaptor::ScalableUDivIOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ScalableUDivIOpAdaptor::ScalableUDivIOpAdaptor(ScalableUDivIOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ScalableUDivIOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ScalableUDivIOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ScalableUDivIOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableUDivIOpAdaptor::src1() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScalableUDivIOpAdaptor::src2() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr ScalableUDivIOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ScalableUDivIOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> ScalableUDivIOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ScalableUDivIOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableUDivIOp::src1() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScalableUDivIOp::src2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange ScalableUDivIOp::src1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScalableUDivIOp::src2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ScalableUDivIOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ScalableUDivIOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalableUDivIOp::dst() {
  return *getODSResults(0).begin();
}

void ScalableUDivIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dst, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  odsState.addTypes(dst);
}

void ScalableUDivIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ScalableUDivIOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ScalableUDivIOp::verify() {
  if (::mlir::failed(ScalableUDivIOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(0).begin()).getType(), (*this->getODSOperands(1).begin()).getType(), (*this->getODSResults(0).begin()).getType()})))))
    return emitOpError("failed to verify that all of {src1, src2, dst} have same type");
  return ::mlir::success();
}

::mlir::ParseResult ScalableUDivIOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType src1RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> src1Operands(src1RawOperands);  ::llvm::SMLoc src1OperandsLoc;
  (void)src1OperandsLoc;
  ::mlir::OpAsmParser::OperandType src2RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> src2Operands(src2RawOperands);  ::llvm::SMLoc src2OperandsLoc;
  (void)src2OperandsLoc;
  ::mlir::Type src1RawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> src1Types(src1RawTypes);

  src1OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src1RawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  src2OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src2RawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(src1RawTypes[0]))
    return ::mlir::failure();
  result.addTypes(src1Types);
  if (parser.resolveOperands(src1Operands, src1Types, src1OperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(src2Operands, src1Types[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ScalableUDivIOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << src1();
  p << ",";
  p << ' ';
  p << src2();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(src1().getType());
}

} // namespace arm_sve
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableUDivIOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::SdotIntrOp definitions
//===----------------------------------------------------------------------===//

SdotIntrOpAdaptor::SdotIntrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

SdotIntrOpAdaptor::SdotIntrOpAdaptor(SdotIntrOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange SdotIntrOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SdotIntrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SdotIntrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr SdotIntrOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult SdotIntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> SdotIntrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SdotIntrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> SdotIntrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SdotIntrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SdotIntrOp::res() {
  return *getODSResults(0).begin();
}

void SdotIntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addOperands(odsArg_2);
  odsState.addTypes(res);
}

void SdotIntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addOperands(odsArg_2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SdotIntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SdotIntrOp::verify() {
  if (::mlir::failed(SdotIntrOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE7(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

} // namespace arm_sve
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_sve::SdotIntrOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::SdotOp definitions
//===----------------------------------------------------------------------===//

SdotOpAdaptor::SdotOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

SdotOpAdaptor::SdotOpAdaptor(SdotOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange SdotOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SdotOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SdotOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SdotOpAdaptor::acc() {
  return *getODSOperands(0).begin();
}

::mlir::Value SdotOpAdaptor::src1() {
  return *getODSOperands(1).begin();
}

::mlir::Value SdotOpAdaptor::src2() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr SdotOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult SdotOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> SdotOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SdotOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SdotOp::acc() {
  return *getODSOperands(0).begin();
}

::mlir::Value SdotOp::src1() {
  return *getODSOperands(1).begin();
}

::mlir::Value SdotOp::src2() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange SdotOp::accMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange SdotOp::src1Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange SdotOp::src2Mutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SdotOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SdotOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SdotOp::dst() {
  return *getODSResults(0).begin();
}

void SdotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dst, ::mlir::Value acc, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(acc);
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  odsState.addTypes(dst);
}

void SdotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value acc, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(acc);
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SdotOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SdotOp::verify() {
  if (::mlir::failed(SdotOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE8(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE9(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE9(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE8(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(1).begin()).getType(), (*this->getODSOperands(2).begin()).getType()})))))
    return emitOpError("failed to verify that all of {src1, src2} have same type");
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(0).begin()).getType(), (*this->getODSResults(0).begin()).getType()})))))
    return emitOpError("failed to verify that all of {acc, dst} have same type");
  return ::mlir::success();
}

::mlir::ParseResult SdotOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType accRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> accOperands(accRawOperands);  ::llvm::SMLoc accOperandsLoc;
  (void)accOperandsLoc;
  ::mlir::OpAsmParser::OperandType src1RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> src1Operands(src1RawOperands);  ::llvm::SMLoc src1OperandsLoc;
  (void)src1OperandsLoc;
  ::mlir::OpAsmParser::OperandType src2RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> src2Operands(src2RawOperands);  ::llvm::SMLoc src2OperandsLoc;
  (void)src2OperandsLoc;
  ::mlir::Type src1RawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> src1Types(src1RawTypes);
  ::mlir::Type dstRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> dstTypes(dstRawTypes);

  accOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(accRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  src1OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src1RawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  src2OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src2RawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(src1RawTypes[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseType(dstRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(dstTypes);
  if (parser.resolveOperands(accOperands, dstTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(src1Operands, src1Types, src1OperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(src2Operands, src1Types[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SdotOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << acc();
  p << ",";
  p << ' ';
  p << src1();
  p << ",";
  p << ' ';
  p << src2();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(src1().getType());
  p << ' ' << "to";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(dst().getType());
}

void SdotOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace arm_sve
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_sve::SdotOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::SmmlaIntrOp definitions
//===----------------------------------------------------------------------===//

SmmlaIntrOpAdaptor::SmmlaIntrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

SmmlaIntrOpAdaptor::SmmlaIntrOpAdaptor(SmmlaIntrOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange SmmlaIntrOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SmmlaIntrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SmmlaIntrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr SmmlaIntrOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult SmmlaIntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> SmmlaIntrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SmmlaIntrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> SmmlaIntrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SmmlaIntrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SmmlaIntrOp::res() {
  return *getODSResults(0).begin();
}

void SmmlaIntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addOperands(odsArg_2);
  odsState.addTypes(res);
}

void SmmlaIntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addOperands(odsArg_2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SmmlaIntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SmmlaIntrOp::verify() {
  if (::mlir::failed(SmmlaIntrOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE7(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

} // namespace arm_sve
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_sve::SmmlaIntrOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::SmmlaOp definitions
//===----------------------------------------------------------------------===//

SmmlaOpAdaptor::SmmlaOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

SmmlaOpAdaptor::SmmlaOpAdaptor(SmmlaOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange SmmlaOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SmmlaOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SmmlaOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SmmlaOpAdaptor::acc() {
  return *getODSOperands(0).begin();
}

::mlir::Value SmmlaOpAdaptor::src1() {
  return *getODSOperands(1).begin();
}

::mlir::Value SmmlaOpAdaptor::src2() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr SmmlaOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult SmmlaOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> SmmlaOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SmmlaOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SmmlaOp::acc() {
  return *getODSOperands(0).begin();
}

::mlir::Value SmmlaOp::src1() {
  return *getODSOperands(1).begin();
}

::mlir::Value SmmlaOp::src2() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange SmmlaOp::accMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange SmmlaOp::src1Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange SmmlaOp::src2Mutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SmmlaOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SmmlaOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SmmlaOp::dst() {
  return *getODSResults(0).begin();
}

void SmmlaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dst, ::mlir::Value acc, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(acc);
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  odsState.addTypes(dst);
}

void SmmlaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value acc, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(acc);
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SmmlaOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SmmlaOp::verify() {
  if (::mlir::failed(SmmlaOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE10(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE11(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE11(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE10(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(1).begin()).getType(), (*this->getODSOperands(2).begin()).getType()})))))
    return emitOpError("failed to verify that all of {src1, src2} have same type");
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(0).begin()).getType(), (*this->getODSResults(0).begin()).getType()})))))
    return emitOpError("failed to verify that all of {acc, dst} have same type");
  return ::mlir::success();
}

::mlir::ParseResult SmmlaOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType accRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> accOperands(accRawOperands);  ::llvm::SMLoc accOperandsLoc;
  (void)accOperandsLoc;
  ::mlir::OpAsmParser::OperandType src1RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> src1Operands(src1RawOperands);  ::llvm::SMLoc src1OperandsLoc;
  (void)src1OperandsLoc;
  ::mlir::OpAsmParser::OperandType src2RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> src2Operands(src2RawOperands);  ::llvm::SMLoc src2OperandsLoc;
  (void)src2OperandsLoc;
  ::mlir::Type src1RawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> src1Types(src1RawTypes);
  ::mlir::Type dstRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> dstTypes(dstRawTypes);

  accOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(accRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  src1OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src1RawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  src2OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src2RawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(src1RawTypes[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseType(dstRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(dstTypes);
  if (parser.resolveOperands(accOperands, dstTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(src1Operands, src1Types, src1OperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(src2Operands, src1Types[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SmmlaOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << acc();
  p << ",";
  p << ' ';
  p << src1();
  p << ",";
  p << ' ';
  p << src2();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(src1().getType());
  p << ' ' << "to";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(dst().getType());
}

void SmmlaOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace arm_sve
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_sve::SmmlaOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::UdotIntrOp definitions
//===----------------------------------------------------------------------===//

UdotIntrOpAdaptor::UdotIntrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

UdotIntrOpAdaptor::UdotIntrOpAdaptor(UdotIntrOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange UdotIntrOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> UdotIntrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange UdotIntrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr UdotIntrOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult UdotIntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> UdotIntrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range UdotIntrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> UdotIntrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range UdotIntrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UdotIntrOp::res() {
  return *getODSResults(0).begin();
}

void UdotIntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addOperands(odsArg_2);
  odsState.addTypes(res);
}

void UdotIntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addOperands(odsArg_2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void UdotIntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult UdotIntrOp::verify() {
  if (::mlir::failed(UdotIntrOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE7(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

} // namespace arm_sve
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_sve::UdotIntrOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::UdotOp definitions
//===----------------------------------------------------------------------===//

UdotOpAdaptor::UdotOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

UdotOpAdaptor::UdotOpAdaptor(UdotOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange UdotOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> UdotOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange UdotOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UdotOpAdaptor::acc() {
  return *getODSOperands(0).begin();
}

::mlir::Value UdotOpAdaptor::src1() {
  return *getODSOperands(1).begin();
}

::mlir::Value UdotOpAdaptor::src2() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr UdotOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult UdotOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> UdotOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range UdotOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UdotOp::acc() {
  return *getODSOperands(0).begin();
}

::mlir::Value UdotOp::src1() {
  return *getODSOperands(1).begin();
}

::mlir::Value UdotOp::src2() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange UdotOp::accMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange UdotOp::src1Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange UdotOp::src2Mutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> UdotOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range UdotOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UdotOp::dst() {
  return *getODSResults(0).begin();
}

void UdotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dst, ::mlir::Value acc, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(acc);
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  odsState.addTypes(dst);
}

void UdotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value acc, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(acc);
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void UdotOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult UdotOp::verify() {
  if (::mlir::failed(UdotOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE8(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE9(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE9(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE8(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(1).begin()).getType(), (*this->getODSOperands(2).begin()).getType()})))))
    return emitOpError("failed to verify that all of {src1, src2} have same type");
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(0).begin()).getType(), (*this->getODSResults(0).begin()).getType()})))))
    return emitOpError("failed to verify that all of {acc, dst} have same type");
  return ::mlir::success();
}

::mlir::ParseResult UdotOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType accRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> accOperands(accRawOperands);  ::llvm::SMLoc accOperandsLoc;
  (void)accOperandsLoc;
  ::mlir::OpAsmParser::OperandType src1RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> src1Operands(src1RawOperands);  ::llvm::SMLoc src1OperandsLoc;
  (void)src1OperandsLoc;
  ::mlir::OpAsmParser::OperandType src2RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> src2Operands(src2RawOperands);  ::llvm::SMLoc src2OperandsLoc;
  (void)src2OperandsLoc;
  ::mlir::Type src1RawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> src1Types(src1RawTypes);
  ::mlir::Type dstRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> dstTypes(dstRawTypes);

  accOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(accRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  src1OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src1RawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  src2OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src2RawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(src1RawTypes[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseType(dstRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(dstTypes);
  if (parser.resolveOperands(accOperands, dstTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(src1Operands, src1Types, src1OperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(src2Operands, src1Types[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void UdotOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << acc();
  p << ",";
  p << ' ';
  p << src1();
  p << ",";
  p << ' ';
  p << src2();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(src1().getType());
  p << ' ' << "to";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(dst().getType());
}

void UdotOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace arm_sve
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_sve::UdotOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::UmmlaIntrOp definitions
//===----------------------------------------------------------------------===//

UmmlaIntrOpAdaptor::UmmlaIntrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

UmmlaIntrOpAdaptor::UmmlaIntrOpAdaptor(UmmlaIntrOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange UmmlaIntrOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> UmmlaIntrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange UmmlaIntrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr UmmlaIntrOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult UmmlaIntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> UmmlaIntrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range UmmlaIntrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> UmmlaIntrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range UmmlaIntrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UmmlaIntrOp::res() {
  return *getODSResults(0).begin();
}

void UmmlaIntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addOperands(odsArg_2);
  odsState.addTypes(res);
}

void UmmlaIntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addOperands(odsArg_2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void UmmlaIntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult UmmlaIntrOp::verify() {
  if (::mlir::failed(UmmlaIntrOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE7(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

} // namespace arm_sve
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_sve::UmmlaIntrOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::UmmlaOp definitions
//===----------------------------------------------------------------------===//

UmmlaOpAdaptor::UmmlaOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

UmmlaOpAdaptor::UmmlaOpAdaptor(UmmlaOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange UmmlaOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> UmmlaOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange UmmlaOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UmmlaOpAdaptor::acc() {
  return *getODSOperands(0).begin();
}

::mlir::Value UmmlaOpAdaptor::src1() {
  return *getODSOperands(1).begin();
}

::mlir::Value UmmlaOpAdaptor::src2() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr UmmlaOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult UmmlaOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> UmmlaOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range UmmlaOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UmmlaOp::acc() {
  return *getODSOperands(0).begin();
}

::mlir::Value UmmlaOp::src1() {
  return *getODSOperands(1).begin();
}

::mlir::Value UmmlaOp::src2() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange UmmlaOp::accMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange UmmlaOp::src1Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange UmmlaOp::src2Mutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> UmmlaOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range UmmlaOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UmmlaOp::dst() {
  return *getODSResults(0).begin();
}

void UmmlaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dst, ::mlir::Value acc, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(acc);
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  odsState.addTypes(dst);
}

void UmmlaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value acc, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(acc);
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void UmmlaOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult UmmlaOp::verify() {
  if (::mlir::failed(UmmlaOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE10(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE11(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE11(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE10(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(1).begin()).getType(), (*this->getODSOperands(2).begin()).getType()})))))
    return emitOpError("failed to verify that all of {src1, src2} have same type");
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(0).begin()).getType(), (*this->getODSResults(0).begin()).getType()})))))
    return emitOpError("failed to verify that all of {acc, dst} have same type");
  return ::mlir::success();
}

::mlir::ParseResult UmmlaOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType accRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> accOperands(accRawOperands);  ::llvm::SMLoc accOperandsLoc;
  (void)accOperandsLoc;
  ::mlir::OpAsmParser::OperandType src1RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> src1Operands(src1RawOperands);  ::llvm::SMLoc src1OperandsLoc;
  (void)src1OperandsLoc;
  ::mlir::OpAsmParser::OperandType src2RawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> src2Operands(src2RawOperands);  ::llvm::SMLoc src2OperandsLoc;
  (void)src2OperandsLoc;
  ::mlir::Type src1RawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> src1Types(src1RawTypes);
  ::mlir::Type dstRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> dstTypes(dstRawTypes);

  accOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(accRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  src1OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src1RawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  src2OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src2RawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(src1RawTypes[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseType(dstRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(dstTypes);
  if (parser.resolveOperands(accOperands, dstTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(src1Operands, src1Types, src1OperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(src2Operands, src1Types[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void UmmlaOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << acc();
  p << ",";
  p << ' ';
  p << src1();
  p << ",";
  p << ' ';
  p << src2();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(src1().getType());
  p << ' ' << "to";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(dst().getType());
}

void UmmlaOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace arm_sve
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_sve::UmmlaOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::VectorScaleIntrOp definitions
//===----------------------------------------------------------------------===//

VectorScaleIntrOpAdaptor::VectorScaleIntrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

VectorScaleIntrOpAdaptor::VectorScaleIntrOpAdaptor(VectorScaleIntrOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange VectorScaleIntrOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VectorScaleIntrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VectorScaleIntrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr VectorScaleIntrOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VectorScaleIntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> VectorScaleIntrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VectorScaleIntrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> VectorScaleIntrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VectorScaleIntrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VectorScaleIntrOp::res() {
  return *getODSResults(0).begin();
}

void VectorScaleIntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res) {
  odsState.addTypes(res);
}

void VectorScaleIntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VectorScaleIntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VectorScaleIntrOp::verify() {
  if (::mlir::failed(VectorScaleIntrOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE7(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

} // namespace arm_sve
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_sve::VectorScaleIntrOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::VectorScaleOp definitions
//===----------------------------------------------------------------------===//

VectorScaleOpAdaptor::VectorScaleOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

VectorScaleOpAdaptor::VectorScaleOpAdaptor(VectorScaleOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange VectorScaleOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VectorScaleOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VectorScaleOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr VectorScaleOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VectorScaleOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> VectorScaleOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VectorScaleOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> VectorScaleOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VectorScaleOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VectorScaleOp::res() {
  return *getODSResults(0).begin();
}

void VectorScaleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res) {
  odsState.addTypes(res);
}

void VectorScaleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VectorScaleOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VectorScaleOp::verify() {
  if (::mlir::failed(VectorScaleOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmSVE4(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult VectorScaleOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resTypes);
  return ::mlir::success();
}

void VectorScaleOp::print(::mlir::OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

void VectorScaleOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace arm_sve
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_sve::VectorScaleOp)


#endif  // GET_OP_CLASSES

