/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::NVVM::Barrier0Op,
::mlir::NVVM::BlockDimXOp,
::mlir::NVVM::BlockDimYOp,
::mlir::NVVM::BlockDimZOp,
::mlir::NVVM::BlockIdXOp,
::mlir::NVVM::BlockIdYOp,
::mlir::NVVM::BlockIdZOp,
::mlir::NVVM::GridDimXOp,
::mlir::NVVM::GridDimYOp,
::mlir::NVVM::GridDimZOp,
::mlir::NVVM::LaneIdOp,
::mlir::NVVM::MmaOp,
::mlir::NVVM::ShflBflyOp,
::mlir::NVVM::ThreadIdXOp,
::mlir::NVVM::ThreadIdYOp,
::mlir::NVVM::ThreadIdZOp,
::mlir::NVVM::VoteBallotOp,
::mlir::NVVM::WMMALoadAM16N16K16Op,
::mlir::NVVM::WMMALoadBM16N16K16Op,
::mlir::NVVM::WMMALoadCF16M16N16K16Op,
::mlir::NVVM::WMMALoadCF32M16N16K16Op,
::mlir::NVVM::WMMAMmaF16F16M16N16K16Op,
::mlir::NVVM::WMMAMmaF32F32M16N16K16Op,
::mlir::NVVM::WMMAStoreF16M16N16K16Op,
::mlir::NVVM::WMMAStoreF32M16N16K16Op,
::mlir::NVVM::WarpSizeOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace NVVM {
static ::mlir::LogicalResult __mlir_ods_local_type_constraint_NVVMOps0(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((::mlir::LLVM::isCompatibleType(type)))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be LLVM dialect-compatible type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_NVVMOps2(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isa<::mlir::LLVM::LLVMStructType>()))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be LLVM structure type, but got " << type;
  }
  return ::mlir::success();
}

} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::Barrier0Op definitions
//===----------------------------------------------------------------------===//

Barrier0OpAdaptor::Barrier0OpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

Barrier0OpAdaptor::Barrier0OpAdaptor(Barrier0Op&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange Barrier0OpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> Barrier0OpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange Barrier0OpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr Barrier0OpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult Barrier0OpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> Barrier0Op::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range Barrier0Op::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> Barrier0Op::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range Barrier0Op::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void Barrier0Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {

}

void Barrier0Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void Barrier0Op::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult Barrier0Op::verify() {
  if (::mlir::failed(Barrier0OpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::ParseResult Barrier0Op::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void Barrier0Op::print(::mlir::OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace NVVM
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::Barrier0Op)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::BlockDimXOp definitions
//===----------------------------------------------------------------------===//

BlockDimXOpAdaptor::BlockDimXOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

BlockDimXOpAdaptor::BlockDimXOpAdaptor(BlockDimXOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange BlockDimXOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BlockDimXOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BlockDimXOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr BlockDimXOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult BlockDimXOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> BlockDimXOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BlockDimXOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> BlockDimXOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BlockDimXOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BlockDimXOp::res() {
  return *getODSResults(0).begin();
}

void BlockDimXOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res) {
  odsState.addTypes(res);
}

void BlockDimXOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BlockDimXOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BlockDimXOp::verify() {
  if (::mlir::failed(BlockDimXOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult BlockDimXOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
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

void BlockDimXOp::print(::mlir::OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

void BlockDimXOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace NVVM
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::BlockDimXOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::BlockDimYOp definitions
//===----------------------------------------------------------------------===//

BlockDimYOpAdaptor::BlockDimYOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

BlockDimYOpAdaptor::BlockDimYOpAdaptor(BlockDimYOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange BlockDimYOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BlockDimYOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BlockDimYOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr BlockDimYOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult BlockDimYOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> BlockDimYOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BlockDimYOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> BlockDimYOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BlockDimYOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BlockDimYOp::res() {
  return *getODSResults(0).begin();
}

void BlockDimYOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res) {
  odsState.addTypes(res);
}

void BlockDimYOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BlockDimYOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BlockDimYOp::verify() {
  if (::mlir::failed(BlockDimYOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult BlockDimYOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
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

void BlockDimYOp::print(::mlir::OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

void BlockDimYOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace NVVM
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::BlockDimYOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::BlockDimZOp definitions
//===----------------------------------------------------------------------===//

BlockDimZOpAdaptor::BlockDimZOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

BlockDimZOpAdaptor::BlockDimZOpAdaptor(BlockDimZOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange BlockDimZOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BlockDimZOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BlockDimZOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr BlockDimZOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult BlockDimZOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> BlockDimZOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BlockDimZOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> BlockDimZOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BlockDimZOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BlockDimZOp::res() {
  return *getODSResults(0).begin();
}

void BlockDimZOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res) {
  odsState.addTypes(res);
}

void BlockDimZOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BlockDimZOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BlockDimZOp::verify() {
  if (::mlir::failed(BlockDimZOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult BlockDimZOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
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

void BlockDimZOp::print(::mlir::OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

void BlockDimZOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace NVVM
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::BlockDimZOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::BlockIdXOp definitions
//===----------------------------------------------------------------------===//

BlockIdXOpAdaptor::BlockIdXOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

BlockIdXOpAdaptor::BlockIdXOpAdaptor(BlockIdXOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange BlockIdXOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BlockIdXOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BlockIdXOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr BlockIdXOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult BlockIdXOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> BlockIdXOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BlockIdXOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> BlockIdXOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BlockIdXOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BlockIdXOp::res() {
  return *getODSResults(0).begin();
}

void BlockIdXOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res) {
  odsState.addTypes(res);
}

void BlockIdXOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BlockIdXOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BlockIdXOp::verify() {
  if (::mlir::failed(BlockIdXOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult BlockIdXOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
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

void BlockIdXOp::print(::mlir::OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

void BlockIdXOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace NVVM
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::BlockIdXOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::BlockIdYOp definitions
//===----------------------------------------------------------------------===//

BlockIdYOpAdaptor::BlockIdYOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

BlockIdYOpAdaptor::BlockIdYOpAdaptor(BlockIdYOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange BlockIdYOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BlockIdYOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BlockIdYOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr BlockIdYOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult BlockIdYOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> BlockIdYOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BlockIdYOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> BlockIdYOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BlockIdYOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BlockIdYOp::res() {
  return *getODSResults(0).begin();
}

void BlockIdYOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res) {
  odsState.addTypes(res);
}

void BlockIdYOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BlockIdYOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BlockIdYOp::verify() {
  if (::mlir::failed(BlockIdYOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult BlockIdYOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
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

void BlockIdYOp::print(::mlir::OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

void BlockIdYOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace NVVM
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::BlockIdYOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::BlockIdZOp definitions
//===----------------------------------------------------------------------===//

BlockIdZOpAdaptor::BlockIdZOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

BlockIdZOpAdaptor::BlockIdZOpAdaptor(BlockIdZOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange BlockIdZOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BlockIdZOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BlockIdZOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr BlockIdZOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult BlockIdZOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> BlockIdZOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BlockIdZOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> BlockIdZOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BlockIdZOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BlockIdZOp::res() {
  return *getODSResults(0).begin();
}

void BlockIdZOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res) {
  odsState.addTypes(res);
}

void BlockIdZOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BlockIdZOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BlockIdZOp::verify() {
  if (::mlir::failed(BlockIdZOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult BlockIdZOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
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

void BlockIdZOp::print(::mlir::OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

void BlockIdZOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace NVVM
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::BlockIdZOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::GridDimXOp definitions
//===----------------------------------------------------------------------===//

GridDimXOpAdaptor::GridDimXOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

GridDimXOpAdaptor::GridDimXOpAdaptor(GridDimXOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange GridDimXOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GridDimXOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GridDimXOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr GridDimXOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult GridDimXOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> GridDimXOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GridDimXOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> GridDimXOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GridDimXOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GridDimXOp::res() {
  return *getODSResults(0).begin();
}

void GridDimXOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res) {
  odsState.addTypes(res);
}

void GridDimXOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GridDimXOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GridDimXOp::verify() {
  if (::mlir::failed(GridDimXOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult GridDimXOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
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

void GridDimXOp::print(::mlir::OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

void GridDimXOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace NVVM
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::GridDimXOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::GridDimYOp definitions
//===----------------------------------------------------------------------===//

GridDimYOpAdaptor::GridDimYOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

GridDimYOpAdaptor::GridDimYOpAdaptor(GridDimYOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange GridDimYOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GridDimYOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GridDimYOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr GridDimYOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult GridDimYOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> GridDimYOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GridDimYOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> GridDimYOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GridDimYOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GridDimYOp::res() {
  return *getODSResults(0).begin();
}

void GridDimYOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res) {
  odsState.addTypes(res);
}

void GridDimYOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GridDimYOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GridDimYOp::verify() {
  if (::mlir::failed(GridDimYOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult GridDimYOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
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

void GridDimYOp::print(::mlir::OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

void GridDimYOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace NVVM
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::GridDimYOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::GridDimZOp definitions
//===----------------------------------------------------------------------===//

GridDimZOpAdaptor::GridDimZOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

GridDimZOpAdaptor::GridDimZOpAdaptor(GridDimZOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange GridDimZOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GridDimZOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GridDimZOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr GridDimZOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult GridDimZOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> GridDimZOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GridDimZOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> GridDimZOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GridDimZOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GridDimZOp::res() {
  return *getODSResults(0).begin();
}

void GridDimZOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res) {
  odsState.addTypes(res);
}

void GridDimZOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GridDimZOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GridDimZOp::verify() {
  if (::mlir::failed(GridDimZOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult GridDimZOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
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

void GridDimZOp::print(::mlir::OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

void GridDimZOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace NVVM
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::GridDimZOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::LaneIdOp definitions
//===----------------------------------------------------------------------===//

LaneIdOpAdaptor::LaneIdOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

LaneIdOpAdaptor::LaneIdOpAdaptor(LaneIdOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange LaneIdOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> LaneIdOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange LaneIdOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr LaneIdOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult LaneIdOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> LaneIdOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range LaneIdOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> LaneIdOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range LaneIdOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LaneIdOp::res() {
  return *getODSResults(0).begin();
}

void LaneIdOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res) {
  odsState.addTypes(res);
}

void LaneIdOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LaneIdOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult LaneIdOp::verify() {
  if (::mlir::failed(LaneIdOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult LaneIdOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
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

void LaneIdOp::print(::mlir::OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

void LaneIdOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace NVVM
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::LaneIdOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::MmaOp definitions
//===----------------------------------------------------------------------===//

MmaOpAdaptor::MmaOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

MmaOpAdaptor::MmaOpAdaptor(MmaOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange MmaOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MmaOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange MmaOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange MmaOpAdaptor::args() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr MmaOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult MmaOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> MmaOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range MmaOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range MmaOp::args() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange MmaOp::argsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> MmaOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MmaOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MmaOp::res() {
  return *getODSResults(0).begin();
}

void MmaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::ValueRange args) {
  odsState.addOperands(args);
  odsState.addTypes(res);
}

void MmaOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MmaOp::verify() {
  if (::mlir::failed(MmaOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::verify(*this);
}

::mlir::ParseResult MmaOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> argsOperands;
  ::llvm::SMLoc argsOperandsLoc;
  (void)argsOperandsLoc;
  ::llvm::ArrayRef<::mlir::Type> argsTypes;
  ::llvm::ArrayRef<::mlir::Type> resTypes;

  argsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(argsOperands))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType args__res_functionType;
  if (parser.parseType(args__res_functionType))
    return ::mlir::failure();
  argsTypes = args__res_functionType.getInputs();
  resTypes = args__res_functionType.getResults();
  result.addTypes(resTypes);
  if (parser.resolveOperands(argsOperands, argsTypes, argsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void MmaOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << args();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p.printFunctionalType(args().getTypes(), ::llvm::ArrayRef<::mlir::Type>(res().getType()));
}

} // namespace NVVM
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::MmaOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::ShflBflyOp definitions
//===----------------------------------------------------------------------===//

ShflBflyOpAdaptor::ShflBflyOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ShflBflyOpAdaptor::ShflBflyOpAdaptor(ShflBflyOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ShflBflyOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ShflBflyOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ShflBflyOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ShflBflyOpAdaptor::dst() {
  return *getODSOperands(0).begin();
}

::mlir::Value ShflBflyOpAdaptor::val() {
  return *getODSOperands(1).begin();
}

::mlir::Value ShflBflyOpAdaptor::offset() {
  return *getODSOperands(2).begin();
}

::mlir::Value ShflBflyOpAdaptor::mask_and_clamp() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr ShflBflyOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::UnitAttr ShflBflyOpAdaptor::return_value_and_is_valid() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::UnitAttr attr = odsAttrs.get("return_value_and_is_valid").dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

::mlir::LogicalResult ShflBflyOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_return_value_and_is_valid = odsAttrs.get("return_value_and_is_valid");
  if (tblgen_return_value_and_is_valid) {
    if (!((tblgen_return_value_and_is_valid.isa<::mlir::UnitAttr>()))) return emitError(loc, "'nvvm.shfl.sync.bfly' op ""attribute 'return_value_and_is_valid' failed to satisfy constraint: unit attribute");
  }
  }
  return ::mlir::success();
}













std::pair<unsigned, unsigned> ShflBflyOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ShflBflyOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ShflBflyOp::dst() {
  return *getODSOperands(0).begin();
}

::mlir::Value ShflBflyOp::val() {
  return *getODSOperands(1).begin();
}

::mlir::Value ShflBflyOp::offset() {
  return *getODSOperands(2).begin();
}

::mlir::Value ShflBflyOp::mask_and_clamp() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange ShflBflyOp::dstMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ShflBflyOp::valMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ShflBflyOp::offsetMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ShflBflyOp::mask_and_clampMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ShflBflyOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ShflBflyOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ShflBflyOp::res() {
  return *getODSResults(0).begin();
}

::mlir::UnitAttr ShflBflyOp::return_value_and_is_validAttr() {
  return (*this)->getAttr(return_value_and_is_validAttrName()).template dyn_cast_or_null<::mlir::UnitAttr>();
}

::llvm::Optional<bool> ShflBflyOp::return_value_and_is_valid() {
  auto attr = return_value_and_is_validAttr();
  return attr ? ::llvm::Optional<bool>(attr != nullptr) : (::llvm::None);
}

void ShflBflyOp::return_value_and_is_validAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(return_value_and_is_validAttrName(), attr);
}

::mlir::Attribute ShflBflyOp::removeReturn_value_and_is_validAttr() {
  return (*this)->removeAttr(return_value_and_is_validAttrName());
}

void ShflBflyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value dst, ::mlir::Value val, ::mlir::Value offset, ::mlir::Value mask_and_clamp, /*optional*/::mlir::UnitAttr return_value_and_is_valid) {
  odsState.addOperands(dst);
  odsState.addOperands(val);
  odsState.addOperands(offset);
  odsState.addOperands(mask_and_clamp);
  if (return_value_and_is_valid) {
  odsState.addAttribute(return_value_and_is_validAttrName(odsState.name), return_value_and_is_valid);
  }
  odsState.addTypes(res);
}

void ShflBflyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value dst, ::mlir::Value val, ::mlir::Value offset, ::mlir::Value mask_and_clamp, /*optional*/::mlir::UnitAttr return_value_and_is_valid) {
  odsState.addOperands(dst);
  odsState.addOperands(val);
  odsState.addOperands(offset);
  odsState.addOperands(mask_and_clamp);
  if (return_value_and_is_valid) {
  odsState.addAttribute(return_value_and_is_validAttrName(odsState.name), return_value_and_is_valid);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ShflBflyOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult ShflBflyOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return parseNVVMShflSyncBflyOp(parser, result);
}

void ShflBflyOp::print(::mlir::OpAsmPrinter &p) {
  printNVVMIntrinsicOp(p, this->getOperation());
}

::mlir::LogicalResult ShflBflyOp::verify() {
  if (::mlir::failed(ShflBflyOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup3 = getODSOperands(3);
    for (::mlir::Value v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  if (!(*this)->getAttrOfType<UnitAttr>("return_value_and_is_valid"))
      return success();
    auto type = getType().dyn_cast<LLVM::LLVMStructType>();
    auto elementType = (type && type.getBody().size() == 2)
                     ? type.getBody()[1].dyn_cast<IntegerType>()
                     : nullptr;
    if (!elementType || elementType.getWidth() != 1)
      return emitError("expected return type to be a two-element struct with "
                       "i1 as the second element");
    return success();
}

} // namespace NVVM
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::ShflBflyOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::ThreadIdXOp definitions
//===----------------------------------------------------------------------===//

ThreadIdXOpAdaptor::ThreadIdXOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ThreadIdXOpAdaptor::ThreadIdXOpAdaptor(ThreadIdXOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ThreadIdXOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ThreadIdXOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ThreadIdXOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr ThreadIdXOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ThreadIdXOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> ThreadIdXOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ThreadIdXOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> ThreadIdXOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ThreadIdXOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ThreadIdXOp::res() {
  return *getODSResults(0).begin();
}

void ThreadIdXOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res) {
  odsState.addTypes(res);
}

void ThreadIdXOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ThreadIdXOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ThreadIdXOp::verify() {
  if (::mlir::failed(ThreadIdXOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult ThreadIdXOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
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

void ThreadIdXOp::print(::mlir::OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

void ThreadIdXOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace NVVM
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::ThreadIdXOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::ThreadIdYOp definitions
//===----------------------------------------------------------------------===//

ThreadIdYOpAdaptor::ThreadIdYOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ThreadIdYOpAdaptor::ThreadIdYOpAdaptor(ThreadIdYOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ThreadIdYOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ThreadIdYOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ThreadIdYOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr ThreadIdYOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ThreadIdYOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> ThreadIdYOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ThreadIdYOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> ThreadIdYOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ThreadIdYOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ThreadIdYOp::res() {
  return *getODSResults(0).begin();
}

void ThreadIdYOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res) {
  odsState.addTypes(res);
}

void ThreadIdYOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ThreadIdYOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ThreadIdYOp::verify() {
  if (::mlir::failed(ThreadIdYOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult ThreadIdYOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
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

void ThreadIdYOp::print(::mlir::OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

void ThreadIdYOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace NVVM
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::ThreadIdYOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::ThreadIdZOp definitions
//===----------------------------------------------------------------------===//

ThreadIdZOpAdaptor::ThreadIdZOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ThreadIdZOpAdaptor::ThreadIdZOpAdaptor(ThreadIdZOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ThreadIdZOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ThreadIdZOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ThreadIdZOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr ThreadIdZOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ThreadIdZOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> ThreadIdZOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ThreadIdZOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> ThreadIdZOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ThreadIdZOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ThreadIdZOp::res() {
  return *getODSResults(0).begin();
}

void ThreadIdZOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res) {
  odsState.addTypes(res);
}

void ThreadIdZOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ThreadIdZOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ThreadIdZOp::verify() {
  if (::mlir::failed(ThreadIdZOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult ThreadIdZOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
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

void ThreadIdZOp::print(::mlir::OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

void ThreadIdZOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace NVVM
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::ThreadIdZOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::VoteBallotOp definitions
//===----------------------------------------------------------------------===//

VoteBallotOpAdaptor::VoteBallotOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

VoteBallotOpAdaptor::VoteBallotOpAdaptor(VoteBallotOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange VoteBallotOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VoteBallotOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VoteBallotOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VoteBallotOpAdaptor::mask() {
  return *getODSOperands(0).begin();
}

::mlir::Value VoteBallotOpAdaptor::pred() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr VoteBallotOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VoteBallotOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> VoteBallotOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VoteBallotOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VoteBallotOp::mask() {
  return *getODSOperands(0).begin();
}

::mlir::Value VoteBallotOp::pred() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange VoteBallotOp::maskMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VoteBallotOp::predMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VoteBallotOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VoteBallotOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VoteBallotOp::res() {
  return *getODSResults(0).begin();
}

void VoteBallotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value mask, ::mlir::Value pred) {
  odsState.addOperands(mask);
  odsState.addOperands(pred);
  odsState.addTypes(res);
}

void VoteBallotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value mask, ::mlir::Value pred) {
  odsState.addOperands(mask);
  odsState.addOperands(pred);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VoteBallotOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult VoteBallotOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return parseNVVMVoteBallotOp(parser, result);
}

void VoteBallotOp::print(::mlir::OpAsmPrinter &p) {
  printNVVMIntrinsicOp(p, this->getOperation());
}

::mlir::LogicalResult VoteBallotOp::verify() {
  if (::mlir::failed(VoteBallotOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

} // namespace NVVM
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::VoteBallotOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::WMMALoadAM16N16K16Op definitions
//===----------------------------------------------------------------------===//

WMMALoadAM16N16K16OpAdaptor::WMMALoadAM16N16K16OpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

WMMALoadAM16N16K16OpAdaptor::WMMALoadAM16N16K16OpAdaptor(WMMALoadAM16N16K16Op&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange WMMALoadAM16N16K16OpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> WMMALoadAM16N16K16OpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange WMMALoadAM16N16K16OpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange WMMALoadAM16N16K16OpAdaptor::args() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr WMMALoadAM16N16K16OpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult WMMALoadAM16N16K16OpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> WMMALoadAM16N16K16Op::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range WMMALoadAM16N16K16Op::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range WMMALoadAM16N16K16Op::args() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange WMMALoadAM16N16K16Op::argsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> WMMALoadAM16N16K16Op::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range WMMALoadAM16N16K16Op::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value WMMALoadAM16N16K16Op::res() {
  return *getODSResults(0).begin();
}

void WMMALoadAM16N16K16Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::ValueRange args) {
  odsState.addOperands(args);
  odsState.addTypes(res);
}

void WMMALoadAM16N16K16Op::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult WMMALoadAM16N16K16Op::verify() {
  if (::mlir::failed(WMMALoadAM16N16K16OpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::verify(*this);
}

::mlir::ParseResult WMMALoadAM16N16K16Op::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> argsOperands;
  ::llvm::SMLoc argsOperandsLoc;
  (void)argsOperandsLoc;
  ::llvm::ArrayRef<::mlir::Type> argsTypes;
  ::llvm::ArrayRef<::mlir::Type> resTypes;

  argsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(argsOperands))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType args__res_functionType;
  if (parser.parseType(args__res_functionType))
    return ::mlir::failure();
  argsTypes = args__res_functionType.getInputs();
  resTypes = args__res_functionType.getResults();
  result.addTypes(resTypes);
  if (parser.resolveOperands(argsOperands, argsTypes, argsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void WMMALoadAM16N16K16Op::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << args();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p.printFunctionalType(args().getTypes(), ::llvm::ArrayRef<::mlir::Type>(res().getType()));
}

} // namespace NVVM
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::WMMALoadAM16N16K16Op)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::WMMALoadBM16N16K16Op definitions
//===----------------------------------------------------------------------===//

WMMALoadBM16N16K16OpAdaptor::WMMALoadBM16N16K16OpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

WMMALoadBM16N16K16OpAdaptor::WMMALoadBM16N16K16OpAdaptor(WMMALoadBM16N16K16Op&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange WMMALoadBM16N16K16OpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> WMMALoadBM16N16K16OpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange WMMALoadBM16N16K16OpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange WMMALoadBM16N16K16OpAdaptor::args() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr WMMALoadBM16N16K16OpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult WMMALoadBM16N16K16OpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> WMMALoadBM16N16K16Op::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range WMMALoadBM16N16K16Op::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range WMMALoadBM16N16K16Op::args() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange WMMALoadBM16N16K16Op::argsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> WMMALoadBM16N16K16Op::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range WMMALoadBM16N16K16Op::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value WMMALoadBM16N16K16Op::res() {
  return *getODSResults(0).begin();
}

void WMMALoadBM16N16K16Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::ValueRange args) {
  odsState.addOperands(args);
  odsState.addTypes(res);
}

void WMMALoadBM16N16K16Op::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult WMMALoadBM16N16K16Op::verify() {
  if (::mlir::failed(WMMALoadBM16N16K16OpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::verify(*this);
}

::mlir::ParseResult WMMALoadBM16N16K16Op::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> argsOperands;
  ::llvm::SMLoc argsOperandsLoc;
  (void)argsOperandsLoc;
  ::llvm::ArrayRef<::mlir::Type> argsTypes;
  ::llvm::ArrayRef<::mlir::Type> resTypes;

  argsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(argsOperands))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType args__res_functionType;
  if (parser.parseType(args__res_functionType))
    return ::mlir::failure();
  argsTypes = args__res_functionType.getInputs();
  resTypes = args__res_functionType.getResults();
  result.addTypes(resTypes);
  if (parser.resolveOperands(argsOperands, argsTypes, argsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void WMMALoadBM16N16K16Op::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << args();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p.printFunctionalType(args().getTypes(), ::llvm::ArrayRef<::mlir::Type>(res().getType()));
}

} // namespace NVVM
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::WMMALoadBM16N16K16Op)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::WMMALoadCF16M16N16K16Op definitions
//===----------------------------------------------------------------------===//

WMMALoadCF16M16N16K16OpAdaptor::WMMALoadCF16M16N16K16OpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

WMMALoadCF16M16N16K16OpAdaptor::WMMALoadCF16M16N16K16OpAdaptor(WMMALoadCF16M16N16K16Op&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange WMMALoadCF16M16N16K16OpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> WMMALoadCF16M16N16K16OpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange WMMALoadCF16M16N16K16OpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange WMMALoadCF16M16N16K16OpAdaptor::args() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr WMMALoadCF16M16N16K16OpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult WMMALoadCF16M16N16K16OpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> WMMALoadCF16M16N16K16Op::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range WMMALoadCF16M16N16K16Op::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range WMMALoadCF16M16N16K16Op::args() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange WMMALoadCF16M16N16K16Op::argsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> WMMALoadCF16M16N16K16Op::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range WMMALoadCF16M16N16K16Op::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value WMMALoadCF16M16N16K16Op::res() {
  return *getODSResults(0).begin();
}

void WMMALoadCF16M16N16K16Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::ValueRange args) {
  odsState.addOperands(args);
  odsState.addTypes(res);
}

void WMMALoadCF16M16N16K16Op::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult WMMALoadCF16M16N16K16Op::verify() {
  if (::mlir::failed(WMMALoadCF16M16N16K16OpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::verify(*this);
}

::mlir::ParseResult WMMALoadCF16M16N16K16Op::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> argsOperands;
  ::llvm::SMLoc argsOperandsLoc;
  (void)argsOperandsLoc;
  ::llvm::ArrayRef<::mlir::Type> argsTypes;
  ::llvm::ArrayRef<::mlir::Type> resTypes;

  argsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(argsOperands))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType args__res_functionType;
  if (parser.parseType(args__res_functionType))
    return ::mlir::failure();
  argsTypes = args__res_functionType.getInputs();
  resTypes = args__res_functionType.getResults();
  result.addTypes(resTypes);
  if (parser.resolveOperands(argsOperands, argsTypes, argsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void WMMALoadCF16M16N16K16Op::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << args();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p.printFunctionalType(args().getTypes(), ::llvm::ArrayRef<::mlir::Type>(res().getType()));
}

} // namespace NVVM
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::WMMALoadCF16M16N16K16Op)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::WMMALoadCF32M16N16K16Op definitions
//===----------------------------------------------------------------------===//

WMMALoadCF32M16N16K16OpAdaptor::WMMALoadCF32M16N16K16OpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

WMMALoadCF32M16N16K16OpAdaptor::WMMALoadCF32M16N16K16OpAdaptor(WMMALoadCF32M16N16K16Op&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange WMMALoadCF32M16N16K16OpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> WMMALoadCF32M16N16K16OpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange WMMALoadCF32M16N16K16OpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange WMMALoadCF32M16N16K16OpAdaptor::args() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr WMMALoadCF32M16N16K16OpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult WMMALoadCF32M16N16K16OpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> WMMALoadCF32M16N16K16Op::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range WMMALoadCF32M16N16K16Op::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range WMMALoadCF32M16N16K16Op::args() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange WMMALoadCF32M16N16K16Op::argsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> WMMALoadCF32M16N16K16Op::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range WMMALoadCF32M16N16K16Op::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value WMMALoadCF32M16N16K16Op::res() {
  return *getODSResults(0).begin();
}

void WMMALoadCF32M16N16K16Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::ValueRange args) {
  odsState.addOperands(args);
  odsState.addTypes(res);
}

void WMMALoadCF32M16N16K16Op::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult WMMALoadCF32M16N16K16Op::verify() {
  if (::mlir::failed(WMMALoadCF32M16N16K16OpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::verify(*this);
}

::mlir::ParseResult WMMALoadCF32M16N16K16Op::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> argsOperands;
  ::llvm::SMLoc argsOperandsLoc;
  (void)argsOperandsLoc;
  ::llvm::ArrayRef<::mlir::Type> argsTypes;
  ::llvm::ArrayRef<::mlir::Type> resTypes;

  argsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(argsOperands))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType args__res_functionType;
  if (parser.parseType(args__res_functionType))
    return ::mlir::failure();
  argsTypes = args__res_functionType.getInputs();
  resTypes = args__res_functionType.getResults();
  result.addTypes(resTypes);
  if (parser.resolveOperands(argsOperands, argsTypes, argsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void WMMALoadCF32M16N16K16Op::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << args();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p.printFunctionalType(args().getTypes(), ::llvm::ArrayRef<::mlir::Type>(res().getType()));
}

} // namespace NVVM
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::WMMALoadCF32M16N16K16Op)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::WMMAMmaF16F16M16N16K16Op definitions
//===----------------------------------------------------------------------===//

WMMAMmaF16F16M16N16K16OpAdaptor::WMMAMmaF16F16M16N16K16OpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

WMMAMmaF16F16M16N16K16OpAdaptor::WMMAMmaF16F16M16N16K16OpAdaptor(WMMAMmaF16F16M16N16K16Op&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange WMMAMmaF16F16M16N16K16OpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> WMMAMmaF16F16M16N16K16OpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange WMMAMmaF16F16M16N16K16OpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange WMMAMmaF16F16M16N16K16OpAdaptor::args() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr WMMAMmaF16F16M16N16K16OpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult WMMAMmaF16F16M16N16K16OpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> WMMAMmaF16F16M16N16K16Op::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range WMMAMmaF16F16M16N16K16Op::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range WMMAMmaF16F16M16N16K16Op::args() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange WMMAMmaF16F16M16N16K16Op::argsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> WMMAMmaF16F16M16N16K16Op::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range WMMAMmaF16F16M16N16K16Op::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value WMMAMmaF16F16M16N16K16Op::res() {
  return *getODSResults(0).begin();
}

void WMMAMmaF16F16M16N16K16Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::ValueRange args) {
  odsState.addOperands(args);
  odsState.addTypes(res);
}

void WMMAMmaF16F16M16N16K16Op::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult WMMAMmaF16F16M16N16K16Op::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return parseWMMAMmaF16F16M16N16K16Op(parser, result);
}

void WMMAMmaF16F16M16N16K16Op::print(::mlir::OpAsmPrinter &p) {
  printWMMAMmaF16F16M16N16K16Op(p, *this);
}

::mlir::LogicalResult WMMAMmaF16F16M16N16K16Op::verify() {
  if (::mlir::failed(WMMAMmaF16F16M16N16K16OpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::verify(*this);
}

} // namespace NVVM
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::WMMAMmaF16F16M16N16K16Op)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::WMMAMmaF32F32M16N16K16Op definitions
//===----------------------------------------------------------------------===//

WMMAMmaF32F32M16N16K16OpAdaptor::WMMAMmaF32F32M16N16K16OpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

WMMAMmaF32F32M16N16K16OpAdaptor::WMMAMmaF32F32M16N16K16OpAdaptor(WMMAMmaF32F32M16N16K16Op&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange WMMAMmaF32F32M16N16K16OpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> WMMAMmaF32F32M16N16K16OpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange WMMAMmaF32F32M16N16K16OpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange WMMAMmaF32F32M16N16K16OpAdaptor::args() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr WMMAMmaF32F32M16N16K16OpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult WMMAMmaF32F32M16N16K16OpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> WMMAMmaF32F32M16N16K16Op::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range WMMAMmaF32F32M16N16K16Op::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range WMMAMmaF32F32M16N16K16Op::args() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange WMMAMmaF32F32M16N16K16Op::argsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> WMMAMmaF32F32M16N16K16Op::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range WMMAMmaF32F32M16N16K16Op::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value WMMAMmaF32F32M16N16K16Op::res() {
  return *getODSResults(0).begin();
}

void WMMAMmaF32F32M16N16K16Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::ValueRange args) {
  odsState.addOperands(args);
  odsState.addTypes(res);
}

void WMMAMmaF32F32M16N16K16Op::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult WMMAMmaF32F32M16N16K16Op::verify() {
  if (::mlir::failed(WMMAMmaF32F32M16N16K16OpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::verify(*this);
}

::mlir::ParseResult WMMAMmaF32F32M16N16K16Op::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> argsOperands;
  ::llvm::SMLoc argsOperandsLoc;
  (void)argsOperandsLoc;
  ::llvm::ArrayRef<::mlir::Type> argsTypes;
  ::llvm::ArrayRef<::mlir::Type> resTypes;

  argsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(argsOperands))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType args__res_functionType;
  if (parser.parseType(args__res_functionType))
    return ::mlir::failure();
  argsTypes = args__res_functionType.getInputs();
  resTypes = args__res_functionType.getResults();
  result.addTypes(resTypes);
  if (parser.resolveOperands(argsOperands, argsTypes, argsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void WMMAMmaF32F32M16N16K16Op::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << args();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p.printFunctionalType(args().getTypes(), ::llvm::ArrayRef<::mlir::Type>(res().getType()));
}

} // namespace NVVM
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::WMMAMmaF32F32M16N16K16Op)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::WMMAStoreF16M16N16K16Op definitions
//===----------------------------------------------------------------------===//

WMMAStoreF16M16N16K16OpAdaptor::WMMAStoreF16M16N16K16OpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

WMMAStoreF16M16N16K16OpAdaptor::WMMAStoreF16M16N16K16OpAdaptor(WMMAStoreF16M16N16K16Op&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange WMMAStoreF16M16N16K16OpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> WMMAStoreF16M16N16K16OpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange WMMAStoreF16M16N16K16OpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange WMMAStoreF16M16N16K16OpAdaptor::args() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr WMMAStoreF16M16N16K16OpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult WMMAStoreF16M16N16K16OpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> WMMAStoreF16M16N16K16Op::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range WMMAStoreF16M16N16K16Op::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range WMMAStoreF16M16N16K16Op::args() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange WMMAStoreF16M16N16K16Op::argsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> WMMAStoreF16M16N16K16Op::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range WMMAStoreF16M16N16K16Op::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void WMMAStoreF16M16N16K16Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange args) {
  odsState.addOperands(args);
}

void WMMAStoreF16M16N16K16Op::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult WMMAStoreF16M16N16K16Op::verify() {
  if (::mlir::failed(WMMAStoreF16M16N16K16OpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::verify(*this);
}

::mlir::ParseResult WMMAStoreF16M16N16K16Op::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> argsOperands;
  ::llvm::SMLoc argsOperandsLoc;
  (void)argsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> argsTypes;

  argsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(argsOperands))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(argsTypes))
    return ::mlir::failure();
  if (parser.resolveOperands(argsOperands, argsTypes, argsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void WMMAStoreF16M16N16K16Op::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << args();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << args().getTypes();
}

} // namespace NVVM
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::WMMAStoreF16M16N16K16Op)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::WMMAStoreF32M16N16K16Op definitions
//===----------------------------------------------------------------------===//

WMMAStoreF32M16N16K16OpAdaptor::WMMAStoreF32M16N16K16OpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

WMMAStoreF32M16N16K16OpAdaptor::WMMAStoreF32M16N16K16OpAdaptor(WMMAStoreF32M16N16K16Op&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange WMMAStoreF32M16N16K16OpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> WMMAStoreF32M16N16K16OpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange WMMAStoreF32M16N16K16OpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange WMMAStoreF32M16N16K16OpAdaptor::args() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr WMMAStoreF32M16N16K16OpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult WMMAStoreF32M16N16K16OpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> WMMAStoreF32M16N16K16Op::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range WMMAStoreF32M16N16K16Op::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range WMMAStoreF32M16N16K16Op::args() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange WMMAStoreF32M16N16K16Op::argsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> WMMAStoreF32M16N16K16Op::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range WMMAStoreF32M16N16K16Op::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void WMMAStoreF32M16N16K16Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange args) {
  odsState.addOperands(args);
}

void WMMAStoreF32M16N16K16Op::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult WMMAStoreF32M16N16K16Op::verify() {
  if (::mlir::failed(WMMAStoreF32M16N16K16OpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::verify(*this);
}

::mlir::ParseResult WMMAStoreF32M16N16K16Op::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> argsOperands;
  ::llvm::SMLoc argsOperandsLoc;
  (void)argsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> argsTypes;

  argsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(argsOperands))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(argsTypes))
    return ::mlir::failure();
  if (parser.resolveOperands(argsOperands, argsTypes, argsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void WMMAStoreF32M16N16K16Op::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << args();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << args().getTypes();
}

} // namespace NVVM
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::WMMAStoreF32M16N16K16Op)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::WarpSizeOp definitions
//===----------------------------------------------------------------------===//

WarpSizeOpAdaptor::WarpSizeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

WarpSizeOpAdaptor::WarpSizeOpAdaptor(WarpSizeOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange WarpSizeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> WarpSizeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange WarpSizeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr WarpSizeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult WarpSizeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> WarpSizeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range WarpSizeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> WarpSizeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range WarpSizeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value WarpSizeOp::res() {
  return *getODSResults(0).begin();
}

void WarpSizeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res) {
  odsState.addTypes(res);
}

void WarpSizeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void WarpSizeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult WarpSizeOp::verify() {
  if (::mlir::failed(WarpSizeOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult WarpSizeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
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

void WarpSizeOp::print(::mlir::OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

void WarpSizeOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace NVVM
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::WarpSizeOp)


#endif  // GET_OP_CLASSES

