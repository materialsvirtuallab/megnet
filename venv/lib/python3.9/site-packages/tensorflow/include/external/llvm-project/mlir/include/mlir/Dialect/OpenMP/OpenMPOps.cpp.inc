/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::omp::BarrierOp,
::mlir::omp::CriticalDeclareOp,
::mlir::omp::CriticalOp,
::mlir::omp::FlushOp,
::mlir::omp::MasterOp,
::mlir::omp::ParallelOp,
::mlir::omp::ReductionDeclareOp,
::mlir::omp::ReductionOp,
::mlir::omp::TargetOp,
::mlir::omp::TaskwaitOp,
::mlir::omp::TaskyieldOp,
::mlir::omp::TerminatorOp,
::mlir::omp::WsLoopOp,
::mlir::omp::YieldOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace omp {
static ::mlir::LogicalResult __mlir_ods_local_type_constraint_OpenMPOps0(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((true))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be any type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_OpenMPOps3(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isa<::mlir::omp::PointerLikeType>()))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be OpenMP-compatible variable type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_OpenMPOps4(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isSignlessInteger(1)))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be 1-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_OpenMPOps5(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isa<::mlir::IntegerType>()))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_OpenMPOps6(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isa<::mlir::IntegerType>())) || ((type.isa<::mlir::IndexType>())))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be integer or index, but got " << type;
  }
  return ::mlir::success();
}

} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::BarrierOp definitions
//===----------------------------------------------------------------------===//

BarrierOpAdaptor::BarrierOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

BarrierOpAdaptor::BarrierOpAdaptor(BarrierOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange BarrierOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BarrierOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BarrierOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr BarrierOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult BarrierOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> BarrierOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BarrierOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> BarrierOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BarrierOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void BarrierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {

}

void BarrierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BarrierOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BarrierOp::verify() {
  if (::mlir::failed(BarrierOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::ParseResult BarrierOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void BarrierOp::print(::mlir::OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace omp
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::BarrierOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::CriticalDeclareOp definitions
//===----------------------------------------------------------------------===//

CriticalDeclareOpAdaptor::CriticalDeclareOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

CriticalDeclareOpAdaptor::CriticalDeclareOpAdaptor(CriticalDeclareOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange CriticalDeclareOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CriticalDeclareOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CriticalDeclareOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr CriticalDeclareOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr CriticalDeclareOpAdaptor::sym_name() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("sym_name").cast<::mlir::StringAttr>();
  return attr;
}

::mlir::LogicalResult CriticalDeclareOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_sym_name = odsAttrs.get("sym_name");
  if (!tblgen_sym_name) return emitError(loc, "'omp.critical.declare' op ""requires attribute 'sym_name'");
    if (!((tblgen_sym_name.isa<::mlir::StringAttr>()))) return emitError(loc, "'omp.critical.declare' op ""attribute 'sym_name' failed to satisfy constraint: string attribute");
  }
  return ::mlir::success();
}













std::pair<unsigned, unsigned> CriticalDeclareOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CriticalDeclareOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> CriticalDeclareOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CriticalDeclareOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::StringAttr CriticalDeclareOp::sym_nameAttr() {
  return (*this)->getAttr(sym_nameAttrName()).template cast<::mlir::StringAttr>();
}

::llvm::StringRef CriticalDeclareOp::sym_name() {
  auto attr = sym_nameAttr();
  return attr.getValue();
}

void CriticalDeclareOp::sym_nameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(sym_nameAttrName(), attr);
}

void CriticalDeclareOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr sym_name) {
  odsState.addAttribute(sym_nameAttrName(odsState.name), sym_name);
}

void CriticalDeclareOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr sym_name) {
  odsState.addAttribute(sym_nameAttrName(odsState.name), sym_name);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CriticalDeclareOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef sym_name) {
  odsState.addAttribute(sym_nameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
}

void CriticalDeclareOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef sym_name) {
  odsState.addAttribute(sym_nameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CriticalDeclareOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CriticalDeclareOp::verify() {
  if (::mlir::failed(CriticalDeclareOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::ParseResult CriticalDeclareOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr sym_nameAttr;

  if (parser.parseSymbolName(sym_nameAttr, "sym_name", result.attributes))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void CriticalDeclareOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p.printSymbolName(sym_nameAttr().getValue());
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"sym_name"});
}

} // namespace omp
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::CriticalDeclareOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::CriticalOp definitions
//===----------------------------------------------------------------------===//

CriticalOpAdaptor::CriticalOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

CriticalOpAdaptor::CriticalOpAdaptor(CriticalOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange CriticalOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CriticalOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CriticalOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr CriticalOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::FlatSymbolRefAttr CriticalOpAdaptor::name() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::FlatSymbolRefAttr attr = odsAttrs.get("name").dyn_cast_or_null<::mlir::FlatSymbolRefAttr>();
  return attr;
}

::mlir::omp::SyncHintKindAttr CriticalOpAdaptor::hint() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::omp::SyncHintKindAttr attr = odsAttrs.get("hint").dyn_cast_or_null<::mlir::omp::SyncHintKindAttr>();
  return attr;
}

::mlir::RegionRange CriticalOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &CriticalOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult CriticalOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_name = odsAttrs.get("name");
  if (tblgen_name) {
    if (!((tblgen_name.isa<::mlir::FlatSymbolRefAttr>()))) return emitError(loc, "'omp.critical' op ""attribute 'name' failed to satisfy constraint: flat symbol reference attribute");
  }
  }
  {
  auto tblgen_hint = odsAttrs.get("hint");
  if (tblgen_hint) {
    if (!((tblgen_hint.isa<::mlir::omp::SyncHintKindAttr>()))) return emitError(loc, "'omp.critical' op ""attribute 'hint' failed to satisfy constraint: OpenMP Sync Hint Kind");
  }
  }
  return ::mlir::success();
}

















std::pair<unsigned, unsigned> CriticalOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CriticalOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> CriticalOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CriticalOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &CriticalOp::region() {
  return (*this)->getRegion(0);
}

::mlir::FlatSymbolRefAttr CriticalOp::nameAttr() {
  return (*this)->getAttr(nameAttrName()).template dyn_cast_or_null<::mlir::FlatSymbolRefAttr>();
}

::llvm::Optional< ::llvm::StringRef > CriticalOp::name() {
  auto attr = nameAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::omp::SyncHintKindAttr CriticalOp::hintAttr() {
  return (*this)->getAttr(hintAttrName()).template dyn_cast_or_null<::mlir::omp::SyncHintKindAttr>();
}

::llvm::Optional<::mlir::omp::SyncHintKind> CriticalOp::hint() {
  auto attr = hintAttr();
  return attr ? ::llvm::Optional<::mlir::omp::SyncHintKind>(attr.getValue()) : (::llvm::None);
}

void CriticalOp::nameAttr(::mlir::FlatSymbolRefAttr attr) {
  (*this)->setAttr(nameAttrName(), attr);
}

void CriticalOp::hintAttr(::mlir::omp::SyncHintKindAttr attr) {
  (*this)->setAttr(hintAttrName(), attr);
}

::mlir::Attribute CriticalOp::removeNameAttr() {
  return (*this)->removeAttr(nameAttrName());
}

::mlir::Attribute CriticalOp::removeHintAttr() {
  return (*this)->removeAttr(hintAttrName());
}

void CriticalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::FlatSymbolRefAttr name, /*optional*/::mlir::omp::SyncHintKindAttr hint) {
  if (name) {
  odsState.addAttribute(nameAttrName(odsState.name), name);
  }
  if (hint) {
  odsState.addAttribute(hintAttrName(odsState.name), hint);
  }
  (void)odsState.addRegion();
}

void CriticalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::FlatSymbolRefAttr name, /*optional*/::mlir::omp::SyncHintKindAttr hint) {
  if (name) {
  odsState.addAttribute(nameAttrName(odsState.name), name);
  }
  if (hint) {
  odsState.addAttribute(hintAttrName(odsState.name), hint);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CriticalOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CriticalOp::verify() {
  if (::mlir::failed(CriticalOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::verifyCriticalOp(*this);
}

::mlir::ParseResult CriticalOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::FlatSymbolRefAttr nameAttr;
  ::mlir::omp::SyncHintKindAttr hintAttr;
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();
  if (succeeded(parser.parseOptionalLParen())) {

  {
    ::mlir::OptionalParseResult parseResult =
      parser.parseOptionalAttribute(nameAttr, parser.getBuilder().getType<::mlir::NoneType>(), "name", result.attributes);
    if (parseResult.hasValue() && failed(*parseResult))
      return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("hint"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::llvm::StringRef attrStr;
    ::mlir::NamedAttrList attrStorage;
    auto loc = parser.getCurrentLocation();
    if (parser.parseOptionalKeyword(&attrStr, {"none","uncontended","contended","nonspeculative","speculative"})) {
      ::mlir::StringAttr attrVal;
      ::mlir::OptionalParseResult parseResult =
        parser.parseOptionalAttribute(attrVal,
                                      parser.getBuilder().getNoneType(),
                                      "hint", attrStorage);
      if (parseResult.hasValue()) {
        if (failed(*parseResult))
          return ::mlir::failure();
        attrStr = attrVal.getValue();
      } else {
        
      }
    }
    if (!attrStr.empty()) {
      auto attrOptional = ::mlir::omp::ConvertToEnum(attrStr);
      if (!attrOptional)
        return parser.emitError(loc, "invalid ")
               << "hint attribute specification: \"" << attrStr << '"';;

      hintAttr = ::mlir::omp::SyncHintKindAttr::get(parser.getBuilder().getContext(), attrOptional.getValue());
      result.addAttribute("hint", hintAttr);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }

  if (parser.parseRegion(*regionRegion))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addRegion(std::move(regionRegion));
  return ::mlir::success();
}

void CriticalOp::print(::mlir::OpAsmPrinter &p) {
  if ((*this)->getAttr("name")) {
  p << "(";
  p.printAttributeWithoutType(nameAttr());
  p << ")";
  }
  if ((*this)->getAttr("hint")) {
  p << ' ' << "hint";
  p << "(";

  {
    auto caseValue = *hint();
    auto caseValueStr = ConvertToString(caseValue);
    p << caseValueStr;
  }
  p << ")";
  }
  p << ' ';
  p.printRegion(region());
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"name", "hint"});
}

} // namespace omp
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::CriticalOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::FlushOp definitions
//===----------------------------------------------------------------------===//

FlushOpAdaptor::FlushOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

FlushOpAdaptor::FlushOpAdaptor(FlushOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange FlushOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FlushOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange FlushOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange FlushOpAdaptor::varList() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr FlushOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult FlushOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> FlushOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range FlushOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range FlushOp::varList() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange FlushOp::varListMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> FlushOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FlushOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void FlushOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange varList) {
  odsState.addOperands(varList);
}

void FlushOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult FlushOp::verify() {
  if (::mlir::failed(FlushOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::ParseResult FlushOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> varListOperands;
  ::llvm::SMLoc varListOperandsLoc;
  (void)varListOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> varListTypes;
  if (succeeded(parser.parseOptionalLParen())) {

  varListOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(varListOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(varListTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.resolveOperands(varListOperands, varListTypes, varListOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void FlushOp::print(::mlir::OpAsmPrinter &p) {
  if (!varList().empty()) {
  p << "(";
  p << varList();
  p << ' ' << ":";
  p << ' ';
  p << varList().getTypes();
  p << ")";
  }
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace omp
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::FlushOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::MasterOp definitions
//===----------------------------------------------------------------------===//

MasterOpAdaptor::MasterOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

MasterOpAdaptor::MasterOpAdaptor(MasterOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange MasterOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MasterOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MasterOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr MasterOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange MasterOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &MasterOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult MasterOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> MasterOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MasterOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> MasterOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MasterOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &MasterOp::region() {
  return (*this)->getRegion(0);
}

void MasterOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
  (void)odsState.addRegion();
}

void MasterOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MasterOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MasterOp::verify() {
  if (::mlir::failed(MasterOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult MasterOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();

  if (parser.parseRegion(*regionRegion))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addRegion(std::move(regionRegion));
  return ::mlir::success();
}

void MasterOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p.printRegion(region());
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace omp
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::MasterOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::ParallelOp definitions
//===----------------------------------------------------------------------===//

ParallelOpAdaptor::ParallelOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ParallelOpAdaptor::ParallelOpAdaptor(ParallelOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ParallelOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ParallelOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange ParallelOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ParallelOpAdaptor::if_expr_var() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value ParallelOpAdaptor::num_threads_var() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::ValueRange ParallelOpAdaptor::private_vars() {
  return getODSOperands(2);
}

::mlir::ValueRange ParallelOpAdaptor::firstprivate_vars() {
  return getODSOperands(3);
}

::mlir::ValueRange ParallelOpAdaptor::shared_vars() {
  return getODSOperands(4);
}

::mlir::ValueRange ParallelOpAdaptor::copyin_vars() {
  return getODSOperands(5);
}

::mlir::ValueRange ParallelOpAdaptor::allocate_vars() {
  return getODSOperands(6);
}

::mlir::ValueRange ParallelOpAdaptor::allocators_vars() {
  return getODSOperands(7);
}

::mlir::DictionaryAttr ParallelOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr ParallelOpAdaptor::default_val() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("default_val").dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::mlir::StringAttr ParallelOpAdaptor::proc_bind_val() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("proc_bind_val").dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::mlir::RegionRange ParallelOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &ParallelOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult ParallelOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 8)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 8 elements, but got ") << numElements;
  }
    {
  auto tblgen_default_val = odsAttrs.get("default_val");
  if (tblgen_default_val) {
    if (!(((tblgen_default_val.isa<::mlir::StringAttr>())) && (((tblgen_default_val.cast<::mlir::StringAttr>().getValue() == "defprivate")) || ((tblgen_default_val.cast<::mlir::StringAttr>().getValue() == "deffirstprivate")) || ((tblgen_default_val.cast<::mlir::StringAttr>().getValue() == "defshared")) || ((tblgen_default_val.cast<::mlir::StringAttr>().getValue() == "defnone"))))) return emitError(loc, "'omp.parallel' op ""attribute 'default_val' failed to satisfy constraint: default clause");
  }
  }
  {
  auto tblgen_proc_bind_val = odsAttrs.get("proc_bind_val");
  if (tblgen_proc_bind_val) {
    if (!(((tblgen_proc_bind_val.isa<::mlir::StringAttr>())) && (((tblgen_proc_bind_val.cast<::mlir::StringAttr>().getValue() == "primary")) || ((tblgen_proc_bind_val.cast<::mlir::StringAttr>().getValue() == "master")) || ((tblgen_proc_bind_val.cast<::mlir::StringAttr>().getValue() == "close")) || ((tblgen_proc_bind_val.cast<::mlir::StringAttr>().getValue() == "spread"))))) return emitError(loc, "'omp.parallel' op ""attribute 'proc_bind_val' failed to satisfy constraint: ProcBindKind Clause");
  }
  }
  return ::mlir::success();
}





















std::pair<unsigned, unsigned> ParallelOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range ParallelOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ParallelOp::if_expr_var() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value ParallelOp::num_threads_var() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Operation::operand_range ParallelOp::private_vars() {
  return getODSOperands(2);
}

::mlir::Operation::operand_range ParallelOp::firstprivate_vars() {
  return getODSOperands(3);
}

::mlir::Operation::operand_range ParallelOp::shared_vars() {
  return getODSOperands(4);
}

::mlir::Operation::operand_range ParallelOp::copyin_vars() {
  return getODSOperands(5);
}

::mlir::Operation::operand_range ParallelOp::allocate_vars() {
  return getODSOperands(6);
}

::mlir::Operation::operand_range ParallelOp::allocators_vars() {
  return getODSOperands(7);
}

::mlir::MutableOperandRange ParallelOp::if_expr_varMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::num_threads_varMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::private_varsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::firstprivate_varsMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::shared_varsMutable() {
  auto range = getODSOperandIndexAndLength(4);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(4u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::copyin_varsMutable() {
  auto range = getODSOperandIndexAndLength(5);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(5u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::allocate_varsMutable() {
  auto range = getODSOperandIndexAndLength(6);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(6u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::allocators_varsMutable() {
  auto range = getODSOperandIndexAndLength(7);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(7u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> ParallelOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ParallelOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &ParallelOp::region() {
  return (*this)->getRegion(0);
}

::mlir::StringAttr ParallelOp::default_valAttr() {
  return (*this)->getAttr(default_valAttrName()).template dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > ParallelOp::default_val() {
  auto attr = default_valAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::StringAttr ParallelOp::proc_bind_valAttr() {
  return (*this)->getAttr(proc_bind_valAttrName()).template dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > ParallelOp::proc_bind_val() {
  auto attr = proc_bind_valAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

void ParallelOp::default_valAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(default_valAttrName(), attr);
}

void ParallelOp::proc_bind_valAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(proc_bind_valAttrName(), attr);
}

::mlir::Attribute ParallelOp::removeDefault_valAttr() {
  return (*this)->removeAttr(default_valAttrName());
}

::mlir::Attribute ParallelOp::removeProc_bind_valAttr() {
  return (*this)->removeAttr(proc_bind_valAttrName());
}



void ParallelOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value if_expr_var, /*optional*/::mlir::Value num_threads_var, /*optional*/::mlir::StringAttr default_val, ::mlir::ValueRange private_vars, ::mlir::ValueRange firstprivate_vars, ::mlir::ValueRange shared_vars, ::mlir::ValueRange copyin_vars, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/::mlir::StringAttr proc_bind_val) {
  if (if_expr_var)
    odsState.addOperands(if_expr_var);
  if (num_threads_var)
    odsState.addOperands(num_threads_var);
  odsState.addOperands(private_vars);
  odsState.addOperands(firstprivate_vars);
  odsState.addOperands(shared_vars);
  odsState.addOperands(copyin_vars);
  odsState.addOperands(allocate_vars);
  odsState.addOperands(allocators_vars);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({(if_expr_var ? 1 : 0), (num_threads_var ? 1 : 0), static_cast<int32_t>(private_vars.size()), static_cast<int32_t>(firstprivate_vars.size()), static_cast<int32_t>(shared_vars.size()), static_cast<int32_t>(copyin_vars.size()), static_cast<int32_t>(allocate_vars.size()), static_cast<int32_t>(allocators_vars.size())}));
  if (default_val) {
  odsState.addAttribute(default_valAttrName(odsState.name), default_val);
  }
  if (proc_bind_val) {
  odsState.addAttribute(proc_bind_valAttrName(odsState.name), proc_bind_val);
  }
  (void)odsState.addRegion();
}

void ParallelOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value if_expr_var, /*optional*/::mlir::Value num_threads_var, /*optional*/::mlir::StringAttr default_val, ::mlir::ValueRange private_vars, ::mlir::ValueRange firstprivate_vars, ::mlir::ValueRange shared_vars, ::mlir::ValueRange copyin_vars, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/::mlir::StringAttr proc_bind_val) {
  if (if_expr_var)
    odsState.addOperands(if_expr_var);
  if (num_threads_var)
    odsState.addOperands(num_threads_var);
  odsState.addOperands(private_vars);
  odsState.addOperands(firstprivate_vars);
  odsState.addOperands(shared_vars);
  odsState.addOperands(copyin_vars);
  odsState.addOperands(allocate_vars);
  odsState.addOperands(allocators_vars);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({(if_expr_var ? 1 : 0), (num_threads_var ? 1 : 0), static_cast<int32_t>(private_vars.size()), static_cast<int32_t>(firstprivate_vars.size()), static_cast<int32_t>(shared_vars.size()), static_cast<int32_t>(copyin_vars.size()), static_cast<int32_t>(allocate_vars.size()), static_cast<int32_t>(allocators_vars.size())}));
  if (default_val) {
  odsState.addAttribute(default_valAttrName(odsState.name), default_val);
  }
  if (proc_bind_val) {
  odsState.addAttribute(proc_bind_valAttrName(odsState.name), proc_bind_val);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ParallelOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult ParallelOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return parseParallelOp(parser, result);
}

void ParallelOp::print(::mlir::OpAsmPrinter &p) {
  return printParallelOp(p, *this);
}

::mlir::LogicalResult ParallelOp::verify() {
  if (::mlir::failed(ParallelOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    if (valueGroup0.size() > 1)
      return emitOpError("operand group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup0.size();
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    if (valueGroup1.size() > 1)
      return emitOpError("operand group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup1.size();
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup3 = getODSOperands(3);
    for (::mlir::Value v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup4 = getODSOperands(4);
    for (::mlir::Value v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup5 = getODSOperands(5);
    for (::mlir::Value v : valueGroup5) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup6 = getODSOperands(6);
    for (::mlir::Value v : valueGroup6) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup7 = getODSOperands(7);
    for (::mlir::Value v : valueGroup7) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::verifyParallelOp(*this);
}

} // namespace omp
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::ParallelOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::ReductionDeclareOp definitions
//===----------------------------------------------------------------------===//

ReductionDeclareOpAdaptor::ReductionDeclareOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ReductionDeclareOpAdaptor::ReductionDeclareOpAdaptor(ReductionDeclareOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ReductionDeclareOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ReductionDeclareOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ReductionDeclareOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr ReductionDeclareOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr ReductionDeclareOpAdaptor::sym_name() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("sym_name").cast<::mlir::StringAttr>();
  return attr;
}

::mlir::TypeAttr ReductionDeclareOpAdaptor::type() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::TypeAttr attr = odsAttrs.get("type").cast<::mlir::TypeAttr>();
  return attr;
}

::mlir::RegionRange ReductionDeclareOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &ReductionDeclareOpAdaptor::initializerRegion() {
  return *odsRegions[0];
}

::mlir::Region &ReductionDeclareOpAdaptor::reductionRegion() {
  return *odsRegions[1];
}

::mlir::Region &ReductionDeclareOpAdaptor::atomicReductionRegion() {
  return *odsRegions[2];
}

::mlir::LogicalResult ReductionDeclareOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_sym_name = odsAttrs.get("sym_name");
  if (!tblgen_sym_name) return emitError(loc, "'omp.reduction.declare' op ""requires attribute 'sym_name'");
    if (!((tblgen_sym_name.isa<::mlir::StringAttr>()))) return emitError(loc, "'omp.reduction.declare' op ""attribute 'sym_name' failed to satisfy constraint: string attribute");
  }
  {
  auto tblgen_type = odsAttrs.get("type");
  if (!tblgen_type) return emitError(loc, "'omp.reduction.declare' op ""requires attribute 'type'");
    if (!(((tblgen_type.isa<::mlir::TypeAttr>())) && ((tblgen_type.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>())))) return emitError(loc, "'omp.reduction.declare' op ""attribute 'type' failed to satisfy constraint: any type attribute");
  }
  return ::mlir::success();
}

















std::pair<unsigned, unsigned> ReductionDeclareOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ReductionDeclareOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> ReductionDeclareOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReductionDeclareOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &ReductionDeclareOp::initializerRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &ReductionDeclareOp::reductionRegion() {
  return (*this)->getRegion(1);
}

::mlir::Region &ReductionDeclareOp::atomicReductionRegion() {
  return (*this)->getRegion(2);
}

::mlir::StringAttr ReductionDeclareOp::sym_nameAttr() {
  return (*this)->getAttr(sym_nameAttrName()).template cast<::mlir::StringAttr>();
}

::llvm::StringRef ReductionDeclareOp::sym_name() {
  auto attr = sym_nameAttr();
  return attr.getValue();
}

::mlir::TypeAttr ReductionDeclareOp::typeAttr() {
  return (*this)->getAttr(typeAttrName()).template cast<::mlir::TypeAttr>();
}

::mlir::Type ReductionDeclareOp::type() {
  auto attr = typeAttr();
  return attr.getValue().cast<::mlir::Type>();
}

void ReductionDeclareOp::sym_nameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(sym_nameAttrName(), attr);
}

void ReductionDeclareOp::typeAttr(::mlir::TypeAttr attr) {
  (*this)->setAttr(typeAttrName(), attr);
}

void ReductionDeclareOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr sym_name, ::mlir::TypeAttr type) {
  odsState.addAttribute(sym_nameAttrName(odsState.name), sym_name);
  odsState.addAttribute(typeAttrName(odsState.name), type);
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  (void)odsState.addRegion();
}

void ReductionDeclareOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr sym_name, ::mlir::TypeAttr type) {
  odsState.addAttribute(sym_nameAttrName(odsState.name), sym_name);
  odsState.addAttribute(typeAttrName(odsState.name), type);
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReductionDeclareOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef sym_name, ::mlir::Type type) {
  odsState.addAttribute(sym_nameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
  odsState.addAttribute(typeAttrName(odsState.name), ::mlir::TypeAttr::get(type));
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  (void)odsState.addRegion();
}

void ReductionDeclareOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef sym_name, ::mlir::Type type) {
  odsState.addAttribute(sym_nameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
  odsState.addAttribute(typeAttrName(odsState.name), ::mlir::TypeAttr::get(type));
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReductionDeclareOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 3; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReductionDeclareOp::verify() {
  if (::mlir::failed(ReductionDeclareOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('initializerRegion') failed to verify constraint: any region";
      }
      ++index;
    }
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(1))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('reductionRegion') failed to verify constraint: any region";
      }
      ++index;
    }
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(2))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('atomicReductionRegion') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::verifyReductionDeclareOp(*this);
}

::mlir::ParseResult ReductionDeclareOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr sym_nameAttr;
  ::mlir::TypeAttr typeAttr;
  std::unique_ptr<::mlir::Region> initializerRegionRegion = std::make_unique<::mlir::Region>();
  std::unique_ptr<::mlir::Region> reductionRegionRegion = std::make_unique<::mlir::Region>();
  std::unique_ptr<::mlir::Region> atomicReductionRegionRegion = std::make_unique<::mlir::Region>();

  if (parser.parseSymbolName(sym_nameAttr, "sym_name", result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseAttribute(typeAttr, parser.getBuilder().getType<::mlir::NoneType>(), "type", result.attributes))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();
  if (parser.parseKeyword("init"))
    return ::mlir::failure();

  if (parser.parseRegion(*initializerRegionRegion))
    return ::mlir::failure();
  if (parser.parseKeyword("combiner"))
    return ::mlir::failure();

  if (parser.parseRegion(*reductionRegionRegion))
    return ::mlir::failure();
  {
    if (parseAtomicReductionRegion(parser, *atomicReductionRegionRegion))
      return ::mlir::failure();
  }
  result.addRegion(std::move(initializerRegionRegion));
  result.addRegion(std::move(reductionRegionRegion));
  result.addRegion(std::move(atomicReductionRegionRegion));
  return ::mlir::success();
}

void ReductionDeclareOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p.printSymbolName(sym_nameAttr().getValue());
  p << ' ' << ":";
  p << ' ';
  p.printAttributeWithoutType(typeAttr());
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(), /*elidedAttrs=*/{"sym_name", "type"});
  p << ' ' << "init";
  p << ' ';
  p.printRegion(initializerRegion());
  p << ' ' << "combiner";
  p << ' ';
  p.printRegion(reductionRegion());
  p << ' ';
  printAtomicReductionRegion(p, *this, atomicReductionRegion());
}

} // namespace omp
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::ReductionDeclareOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::ReductionOp definitions
//===----------------------------------------------------------------------===//

ReductionOpAdaptor::ReductionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ReductionOpAdaptor::ReductionOpAdaptor(ReductionOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ReductionOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ReductionOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ReductionOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReductionOpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::Value ReductionOpAdaptor::accumulator() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr ReductionOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ReductionOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> ReductionOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ReductionOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReductionOp::operand() {
  return *getODSOperands(0).begin();
}

::mlir::Value ReductionOp::accumulator() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange ReductionOp::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ReductionOp::accumulatorMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ReductionOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReductionOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void ReductionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand, ::mlir::Value accumulator) {
  odsState.addOperands(operand);
  odsState.addOperands(accumulator);
}

void ReductionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand, ::mlir::Value accumulator) {
  odsState.addOperands(operand);
  odsState.addOperands(accumulator);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReductionOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReductionOp::verify() {
  if (::mlir::failed(ReductionOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  if (!((std::equal_to<>()((*this->getODSOperands(1).begin()).getType().cast<::mlir::omp::PointerLikeType>().getElementType(), (*this->getODSOperands(0).begin()).getType()))))
    return emitOpError("failed to verify that value types matches accumulator element type");
  return ::verifyReductionOp(*this);
}

::mlir::ParseResult ReductionOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType operandRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> operandOperands(operandRawOperands);  ::llvm::SMLoc operandOperandsLoc;
  (void)operandOperandsLoc;
  ::mlir::OpAsmParser::OperandType accumulatorRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> accumulatorOperands(accumulatorRawOperands);  ::llvm::SMLoc accumulatorOperandsLoc;
  (void)accumulatorOperandsLoc;
  ::mlir::Type accumulatorRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> accumulatorTypes(accumulatorRawTypes);

  operandOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operandRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  accumulatorOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(accumulatorRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(accumulatorRawTypes[0]))
    return ::mlir::failure();
  for (::mlir::Type type : accumulatorTypes) {
    (void)type;
    if (!((type.isa<::mlir::omp::PointerLikeType>()))) {
      return parser.emitError(parser.getNameLoc()) << "'accumulator' must be OpenMP-compatible variable type, but got " << type;
    }
  }
  if (parser.resolveOperands(operandOperands, accumulatorTypes[0].cast<::mlir::omp::PointerLikeType>().getElementType(), operandOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(accumulatorOperands, accumulatorTypes, accumulatorOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ReductionOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << operand();
  p << ",";
  p << ' ';
  p << accumulator();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(accumulator().getType());
}

} // namespace omp
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::ReductionOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::TargetOp definitions
//===----------------------------------------------------------------------===//

TargetOpAdaptor::TargetOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

TargetOpAdaptor::TargetOpAdaptor(TargetOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange TargetOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> TargetOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange TargetOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TargetOpAdaptor::if_expr() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value TargetOpAdaptor::device() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value TargetOpAdaptor::thread_limit() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::DictionaryAttr TargetOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::UnitAttr TargetOpAdaptor::nowait() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::UnitAttr attr = odsAttrs.get("nowait").dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

::mlir::RegionRange TargetOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &TargetOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult TargetOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 3 elements, but got ") << numElements;
  }
    {
  auto tblgen_nowait = odsAttrs.get("nowait");
  if (tblgen_nowait) {
    if (!((tblgen_nowait.isa<::mlir::UnitAttr>()))) return emitError(loc, "'omp.target' op ""attribute 'nowait' failed to satisfy constraint: unit attribute");
  }
  }
  return ::mlir::success();
}

















std::pair<unsigned, unsigned> TargetOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range TargetOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TargetOp::if_expr() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value TargetOp::device() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value TargetOp::thread_limit() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::MutableOperandRange TargetOp::if_exprMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TargetOp::deviceMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TargetOp::thread_limitMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> TargetOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TargetOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &TargetOp::region() {
  return (*this)->getRegion(0);
}

::mlir::UnitAttr TargetOp::nowaitAttr() {
  return (*this)->getAttr(nowaitAttrName()).template dyn_cast_or_null<::mlir::UnitAttr>();
}

bool TargetOp::nowait() {
  auto attr = nowaitAttr();
  return attr != nullptr;
}

void TargetOp::nowaitAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(nowaitAttrName(), attr);
}

::mlir::Attribute TargetOp::removeNowaitAttr() {
  return (*this)->removeAttr(nowaitAttrName());
}

void TargetOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, /*optional*/::mlir::Value thread_limit, /*optional*/::mlir::UnitAttr nowait) {
  if (if_expr)
    odsState.addOperands(if_expr);
  if (device)
    odsState.addOperands(device);
  if (thread_limit)
    odsState.addOperands(thread_limit);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({(if_expr ? 1 : 0), (device ? 1 : 0), (thread_limit ? 1 : 0)}));
  if (nowait) {
  odsState.addAttribute(nowaitAttrName(odsState.name), nowait);
  }
  (void)odsState.addRegion();
}

void TargetOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, /*optional*/::mlir::Value thread_limit, /*optional*/::mlir::UnitAttr nowait) {
  if (if_expr)
    odsState.addOperands(if_expr);
  if (device)
    odsState.addOperands(device);
  if (thread_limit)
    odsState.addOperands(thread_limit);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({(if_expr ? 1 : 0), (device ? 1 : 0), (thread_limit ? 1 : 0)}));
  if (nowait) {
  odsState.addAttribute(nowaitAttrName(odsState.name), nowait);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TargetOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, /*optional*/::mlir::Value thread_limit, /*optional*/bool nowait) {
  if (if_expr)
    odsState.addOperands(if_expr);
  if (device)
    odsState.addOperands(device);
  if (thread_limit)
    odsState.addOperands(thread_limit);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({(if_expr ? 1 : 0), (device ? 1 : 0), (thread_limit ? 1 : 0)}));
  if (nowait) {
  odsState.addAttribute(nowaitAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  (void)odsState.addRegion();
}

void TargetOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, /*optional*/::mlir::Value thread_limit, /*optional*/bool nowait) {
  if (if_expr)
    odsState.addOperands(if_expr);
  if (device)
    odsState.addOperands(device);
  if (thread_limit)
    odsState.addOperands(thread_limit);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({(if_expr ? 1 : 0), (device ? 1 : 0), (thread_limit ? 1 : 0)}));
  if (nowait) {
  odsState.addAttribute(nowaitAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TargetOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TargetOp::verify() {
  if (::mlir::failed(TargetOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    if (valueGroup0.size() > 1)
      return emitOpError("operand group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup0.size();
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps4(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    if (valueGroup1.size() > 1)
      return emitOpError("operand group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup1.size();
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps5(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    if (valueGroup2.size() > 1)
      return emitOpError("operand group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup2.size();
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps5(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::mlir::success();
}

} // namespace omp
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::TargetOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::TaskwaitOp definitions
//===----------------------------------------------------------------------===//

TaskwaitOpAdaptor::TaskwaitOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

TaskwaitOpAdaptor::TaskwaitOpAdaptor(TaskwaitOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange TaskwaitOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> TaskwaitOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange TaskwaitOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr TaskwaitOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult TaskwaitOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> TaskwaitOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TaskwaitOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> TaskwaitOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TaskwaitOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void TaskwaitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {

}

void TaskwaitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TaskwaitOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TaskwaitOp::verify() {
  if (::mlir::failed(TaskwaitOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::ParseResult TaskwaitOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void TaskwaitOp::print(::mlir::OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace omp
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::TaskwaitOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::TaskyieldOp definitions
//===----------------------------------------------------------------------===//

TaskyieldOpAdaptor::TaskyieldOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

TaskyieldOpAdaptor::TaskyieldOpAdaptor(TaskyieldOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange TaskyieldOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> TaskyieldOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange TaskyieldOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr TaskyieldOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult TaskyieldOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> TaskyieldOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TaskyieldOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> TaskyieldOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TaskyieldOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void TaskyieldOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {

}

void TaskyieldOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TaskyieldOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TaskyieldOp::verify() {
  if (::mlir::failed(TaskyieldOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::ParseResult TaskyieldOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void TaskyieldOp::print(::mlir::OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace omp
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::TaskyieldOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::TerminatorOp definitions
//===----------------------------------------------------------------------===//

TerminatorOpAdaptor::TerminatorOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

TerminatorOpAdaptor::TerminatorOpAdaptor(TerminatorOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange TerminatorOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> TerminatorOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange TerminatorOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr TerminatorOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult TerminatorOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> TerminatorOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TerminatorOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> TerminatorOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TerminatorOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void TerminatorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {

}

void TerminatorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TerminatorOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TerminatorOp::verify() {
  if (::mlir::failed(TerminatorOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::ParseResult TerminatorOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void TerminatorOp::print(::mlir::OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace omp
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::TerminatorOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::WsLoopOp definitions
//===----------------------------------------------------------------------===//

WsLoopOpAdaptor::WsLoopOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

WsLoopOpAdaptor::WsLoopOpAdaptor(WsLoopOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange WsLoopOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> WsLoopOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange WsLoopOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange WsLoopOpAdaptor::lowerBound() {
  return getODSOperands(0);
}

::mlir::ValueRange WsLoopOpAdaptor::upperBound() {
  return getODSOperands(1);
}

::mlir::ValueRange WsLoopOpAdaptor::step() {
  return getODSOperands(2);
}

::mlir::ValueRange WsLoopOpAdaptor::private_vars() {
  return getODSOperands(3);
}

::mlir::ValueRange WsLoopOpAdaptor::firstprivate_vars() {
  return getODSOperands(4);
}

::mlir::ValueRange WsLoopOpAdaptor::lastprivate_vars() {
  return getODSOperands(5);
}

::mlir::ValueRange WsLoopOpAdaptor::linear_vars() {
  return getODSOperands(6);
}

::mlir::ValueRange WsLoopOpAdaptor::linear_step_vars() {
  return getODSOperands(7);
}

::mlir::ValueRange WsLoopOpAdaptor::reduction_vars() {
  return getODSOperands(8);
}

::mlir::Value WsLoopOpAdaptor::schedule_chunk_var() {
  auto operands = getODSOperands(9);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::DictionaryAttr WsLoopOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr WsLoopOpAdaptor::reductions() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("reductions").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::StringAttr WsLoopOpAdaptor::schedule_val() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("schedule_val").dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::mlir::IntegerAttr WsLoopOpAdaptor::collapse_val() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("collapse_val").dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::mlir::UnitAttr WsLoopOpAdaptor::nowait() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::UnitAttr attr = odsAttrs.get("nowait").dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

::mlir::IntegerAttr WsLoopOpAdaptor::ordered_val() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("ordered_val").dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::mlir::StringAttr WsLoopOpAdaptor::order_val() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("order_val").dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::mlir::UnitAttr WsLoopOpAdaptor::inclusive() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::UnitAttr attr = odsAttrs.get("inclusive").dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

::mlir::RegionRange WsLoopOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &WsLoopOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult WsLoopOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 10)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 10 elements, but got ") << numElements;
  }
    {
  auto tblgen_reductions = odsAttrs.get("reductions");
  if (tblgen_reductions) {
    if (!(((tblgen_reductions.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_reductions.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return (attr.isa<::mlir::SymbolRefAttr>()); })))) return emitError(loc, "'omp.wsloop' op ""attribute 'reductions' failed to satisfy constraint: array of symbol references");
  }
  }
  {
  auto tblgen_schedule_val = odsAttrs.get("schedule_val");
  if (tblgen_schedule_val) {
    if (!(((tblgen_schedule_val.isa<::mlir::StringAttr>())) && (((tblgen_schedule_val.cast<::mlir::StringAttr>().getValue() == "Static")) || ((tblgen_schedule_val.cast<::mlir::StringAttr>().getValue() == "Dynamic")) || ((tblgen_schedule_val.cast<::mlir::StringAttr>().getValue() == "Guided")) || ((tblgen_schedule_val.cast<::mlir::StringAttr>().getValue() == "Auto")) || ((tblgen_schedule_val.cast<::mlir::StringAttr>().getValue() == "Runtime"))))) return emitError(loc, "'omp.wsloop' op ""attribute 'schedule_val' failed to satisfy constraint: ScheduleKind Clause");
  }
  }
  {
  auto tblgen_collapse_val = odsAttrs.get("collapse_val");
  if (tblgen_collapse_val) {
    if (!((((tblgen_collapse_val.isa<::mlir::IntegerAttr>())) && ((tblgen_collapse_val.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && ((tblgen_collapse_val.cast<::mlir::IntegerAttr>().getInt() >= 0)))) return emitError(loc, "'omp.wsloop' op ""attribute 'collapse_val' failed to satisfy constraint: 64-bit signless integer attribute whose minimum value is 0");
  }
  }
  {
  auto tblgen_nowait = odsAttrs.get("nowait");
  if (tblgen_nowait) {
    if (!((tblgen_nowait.isa<::mlir::UnitAttr>()))) return emitError(loc, "'omp.wsloop' op ""attribute 'nowait' failed to satisfy constraint: unit attribute");
  }
  }
  {
  auto tblgen_ordered_val = odsAttrs.get("ordered_val");
  if (tblgen_ordered_val) {
    if (!((((tblgen_ordered_val.isa<::mlir::IntegerAttr>())) && ((tblgen_ordered_val.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && ((tblgen_ordered_val.cast<::mlir::IntegerAttr>().getInt() >= 0)))) return emitError(loc, "'omp.wsloop' op ""attribute 'ordered_val' failed to satisfy constraint: 64-bit signless integer attribute whose minimum value is 0");
  }
  }
  {
  auto tblgen_order_val = odsAttrs.get("order_val");
  if (tblgen_order_val) {
    if (!(((tblgen_order_val.isa<::mlir::StringAttr>())) && (false))) return emitError(loc, "'omp.wsloop' op ""attribute 'order_val' failed to satisfy constraint: OrderKind Clause");
  }
  }
  {
  auto tblgen_inclusive = odsAttrs.get("inclusive");
  if (tblgen_inclusive) {
    if (!((tblgen_inclusive.isa<::mlir::UnitAttr>()))) return emitError(loc, "'omp.wsloop' op ""attribute 'inclusive' failed to satisfy constraint: unit attribute");
  }
  }
  return ::mlir::success();
}









































std::pair<unsigned, unsigned> WsLoopOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range WsLoopOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range WsLoopOp::lowerBound() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range WsLoopOp::upperBound() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range WsLoopOp::step() {
  return getODSOperands(2);
}

::mlir::Operation::operand_range WsLoopOp::private_vars() {
  return getODSOperands(3);
}

::mlir::Operation::operand_range WsLoopOp::firstprivate_vars() {
  return getODSOperands(4);
}

::mlir::Operation::operand_range WsLoopOp::lastprivate_vars() {
  return getODSOperands(5);
}

::mlir::Operation::operand_range WsLoopOp::linear_vars() {
  return getODSOperands(6);
}

::mlir::Operation::operand_range WsLoopOp::linear_step_vars() {
  return getODSOperands(7);
}

::mlir::Operation::operand_range WsLoopOp::reduction_vars() {
  return getODSOperands(8);
}

::mlir::Value WsLoopOp::schedule_chunk_var() {
  auto operands = getODSOperands(9);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::MutableOperandRange WsLoopOp::lowerBoundMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange WsLoopOp::upperBoundMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange WsLoopOp::stepMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange WsLoopOp::private_varsMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange WsLoopOp::firstprivate_varsMutable() {
  auto range = getODSOperandIndexAndLength(4);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(4u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange WsLoopOp::lastprivate_varsMutable() {
  auto range = getODSOperandIndexAndLength(5);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(5u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange WsLoopOp::linear_varsMutable() {
  auto range = getODSOperandIndexAndLength(6);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(6u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange WsLoopOp::linear_step_varsMutable() {
  auto range = getODSOperandIndexAndLength(7);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(7u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange WsLoopOp::reduction_varsMutable() {
  auto range = getODSOperandIndexAndLength(8);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(8u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange WsLoopOp::schedule_chunk_varMutable() {
  auto range = getODSOperandIndexAndLength(9);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(9u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> WsLoopOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range WsLoopOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &WsLoopOp::region() {
  return (*this)->getRegion(0);
}

::mlir::ArrayAttr WsLoopOp::reductionsAttr() {
  return (*this)->getAttr(reductionsAttrName()).template dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > WsLoopOp::reductions() {
  auto attr = reductionsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::StringAttr WsLoopOp::schedule_valAttr() {
  return (*this)->getAttr(schedule_valAttrName()).template dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > WsLoopOp::schedule_val() {
  auto attr = schedule_valAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::IntegerAttr WsLoopOp::collapse_valAttr() {
  return (*this)->getAttr(collapse_valAttrName()).template dyn_cast_or_null<::mlir::IntegerAttr>();
}

::llvm::Optional<uint64_t> WsLoopOp::collapse_val() {
  auto attr = collapse_valAttr();
  return attr ? ::llvm::Optional<uint64_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

::mlir::UnitAttr WsLoopOp::nowaitAttr() {
  return (*this)->getAttr(nowaitAttrName()).template dyn_cast_or_null<::mlir::UnitAttr>();
}

bool WsLoopOp::nowait() {
  auto attr = nowaitAttr();
  return attr != nullptr;
}

::mlir::IntegerAttr WsLoopOp::ordered_valAttr() {
  return (*this)->getAttr(ordered_valAttrName()).template dyn_cast_or_null<::mlir::IntegerAttr>();
}

::llvm::Optional<uint64_t> WsLoopOp::ordered_val() {
  auto attr = ordered_valAttr();
  return attr ? ::llvm::Optional<uint64_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

::mlir::StringAttr WsLoopOp::order_valAttr() {
  return (*this)->getAttr(order_valAttrName()).template dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > WsLoopOp::order_val() {
  auto attr = order_valAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::UnitAttr WsLoopOp::inclusiveAttr() {
  return (*this)->getAttr(inclusiveAttrName()).template dyn_cast_or_null<::mlir::UnitAttr>();
}

bool WsLoopOp::inclusive() {
  auto attr = inclusiveAttr();
  return attr != nullptr;
}

void WsLoopOp::reductionsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(reductionsAttrName(), attr);
}

void WsLoopOp::schedule_valAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(schedule_valAttrName(), attr);
}

void WsLoopOp::collapse_valAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(collapse_valAttrName(), attr);
}

void WsLoopOp::nowaitAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(nowaitAttrName(), attr);
}

void WsLoopOp::ordered_valAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(ordered_valAttrName(), attr);
}

void WsLoopOp::order_valAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(order_valAttrName(), attr);
}

void WsLoopOp::inclusiveAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(inclusiveAttrName(), attr);
}

::mlir::Attribute WsLoopOp::removeReductionsAttr() {
  return (*this)->removeAttr(reductionsAttrName());
}

::mlir::Attribute WsLoopOp::removeSchedule_valAttr() {
  return (*this)->removeAttr(schedule_valAttrName());
}

::mlir::Attribute WsLoopOp::removeCollapse_valAttr() {
  return (*this)->removeAttr(collapse_valAttrName());
}

::mlir::Attribute WsLoopOp::removeNowaitAttr() {
  return (*this)->removeAttr(nowaitAttrName());
}

::mlir::Attribute WsLoopOp::removeOrdered_valAttr() {
  return (*this)->removeAttr(ordered_valAttrName());
}

::mlir::Attribute WsLoopOp::removeOrder_valAttr() {
  return (*this)->removeAttr(order_valAttrName());
}

::mlir::Attribute WsLoopOp::removeInclusiveAttr() {
  return (*this)->removeAttr(inclusiveAttrName());
}







::mlir::ParseResult WsLoopOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return parseWsLoopOp(parser, result);
}

void WsLoopOp::print(::mlir::OpAsmPrinter &p) {
  return printWsLoopOp(p, *this);
}

::mlir::LogicalResult WsLoopOp::verify() {
  if (::mlir::failed(WsLoopOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup3 = getODSOperands(3);
    for (::mlir::Value v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup4 = getODSOperands(4);
    for (::mlir::Value v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup5 = getODSOperands(5);
    for (::mlir::Value v : valueGroup5) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup6 = getODSOperands(6);
    for (::mlir::Value v : valueGroup6) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup7 = getODSOperands(7);
    for (::mlir::Value v : valueGroup7) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup8 = getODSOperands(8);
    for (::mlir::Value v : valueGroup8) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup9 = getODSOperands(9);
    if (valueGroup9.size() > 1)
      return emitOpError("operand group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup9.size();
    for (::mlir::Value v : valueGroup9) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({this->getODSOperands(0).getType(), this->getODSOperands(1).getType(), this->getODSOperands(2).getType()})))))
    return emitOpError("failed to verify that all of {lowerBound, upperBound, step} have same type");
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::verifyWsLoopOp(*this);
}

} // namespace omp
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::WsLoopOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::YieldOp definitions
//===----------------------------------------------------------------------===//

YieldOpAdaptor::YieldOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

YieldOpAdaptor::YieldOpAdaptor(YieldOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange YieldOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> YieldOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange YieldOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange YieldOpAdaptor::results() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr YieldOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult YieldOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> YieldOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range YieldOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range YieldOp::results() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange YieldOp::resultsMutable() {
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

void YieldOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange results) {
  odsState.addOperands(results);
}

void YieldOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
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
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(getOperation(), v.getType(), "operand", index)))
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
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> resultsOperands;
  ::llvm::SMLoc resultsOperandsLoc;
  (void)resultsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> resultsTypes;
  if (succeeded(parser.parseOptionalLParen())) {

  resultsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(resultsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(resultsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.resolveOperands(resultsOperands, resultsTypes, resultsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void YieldOp::print(::mlir::OpAsmPrinter &p) {
  if (!results().empty()) {
  p << "(";
  p << results();
  p << ' ' << ":";
  p << ' ';
  p << results().getTypes();
  p << ")";
  }
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

void YieldOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace omp
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::YieldOp)


#endif  // GET_OP_CLASSES

