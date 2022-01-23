/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::gpu::AllReduceOp,
::mlir::gpu::AllocOp,
::mlir::gpu::BarrierOp,
::mlir::gpu::BlockDimOp,
::mlir::gpu::BlockIdOp,
::mlir::gpu::DeallocOp,
::mlir::gpu::GPUFuncOp,
::mlir::gpu::GPUModuleOp,
::mlir::gpu::GridDimOp,
::mlir::gpu::HostRegisterOp,
::mlir::gpu::LaunchFuncOp,
::mlir::gpu::LaunchOp,
::mlir::gpu::MemcpyOp,
::mlir::gpu::MemsetOp,
::mlir::gpu::ModuleEndOp,
::mlir::gpu::NumSubgroupsOp,
::mlir::gpu::ReturnOp,
::mlir::gpu::ShuffleOp,
::mlir::gpu::SubgroupIdOp,
::mlir::gpu::SubgroupMmaComputeOp,
::mlir::gpu::SubgroupMmaConstantMatrixOp,
::mlir::gpu::SubgroupMmaLoadMatrixOp,
::mlir::gpu::SubgroupMmaStoreMatrixOp,
::mlir::gpu::SubgroupSizeOp,
::mlir::gpu::TerminatorOp,
::mlir::gpu::ThreadIdOp,
::mlir::gpu::WaitOp,
::mlir::gpu::YieldOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace gpu {
static ::mlir::LogicalResult __mlir_ods_local_type_constraint_GPUOps0(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((true))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be any type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_GPUOps1(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isa<::mlir::gpu::AsyncTokenType>()))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be async token type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_GPUOps2(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isa<::mlir::IndexType>()))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be index, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_GPUOps3(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isa<::mlir::MemRefType>())) && ((true)))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be memref of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_GPUOps6(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isa<::mlir::UnrankedMemRefType>())) && ((true)))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be unranked.memref of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_GPUOps8(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isSignlessInteger(32)))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be 32-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_GPUOps9(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isSignlessInteger(1)))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be 1-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_GPUOps10(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isa<::mlir::gpu::MMAMatrixType>())) && ((type.cast<::mlir::gpu::MMAMatrixType>().getElementType().isF16())))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be gpu.mma_matrix of 16-bit float values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_GPUOps11(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isa<::mlir::gpu::MMAMatrixType>())) && (((type.cast<::mlir::gpu::MMAMatrixType>().getElementType().isF16())) || ((type.cast<::mlir::gpu::MMAMatrixType>().getElementType().isF32()))))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be gpu.mma_matrix of 16-bit float or 32-bit float values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_GPUOps12(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((type.isa<::mlir::gpu::MMAMatrixType>()))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be MMAMatrix type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_GPUOps13(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isF16())) || ((type.isF32())))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be 16-bit float or 32-bit float, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_GPUOps14(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((((type.isa<::mlir::MemRefType>())) && (((type.cast<::mlir::ShapedType>().getElementType().isF16())) || ((type.cast<::mlir::ShapedType>().getElementType().isF32())))) && (((type.cast<::mlir::ShapedType>().hasRank())) && ((type.cast<::mlir::ShapedType>().getRank()
                           == 2))))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be 2D memref of 16-bit float or 32-bit float values, but got " << type;
  }
  return ::mlir::success();
}

} // namespace gpu
} // namespace mlir
namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::AllReduceOp definitions
//===----------------------------------------------------------------------===//

AllReduceOpAdaptor::AllReduceOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

AllReduceOpAdaptor::AllReduceOpAdaptor(AllReduceOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange AllReduceOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AllReduceOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AllReduceOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AllReduceOpAdaptor::value() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr AllReduceOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr AllReduceOpAdaptor::op() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("op").dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::mlir::RegionRange AllReduceOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &AllReduceOpAdaptor::body() {
  return *odsRegions[0];
}

::mlir::LogicalResult AllReduceOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_op = odsAttrs.get("op");
  if (tblgen_op) {
    if (!(((tblgen_op.isa<::mlir::StringAttr>())) && (((tblgen_op.cast<::mlir::StringAttr>().getValue() == "add")) || ((tblgen_op.cast<::mlir::StringAttr>().getValue() == "and")) || ((tblgen_op.cast<::mlir::StringAttr>().getValue() == "max")) || ((tblgen_op.cast<::mlir::StringAttr>().getValue() == "min")) || ((tblgen_op.cast<::mlir::StringAttr>().getValue() == "mul")) || ((tblgen_op.cast<::mlir::StringAttr>().getValue() == "or")) || ((tblgen_op.cast<::mlir::StringAttr>().getValue() == "xor"))))) return emitError(loc, "'gpu.all_reduce' op ""attribute 'op' failed to satisfy constraint: built-in reduction operations supported by gpu.allreduce.");
  }
  }
  return ::mlir::success();
}













std::pair<unsigned, unsigned> AllReduceOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AllReduceOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AllReduceOp::value() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange AllReduceOp::valueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AllReduceOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AllReduceOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &AllReduceOp::body() {
  return (*this)->getRegion(0);
}

::mlir::StringAttr AllReduceOp::opAttr() {
  return (*this)->getAttr(opAttrName()).template dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > AllReduceOp::op() {
  auto attr = opAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

void AllReduceOp::opAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(opAttrName(), attr);
}

::mlir::Attribute AllReduceOp::removeOpAttr() {
  return (*this)->removeAttr(opAttrName());
}

void AllReduceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value value, /*optional*/::mlir::StringAttr op) {
  odsState.addOperands(value);
  if (op) {
  odsState.addAttribute(opAttrName(odsState.name), op);
  }
  (void)odsState.addRegion();
  odsState.addTypes(resultType0);
}

void AllReduceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, /*optional*/::mlir::StringAttr op) {
  odsState.addOperands(value);
  if (op) {
  odsState.addAttribute(opAttrName(odsState.name), op);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AllReduceOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void AllReduceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, /*optional*/::mlir::StringAttr op) {
  odsState.addOperands(value);
  if (op) {
  odsState.addAttribute(opAttrName(odsState.name), op);
  }
  (void)odsState.addRegion();
  odsState.addTypes({value.getType()});

}

void AllReduceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult AllReduceOp::verify() {
  if (::mlir::failed(AllReduceOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>((*this)->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('body') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::verifyAllReduce(*this);
}

} // namespace gpu
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::AllReduceOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::AllocOp definitions
//===----------------------------------------------------------------------===//

AllocOpAdaptor::AllocOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

AllocOpAdaptor::AllocOpAdaptor(AllocOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange AllocOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AllocOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange AllocOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange AllocOpAdaptor::asyncDependencies() {
  return getODSOperands(0);
}

::mlir::ValueRange AllocOpAdaptor::dynamicSizes() {
  return getODSOperands(1);
}

::mlir::ValueRange AllocOpAdaptor::symbolOperands() {
  return getODSOperands(2);
}

::mlir::DictionaryAttr AllocOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult AllocOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 3 elements, but got ") << numElements;
  }
    return ::mlir::success();
}











void AllocOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!llvm::empty(resultGroup0))
    setNameFn(*resultGroup0.begin(), "memref");
  auto resultGroup1 = getODSResults(1);
  if (!llvm::empty(resultGroup1))
    setNameFn(*resultGroup1.begin(), "asyncToken");
}



std::pair<unsigned, unsigned> AllocOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range AllocOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range AllocOp::asyncDependencies() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range AllocOp::dynamicSizes() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range AllocOp::symbolOperands() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange AllocOp::asyncDependenciesMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange AllocOp::dynamicSizesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange AllocOp::symbolOperandsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> AllocOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range AllocOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AllocOp::memref() {
  return *getODSResults(0).begin();
}

::mlir::Value AllocOp::asyncToken() {
  auto results = getODSResults(1);
  return results.empty() ? ::mlir::Value() : *results.begin();
}

void AllocOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type memref, /*optional*/::mlir::Type asyncToken, ::mlir::ValueRange asyncDependencies, ::mlir::ValueRange dynamicSizes, ::mlir::ValueRange symbolOperands) {
  odsState.addOperands(asyncDependencies);
  odsState.addOperands(dynamicSizes);
  odsState.addOperands(symbolOperands);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({static_cast<int32_t>(asyncDependencies.size()), static_cast<int32_t>(dynamicSizes.size()), static_cast<int32_t>(symbolOperands.size())}));
  odsState.addTypes(memref);
  if (asyncToken)
    odsState.addTypes(asyncToken);
}

void AllocOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange asyncDependencies, ::mlir::ValueRange dynamicSizes, ::mlir::ValueRange symbolOperands) {
  odsState.addOperands(asyncDependencies);
  odsState.addOperands(dynamicSizes);
  odsState.addOperands(symbolOperands);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({static_cast<int32_t>(asyncDependencies.size()), static_cast<int32_t>(dynamicSizes.size()), static_cast<int32_t>(symbolOperands.size())}));
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AllocOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AllocOp::verify() {
  if (::mlir::failed(AllocOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps3(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSResults(1);
    if (valueGroup1.size() > 1)
      return emitOpError("result group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup1.size();
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}



::mlir::ParseResult AllocOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::Type, 1> asyncTokenTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> asyncDependenciesOperands;
  ::llvm::SMLoc asyncDependenciesOperandsLoc;
  (void)asyncDependenciesOperandsLoc;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> dynamicSizesOperands;
  ::llvm::SMLoc dynamicSizesOperandsLoc;
  (void)dynamicSizesOperandsLoc;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> symbolOperandsOperands;
  ::llvm::SMLoc symbolOperandsOperandsLoc;
  (void)symbolOperandsOperandsLoc;
  ::mlir::Type memrefRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> memrefTypes(memrefRawTypes);
  {
    ::mlir::Type asyncTokenType;
    asyncDependenciesOperandsLoc = parser.getCurrentLocation();
    if (parseAsyncDependencies(parser, asyncTokenType, asyncDependenciesOperands))
      return ::mlir::failure();
    if (asyncTokenType)
      asyncTokenTypes.push_back(asyncTokenType);
  }
  if (parser.parseLParen())
    return ::mlir::failure();

  dynamicSizesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(dynamicSizesOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalLSquare())) {

  symbolOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(symbolOperandsOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(memrefRawTypes[0]))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = mlir::gpu::AsyncTokenType::get(parser.getBuilder().getContext());
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getIndexType();
  result.addTypes(memrefTypes);
  result.addTypes(asyncTokenTypes);
  if (parser.resolveOperands(asyncDependenciesOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(dynamicSizesOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(symbolOperandsOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getI32VectorAttr({static_cast<int32_t>(asyncDependenciesOperands.size()), static_cast<int32_t>(dynamicSizesOperands.size()), static_cast<int32_t>(symbolOperandsOperands.size())}));
  return ::mlir::success();
}

void AllocOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  printAsyncDependencies(p, *this, (asyncToken() ? asyncToken().getType() : Type()), asyncDependencies());
  p << ' ';
  p << "(";
  p << dynamicSizes();
  p << ")";
  if (!symbolOperands().empty()) {
  p << "[";
  p << symbolOperands();
  p << "]";
  }
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", });
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(memref().getType());
}

void AllocOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSResults(0))
    effects.emplace_back(::mlir::MemoryEffects::Allocate::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace gpu
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::AllocOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::BarrierOp definitions
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

} // namespace gpu
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::BarrierOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::BlockDimOp definitions
//===----------------------------------------------------------------------===//

BlockDimOpAdaptor::BlockDimOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

BlockDimOpAdaptor::BlockDimOpAdaptor(BlockDimOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange BlockDimOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BlockDimOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BlockDimOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr BlockDimOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr BlockDimOpAdaptor::dimension() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("dimension").cast<::mlir::StringAttr>();
  return attr;
}

::mlir::LogicalResult BlockDimOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_dimension = odsAttrs.get("dimension");
  if (!tblgen_dimension) return emitError(loc, "'gpu.block_dim' op ""requires attribute 'dimension'");
    if (!((tblgen_dimension.isa<::mlir::StringAttr>()))) return emitError(loc, "'gpu.block_dim' op ""attribute 'dimension' failed to satisfy constraint: string attribute");
  }
  return ::mlir::success();
}













std::pair<unsigned, unsigned> BlockDimOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BlockDimOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> BlockDimOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BlockDimOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::StringAttr BlockDimOp::dimensionAttr() {
  return (*this)->getAttr(dimensionAttrName()).template cast<::mlir::StringAttr>();
}

::llvm::StringRef BlockDimOp::dimension() {
  auto attr = dimensionAttr();
  return attr.getValue();
}

void BlockDimOp::dimensionAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(dimensionAttrName(), attr);
}

void BlockDimOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::StringAttr dimension) {
  odsState.addAttribute(dimensionAttrName(odsState.name), dimension);
  odsState.addTypes(resultType0);
}

void BlockDimOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr dimension) {
  odsState.addAttribute(dimensionAttrName(odsState.name), dimension);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BlockDimOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::llvm::StringRef dimension) {
  odsState.addAttribute(dimensionAttrName(odsState.name), odsBuilder.getStringAttr(dimension));
  odsState.addTypes(resultType0);
}

void BlockDimOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef dimension) {
  odsState.addAttribute(dimensionAttrName(odsState.name), odsBuilder.getStringAttr(dimension));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BlockDimOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BlockDimOp::verify() {
  if (::mlir::failed(BlockDimOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::verifyIndexOp(*this);
}

void BlockDimOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace gpu
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::BlockDimOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::BlockIdOp definitions
//===----------------------------------------------------------------------===//

BlockIdOpAdaptor::BlockIdOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

BlockIdOpAdaptor::BlockIdOpAdaptor(BlockIdOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange BlockIdOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BlockIdOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BlockIdOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr BlockIdOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr BlockIdOpAdaptor::dimension() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("dimension").cast<::mlir::StringAttr>();
  return attr;
}

::mlir::LogicalResult BlockIdOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_dimension = odsAttrs.get("dimension");
  if (!tblgen_dimension) return emitError(loc, "'gpu.block_id' op ""requires attribute 'dimension'");
    if (!((tblgen_dimension.isa<::mlir::StringAttr>()))) return emitError(loc, "'gpu.block_id' op ""attribute 'dimension' failed to satisfy constraint: string attribute");
  }
  return ::mlir::success();
}













std::pair<unsigned, unsigned> BlockIdOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BlockIdOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> BlockIdOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BlockIdOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::StringAttr BlockIdOp::dimensionAttr() {
  return (*this)->getAttr(dimensionAttrName()).template cast<::mlir::StringAttr>();
}

::llvm::StringRef BlockIdOp::dimension() {
  auto attr = dimensionAttr();
  return attr.getValue();
}

void BlockIdOp::dimensionAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(dimensionAttrName(), attr);
}

void BlockIdOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::StringAttr dimension) {
  odsState.addAttribute(dimensionAttrName(odsState.name), dimension);
  odsState.addTypes(resultType0);
}

void BlockIdOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr dimension) {
  odsState.addAttribute(dimensionAttrName(odsState.name), dimension);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BlockIdOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::llvm::StringRef dimension) {
  odsState.addAttribute(dimensionAttrName(odsState.name), odsBuilder.getStringAttr(dimension));
  odsState.addTypes(resultType0);
}

void BlockIdOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef dimension) {
  odsState.addAttribute(dimensionAttrName(odsState.name), odsBuilder.getStringAttr(dimension));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BlockIdOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BlockIdOp::verify() {
  if (::mlir::failed(BlockIdOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::verifyIndexOp(*this);
}

void BlockIdOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace gpu
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::BlockIdOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::DeallocOp definitions
//===----------------------------------------------------------------------===//

DeallocOpAdaptor::DeallocOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

DeallocOpAdaptor::DeallocOpAdaptor(DeallocOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange DeallocOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> DeallocOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false};
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

::mlir::ValueRange DeallocOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange DeallocOpAdaptor::asyncDependencies() {
  return getODSOperands(0);
}

::mlir::Value DeallocOpAdaptor::memref() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr DeallocOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult DeallocOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> DeallocOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false};
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

::mlir::Operation::operand_range DeallocOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range DeallocOp::asyncDependencies() {
  return getODSOperands(0);
}

::mlir::Value DeallocOp::memref() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange DeallocOp::asyncDependenciesMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange DeallocOp::memrefMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> DeallocOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range DeallocOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value DeallocOp::asyncToken() {
  auto results = getODSResults(0);
  return results.empty() ? ::mlir::Value() : *results.begin();
}

void DeallocOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type asyncToken, ::mlir::ValueRange asyncDependencies, ::mlir::Value memref) {
  odsState.addOperands(asyncDependencies);
  odsState.addOperands(memref);
  if (asyncToken)
    odsState.addTypes(asyncToken);
}

void DeallocOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange asyncDependencies, ::mlir::Value memref) {
  odsState.addOperands(asyncDependencies);
  odsState.addOperands(memref);
  odsState.addTypes(resultTypes);
}

void DeallocOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult DeallocOp::verify() {
  if (::mlir::failed(DeallocOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    if (valueGroup0.size() > 1)
      return emitOpError("result group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup0.size();
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult DeallocOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::Type, 1> asyncTokenTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> asyncDependenciesOperands;
  ::llvm::SMLoc asyncDependenciesOperandsLoc;
  (void)asyncDependenciesOperandsLoc;
  ::mlir::OpAsmParser::OperandType memrefRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> memrefOperands(memrefRawOperands);  ::llvm::SMLoc memrefOperandsLoc;
  (void)memrefOperandsLoc;
  ::mlir::Type memrefRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> memrefTypes(memrefRawTypes);
  {
    ::mlir::Type asyncTokenType;
    asyncDependenciesOperandsLoc = parser.getCurrentLocation();
    if (parseAsyncDependencies(parser, asyncTokenType, asyncDependenciesOperands))
      return ::mlir::failure();
    if (asyncTokenType)
      asyncTokenTypes.push_back(asyncTokenType);
  }

  memrefOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(memrefRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(memrefRawTypes[0]))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = mlir::gpu::AsyncTokenType::get(parser.getBuilder().getContext());
  result.addTypes(asyncTokenTypes);
  if (parser.resolveOperands(asyncDependenciesOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(memrefOperands, memrefTypes, memrefOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void DeallocOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  printAsyncDependencies(p, *this, (asyncToken() ? asyncToken().getType() : Type()), asyncDependencies());
  p << ' ';
  p << memref();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(memref().getType());
}

void DeallocOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(1))
    effects.emplace_back(::mlir::MemoryEffects::Free::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace gpu
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::DeallocOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::GPUFuncOp definitions
//===----------------------------------------------------------------------===//

GPUFuncOpAdaptor::GPUFuncOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

GPUFuncOpAdaptor::GPUFuncOpAdaptor(GPUFuncOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange GPUFuncOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GPUFuncOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GPUFuncOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr GPUFuncOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange GPUFuncOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &GPUFuncOpAdaptor::body() {
  return *odsRegions[0];
}

::mlir::LogicalResult GPUFuncOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> GPUFuncOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GPUFuncOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> GPUFuncOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GPUFuncOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &GPUFuncOp::body() {
  return (*this)->getRegion(0);
}



::mlir::ParseResult GPUFuncOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return parseGPUFuncOp(parser, result);
}

void GPUFuncOp::print(::mlir::OpAsmPrinter &p) {
  printGPUFuncOp(p, *this);
}

::mlir::LogicalResult GPUFuncOp::verify() {
  if (::mlir::failed(GPUFuncOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
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
        return emitOpError("region #") << index << " ('body') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::mlir::success();
}

} // namespace gpu
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::GPUFuncOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::GPUModuleOp definitions
//===----------------------------------------------------------------------===//

GPUModuleOpAdaptor::GPUModuleOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

GPUModuleOpAdaptor::GPUModuleOpAdaptor(GPUModuleOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange GPUModuleOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GPUModuleOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GPUModuleOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr GPUModuleOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange GPUModuleOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &GPUModuleOpAdaptor::body() {
  return *odsRegions[0];
}

::mlir::LogicalResult GPUModuleOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> GPUModuleOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GPUModuleOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> GPUModuleOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GPUModuleOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &GPUModuleOp::body() {
  return (*this)->getRegion(0);
}



::mlir::ParseResult GPUModuleOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseGPUModuleOp(parser, result);
}

void GPUModuleOp::print(::mlir::OpAsmPrinter &p) {
  return ::print(p, *this);
}

::mlir::LogicalResult GPUModuleOp::verify() {
  if (::mlir::failed(GPUModuleOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
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
      if (!((::llvm::hasNItems(region, 1)))) {
        return emitOpError("region #") << index << " ('body') failed to verify constraint: region with 1 blocks";
      }
      ++index;
    }
  }
  return ::mlir::success();
}

} // namespace gpu
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::GPUModuleOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::GridDimOp definitions
//===----------------------------------------------------------------------===//

GridDimOpAdaptor::GridDimOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

GridDimOpAdaptor::GridDimOpAdaptor(GridDimOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange GridDimOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GridDimOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GridDimOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr GridDimOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr GridDimOpAdaptor::dimension() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("dimension").cast<::mlir::StringAttr>();
  return attr;
}

::mlir::LogicalResult GridDimOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_dimension = odsAttrs.get("dimension");
  if (!tblgen_dimension) return emitError(loc, "'gpu.grid_dim' op ""requires attribute 'dimension'");
    if (!((tblgen_dimension.isa<::mlir::StringAttr>()))) return emitError(loc, "'gpu.grid_dim' op ""attribute 'dimension' failed to satisfy constraint: string attribute");
  }
  return ::mlir::success();
}













std::pair<unsigned, unsigned> GridDimOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GridDimOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> GridDimOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GridDimOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::StringAttr GridDimOp::dimensionAttr() {
  return (*this)->getAttr(dimensionAttrName()).template cast<::mlir::StringAttr>();
}

::llvm::StringRef GridDimOp::dimension() {
  auto attr = dimensionAttr();
  return attr.getValue();
}

void GridDimOp::dimensionAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(dimensionAttrName(), attr);
}

void GridDimOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::StringAttr dimension) {
  odsState.addAttribute(dimensionAttrName(odsState.name), dimension);
  odsState.addTypes(resultType0);
}

void GridDimOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr dimension) {
  odsState.addAttribute(dimensionAttrName(odsState.name), dimension);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GridDimOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::llvm::StringRef dimension) {
  odsState.addAttribute(dimensionAttrName(odsState.name), odsBuilder.getStringAttr(dimension));
  odsState.addTypes(resultType0);
}

void GridDimOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef dimension) {
  odsState.addAttribute(dimensionAttrName(odsState.name), odsBuilder.getStringAttr(dimension));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GridDimOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GridDimOp::verify() {
  if (::mlir::failed(GridDimOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::verifyIndexOp(*this);
}

void GridDimOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace gpu
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::GridDimOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::HostRegisterOp definitions
//===----------------------------------------------------------------------===//

HostRegisterOpAdaptor::HostRegisterOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

HostRegisterOpAdaptor::HostRegisterOpAdaptor(HostRegisterOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange HostRegisterOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> HostRegisterOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange HostRegisterOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value HostRegisterOpAdaptor::value() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr HostRegisterOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult HostRegisterOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> HostRegisterOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range HostRegisterOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value HostRegisterOp::value() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange HostRegisterOp::valueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> HostRegisterOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range HostRegisterOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void HostRegisterOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value) {
  odsState.addOperands(value);
}

void HostRegisterOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value) {
  odsState.addOperands(value);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void HostRegisterOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult HostRegisterOp::verify() {
  if (::mlir::failed(HostRegisterOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps6(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return success();
}

::mlir::ParseResult HostRegisterOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
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

void HostRegisterOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << value();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(value().getType());
}

} // namespace gpu
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::HostRegisterOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::LaunchFuncOp definitions
//===----------------------------------------------------------------------===//

LaunchFuncOpAdaptor::LaunchFuncOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

LaunchFuncOpAdaptor::LaunchFuncOpAdaptor(LaunchFuncOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange LaunchFuncOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> LaunchFuncOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange LaunchFuncOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange LaunchFuncOpAdaptor::asyncDependencies() {
  return getODSOperands(0);
}

::mlir::Value LaunchFuncOpAdaptor::gridSizeX() {
  return *getODSOperands(1).begin();
}

::mlir::Value LaunchFuncOpAdaptor::gridSizeY() {
  return *getODSOperands(2).begin();
}

::mlir::Value LaunchFuncOpAdaptor::gridSizeZ() {
  return *getODSOperands(3).begin();
}

::mlir::Value LaunchFuncOpAdaptor::blockSizeX() {
  return *getODSOperands(4).begin();
}

::mlir::Value LaunchFuncOpAdaptor::blockSizeY() {
  return *getODSOperands(5).begin();
}

::mlir::Value LaunchFuncOpAdaptor::blockSizeZ() {
  return *getODSOperands(6).begin();
}

::mlir::ValueRange LaunchFuncOpAdaptor::operands() {
  return getODSOperands(7);
}

::mlir::DictionaryAttr LaunchFuncOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::SymbolRefAttr LaunchFuncOpAdaptor::kernel() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::SymbolRefAttr attr = odsAttrs.get("kernel").cast<::mlir::SymbolRefAttr>();
  return attr;
}

::mlir::LogicalResult LaunchFuncOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 8)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 8 elements, but got ") << numElements;
  }
    {
  auto tblgen_kernel = odsAttrs.get("kernel");
  if (!tblgen_kernel) return emitError(loc, "'gpu.launch_func' op ""requires attribute 'kernel'");
    if (!((tblgen_kernel.isa<::mlir::SymbolRefAttr>()))) return emitError(loc, "'gpu.launch_func' op ""attribute 'kernel' failed to satisfy constraint: symbol reference attribute");
  }
  return ::mlir::success();
}

















std::pair<unsigned, unsigned> LaunchFuncOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range LaunchFuncOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range LaunchFuncOp::asyncDependencies() {
  return getODSOperands(0);
}

::mlir::Value LaunchFuncOp::gridSizeX() {
  return *getODSOperands(1).begin();
}

::mlir::Value LaunchFuncOp::gridSizeY() {
  return *getODSOperands(2).begin();
}

::mlir::Value LaunchFuncOp::gridSizeZ() {
  return *getODSOperands(3).begin();
}

::mlir::Value LaunchFuncOp::blockSizeX() {
  return *getODSOperands(4).begin();
}

::mlir::Value LaunchFuncOp::blockSizeY() {
  return *getODSOperands(5).begin();
}

::mlir::Value LaunchFuncOp::blockSizeZ() {
  return *getODSOperands(6).begin();
}

::mlir::Operation::operand_range LaunchFuncOp::operands() {
  return getODSOperands(7);
}

::mlir::MutableOperandRange LaunchFuncOp::asyncDependenciesMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange LaunchFuncOp::gridSizeXMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange LaunchFuncOp::gridSizeYMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange LaunchFuncOp::gridSizeZMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange LaunchFuncOp::blockSizeXMutable() {
  auto range = getODSOperandIndexAndLength(4);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(4u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange LaunchFuncOp::blockSizeYMutable() {
  auto range = getODSOperandIndexAndLength(5);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(5u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange LaunchFuncOp::blockSizeZMutable() {
  auto range = getODSOperandIndexAndLength(6);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(6u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange LaunchFuncOp::operandsMutable() {
  auto range = getODSOperandIndexAndLength(7);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(7u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> LaunchFuncOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range LaunchFuncOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LaunchFuncOp::asyncToken() {
  auto results = getODSResults(0);
  return results.empty() ? ::mlir::Value() : *results.begin();
}

::mlir::SymbolRefAttr LaunchFuncOp::kernelAttr() {
  return (*this)->getAttr(kernelAttrName()).template cast<::mlir::SymbolRefAttr>();
}

::mlir::SymbolRefAttr LaunchFuncOp::kernel() {
  auto attr = kernelAttr();
  return attr;
}

void LaunchFuncOp::kernelAttr(::mlir::SymbolRefAttr attr) {
  (*this)->setAttr(kernelAttrName(), attr);
}



::mlir::LogicalResult LaunchFuncOp::verify() {
  if (::mlir::failed(LaunchFuncOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup3 = getODSOperands(3);
    for (::mlir::Value v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup4 = getODSOperands(4);
    for (::mlir::Value v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup5 = getODSOperands(5);
    for (::mlir::Value v : valueGroup5) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup6 = getODSOperands(6);
    for (::mlir::Value v : valueGroup6) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup7 = getODSOperands(7);
    for (::mlir::Value v : valueGroup7) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    if (valueGroup0.size() > 1)
      return emitOpError("result group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup0.size();
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::verify(*this);
}

::mlir::ParseResult LaunchFuncOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::Type, 1> asyncTokenTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> asyncDependenciesOperands;
  ::llvm::SMLoc asyncDependenciesOperandsLoc;
  (void)asyncDependenciesOperandsLoc;
  ::mlir::SymbolRefAttr kernelAttr;
  ::mlir::OpAsmParser::OperandType gridSizeXRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> gridSizeXOperands(gridSizeXRawOperands);  ::llvm::SMLoc gridSizeXOperandsLoc;
  (void)gridSizeXOperandsLoc;
  ::mlir::OpAsmParser::OperandType gridSizeYRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> gridSizeYOperands(gridSizeYRawOperands);  ::llvm::SMLoc gridSizeYOperandsLoc;
  (void)gridSizeYOperandsLoc;
  ::mlir::OpAsmParser::OperandType gridSizeZRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> gridSizeZOperands(gridSizeZRawOperands);  ::llvm::SMLoc gridSizeZOperandsLoc;
  (void)gridSizeZOperandsLoc;
  ::mlir::OpAsmParser::OperandType blockSizeXRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> blockSizeXOperands(blockSizeXRawOperands);  ::llvm::SMLoc blockSizeXOperandsLoc;
  (void)blockSizeXOperandsLoc;
  ::mlir::OpAsmParser::OperandType blockSizeYRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> blockSizeYOperands(blockSizeYRawOperands);  ::llvm::SMLoc blockSizeYOperandsLoc;
  (void)blockSizeYOperandsLoc;
  ::mlir::OpAsmParser::OperandType blockSizeZRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> blockSizeZOperands(blockSizeZRawOperands);  ::llvm::SMLoc blockSizeZOperandsLoc;
  (void)blockSizeZOperandsLoc;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> operandsOperands;
  ::llvm::SMLoc operandsOperandsLoc;
  (void)operandsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> operandsTypes;
  {
    ::mlir::Type asyncTokenType;
    asyncDependenciesOperandsLoc = parser.getCurrentLocation();
    if (parseAsyncDependencies(parser, asyncTokenType, asyncDependenciesOperands))
      return ::mlir::failure();
    if (asyncTokenType)
      asyncTokenTypes.push_back(asyncTokenType);
  }

  if (parser.parseAttribute(kernelAttr, parser.getBuilder().getType<::mlir::NoneType>(), "kernel", result.attributes))
    return ::mlir::failure();
  if (parser.parseKeyword("blocks"))
    return ::mlir::failure();
  if (parser.parseKeyword("in"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  gridSizeXOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(gridSizeXRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  gridSizeYOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(gridSizeYRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  gridSizeZOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(gridSizeZRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseKeyword("threads"))
    return ::mlir::failure();
  if (parser.parseKeyword("in"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  blockSizeXOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(blockSizeXRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  blockSizeYOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(blockSizeYRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  blockSizeZOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(blockSizeZRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  {
    operandsOperandsLoc = parser.getCurrentLocation();
    if (parseLaunchFuncOperands(parser, operandsOperands, operandsTypes))
      return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = mlir::gpu::AsyncTokenType::get(parser.getBuilder().getContext());
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getIndexType();
  result.addTypes(asyncTokenTypes);
  if (parser.resolveOperands(asyncDependenciesOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(gridSizeXOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(gridSizeYOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(gridSizeZOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(blockSizeXOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(blockSizeYOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(blockSizeZOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(operandsOperands, operandsTypes, operandsOperandsLoc, result.operands))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getI32VectorAttr({static_cast<int32_t>(asyncDependenciesOperands.size()), 1, 1, 1, 1, 1, 1, static_cast<int32_t>(operandsOperands.size())}));
  return ::mlir::success();
}

void LaunchFuncOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  printAsyncDependencies(p, *this, (asyncToken() ? asyncToken().getType() : Type()), asyncDependencies());
  p << ' ';
  p.printAttributeWithoutType(kernelAttr());
  p << ' ' << "blocks";
  p << ' ' << "in";
  p << ' ';
  p << "(";
  p << gridSizeX();
  p << ",";
  p << ' ';
  p << gridSizeY();
  p << ",";
  p << ' ';
  p << gridSizeZ();
  p << ")";
  p << ' ' << "threads";
  p << ' ' << "in";
  p << ' ';
  p << "(";
  p << blockSizeX();
  p << ",";
  p << ' ';
  p << blockSizeY();
  p << ",";
  p << ' ';
  p << blockSizeZ();
  p << ")";
  p << ' ';
  printLaunchFuncOperands(p, *this, operands(), operands().getTypes());
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", "kernel"});
}

} // namespace gpu
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::LaunchFuncOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::LaunchOp definitions
//===----------------------------------------------------------------------===//

LaunchOpAdaptor::LaunchOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

LaunchOpAdaptor::LaunchOpAdaptor(LaunchOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange LaunchOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> LaunchOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange LaunchOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LaunchOpAdaptor::gridSizeX() {
  return *getODSOperands(0).begin();
}

::mlir::Value LaunchOpAdaptor::gridSizeY() {
  return *getODSOperands(1).begin();
}

::mlir::Value LaunchOpAdaptor::gridSizeZ() {
  return *getODSOperands(2).begin();
}

::mlir::Value LaunchOpAdaptor::blockSizeX() {
  return *getODSOperands(3).begin();
}

::mlir::Value LaunchOpAdaptor::blockSizeY() {
  return *getODSOperands(4).begin();
}

::mlir::Value LaunchOpAdaptor::blockSizeZ() {
  return *getODSOperands(5).begin();
}

::mlir::DictionaryAttr LaunchOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange LaunchOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &LaunchOpAdaptor::body() {
  return *odsRegions[0];
}

::mlir::LogicalResult LaunchOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> LaunchOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range LaunchOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LaunchOp::gridSizeX() {
  return *getODSOperands(0).begin();
}

::mlir::Value LaunchOp::gridSizeY() {
  return *getODSOperands(1).begin();
}

::mlir::Value LaunchOp::gridSizeZ() {
  return *getODSOperands(2).begin();
}

::mlir::Value LaunchOp::blockSizeX() {
  return *getODSOperands(3).begin();
}

::mlir::Value LaunchOp::blockSizeY() {
  return *getODSOperands(4).begin();
}

::mlir::Value LaunchOp::blockSizeZ() {
  return *getODSOperands(5).begin();
}

::mlir::MutableOperandRange LaunchOp::gridSizeXMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange LaunchOp::gridSizeYMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange LaunchOp::gridSizeZMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange LaunchOp::blockSizeXMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange LaunchOp::blockSizeYMutable() {
  auto range = getODSOperandIndexAndLength(4);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange LaunchOp::blockSizeZMutable() {
  auto range = getODSOperandIndexAndLength(5);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> LaunchOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range LaunchOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &LaunchOp::body() {
  return (*this)->getRegion(0);
}



::mlir::ParseResult LaunchOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return parseLaunchOp(parser, result);
}

void LaunchOp::print(::mlir::OpAsmPrinter &p) {
  printLaunchOp(p, *this);
}

::mlir::LogicalResult LaunchOp::verify() {
  if (::mlir::failed(LaunchOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup3 = getODSOperands(3);
    for (::mlir::Value v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup4 = getODSOperands(4);
    for (::mlir::Value v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup5 = getODSOperands(5);
    for (::mlir::Value v : valueGroup5) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps2(getOperation(), v.getType(), "operand", index)))
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
        return emitOpError("region #") << index << " ('body') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::verify(*this);
}



} // namespace gpu
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::LaunchOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::MemcpyOp definitions
//===----------------------------------------------------------------------===//

MemcpyOpAdaptor::MemcpyOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

MemcpyOpAdaptor::MemcpyOpAdaptor(MemcpyOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange MemcpyOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MemcpyOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false, false};
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

::mlir::ValueRange MemcpyOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange MemcpyOpAdaptor::asyncDependencies() {
  return getODSOperands(0);
}

::mlir::Value MemcpyOpAdaptor::dst() {
  return *getODSOperands(1).begin();
}

::mlir::Value MemcpyOpAdaptor::src() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr MemcpyOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult MemcpyOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> MemcpyOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false, false};
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

::mlir::Operation::operand_range MemcpyOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range MemcpyOp::asyncDependencies() {
  return getODSOperands(0);
}

::mlir::Value MemcpyOp::dst() {
  return *getODSOperands(1).begin();
}

::mlir::Value MemcpyOp::src() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange MemcpyOp::asyncDependenciesMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MemcpyOp::dstMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MemcpyOp::srcMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> MemcpyOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range MemcpyOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MemcpyOp::asyncToken() {
  auto results = getODSResults(0);
  return results.empty() ? ::mlir::Value() : *results.begin();
}

void MemcpyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type asyncToken, ::mlir::ValueRange asyncDependencies, ::mlir::Value dst, ::mlir::Value src) {
  odsState.addOperands(asyncDependencies);
  odsState.addOperands(dst);
  odsState.addOperands(src);
  if (asyncToken)
    odsState.addTypes(asyncToken);
}

void MemcpyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange asyncDependencies, ::mlir::Value dst, ::mlir::Value src) {
  odsState.addOperands(asyncDependencies);
  odsState.addOperands(dst);
  odsState.addOperands(src);
  odsState.addTypes(resultTypes);
}

void MemcpyOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MemcpyOp::verify() {
  if (::mlir::failed(MemcpyOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    if (valueGroup0.size() > 1)
      return emitOpError("result group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup0.size();
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::verify(*this);
}



::mlir::ParseResult MemcpyOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::Type, 1> asyncTokenTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> asyncDependenciesOperands;
  ::llvm::SMLoc asyncDependenciesOperandsLoc;
  (void)asyncDependenciesOperandsLoc;
  ::mlir::OpAsmParser::OperandType dstRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> dstOperands(dstRawOperands);  ::llvm::SMLoc dstOperandsLoc;
  (void)dstOperandsLoc;
  ::mlir::OpAsmParser::OperandType srcRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> srcOperands(srcRawOperands);  ::llvm::SMLoc srcOperandsLoc;
  (void)srcOperandsLoc;
  ::mlir::Type dstRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> dstTypes(dstRawTypes);
  ::mlir::Type srcRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> srcTypes(srcRawTypes);
  {
    ::mlir::Type asyncTokenType;
    asyncDependenciesOperandsLoc = parser.getCurrentLocation();
    if (parseAsyncDependencies(parser, asyncTokenType, asyncDependenciesOperands))
      return ::mlir::failure();
    if (asyncTokenType)
      asyncTokenTypes.push_back(asyncTokenType);
  }

  dstOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(dstRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  srcOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(srcRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(dstRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(srcRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = mlir::gpu::AsyncTokenType::get(parser.getBuilder().getContext());
  result.addTypes(asyncTokenTypes);
  if (parser.resolveOperands(asyncDependenciesOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(dstOperands, dstTypes, dstOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(srcOperands, srcTypes, srcOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void MemcpyOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  printAsyncDependencies(p, *this, (asyncToken() ? asyncToken().getType() : Type()), asyncDependencies());
  p << ' ';
  p << dst();
  p << ",";
  p << ' ';
  p << src();
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(dst().getType());
  p << ",";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(src().getType());
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

void MemcpyOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(1))
    effects.emplace_back(::mlir::MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(2))
    effects.emplace_back(::mlir::MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace gpu
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::MemcpyOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::MemsetOp definitions
//===----------------------------------------------------------------------===//

MemsetOpAdaptor::MemsetOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

MemsetOpAdaptor::MemsetOpAdaptor(MemsetOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange MemsetOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MemsetOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false, false};
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

::mlir::ValueRange MemsetOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange MemsetOpAdaptor::asyncDependencies() {
  return getODSOperands(0);
}

::mlir::Value MemsetOpAdaptor::dst() {
  return *getODSOperands(1).begin();
}

::mlir::Value MemsetOpAdaptor::value() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr MemsetOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult MemsetOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> MemsetOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false, false};
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

::mlir::Operation::operand_range MemsetOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range MemsetOp::asyncDependencies() {
  return getODSOperands(0);
}

::mlir::Value MemsetOp::dst() {
  return *getODSOperands(1).begin();
}

::mlir::Value MemsetOp::value() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange MemsetOp::asyncDependenciesMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MemsetOp::dstMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MemsetOp::valueMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> MemsetOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range MemsetOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MemsetOp::asyncToken() {
  auto results = getODSResults(0);
  return results.empty() ? ::mlir::Value() : *results.begin();
}

void MemsetOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type asyncToken, ::mlir::ValueRange asyncDependencies, ::mlir::Value dst, ::mlir::Value value) {
  odsState.addOperands(asyncDependencies);
  odsState.addOperands(dst);
  odsState.addOperands(value);
  if (asyncToken)
    odsState.addTypes(asyncToken);
}

void MemsetOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange asyncDependencies, ::mlir::Value dst, ::mlir::Value value) {
  odsState.addOperands(asyncDependencies);
  odsState.addOperands(dst);
  odsState.addOperands(value);
  odsState.addTypes(resultTypes);
}

void MemsetOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MemsetOp::verify() {
  if (::mlir::failed(MemsetOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    if (valueGroup0.size() > 1)
      return emitOpError("result group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup0.size();
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({getElementTypeOrSelf((*this->getODSOperands(1).begin())), getElementTypeOrSelf((*this->getODSOperands(2).begin()))})))))
    return emitOpError("failed to verify that all of {dst, value} have same element type");
  return success();
}



::mlir::ParseResult MemsetOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::Type, 1> asyncTokenTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> asyncDependenciesOperands;
  ::llvm::SMLoc asyncDependenciesOperandsLoc;
  (void)asyncDependenciesOperandsLoc;
  ::mlir::OpAsmParser::OperandType dstRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> dstOperands(dstRawOperands);  ::llvm::SMLoc dstOperandsLoc;
  (void)dstOperandsLoc;
  ::mlir::OpAsmParser::OperandType valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::Type dstRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> dstTypes(dstRawTypes);
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);
  {
    ::mlir::Type asyncTokenType;
    asyncDependenciesOperandsLoc = parser.getCurrentLocation();
    if (parseAsyncDependencies(parser, asyncTokenType, asyncDependenciesOperands))
      return ::mlir::failure();
    if (asyncTokenType)
      asyncTokenTypes.push_back(asyncTokenType);
  }

  dstOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(dstRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(dstRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(valueRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = mlir::gpu::AsyncTokenType::get(parser.getBuilder().getContext());
  result.addTypes(asyncTokenTypes);
  if (parser.resolveOperands(asyncDependenciesOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(dstOperands, dstTypes, dstOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void MemsetOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  printAsyncDependencies(p, *this, (asyncToken() ? asyncToken().getType() : Type()), asyncDependencies());
  p << ' ';
  p << dst();
  p << ",";
  p << ' ';
  p << value();
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(dst().getType());
  p << ",";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(value().getType());
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

void MemsetOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(1))
    effects.emplace_back(::mlir::MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace gpu
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::MemsetOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::ModuleEndOp definitions
//===----------------------------------------------------------------------===//

ModuleEndOpAdaptor::ModuleEndOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ModuleEndOpAdaptor::ModuleEndOpAdaptor(ModuleEndOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ModuleEndOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ModuleEndOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ModuleEndOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr ModuleEndOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ModuleEndOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> ModuleEndOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ModuleEndOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> ModuleEndOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ModuleEndOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void ModuleEndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {

}

void ModuleEndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ModuleEndOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ModuleEndOp::verify() {
  if (::mlir::failed(ModuleEndOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::ParseResult ModuleEndOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void ModuleEndOp::print(::mlir::OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace gpu
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::ModuleEndOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::NumSubgroupsOp definitions
//===----------------------------------------------------------------------===//

NumSubgroupsOpAdaptor::NumSubgroupsOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

NumSubgroupsOpAdaptor::NumSubgroupsOpAdaptor(NumSubgroupsOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange NumSubgroupsOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> NumSubgroupsOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange NumSubgroupsOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr NumSubgroupsOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult NumSubgroupsOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> NumSubgroupsOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range NumSubgroupsOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> NumSubgroupsOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range NumSubgroupsOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value NumSubgroupsOp::result() {
  return *getODSResults(0).begin();
}

void NumSubgroupsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result) {
  odsState.addTypes(result);
}

void NumSubgroupsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void NumSubgroupsOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult NumSubgroupsOp::verify() {
  if (::mlir::failed(NumSubgroupsOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return success();
}

::mlir::ParseResult NumSubgroupsOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resultRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resultTypes);
  return ::mlir::success();
}

void NumSubgroupsOp::print(::mlir::OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(result().getType());
}

void NumSubgroupsOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace gpu
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::NumSubgroupsOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::ReturnOp definitions
//===----------------------------------------------------------------------===//

ReturnOpAdaptor::ReturnOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ReturnOpAdaptor::ReturnOpAdaptor(ReturnOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ReturnOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ReturnOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange ReturnOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ReturnOpAdaptor::operands() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr ReturnOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ReturnOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> ReturnOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range ReturnOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ReturnOp::operands() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange ReturnOp::operandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ReturnOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReturnOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void ReturnOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
 // empty
}

void ReturnOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
}

void ReturnOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReturnOp::verify() {
  if (::mlir::failed(ReturnOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::verify(*this);
}

::mlir::ParseResult ReturnOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> operandsOperands;
  ::llvm::SMLoc operandsOperandsLoc;
  (void)operandsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> operandsTypes;
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();

  operandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(operandsOperands))
    return ::mlir::failure();
  if (!operandsOperands.empty()) {
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(operandsTypes))
    return ::mlir::failure();
  }
  if (parser.resolveOperands(operandsOperands, operandsTypes, operandsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ReturnOp::print(::mlir::OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  if (!operands().empty()) {
  p << ' ';
  p << operands();
  p << ' ' << ":";
  p << ' ';
  p << operands().getTypes();
  }
}

void ReturnOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace gpu
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::ReturnOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::ShuffleOp definitions
//===----------------------------------------------------------------------===//

ShuffleOpAdaptor::ShuffleOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ShuffleOpAdaptor::ShuffleOpAdaptor(ShuffleOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ShuffleOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ShuffleOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ShuffleOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ShuffleOpAdaptor::value() {
  return *getODSOperands(0).begin();
}

::mlir::Value ShuffleOpAdaptor::offset() {
  return *getODSOperands(1).begin();
}

::mlir::Value ShuffleOpAdaptor::width() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr ShuffleOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr ShuffleOpAdaptor::mode() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("mode").cast<::mlir::StringAttr>();
  return attr;
}

::mlir::LogicalResult ShuffleOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_mode = odsAttrs.get("mode");
  if (!tblgen_mode) return emitError(loc, "'gpu.shuffle' op ""requires attribute 'mode'");
    if (!(((tblgen_mode.isa<::mlir::StringAttr>())) && ((tblgen_mode.cast<::mlir::StringAttr>().getValue() == "xor")))) return emitError(loc, "'gpu.shuffle' op ""attribute 'mode' failed to satisfy constraint: Indexing modes supported by gpu.shuffle.");
  }
  return ::mlir::success();
}











void ShuffleOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!llvm::empty(resultGroup0))
    setNameFn(*resultGroup0.begin(), "result");
  auto resultGroup1 = getODSResults(1);
  if (!llvm::empty(resultGroup1))
    setNameFn(*resultGroup1.begin(), "valid");
}



std::pair<unsigned, unsigned> ShuffleOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ShuffleOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ShuffleOp::value() {
  return *getODSOperands(0).begin();
}

::mlir::Value ShuffleOp::offset() {
  return *getODSOperands(1).begin();
}

::mlir::Value ShuffleOp::width() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange ShuffleOp::valueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ShuffleOp::offsetMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ShuffleOp::widthMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ShuffleOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ShuffleOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ShuffleOp::result() {
  return *getODSResults(0).begin();
}

::mlir::Value ShuffleOp::valid() {
  return *getODSResults(1).begin();
}

::mlir::StringAttr ShuffleOp::modeAttr() {
  return (*this)->getAttr(modeAttrName()).template cast<::mlir::StringAttr>();
}

::llvm::StringRef ShuffleOp::mode() {
  auto attr = modeAttr();
  return attr.getValue();
}

void ShuffleOp::modeAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(modeAttrName(), attr);
}

void ShuffleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Type valid, ::mlir::Value value, ::mlir::Value offset, ::mlir::Value width, ::mlir::StringAttr mode) {
  odsState.addOperands(value);
  odsState.addOperands(offset);
  odsState.addOperands(width);
  odsState.addAttribute(modeAttrName(odsState.name), mode);
  odsState.addTypes(result);
  odsState.addTypes(valid);
}

void ShuffleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value offset, ::mlir::Value width, ::mlir::StringAttr mode) {
  odsState.addOperands(value);
  odsState.addOperands(offset);
  odsState.addOperands(width);
  odsState.addAttribute(modeAttrName(odsState.name), mode);
  assert(resultTypes.size() == 2u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ShuffleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Type valid, ::mlir::Value value, ::mlir::Value offset, ::mlir::Value width, ::llvm::StringRef mode) {
  odsState.addOperands(value);
  odsState.addOperands(offset);
  odsState.addOperands(width);
  odsState.addAttribute(modeAttrName(odsState.name), odsBuilder.getStringAttr(mode));
  odsState.addTypes(result);
  odsState.addTypes(valid);
}

void ShuffleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value offset, ::mlir::Value width, ::llvm::StringRef mode) {
  odsState.addOperands(value);
  odsState.addOperands(offset);
  odsState.addOperands(width);
  odsState.addAttribute(modeAttrName(odsState.name), odsBuilder.getStringAttr(mode));
  assert(resultTypes.size() == 2u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ShuffleOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 2u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult ShuffleOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return parseShuffleOp(parser, result);
}

void ShuffleOp::print(::mlir::OpAsmPrinter &p) {
  printShuffleOp(p, *this);
}

::mlir::LogicalResult ShuffleOp::verify() {
  if (::mlir::failed(ShuffleOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps8(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps8(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps0(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSResults(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps9(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::verifyShuffleOp(*this);
}

void ShuffleOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace gpu
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::ShuffleOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::SubgroupIdOp definitions
//===----------------------------------------------------------------------===//

SubgroupIdOpAdaptor::SubgroupIdOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

SubgroupIdOpAdaptor::SubgroupIdOpAdaptor(SubgroupIdOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange SubgroupIdOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SubgroupIdOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SubgroupIdOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr SubgroupIdOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult SubgroupIdOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> SubgroupIdOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SubgroupIdOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> SubgroupIdOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SubgroupIdOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SubgroupIdOp::result() {
  return *getODSResults(0).begin();
}

void SubgroupIdOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result) {
  odsState.addTypes(result);
}

void SubgroupIdOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SubgroupIdOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SubgroupIdOp::verify() {
  if (::mlir::failed(SubgroupIdOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return success();
}

::mlir::ParseResult SubgroupIdOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resultRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resultTypes);
  return ::mlir::success();
}

void SubgroupIdOp::print(::mlir::OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(result().getType());
}

void SubgroupIdOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace gpu
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::SubgroupIdOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::SubgroupMmaComputeOp definitions
//===----------------------------------------------------------------------===//

SubgroupMmaComputeOpAdaptor::SubgroupMmaComputeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

SubgroupMmaComputeOpAdaptor::SubgroupMmaComputeOpAdaptor(SubgroupMmaComputeOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange SubgroupMmaComputeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SubgroupMmaComputeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SubgroupMmaComputeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SubgroupMmaComputeOpAdaptor::opA() {
  return *getODSOperands(0).begin();
}

::mlir::Value SubgroupMmaComputeOpAdaptor::opB() {
  return *getODSOperands(1).begin();
}

::mlir::Value SubgroupMmaComputeOpAdaptor::opC() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr SubgroupMmaComputeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult SubgroupMmaComputeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> SubgroupMmaComputeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SubgroupMmaComputeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SubgroupMmaComputeOp::opA() {
  return *getODSOperands(0).begin();
}

::mlir::Value SubgroupMmaComputeOp::opB() {
  return *getODSOperands(1).begin();
}

::mlir::Value SubgroupMmaComputeOp::opC() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange SubgroupMmaComputeOp::opAMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange SubgroupMmaComputeOp::opBMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange SubgroupMmaComputeOp::opCMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SubgroupMmaComputeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SubgroupMmaComputeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SubgroupMmaComputeOp::res() {
  return *getODSResults(0).begin();
}

void SubgroupMmaComputeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value opA, ::mlir::Value opB, ::mlir::Value opC) {
  odsState.addOperands(opA);
  odsState.addOperands(opB);
  odsState.addOperands(opC);
  odsState.addTypes(res);
}

void SubgroupMmaComputeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value opA, ::mlir::Value opB, ::mlir::Value opC) {
  odsState.addOperands(opA);
  odsState.addOperands(opB);
  odsState.addOperands(opC);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SubgroupMmaComputeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SubgroupMmaComputeOp::verify() {
  if (::mlir::failed(SubgroupMmaComputeOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps10(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps10(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps11(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps12(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(2).begin()).getType(), (*this->getODSResults(0).begin()).getType()})))))
    return emitOpError("failed to verify that all of {opC, res} have same type");
  return ::verify(*this);
}

::mlir::ParseResult SubgroupMmaComputeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType opARawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> opAOperands(opARawOperands);  ::llvm::SMLoc opAOperandsLoc;
  (void)opAOperandsLoc;
  ::mlir::OpAsmParser::OperandType opBRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> opBOperands(opBRawOperands);  ::llvm::SMLoc opBOperandsLoc;
  (void)opBOperandsLoc;
  ::mlir::OpAsmParser::OperandType opCRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> opCOperands(opCRawOperands);  ::llvm::SMLoc opCOperandsLoc;
  (void)opCOperandsLoc;
  ::mlir::Type opARawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> opATypes(opARawTypes);
  ::mlir::Type opBRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> opBTypes(opBRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  opAOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(opARawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  opBOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(opBRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  opCOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(opCRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(opARawTypes[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(opBRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resTypes);
  if (parser.resolveOperands(opAOperands, opATypes, opAOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(opBOperands, opBTypes, opBOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(opCOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SubgroupMmaComputeOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << opA();
  p << ",";
  p << ' ';
  p << opB();
  p << ",";
  p << ' ';
  p << opC();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(opA().getType());
  p << ",";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(opB().getType());
  p << ' ' << "->";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

void SubgroupMmaComputeOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace gpu
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::SubgroupMmaComputeOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::SubgroupMmaConstantMatrixOp definitions
//===----------------------------------------------------------------------===//

SubgroupMmaConstantMatrixOpAdaptor::SubgroupMmaConstantMatrixOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

SubgroupMmaConstantMatrixOpAdaptor::SubgroupMmaConstantMatrixOpAdaptor(SubgroupMmaConstantMatrixOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange SubgroupMmaConstantMatrixOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SubgroupMmaConstantMatrixOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SubgroupMmaConstantMatrixOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SubgroupMmaConstantMatrixOpAdaptor::value() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr SubgroupMmaConstantMatrixOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult SubgroupMmaConstantMatrixOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> SubgroupMmaConstantMatrixOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SubgroupMmaConstantMatrixOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SubgroupMmaConstantMatrixOp::value() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange SubgroupMmaConstantMatrixOp::valueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SubgroupMmaConstantMatrixOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SubgroupMmaConstantMatrixOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SubgroupMmaConstantMatrixOp::res() {
  return *getODSResults(0).begin();
}

void SubgroupMmaConstantMatrixOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value value) {
  odsState.addOperands(value);
  odsState.addTypes(res);
}

void SubgroupMmaConstantMatrixOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value) {
  odsState.addOperands(value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SubgroupMmaConstantMatrixOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SubgroupMmaConstantMatrixOp::verify() {
  if (::mlir::failed(SubgroupMmaConstantMatrixOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps13(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps12(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  if (!((std::equal_to<>()((*this->getODSResults(0).begin()).getType().cast<gpu::MMAMatrixType>().getElementType(), (*this->getODSOperands(0).begin()).getType()))))
    return emitOpError("failed to verify that value type matches element type of mma_matrix");
  return ::mlir::success();
}

::mlir::ParseResult SubgroupMmaConstantMatrixOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  for (::mlir::Type type : resTypes) {
    (void)type;
    if (!((type.isa<::mlir::gpu::MMAMatrixType>()))) {
      return parser.emitError(parser.getNameLoc()) << "'res' must be MMAMatrix type, but got " << type;
    }
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(valueOperands, resTypes[0].cast<gpu::MMAMatrixType>().getElementType(), valueOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SubgroupMmaConstantMatrixOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << value();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

void SubgroupMmaConstantMatrixOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace gpu
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::SubgroupMmaConstantMatrixOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::SubgroupMmaLoadMatrixOp definitions
//===----------------------------------------------------------------------===//

SubgroupMmaLoadMatrixOpAdaptor::SubgroupMmaLoadMatrixOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

SubgroupMmaLoadMatrixOpAdaptor::SubgroupMmaLoadMatrixOpAdaptor(SubgroupMmaLoadMatrixOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange SubgroupMmaLoadMatrixOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SubgroupMmaLoadMatrixOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange SubgroupMmaLoadMatrixOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SubgroupMmaLoadMatrixOpAdaptor::srcMemref() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange SubgroupMmaLoadMatrixOpAdaptor::indices() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr SubgroupMmaLoadMatrixOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr SubgroupMmaLoadMatrixOpAdaptor::leadDimension() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("leadDimension").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::LogicalResult SubgroupMmaLoadMatrixOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_leadDimension = odsAttrs.get("leadDimension");
  if (!tblgen_leadDimension) return emitError(loc, "'gpu.subgroup_mma_load_matrix' op ""requires attribute 'leadDimension'");
    if (!(((tblgen_leadDimension.isa<::mlir::IntegerAttr>())) && ((tblgen_leadDimension.cast<::mlir::IntegerAttr>().getType().isa<::mlir::IndexType>())))) return emitError(loc, "'gpu.subgroup_mma_load_matrix' op ""attribute 'leadDimension' failed to satisfy constraint: index attribute");
  }
  return ::mlir::success();
}













std::pair<unsigned, unsigned> SubgroupMmaLoadMatrixOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range SubgroupMmaLoadMatrixOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SubgroupMmaLoadMatrixOp::srcMemref() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range SubgroupMmaLoadMatrixOp::indices() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange SubgroupMmaLoadMatrixOp::srcMemrefMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange SubgroupMmaLoadMatrixOp::indicesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SubgroupMmaLoadMatrixOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SubgroupMmaLoadMatrixOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SubgroupMmaLoadMatrixOp::res() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr SubgroupMmaLoadMatrixOp::leadDimensionAttr() {
  return (*this)->getAttr(leadDimensionAttrName()).template cast<::mlir::IntegerAttr>();
}

::llvm::APInt SubgroupMmaLoadMatrixOp::leadDimension() {
  auto attr = leadDimensionAttr();
  return attr.getValue();
}

void SubgroupMmaLoadMatrixOp::leadDimensionAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(leadDimensionAttrName(), attr);
}

void SubgroupMmaLoadMatrixOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value srcMemref, ::mlir::ValueRange indices, ::mlir::IntegerAttr leadDimension) {
  odsState.addOperands(srcMemref);
  odsState.addOperands(indices);
  odsState.addAttribute(leadDimensionAttrName(odsState.name), leadDimension);
  odsState.addTypes(res);
}

void SubgroupMmaLoadMatrixOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value srcMemref, ::mlir::ValueRange indices, ::mlir::IntegerAttr leadDimension) {
  odsState.addOperands(srcMemref);
  odsState.addOperands(indices);
  odsState.addAttribute(leadDimensionAttrName(odsState.name), leadDimension);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SubgroupMmaLoadMatrixOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value srcMemref, ::mlir::ValueRange indices, ::llvm::APInt leadDimension) {
  odsState.addOperands(srcMemref);
  odsState.addOperands(indices);
  odsState.addAttribute(leadDimensionAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIndexType(), leadDimension));
  odsState.addTypes(res);
}

void SubgroupMmaLoadMatrixOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value srcMemref, ::mlir::ValueRange indices, ::llvm::APInt leadDimension) {
  odsState.addOperands(srcMemref);
  odsState.addOperands(indices);
  odsState.addAttribute(leadDimensionAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIndexType(), leadDimension));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SubgroupMmaLoadMatrixOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SubgroupMmaLoadMatrixOp::verify() {
  if (::mlir::failed(SubgroupMmaLoadMatrixOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps14(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps12(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::verify(*this);
}

::mlir::ParseResult SubgroupMmaLoadMatrixOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType srcMemrefRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> srcMemrefOperands(srcMemrefRawOperands);  ::llvm::SMLoc srcMemrefOperandsLoc;
  (void)srcMemrefOperandsLoc;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> indicesOperands;
  ::llvm::SMLoc indicesOperandsLoc;
  (void)indicesOperandsLoc;
  ::mlir::Type srcMemrefRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> srcMemrefTypes(srcMemrefRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  srcMemrefOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(srcMemrefRawOperands[0]))
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

  if (parser.parseType(srcMemrefRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseType(resRawTypes[0]))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(resTypes);
  if (parser.resolveOperands(srcMemrefOperands, srcMemrefTypes, srcMemrefOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(indicesOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SubgroupMmaLoadMatrixOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << srcMemref();
  p << "[";
  p << indices();
  p << "]";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(srcMemref().getType());
  p << ' ' << "->";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(res().getType());
}

void SubgroupMmaLoadMatrixOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  effects.emplace_back(::mlir::MemoryEffects::Read::get(), ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(::mlir::MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace gpu
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::SubgroupMmaLoadMatrixOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::SubgroupMmaStoreMatrixOp definitions
//===----------------------------------------------------------------------===//

SubgroupMmaStoreMatrixOpAdaptor::SubgroupMmaStoreMatrixOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

SubgroupMmaStoreMatrixOpAdaptor::SubgroupMmaStoreMatrixOpAdaptor(SubgroupMmaStoreMatrixOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange SubgroupMmaStoreMatrixOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SubgroupMmaStoreMatrixOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange SubgroupMmaStoreMatrixOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SubgroupMmaStoreMatrixOpAdaptor::src() {
  return *getODSOperands(0).begin();
}

::mlir::Value SubgroupMmaStoreMatrixOpAdaptor::dstMemref() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange SubgroupMmaStoreMatrixOpAdaptor::indices() {
  return getODSOperands(2);
}

::mlir::DictionaryAttr SubgroupMmaStoreMatrixOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr SubgroupMmaStoreMatrixOpAdaptor::leadDimension() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("leadDimension").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::LogicalResult SubgroupMmaStoreMatrixOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_leadDimension = odsAttrs.get("leadDimension");
  if (!tblgen_leadDimension) return emitError(loc, "'gpu.subgroup_mma_store_matrix' op ""requires attribute 'leadDimension'");
    if (!(((tblgen_leadDimension.isa<::mlir::IntegerAttr>())) && ((tblgen_leadDimension.cast<::mlir::IntegerAttr>().getType().isa<::mlir::IndexType>())))) return emitError(loc, "'gpu.subgroup_mma_store_matrix' op ""attribute 'leadDimension' failed to satisfy constraint: index attribute");
  }
  return ::mlir::success();
}













std::pair<unsigned, unsigned> SubgroupMmaStoreMatrixOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range SubgroupMmaStoreMatrixOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SubgroupMmaStoreMatrixOp::src() {
  return *getODSOperands(0).begin();
}

::mlir::Value SubgroupMmaStoreMatrixOp::dstMemref() {
  return *getODSOperands(1).begin();
}

::mlir::Operation::operand_range SubgroupMmaStoreMatrixOp::indices() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange SubgroupMmaStoreMatrixOp::srcMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange SubgroupMmaStoreMatrixOp::dstMemrefMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange SubgroupMmaStoreMatrixOp::indicesMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SubgroupMmaStoreMatrixOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SubgroupMmaStoreMatrixOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::IntegerAttr SubgroupMmaStoreMatrixOp::leadDimensionAttr() {
  return (*this)->getAttr(leadDimensionAttrName()).template cast<::mlir::IntegerAttr>();
}

::llvm::APInt SubgroupMmaStoreMatrixOp::leadDimension() {
  auto attr = leadDimensionAttr();
  return attr.getValue();
}

void SubgroupMmaStoreMatrixOp::leadDimensionAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(leadDimensionAttrName(), attr);
}

void SubgroupMmaStoreMatrixOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value src, ::mlir::Value dstMemref, ::mlir::ValueRange indices, ::mlir::IntegerAttr leadDimension) {
  odsState.addOperands(src);
  odsState.addOperands(dstMemref);
  odsState.addOperands(indices);
  odsState.addAttribute(leadDimensionAttrName(odsState.name), leadDimension);
}

void SubgroupMmaStoreMatrixOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value dstMemref, ::mlir::ValueRange indices, ::mlir::IntegerAttr leadDimension) {
  odsState.addOperands(src);
  odsState.addOperands(dstMemref);
  odsState.addOperands(indices);
  odsState.addAttribute(leadDimensionAttrName(odsState.name), leadDimension);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SubgroupMmaStoreMatrixOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value src, ::mlir::Value dstMemref, ::mlir::ValueRange indices, ::llvm::APInt leadDimension) {
  odsState.addOperands(src);
  odsState.addOperands(dstMemref);
  odsState.addOperands(indices);
  odsState.addAttribute(leadDimensionAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIndexType(), leadDimension));
}

void SubgroupMmaStoreMatrixOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value dstMemref, ::mlir::ValueRange indices, ::llvm::APInt leadDimension) {
  odsState.addOperands(src);
  odsState.addOperands(dstMemref);
  odsState.addOperands(indices);
  odsState.addAttribute(leadDimensionAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIndexType(), leadDimension));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SubgroupMmaStoreMatrixOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SubgroupMmaStoreMatrixOp::verify() {
  if (::mlir::failed(SubgroupMmaStoreMatrixOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps11(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps14(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps2(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::verify(*this);
}

::mlir::ParseResult SubgroupMmaStoreMatrixOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType srcRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> srcOperands(srcRawOperands);  ::llvm::SMLoc srcOperandsLoc;
  (void)srcOperandsLoc;
  ::mlir::OpAsmParser::OperandType dstMemrefRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> dstMemrefOperands(dstMemrefRawOperands);  ::llvm::SMLoc dstMemrefOperandsLoc;
  (void)dstMemrefOperandsLoc;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> indicesOperands;
  ::llvm::SMLoc indicesOperandsLoc;
  (void)indicesOperandsLoc;
  ::mlir::Type srcRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> srcTypes(srcRawTypes);
  ::mlir::Type dstMemrefRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> dstMemrefTypes(dstMemrefRawTypes);

  srcOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(srcRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  dstMemrefOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(dstMemrefRawOperands[0]))
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

  if (parser.parseType(srcRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(dstMemrefRawTypes[0]))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  if (parser.resolveOperands(srcOperands, srcTypes, srcOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(dstMemrefOperands, dstMemrefTypes, dstMemrefOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(indicesOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SubgroupMmaStoreMatrixOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  p << src();
  p << ",";
  p << ' ';
  p << dstMemref();
  p << "[";
  p << indices();
  p << "]";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(src().getType());
  p << ",";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(dstMemref().getType());
}

void SubgroupMmaStoreMatrixOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  effects.emplace_back(::mlir::MemoryEffects::Write::get(), ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(1))
    effects.emplace_back(::mlir::MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace gpu
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::SubgroupMmaStoreMatrixOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::SubgroupSizeOp definitions
//===----------------------------------------------------------------------===//

SubgroupSizeOpAdaptor::SubgroupSizeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

SubgroupSizeOpAdaptor::SubgroupSizeOpAdaptor(SubgroupSizeOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange SubgroupSizeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SubgroupSizeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SubgroupSizeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr SubgroupSizeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult SubgroupSizeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> SubgroupSizeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SubgroupSizeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> SubgroupSizeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SubgroupSizeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SubgroupSizeOp::result() {
  return *getODSResults(0).begin();
}

void SubgroupSizeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result) {
  odsState.addTypes(result);
}

void SubgroupSizeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SubgroupSizeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SubgroupSizeOp::verify() {
  if (::mlir::failed(SubgroupSizeOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return success();
}

::mlir::ParseResult SubgroupSizeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resultRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resultTypes);
  return ::mlir::success();
}

void SubgroupSizeOp::print(::mlir::OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(result().getType());
}

void SubgroupSizeOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace gpu
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::SubgroupSizeOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::TerminatorOp definitions
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

void TerminatorOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace gpu
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::TerminatorOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::ThreadIdOp definitions
//===----------------------------------------------------------------------===//

ThreadIdOpAdaptor::ThreadIdOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ThreadIdOpAdaptor::ThreadIdOpAdaptor(ThreadIdOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ThreadIdOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ThreadIdOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ThreadIdOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr ThreadIdOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr ThreadIdOpAdaptor::dimension() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("dimension").cast<::mlir::StringAttr>();
  return attr;
}

::mlir::LogicalResult ThreadIdOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_dimension = odsAttrs.get("dimension");
  if (!tblgen_dimension) return emitError(loc, "'gpu.thread_id' op ""requires attribute 'dimension'");
    if (!((tblgen_dimension.isa<::mlir::StringAttr>()))) return emitError(loc, "'gpu.thread_id' op ""attribute 'dimension' failed to satisfy constraint: string attribute");
  }
  return ::mlir::success();
}













std::pair<unsigned, unsigned> ThreadIdOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ThreadIdOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> ThreadIdOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ThreadIdOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::StringAttr ThreadIdOp::dimensionAttr() {
  return (*this)->getAttr(dimensionAttrName()).template cast<::mlir::StringAttr>();
}

::llvm::StringRef ThreadIdOp::dimension() {
  auto attr = dimensionAttr();
  return attr.getValue();
}

void ThreadIdOp::dimensionAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(dimensionAttrName(), attr);
}

void ThreadIdOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::StringAttr dimension) {
  odsState.addAttribute(dimensionAttrName(odsState.name), dimension);
  odsState.addTypes(resultType0);
}

void ThreadIdOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr dimension) {
  odsState.addAttribute(dimensionAttrName(odsState.name), dimension);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ThreadIdOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::llvm::StringRef dimension) {
  odsState.addAttribute(dimensionAttrName(odsState.name), odsBuilder.getStringAttr(dimension));
  odsState.addTypes(resultType0);
}

void ThreadIdOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef dimension) {
  odsState.addAttribute(dimensionAttrName(odsState.name), odsBuilder.getStringAttr(dimension));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ThreadIdOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ThreadIdOp::verify() {
  if (::mlir::failed(ThreadIdOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::verifyIndexOp(*this);
}

void ThreadIdOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace gpu
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::ThreadIdOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::WaitOp definitions
//===----------------------------------------------------------------------===//

WaitOpAdaptor::WaitOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

WaitOpAdaptor::WaitOpAdaptor(WaitOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange WaitOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> WaitOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange WaitOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange WaitOpAdaptor::asyncDependencies() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr WaitOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult WaitOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> WaitOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range WaitOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range WaitOp::asyncDependencies() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange WaitOp::asyncDependenciesMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> WaitOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range WaitOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value WaitOp::asyncToken() {
  auto results = getODSResults(0);
  return results.empty() ? ::mlir::Value() : *results.begin();
}

void WaitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type asyncToken, ::mlir::ValueRange asyncDependencies) {
  odsState.addOperands(asyncDependencies);
  if (asyncToken)
    odsState.addTypes(asyncToken);
}

void WaitOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult WaitOp::verify() {
  if (::mlir::failed(WaitOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    if (valueGroup0.size() > 1)
      return emitOpError("result group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup0.size();
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps1(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult WaitOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::Type, 1> asyncTokenTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> asyncDependenciesOperands;
  ::llvm::SMLoc asyncDependenciesOperandsLoc;
  (void)asyncDependenciesOperandsLoc;
  {
    ::mlir::Type asyncTokenType;
    asyncDependenciesOperandsLoc = parser.getCurrentLocation();
    if (parseAsyncDependencies(parser, asyncTokenType, asyncDependenciesOperands))
      return ::mlir::failure();
    if (asyncTokenType)
      asyncTokenTypes.push_back(asyncTokenType);
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = mlir::gpu::AsyncTokenType::get(parser.getBuilder().getContext());
  result.addTypes(asyncTokenTypes);
  if (parser.resolveOperands(asyncDependenciesOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void WaitOp::print(::mlir::OpAsmPrinter &p) {
  p << ' ';
  printAsyncDependencies(p, *this, (asyncToken() ? asyncToken().getType() : Type()), asyncDependencies());
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace gpu
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::WaitOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::YieldOp definitions
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

::mlir::ValueRange YieldOpAdaptor::values() {
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

::mlir::Operation::operand_range YieldOp::values() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange YieldOp::valuesMutable() {
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

void YieldOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange values) {
  odsState.addOperands(values);
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
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

void YieldOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace gpu
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::YieldOp)


#endif  // GET_OP_CLASSES

