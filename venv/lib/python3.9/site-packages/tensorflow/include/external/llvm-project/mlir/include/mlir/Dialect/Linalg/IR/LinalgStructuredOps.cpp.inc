/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::linalg::BatchMatmulOp,
::mlir::linalg::BatchMatvecOp,
::mlir::linalg::Conv1DNwcWcfOp,
::mlir::linalg::Conv1DOp,
::mlir::linalg::Conv2DNchwFchwOp,
::mlir::linalg::Conv2DNhwcHwcfOp,
::mlir::linalg::Conv2DNhwcHwcfQOp,
::mlir::linalg::Conv2DOp,
::mlir::linalg::Conv3DNdhwcDhwcfOp,
::mlir::linalg::Conv3DOp,
::mlir::linalg::ConvOp,
::mlir::linalg::CopyOp,
::mlir::linalg::DepthwiseConv2DNhwOp,
::mlir::linalg::DepthwiseConv2DNhwQOp,
::mlir::linalg::DepthwiseConv2DNhwcOp,
::mlir::linalg::DepthwiseConv2DNhwcQOp,
::mlir::linalg::DotOp,
::mlir::linalg::FillOp,
::mlir::linalg::FillRng2DOp,
::mlir::linalg::GenericOp,
::mlir::linalg::MatmulI8I8I32Op,
::mlir::linalg::MatmulOp,
::mlir::linalg::MatvecOp,
::mlir::linalg::Mmt4DOp,
::mlir::linalg::PoolingMaxOp,
::mlir::linalg::PoolingMinOp,
::mlir::linalg::PoolingNchwMaxOp,
::mlir::linalg::PoolingNdhwcMaxOp,
::mlir::linalg::PoolingNdhwcMinOp,
::mlir::linalg::PoolingNdhwcSumOp,
::mlir::linalg::PoolingNhwcMaxOp,
::mlir::linalg::PoolingNhwcMinOp,
::mlir::linalg::PoolingNhwcSumOp,
::mlir::linalg::PoolingSumOp,
::mlir::linalg::QuantizedBatchMatmulOp,
::mlir::linalg::QuantizedMatmulOp,
::mlir::linalg::SoftPlus2DOp,
::mlir::linalg::VecmatOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace linalg {
static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LinalgStructuredOps0(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((true))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be any type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LinalgStructuredOps1(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isa<::mlir::ShapedType>())) && ((true)))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be shaped of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LinalgStructuredOps2(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((((type.isa<::mlir::TensorType>())) && ((type.cast<::mlir::ShapedType>().hasRank()))) && ((true)))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be ranked tensor of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LinalgStructuredOps3(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!((((type.isa<::mlir::MemRefType>())) && ((true))) && (( isStrided(type.cast<::mlir::MemRefType>()) )))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be strided memref of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LinalgStructuredOps4(::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex) {
  if (!(((type.isa<::mlir::ComplexType>())) || ((type.isa<::mlir::FloatType>())) || ((type.isSignlessInteger())) || (((type.isa<::mlir::VectorType>())) && ((true))))) {
    return  op->emitOpError(valueKind) << " #" << valueGroupStartIndex << " must be complex-type or floating-point or signless integer or vector of any type values, but got " << type;
  }
  return ::mlir::success();
}

} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::BatchMatmulOp definitions
//===----------------------------------------------------------------------===//

BatchMatmulOpAdaptor::BatchMatmulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

BatchMatmulOpAdaptor::BatchMatmulOpAdaptor(BatchMatmulOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange BatchMatmulOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BatchMatmulOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange BatchMatmulOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange BatchMatmulOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange BatchMatmulOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr BatchMatmulOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange BatchMatmulOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &BatchMatmulOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult BatchMatmulOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    return ::mlir::success();
}













std::pair<unsigned, unsigned> BatchMatmulOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range BatchMatmulOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range BatchMatmulOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range BatchMatmulOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange BatchMatmulOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange BatchMatmulOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> BatchMatmulOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range BatchMatmulOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range BatchMatmulOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &BatchMatmulOp::region() {
  return (*this)->getRegion(0);
}

void BatchMatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<BatchMatmulOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void BatchMatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<BatchMatmulOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void BatchMatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::ParseResult BatchMatmulOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<BatchMatmulOp>(parser, result);
}

void BatchMatmulOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult BatchMatmulOp::verify() {
  if (::mlir::failed(BatchMatmulOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
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





} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::BatchMatmulOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::BatchMatvecOp definitions
//===----------------------------------------------------------------------===//

BatchMatvecOpAdaptor::BatchMatvecOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

BatchMatvecOpAdaptor::BatchMatvecOpAdaptor(BatchMatvecOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange BatchMatvecOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BatchMatvecOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange BatchMatvecOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange BatchMatvecOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange BatchMatvecOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr BatchMatvecOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange BatchMatvecOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &BatchMatvecOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult BatchMatvecOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    return ::mlir::success();
}













std::pair<unsigned, unsigned> BatchMatvecOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range BatchMatvecOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range BatchMatvecOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range BatchMatvecOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange BatchMatvecOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange BatchMatvecOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> BatchMatvecOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range BatchMatvecOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range BatchMatvecOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &BatchMatvecOp::region() {
  return (*this)->getRegion(0);
}

void BatchMatvecOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<BatchMatvecOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void BatchMatvecOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<BatchMatvecOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void BatchMatvecOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::ParseResult BatchMatvecOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<BatchMatvecOp>(parser, result);
}

void BatchMatvecOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult BatchMatvecOp::verify() {
  if (::mlir::failed(BatchMatvecOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
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





} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::BatchMatvecOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Conv1DNwcWcfOp definitions
//===----------------------------------------------------------------------===//

Conv1DNwcWcfOpAdaptor::Conv1DNwcWcfOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

Conv1DNwcWcfOpAdaptor::Conv1DNwcWcfOpAdaptor(Conv1DNwcWcfOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange Conv1DNwcWcfOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> Conv1DNwcWcfOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange Conv1DNwcWcfOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange Conv1DNwcWcfOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange Conv1DNwcWcfOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr Conv1DNwcWcfOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr Conv1DNwcWcfOpAdaptor::strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr Conv1DNwcWcfOpAdaptor::dilations() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("dilations").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::RegionRange Conv1DNwcWcfOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &Conv1DNwcWcfOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult Conv1DNwcWcfOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    {
  auto tblgen_strides = odsAttrs.get("strides");
  if (!tblgen_strides) return emitError(loc, "'linalg.conv_1d_nwc_wcf' op ""requires attribute 'strides'");
    if (!((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({1}))))) return emitError(loc, "'linalg.conv_1d_nwc_wcf' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [1]");
  }
  {
  auto tblgen_dilations = odsAttrs.get("dilations");
  if (!tblgen_dilations) return emitError(loc, "'linalg.conv_1d_nwc_wcf' op ""requires attribute 'dilations'");
    if (!((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({1}))))) return emitError(loc, "'linalg.conv_1d_nwc_wcf' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [1]");
  }
  return ::mlir::success();
}





















std::pair<unsigned, unsigned> Conv1DNwcWcfOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range Conv1DNwcWcfOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range Conv1DNwcWcfOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range Conv1DNwcWcfOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange Conv1DNwcWcfOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv1DNwcWcfOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> Conv1DNwcWcfOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range Conv1DNwcWcfOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range Conv1DNwcWcfOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &Conv1DNwcWcfOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr Conv1DNwcWcfOp::stridesAttr() {
  return (*this)->getAttr(stridesAttrName()).template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv1DNwcWcfOp::strides() {
  auto attr = stridesAttr();
  return attr;
}

::mlir::DenseIntElementsAttr Conv1DNwcWcfOp::dilationsAttr() {
  return (*this)->getAttr(dilationsAttrName()).template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv1DNwcWcfOp::dilations() {
  auto attr = dilationsAttr();
  return attr;
}

void Conv1DNwcWcfOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void Conv1DNwcWcfOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

void Conv1DNwcWcfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<Conv1DNwcWcfOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void Conv1DNwcWcfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<Conv1DNwcWcfOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void Conv1DNwcWcfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void Conv1DNwcWcfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes ) {
    odsState.addOperands(inputs);
    odsState.addOperands(outputs);
    odsState.addTypes(resultTensorTypes);
    odsState.addAttribute(
      "operand_segment_sizes",
      odsBuilder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<Conv1DNwcWcfOp>(
      odsBuilder,
      odsState,
      TypeRange(inputs),
      TypeRange(outputs));
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    odsState.addAttributes(attributes);
  
}

::mlir::ParseResult Conv1DNwcWcfOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<Conv1DNwcWcfOp>(parser, result);
}

void Conv1DNwcWcfOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult Conv1DNwcWcfOp::verify() {
  if (::mlir::failed(Conv1DNwcWcfOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
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





} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::Conv1DNwcWcfOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Conv1DOp definitions
//===----------------------------------------------------------------------===//

Conv1DOpAdaptor::Conv1DOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

Conv1DOpAdaptor::Conv1DOpAdaptor(Conv1DOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange Conv1DOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> Conv1DOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange Conv1DOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange Conv1DOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange Conv1DOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr Conv1DOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange Conv1DOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &Conv1DOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult Conv1DOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    return ::mlir::success();
}













std::pair<unsigned, unsigned> Conv1DOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range Conv1DOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range Conv1DOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range Conv1DOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange Conv1DOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv1DOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> Conv1DOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range Conv1DOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range Conv1DOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &Conv1DOp::region() {
  return (*this)->getRegion(0);
}

void Conv1DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<Conv1DOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void Conv1DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<Conv1DOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void Conv1DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::ParseResult Conv1DOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<Conv1DOp>(parser, result);
}

void Conv1DOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult Conv1DOp::verify() {
  if (::mlir::failed(Conv1DOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
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





} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::Conv1DOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Conv2DNchwFchwOp definitions
//===----------------------------------------------------------------------===//

Conv2DNchwFchwOpAdaptor::Conv2DNchwFchwOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

Conv2DNchwFchwOpAdaptor::Conv2DNchwFchwOpAdaptor(Conv2DNchwFchwOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange Conv2DNchwFchwOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> Conv2DNchwFchwOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange Conv2DNchwFchwOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange Conv2DNchwFchwOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange Conv2DNchwFchwOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr Conv2DNchwFchwOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr Conv2DNchwFchwOpAdaptor::strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNchwFchwOpAdaptor::dilations() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("dilations").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::RegionRange Conv2DNchwFchwOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &Conv2DNchwFchwOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult Conv2DNchwFchwOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    {
  auto tblgen_strides = odsAttrs.get("strides");
  if (!tblgen_strides) return emitError(loc, "'linalg.conv_2d_nchw_fchw' op ""requires attribute 'strides'");
    if (!((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2}))))) return emitError(loc, "'linalg.conv_2d_nchw_fchw' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  {
  auto tblgen_dilations = odsAttrs.get("dilations");
  if (!tblgen_dilations) return emitError(loc, "'linalg.conv_2d_nchw_fchw' op ""requires attribute 'dilations'");
    if (!((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2}))))) return emitError(loc, "'linalg.conv_2d_nchw_fchw' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  return ::mlir::success();
}





















std::pair<unsigned, unsigned> Conv2DNchwFchwOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range Conv2DNchwFchwOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range Conv2DNchwFchwOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range Conv2DNchwFchwOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange Conv2DNchwFchwOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv2DNchwFchwOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> Conv2DNchwFchwOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range Conv2DNchwFchwOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range Conv2DNchwFchwOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &Conv2DNchwFchwOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr Conv2DNchwFchwOp::stridesAttr() {
  return (*this)->getAttr(stridesAttrName()).template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv2DNchwFchwOp::strides() {
  auto attr = stridesAttr();
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNchwFchwOp::dilationsAttr() {
  return (*this)->getAttr(dilationsAttrName()).template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv2DNchwFchwOp::dilations() {
  auto attr = dilationsAttr();
  return attr;
}

void Conv2DNchwFchwOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void Conv2DNchwFchwOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

void Conv2DNchwFchwOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<Conv2DNchwFchwOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void Conv2DNchwFchwOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<Conv2DNchwFchwOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void Conv2DNchwFchwOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void Conv2DNchwFchwOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes ) {
    odsState.addOperands(inputs);
    odsState.addOperands(outputs);
    odsState.addTypes(resultTensorTypes);
    odsState.addAttribute(
      "operand_segment_sizes",
      odsBuilder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<Conv2DNchwFchwOp>(
      odsBuilder,
      odsState,
      TypeRange(inputs),
      TypeRange(outputs));
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    odsState.addAttributes(attributes);
  
}

::mlir::ParseResult Conv2DNchwFchwOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<Conv2DNchwFchwOp>(parser, result);
}

void Conv2DNchwFchwOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult Conv2DNchwFchwOp::verify() {
  if (::mlir::failed(Conv2DNchwFchwOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
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





} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::Conv2DNchwFchwOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Conv2DNhwcHwcfOp definitions
//===----------------------------------------------------------------------===//

Conv2DNhwcHwcfOpAdaptor::Conv2DNhwcHwcfOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

Conv2DNhwcHwcfOpAdaptor::Conv2DNhwcHwcfOpAdaptor(Conv2DNhwcHwcfOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange Conv2DNhwcHwcfOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> Conv2DNhwcHwcfOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange Conv2DNhwcHwcfOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange Conv2DNhwcHwcfOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange Conv2DNhwcHwcfOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr Conv2DNhwcHwcfOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfOpAdaptor::strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfOpAdaptor::dilations() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("dilations").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::RegionRange Conv2DNhwcHwcfOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &Conv2DNhwcHwcfOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult Conv2DNhwcHwcfOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    {
  auto tblgen_strides = odsAttrs.get("strides");
  if (!tblgen_strides) return emitError(loc, "'linalg.conv_2d_nhwc_hwcf' op ""requires attribute 'strides'");
    if (!((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2}))))) return emitError(loc, "'linalg.conv_2d_nhwc_hwcf' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  {
  auto tblgen_dilations = odsAttrs.get("dilations");
  if (!tblgen_dilations) return emitError(loc, "'linalg.conv_2d_nhwc_hwcf' op ""requires attribute 'dilations'");
    if (!((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2}))))) return emitError(loc, "'linalg.conv_2d_nhwc_hwcf' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  return ::mlir::success();
}





















std::pair<unsigned, unsigned> Conv2DNhwcHwcfOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range Conv2DNhwcHwcfOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range Conv2DNhwcHwcfOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range Conv2DNhwcHwcfOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange Conv2DNhwcHwcfOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv2DNhwcHwcfOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> Conv2DNhwcHwcfOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range Conv2DNhwcHwcfOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range Conv2DNhwcHwcfOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &Conv2DNhwcHwcfOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfOp::stridesAttr() {
  return (*this)->getAttr(stridesAttrName()).template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfOp::strides() {
  auto attr = stridesAttr();
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfOp::dilationsAttr() {
  return (*this)->getAttr(dilationsAttrName()).template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfOp::dilations() {
  auto attr = dilationsAttr();
  return attr;
}

void Conv2DNhwcHwcfOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void Conv2DNhwcHwcfOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

void Conv2DNhwcHwcfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<Conv2DNhwcHwcfOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void Conv2DNhwcHwcfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<Conv2DNhwcHwcfOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void Conv2DNhwcHwcfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void Conv2DNhwcHwcfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes ) {
    odsState.addOperands(inputs);
    odsState.addOperands(outputs);
    odsState.addTypes(resultTensorTypes);
    odsState.addAttribute(
      "operand_segment_sizes",
      odsBuilder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<Conv2DNhwcHwcfOp>(
      odsBuilder,
      odsState,
      TypeRange(inputs),
      TypeRange(outputs));
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    odsState.addAttributes(attributes);
  
}

::mlir::ParseResult Conv2DNhwcHwcfOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<Conv2DNhwcHwcfOp>(parser, result);
}

void Conv2DNhwcHwcfOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult Conv2DNhwcHwcfOp::verify() {
  if (::mlir::failed(Conv2DNhwcHwcfOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
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





} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::Conv2DNhwcHwcfOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Conv2DNhwcHwcfQOp definitions
//===----------------------------------------------------------------------===//

Conv2DNhwcHwcfQOpAdaptor::Conv2DNhwcHwcfQOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

Conv2DNhwcHwcfQOpAdaptor::Conv2DNhwcHwcfQOpAdaptor(Conv2DNhwcHwcfQOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange Conv2DNhwcHwcfQOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> Conv2DNhwcHwcfQOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange Conv2DNhwcHwcfQOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange Conv2DNhwcHwcfQOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange Conv2DNhwcHwcfQOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr Conv2DNhwcHwcfQOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfQOpAdaptor::strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfQOpAdaptor::dilations() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("dilations").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::RegionRange Conv2DNhwcHwcfQOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &Conv2DNhwcHwcfQOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult Conv2DNhwcHwcfQOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    {
  auto tblgen_strides = odsAttrs.get("strides");
  if (!tblgen_strides) return emitError(loc, "'linalg.conv_2d_nhwc_hwcf_q' op ""requires attribute 'strides'");
    if (!((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2}))))) return emitError(loc, "'linalg.conv_2d_nhwc_hwcf_q' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  {
  auto tblgen_dilations = odsAttrs.get("dilations");
  if (!tblgen_dilations) return emitError(loc, "'linalg.conv_2d_nhwc_hwcf_q' op ""requires attribute 'dilations'");
    if (!((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2}))))) return emitError(loc, "'linalg.conv_2d_nhwc_hwcf_q' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  return ::mlir::success();
}





















std::pair<unsigned, unsigned> Conv2DNhwcHwcfQOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range Conv2DNhwcHwcfQOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range Conv2DNhwcHwcfQOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range Conv2DNhwcHwcfQOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange Conv2DNhwcHwcfQOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv2DNhwcHwcfQOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> Conv2DNhwcHwcfQOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range Conv2DNhwcHwcfQOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range Conv2DNhwcHwcfQOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &Conv2DNhwcHwcfQOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfQOp::stridesAttr() {
  return (*this)->getAttr(stridesAttrName()).template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfQOp::strides() {
  auto attr = stridesAttr();
  return attr;
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfQOp::dilationsAttr() {
  return (*this)->getAttr(dilationsAttrName()).template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv2DNhwcHwcfQOp::dilations() {
  auto attr = dilationsAttr();
  return attr;
}

void Conv2DNhwcHwcfQOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void Conv2DNhwcHwcfQOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

void Conv2DNhwcHwcfQOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<Conv2DNhwcHwcfQOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void Conv2DNhwcHwcfQOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<Conv2DNhwcHwcfQOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void Conv2DNhwcHwcfQOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void Conv2DNhwcHwcfQOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes ) {
    odsState.addOperands(inputs);
    odsState.addOperands(outputs);
    odsState.addTypes(resultTensorTypes);
    odsState.addAttribute(
      "operand_segment_sizes",
      odsBuilder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<Conv2DNhwcHwcfQOp>(
      odsBuilder,
      odsState,
      TypeRange(inputs),
      TypeRange(outputs));
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    odsState.addAttributes(attributes);
  
}

::mlir::ParseResult Conv2DNhwcHwcfQOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<Conv2DNhwcHwcfQOp>(parser, result);
}

void Conv2DNhwcHwcfQOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult Conv2DNhwcHwcfQOp::verify() {
  if (::mlir::failed(Conv2DNhwcHwcfQOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
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





} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::Conv2DNhwcHwcfQOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Conv2DOp definitions
//===----------------------------------------------------------------------===//

Conv2DOpAdaptor::Conv2DOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

Conv2DOpAdaptor::Conv2DOpAdaptor(Conv2DOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange Conv2DOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> Conv2DOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange Conv2DOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange Conv2DOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange Conv2DOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr Conv2DOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange Conv2DOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &Conv2DOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult Conv2DOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    return ::mlir::success();
}













std::pair<unsigned, unsigned> Conv2DOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range Conv2DOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range Conv2DOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range Conv2DOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange Conv2DOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv2DOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> Conv2DOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range Conv2DOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range Conv2DOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &Conv2DOp::region() {
  return (*this)->getRegion(0);
}

void Conv2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<Conv2DOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void Conv2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<Conv2DOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void Conv2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::ParseResult Conv2DOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<Conv2DOp>(parser, result);
}

void Conv2DOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult Conv2DOp::verify() {
  if (::mlir::failed(Conv2DOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
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





} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::Conv2DOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Conv3DNdhwcDhwcfOp definitions
//===----------------------------------------------------------------------===//

Conv3DNdhwcDhwcfOpAdaptor::Conv3DNdhwcDhwcfOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

Conv3DNdhwcDhwcfOpAdaptor::Conv3DNdhwcDhwcfOpAdaptor(Conv3DNdhwcDhwcfOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange Conv3DNdhwcDhwcfOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> Conv3DNdhwcDhwcfOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange Conv3DNdhwcDhwcfOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange Conv3DNdhwcDhwcfOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange Conv3DNdhwcDhwcfOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr Conv3DNdhwcDhwcfOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr Conv3DNdhwcDhwcfOpAdaptor::strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr Conv3DNdhwcDhwcfOpAdaptor::dilations() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("dilations").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::RegionRange Conv3DNdhwcDhwcfOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &Conv3DNdhwcDhwcfOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult Conv3DNdhwcDhwcfOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    {
  auto tblgen_strides = odsAttrs.get("strides");
  if (!tblgen_strides) return emitError(loc, "'linalg.conv_3d_ndhwc_dhwcf' op ""requires attribute 'strides'");
    if (!((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({3}))))) return emitError(loc, "'linalg.conv_3d_ndhwc_dhwcf' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [3]");
  }
  {
  auto tblgen_dilations = odsAttrs.get("dilations");
  if (!tblgen_dilations) return emitError(loc, "'linalg.conv_3d_ndhwc_dhwcf' op ""requires attribute 'dilations'");
    if (!((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({3}))))) return emitError(loc, "'linalg.conv_3d_ndhwc_dhwcf' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [3]");
  }
  return ::mlir::success();
}





















std::pair<unsigned, unsigned> Conv3DNdhwcDhwcfOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range Conv3DNdhwcDhwcfOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range Conv3DNdhwcDhwcfOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range Conv3DNdhwcDhwcfOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange Conv3DNdhwcDhwcfOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv3DNdhwcDhwcfOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> Conv3DNdhwcDhwcfOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range Conv3DNdhwcDhwcfOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range Conv3DNdhwcDhwcfOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &Conv3DNdhwcDhwcfOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr Conv3DNdhwcDhwcfOp::stridesAttr() {
  return (*this)->getAttr(stridesAttrName()).template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv3DNdhwcDhwcfOp::strides() {
  auto attr = stridesAttr();
  return attr;
}

::mlir::DenseIntElementsAttr Conv3DNdhwcDhwcfOp::dilationsAttr() {
  return (*this)->getAttr(dilationsAttrName()).template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr Conv3DNdhwcDhwcfOp::dilations() {
  auto attr = dilationsAttr();
  return attr;
}

void Conv3DNdhwcDhwcfOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void Conv3DNdhwcDhwcfOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

void Conv3DNdhwcDhwcfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<Conv3DNdhwcDhwcfOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void Conv3DNdhwcDhwcfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<Conv3DNdhwcDhwcfOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void Conv3DNdhwcDhwcfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void Conv3DNdhwcDhwcfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes ) {
    odsState.addOperands(inputs);
    odsState.addOperands(outputs);
    odsState.addTypes(resultTensorTypes);
    odsState.addAttribute(
      "operand_segment_sizes",
      odsBuilder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<Conv3DNdhwcDhwcfOp>(
      odsBuilder,
      odsState,
      TypeRange(inputs),
      TypeRange(outputs));
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    odsState.addAttributes(attributes);
  
}

::mlir::ParseResult Conv3DNdhwcDhwcfOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<Conv3DNdhwcDhwcfOp>(parser, result);
}

void Conv3DNdhwcDhwcfOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult Conv3DNdhwcDhwcfOp::verify() {
  if (::mlir::failed(Conv3DNdhwcDhwcfOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
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





} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::Conv3DNdhwcDhwcfOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Conv3DOp definitions
//===----------------------------------------------------------------------===//

Conv3DOpAdaptor::Conv3DOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

Conv3DOpAdaptor::Conv3DOpAdaptor(Conv3DOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange Conv3DOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> Conv3DOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange Conv3DOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange Conv3DOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange Conv3DOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr Conv3DOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange Conv3DOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &Conv3DOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult Conv3DOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    return ::mlir::success();
}













std::pair<unsigned, unsigned> Conv3DOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range Conv3DOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range Conv3DOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range Conv3DOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange Conv3DOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Conv3DOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> Conv3DOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range Conv3DOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range Conv3DOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &Conv3DOp::region() {
  return (*this)->getRegion(0);
}

void Conv3DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<Conv3DOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void Conv3DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<Conv3DOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void Conv3DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::ParseResult Conv3DOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<Conv3DOp>(parser, result);
}

void Conv3DOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult Conv3DOp::verify() {
  if (::mlir::failed(Conv3DOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
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





} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::Conv3DOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::ConvOp definitions
//===----------------------------------------------------------------------===//

ConvOpAdaptor::ConvOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

ConvOpAdaptor::ConvOpAdaptor(ConvOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange ConvOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ConvOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ConvOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ConvOpAdaptor::filter() {
  return *getODSOperands(0).begin();
}

::mlir::Value ConvOpAdaptor::input() {
  return *getODSOperands(1).begin();
}

::mlir::Value ConvOpAdaptor::output() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr ConvOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr ConvOpAdaptor::strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("strides").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr ConvOpAdaptor::dilations() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("dilations").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr ConvOpAdaptor::padding() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("padding").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::LogicalResult ConvOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_strides = odsAttrs.get("strides");
  if (tblgen_strides) {
    if (!(((tblgen_strides.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_strides.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); })))) return emitError(loc, "'linalg.conv' op ""attribute 'strides' failed to satisfy constraint: 64-bit integer array attribute");
  }
  }
  {
  auto tblgen_dilations = odsAttrs.get("dilations");
  if (tblgen_dilations) {
    if (!(((tblgen_dilations.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_dilations.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); })))) return emitError(loc, "'linalg.conv' op ""attribute 'dilations' failed to satisfy constraint: 64-bit integer array attribute");
  }
  }
  {
  auto tblgen_padding = odsAttrs.get("padding");
  if (tblgen_padding) {
    if (!(((tblgen_padding.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_padding.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) return emitError(loc, "'linalg.conv' op ""attribute 'padding' failed to satisfy constraint: 64-bit signless integer elements attribute");
  }
  }
  return ::mlir::success();
}





















std::pair<unsigned, unsigned> ConvOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ConvOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ConvOp::filter() {
  return *getODSOperands(0).begin();
}

::mlir::Value ConvOp::input() {
  return *getODSOperands(1).begin();
}

::mlir::Value ConvOp::output() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange ConvOp::filterMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ConvOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ConvOp::outputMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ConvOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ConvOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::ArrayAttr ConvOp::stridesAttr() {
  return (*this)->getAttr(stridesAttrName()).template dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > ConvOp::strides() {
  auto attr = stridesAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr ConvOp::dilationsAttr() {
  return (*this)->getAttr(dilationsAttrName()).template dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > ConvOp::dilations() {
  auto attr = dilationsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr ConvOp::paddingAttr() {
  return (*this)->getAttr(paddingAttrName()).template dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > ConvOp::padding() {
  auto attr = paddingAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void ConvOp::stridesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void ConvOp::dilationsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

void ConvOp::paddingAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(paddingAttrName(), attr);
}

::mlir::Attribute ConvOp::removeStridesAttr() {
  return (*this)->removeAttr(stridesAttrName());
}

::mlir::Attribute ConvOp::removeDilationsAttr() {
  return (*this)->removeAttr(dilationsAttrName());
}

::mlir::Attribute ConvOp::removePaddingAttr() {
  return (*this)->removeAttr(paddingAttrName());
}

void ConvOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value filter, ::mlir::Value input, ::mlir::Value output, /*optional*/::mlir::ArrayAttr strides, /*optional*/::mlir::ArrayAttr dilations, /*optional*/::mlir::DenseIntElementsAttr padding) {
  odsState.addOperands(filter);
  odsState.addOperands(input);
  odsState.addOperands(output);
  if (strides) {
  odsState.addAttribute(stridesAttrName(odsState.name), strides);
  }
  if (dilations) {
  odsState.addAttribute(dilationsAttrName(odsState.name), dilations);
  }
  if (padding) {
  odsState.addAttribute(paddingAttrName(odsState.name), padding);
  }
}

void ConvOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value filter, ::mlir::Value input, ::mlir::Value output, /*optional*/::mlir::ArrayAttr strides, /*optional*/::mlir::ArrayAttr dilations, /*optional*/::mlir::DenseIntElementsAttr padding) {
  odsState.addOperands(filter);
  odsState.addOperands(input);
  odsState.addOperands(output);
  if (strides) {
  odsState.addAttribute(stridesAttrName(odsState.name), strides);
  }
  if (dilations) {
  odsState.addAttribute(dilationsAttrName(odsState.name), dilations);
  }
  if (padding) {
  odsState.addAttribute(paddingAttrName(odsState.name), padding);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConvOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ConvOp::verify() {
  if (::mlir::failed(ConvOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::verify(*this);
}





::mlir::ParseResult ConvOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> allOperands;
  ::mlir::SmallVector<::mlir::Type, 1> allOperandTypes;
  if (parser.parseLParen())
    return ::mlir::failure();
  ::llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(allOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(allOperandTypes))
    return ::mlir::failure();
  if (parser.resolveOperands(allOperands, allOperandTypes, allOperandLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ConvOp::print(::mlir::OpAsmPrinter &p) {
  p << "(";
  p << getOperation()->getOperands();
  p << ")";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << getOperation()->getOperandTypes();
}

} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::ConvOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::CopyOp definitions
//===----------------------------------------------------------------------===//

CopyOpAdaptor::CopyOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

CopyOpAdaptor::CopyOpAdaptor(CopyOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange CopyOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CopyOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CopyOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CopyOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::Value CopyOpAdaptor::output() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr CopyOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::AffineMapAttr CopyOpAdaptor::inputPermutation() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::AffineMapAttr attr = odsAttrs.get("inputPermutation").dyn_cast_or_null<::mlir::AffineMapAttr>();
  return attr;
}

::mlir::AffineMapAttr CopyOpAdaptor::outputPermutation() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::AffineMapAttr attr = odsAttrs.get("outputPermutation").dyn_cast_or_null<::mlir::AffineMapAttr>();
  return attr;
}

::mlir::RegionRange CopyOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &CopyOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult CopyOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_inputPermutation = odsAttrs.get("inputPermutation");
  if (tblgen_inputPermutation) {
    if (!((tblgen_inputPermutation.isa<::mlir::AffineMapAttr>()))) return emitError(loc, "'linalg.copy' op ""attribute 'inputPermutation' failed to satisfy constraint: AffineMap attribute");
  }
  }
  {
  auto tblgen_outputPermutation = odsAttrs.get("outputPermutation");
  if (tblgen_outputPermutation) {
    if (!((tblgen_outputPermutation.isa<::mlir::AffineMapAttr>()))) return emitError(loc, "'linalg.copy' op ""attribute 'outputPermutation' failed to satisfy constraint: AffineMap attribute");
  }
  }
  return ::mlir::success();
}

















std::pair<unsigned, unsigned> CopyOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CopyOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CopyOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::Value CopyOp::output() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange CopyOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange CopyOp::outputMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CopyOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CopyOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &CopyOp::region() {
  return (*this)->getRegion(0);
}

::mlir::AffineMapAttr CopyOp::inputPermutationAttr() {
  return (*this)->getAttr(inputPermutationAttrName()).template dyn_cast_or_null<::mlir::AffineMapAttr>();
}

::llvm::Optional< ::mlir::AffineMap > CopyOp::inputPermutation() {
  auto attr = inputPermutationAttr();
  return attr ? ::llvm::Optional< ::mlir::AffineMap >(attr.getValue()) : (::llvm::None);
}

::mlir::AffineMapAttr CopyOp::outputPermutationAttr() {
  return (*this)->getAttr(outputPermutationAttrName()).template dyn_cast_or_null<::mlir::AffineMapAttr>();
}

::llvm::Optional< ::mlir::AffineMap > CopyOp::outputPermutation() {
  auto attr = outputPermutationAttr();
  return attr ? ::llvm::Optional< ::mlir::AffineMap >(attr.getValue()) : (::llvm::None);
}

void CopyOp::inputPermutationAttr(::mlir::AffineMapAttr attr) {
  (*this)->setAttr(inputPermutationAttrName(), attr);
}

void CopyOp::outputPermutationAttr(::mlir::AffineMapAttr attr) {
  (*this)->setAttr(outputPermutationAttrName(), attr);
}

::mlir::Attribute CopyOp::removeInputPermutationAttr() {
  return (*this)->removeAttr(inputPermutationAttrName());
}

::mlir::Attribute CopyOp::removeOutputPermutationAttr() {
  return (*this)->removeAttr(outputPermutationAttrName());
}



::mlir::LogicalResult CopyOp::verify() {
  if (::mlir::failed(CopyOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps3(getOperation(), v.getType(), "operand", index)))
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
  return ::verify(*this);
}







::mlir::ParseResult CopyOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType inputRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> inputOperands(inputRawOperands);  ::llvm::SMLoc inputOperandsLoc;
  (void)inputOperandsLoc;
  ::mlir::OpAsmParser::OperandType outputRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> outputOperands(outputRawOperands);  ::llvm::SMLoc outputOperandsLoc;
  (void)outputOperandsLoc;
  ::mlir::Type inputRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> inputTypes(inputRawTypes);
  ::mlir::Type outputRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> outputTypes(outputRawTypes);
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();
  if (parser.parseLParen())
    return ::mlir::failure();

  inputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  outputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(outputRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(inputRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(outputRawTypes[0]))
    return ::mlir::failure();
  {
    if (parseCopyOpRegion(parser, *regionRegion, inputRawTypes[0], inputRawTypes[0]))
      return ::mlir::failure();
  }
  if (parser.resolveOperands(inputOperands, inputTypes, inputOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(outputOperands, outputTypes, outputOperandsLoc, result.operands))
    return ::mlir::failure();
  result.addRegion(std::move(regionRegion));
  return ::mlir::success();
}

void CopyOp::print(::mlir::OpAsmPrinter &p) {
  p << "(";
  p << input();
  p << ",";
  p << ' ';
  p << output();
  p << ")";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(input().getType());
  p << ",";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(output().getType());
  p << ' ';
  printCopyOpRegion(p, *this, region(), input().getType(), input().getType());
}

} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::CopyOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::DepthwiseConv2DNhwOp definitions
//===----------------------------------------------------------------------===//

DepthwiseConv2DNhwOpAdaptor::DepthwiseConv2DNhwOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

DepthwiseConv2DNhwOpAdaptor::DepthwiseConv2DNhwOpAdaptor(DepthwiseConv2DNhwOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange DepthwiseConv2DNhwOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> DepthwiseConv2DNhwOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange DepthwiseConv2DNhwOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange DepthwiseConv2DNhwOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange DepthwiseConv2DNhwOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr DepthwiseConv2DNhwOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwOpAdaptor::strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwOpAdaptor::dilations() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("dilations").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::RegionRange DepthwiseConv2DNhwOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &DepthwiseConv2DNhwOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult DepthwiseConv2DNhwOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    {
  auto tblgen_strides = odsAttrs.get("strides");
  if (!tblgen_strides) return emitError(loc, "'linalg.depthwise_conv2D_nhw' op ""requires attribute 'strides'");
    if (!((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2}))))) return emitError(loc, "'linalg.depthwise_conv2D_nhw' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  {
  auto tblgen_dilations = odsAttrs.get("dilations");
  if (!tblgen_dilations) return emitError(loc, "'linalg.depthwise_conv2D_nhw' op ""requires attribute 'dilations'");
    if (!((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2}))))) return emitError(loc, "'linalg.depthwise_conv2D_nhw' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  return ::mlir::success();
}





















std::pair<unsigned, unsigned> DepthwiseConv2DNhwOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range DepthwiseConv2DNhwOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range DepthwiseConv2DNhwOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range DepthwiseConv2DNhwOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange DepthwiseConv2DNhwOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv2DNhwOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> DepthwiseConv2DNhwOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range DepthwiseConv2DNhwOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range DepthwiseConv2DNhwOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &DepthwiseConv2DNhwOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwOp::stridesAttr() {
  return (*this)->getAttr(stridesAttrName()).template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwOp::strides() {
  auto attr = stridesAttr();
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwOp::dilationsAttr() {
  return (*this)->getAttr(dilationsAttrName()).template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwOp::dilations() {
  auto attr = dilationsAttr();
  return attr;
}

void DepthwiseConv2DNhwOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void DepthwiseConv2DNhwOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

void DepthwiseConv2DNhwOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<DepthwiseConv2DNhwOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void DepthwiseConv2DNhwOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<DepthwiseConv2DNhwOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void DepthwiseConv2DNhwOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void DepthwiseConv2DNhwOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes ) {
    odsState.addOperands(inputs);
    odsState.addOperands(outputs);
    odsState.addTypes(resultTensorTypes);
    odsState.addAttribute(
      "operand_segment_sizes",
      odsBuilder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<DepthwiseConv2DNhwOp>(
      odsBuilder,
      odsState,
      TypeRange(inputs),
      TypeRange(outputs));
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    odsState.addAttributes(attributes);
  
}

::mlir::ParseResult DepthwiseConv2DNhwOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<DepthwiseConv2DNhwOp>(parser, result);
}

void DepthwiseConv2DNhwOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult DepthwiseConv2DNhwOp::verify() {
  if (::mlir::failed(DepthwiseConv2DNhwOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
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





} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::DepthwiseConv2DNhwOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::DepthwiseConv2DNhwQOp definitions
//===----------------------------------------------------------------------===//

DepthwiseConv2DNhwQOpAdaptor::DepthwiseConv2DNhwQOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

DepthwiseConv2DNhwQOpAdaptor::DepthwiseConv2DNhwQOpAdaptor(DepthwiseConv2DNhwQOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange DepthwiseConv2DNhwQOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> DepthwiseConv2DNhwQOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange DepthwiseConv2DNhwQOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange DepthwiseConv2DNhwQOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange DepthwiseConv2DNhwQOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr DepthwiseConv2DNhwQOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwQOpAdaptor::strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwQOpAdaptor::dilations() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("dilations").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::RegionRange DepthwiseConv2DNhwQOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &DepthwiseConv2DNhwQOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult DepthwiseConv2DNhwQOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    {
  auto tblgen_strides = odsAttrs.get("strides");
  if (!tblgen_strides) return emitError(loc, "'linalg.depthwise_conv2D_nhw_q' op ""requires attribute 'strides'");
    if (!((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2}))))) return emitError(loc, "'linalg.depthwise_conv2D_nhw_q' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  {
  auto tblgen_dilations = odsAttrs.get("dilations");
  if (!tblgen_dilations) return emitError(loc, "'linalg.depthwise_conv2D_nhw_q' op ""requires attribute 'dilations'");
    if (!((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2}))))) return emitError(loc, "'linalg.depthwise_conv2D_nhw_q' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  return ::mlir::success();
}





















std::pair<unsigned, unsigned> DepthwiseConv2DNhwQOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range DepthwiseConv2DNhwQOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range DepthwiseConv2DNhwQOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range DepthwiseConv2DNhwQOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange DepthwiseConv2DNhwQOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv2DNhwQOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> DepthwiseConv2DNhwQOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range DepthwiseConv2DNhwQOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range DepthwiseConv2DNhwQOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &DepthwiseConv2DNhwQOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwQOp::stridesAttr() {
  return (*this)->getAttr(stridesAttrName()).template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwQOp::strides() {
  auto attr = stridesAttr();
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwQOp::dilationsAttr() {
  return (*this)->getAttr(dilationsAttrName()).template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwQOp::dilations() {
  auto attr = dilationsAttr();
  return attr;
}

void DepthwiseConv2DNhwQOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void DepthwiseConv2DNhwQOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

void DepthwiseConv2DNhwQOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<DepthwiseConv2DNhwQOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void DepthwiseConv2DNhwQOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<DepthwiseConv2DNhwQOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void DepthwiseConv2DNhwQOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void DepthwiseConv2DNhwQOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes ) {
    odsState.addOperands(inputs);
    odsState.addOperands(outputs);
    odsState.addTypes(resultTensorTypes);
    odsState.addAttribute(
      "operand_segment_sizes",
      odsBuilder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<DepthwiseConv2DNhwQOp>(
      odsBuilder,
      odsState,
      TypeRange(inputs),
      TypeRange(outputs));
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    odsState.addAttributes(attributes);
  
}

::mlir::ParseResult DepthwiseConv2DNhwQOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<DepthwiseConv2DNhwQOp>(parser, result);
}

void DepthwiseConv2DNhwQOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult DepthwiseConv2DNhwQOp::verify() {
  if (::mlir::failed(DepthwiseConv2DNhwQOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
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





} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::DepthwiseConv2DNhwQOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::DepthwiseConv2DNhwcOp definitions
//===----------------------------------------------------------------------===//

DepthwiseConv2DNhwcOpAdaptor::DepthwiseConv2DNhwcOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

DepthwiseConv2DNhwcOpAdaptor::DepthwiseConv2DNhwcOpAdaptor(DepthwiseConv2DNhwcOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange DepthwiseConv2DNhwcOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> DepthwiseConv2DNhwcOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange DepthwiseConv2DNhwcOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange DepthwiseConv2DNhwcOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange DepthwiseConv2DNhwcOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr DepthwiseConv2DNhwcOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcOpAdaptor::strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcOpAdaptor::dilations() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("dilations").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::RegionRange DepthwiseConv2DNhwcOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &DepthwiseConv2DNhwcOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult DepthwiseConv2DNhwcOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    {
  auto tblgen_strides = odsAttrs.get("strides");
  if (!tblgen_strides) return emitError(loc, "'linalg.depthwise_conv2D_nhwc' op ""requires attribute 'strides'");
    if (!((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2}))))) return emitError(loc, "'linalg.depthwise_conv2D_nhwc' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  {
  auto tblgen_dilations = odsAttrs.get("dilations");
  if (!tblgen_dilations) return emitError(loc, "'linalg.depthwise_conv2D_nhwc' op ""requires attribute 'dilations'");
    if (!((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2}))))) return emitError(loc, "'linalg.depthwise_conv2D_nhwc' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  return ::mlir::success();
}





















std::pair<unsigned, unsigned> DepthwiseConv2DNhwcOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range DepthwiseConv2DNhwcOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range DepthwiseConv2DNhwcOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range DepthwiseConv2DNhwcOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange DepthwiseConv2DNhwcOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv2DNhwcOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> DepthwiseConv2DNhwcOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range DepthwiseConv2DNhwcOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range DepthwiseConv2DNhwcOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &DepthwiseConv2DNhwcOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcOp::stridesAttr() {
  return (*this)->getAttr(stridesAttrName()).template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcOp::strides() {
  auto attr = stridesAttr();
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcOp::dilationsAttr() {
  return (*this)->getAttr(dilationsAttrName()).template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcOp::dilations() {
  auto attr = dilationsAttr();
  return attr;
}

void DepthwiseConv2DNhwcOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void DepthwiseConv2DNhwcOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

void DepthwiseConv2DNhwcOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<DepthwiseConv2DNhwcOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void DepthwiseConv2DNhwcOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<DepthwiseConv2DNhwcOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void DepthwiseConv2DNhwcOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void DepthwiseConv2DNhwcOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes ) {
    odsState.addOperands(inputs);
    odsState.addOperands(outputs);
    odsState.addTypes(resultTensorTypes);
    odsState.addAttribute(
      "operand_segment_sizes",
      odsBuilder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<DepthwiseConv2DNhwcOp>(
      odsBuilder,
      odsState,
      TypeRange(inputs),
      TypeRange(outputs));
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    odsState.addAttributes(attributes);
  
}

::mlir::ParseResult DepthwiseConv2DNhwcOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<DepthwiseConv2DNhwcOp>(parser, result);
}

void DepthwiseConv2DNhwcOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult DepthwiseConv2DNhwcOp::verify() {
  if (::mlir::failed(DepthwiseConv2DNhwcOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
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





} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::DepthwiseConv2DNhwcOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::DepthwiseConv2DNhwcQOp definitions
//===----------------------------------------------------------------------===//

DepthwiseConv2DNhwcQOpAdaptor::DepthwiseConv2DNhwcQOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

DepthwiseConv2DNhwcQOpAdaptor::DepthwiseConv2DNhwcQOpAdaptor(DepthwiseConv2DNhwcQOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange DepthwiseConv2DNhwcQOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> DepthwiseConv2DNhwcQOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange DepthwiseConv2DNhwcQOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange DepthwiseConv2DNhwcQOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange DepthwiseConv2DNhwcQOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr DepthwiseConv2DNhwcQOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcQOpAdaptor::strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcQOpAdaptor::dilations() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("dilations").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::RegionRange DepthwiseConv2DNhwcQOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &DepthwiseConv2DNhwcQOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult DepthwiseConv2DNhwcQOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    {
  auto tblgen_strides = odsAttrs.get("strides");
  if (!tblgen_strides) return emitError(loc, "'linalg.depthwise_conv2D_nhwc_q' op ""requires attribute 'strides'");
    if (!((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2}))))) return emitError(loc, "'linalg.depthwise_conv2D_nhwc_q' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  {
  auto tblgen_dilations = odsAttrs.get("dilations");
  if (!tblgen_dilations) return emitError(loc, "'linalg.depthwise_conv2D_nhwc_q' op ""requires attribute 'dilations'");
    if (!((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2}))))) return emitError(loc, "'linalg.depthwise_conv2D_nhwc_q' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  return ::mlir::success();
}





















std::pair<unsigned, unsigned> DepthwiseConv2DNhwcQOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range DepthwiseConv2DNhwcQOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range DepthwiseConv2DNhwcQOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range DepthwiseConv2DNhwcQOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange DepthwiseConv2DNhwcQOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv2DNhwcQOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> DepthwiseConv2DNhwcQOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range DepthwiseConv2DNhwcQOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range DepthwiseConv2DNhwcQOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &DepthwiseConv2DNhwcQOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcQOp::stridesAttr() {
  return (*this)->getAttr(stridesAttrName()).template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcQOp::strides() {
  auto attr = stridesAttr();
  return attr;
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcQOp::dilationsAttr() {
  return (*this)->getAttr(dilationsAttrName()).template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DepthwiseConv2DNhwcQOp::dilations() {
  auto attr = dilationsAttr();
  return attr;
}

void DepthwiseConv2DNhwcQOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void DepthwiseConv2DNhwcQOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

void DepthwiseConv2DNhwcQOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<DepthwiseConv2DNhwcQOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void DepthwiseConv2DNhwcQOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<DepthwiseConv2DNhwcQOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void DepthwiseConv2DNhwcQOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void DepthwiseConv2DNhwcQOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes ) {
    odsState.addOperands(inputs);
    odsState.addOperands(outputs);
    odsState.addTypes(resultTensorTypes);
    odsState.addAttribute(
      "operand_segment_sizes",
      odsBuilder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<DepthwiseConv2DNhwcQOp>(
      odsBuilder,
      odsState,
      TypeRange(inputs),
      TypeRange(outputs));
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    odsState.addAttributes(attributes);
  
}

::mlir::ParseResult DepthwiseConv2DNhwcQOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<DepthwiseConv2DNhwcQOp>(parser, result);
}

void DepthwiseConv2DNhwcQOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult DepthwiseConv2DNhwcQOp::verify() {
  if (::mlir::failed(DepthwiseConv2DNhwcQOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
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





} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::DepthwiseConv2DNhwcQOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::DotOp definitions
//===----------------------------------------------------------------------===//

DotOpAdaptor::DotOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

DotOpAdaptor::DotOpAdaptor(DotOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange DotOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> DotOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange DotOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange DotOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange DotOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr DotOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange DotOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &DotOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult DotOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    return ::mlir::success();
}













std::pair<unsigned, unsigned> DotOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range DotOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range DotOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range DotOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange DotOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DotOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> DotOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range DotOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range DotOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &DotOp::region() {
  return (*this)->getRegion(0);
}

void DotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<DotOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void DotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<DotOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void DotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::ParseResult DotOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<DotOp>(parser, result);
}

void DotOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult DotOp::verify() {
  if (::mlir::failed(DotOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
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





} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::DotOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::FillOp definitions
//===----------------------------------------------------------------------===//

FillOpAdaptor::FillOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

FillOpAdaptor::FillOpAdaptor(FillOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange FillOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FillOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FillOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FillOpAdaptor::value() {
  return *getODSOperands(0).begin();
}

::mlir::Value FillOpAdaptor::output() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr FillOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange FillOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &FillOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult FillOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}





std::pair<unsigned, unsigned> FillOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FillOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FillOp::value() {
  return *getODSOperands(0).begin();
}

::mlir::Value FillOp::output() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange FillOp::valueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange FillOp::outputMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> FillOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range FillOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FillOp::result() {
  auto results = getODSResults(0);
  return results.empty() ? ::mlir::Value() : *results.begin();
}

::mlir::Region &FillOp::region() {
  return (*this)->getRegion(0);
}



void FillOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type result, ::mlir::Value value, ::mlir::Value output) {
  odsState.addOperands(value);
  odsState.addOperands(output);
  (void)odsState.addRegion();
  if (result)
    odsState.addTypes(result);
}

void FillOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value output) {
  odsState.addOperands(value);
  odsState.addOperands(output);
  (void)odsState.addRegion();
  odsState.addTypes(resultTypes);
}

void FillOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult FillOp::verify() {
  if (::mlir::failed(FillOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps4(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "operand", index)))
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
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
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
  return ::verify(*this);
}





::mlir::ParseResult FillOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::OpAsmParser::OperandType outputRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> outputOperands(outputRawOperands);  ::llvm::SMLoc outputOperandsLoc;
  (void)outputOperandsLoc;
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);
  ::mlir::Type outputRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> outputTypes(outputRawTypes);
  ::mlir::SmallVector<::mlir::Type, 1> resultTypes;
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();
  if (parser.parseLParen())
    return ::mlir::failure();

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  outputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(outputRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(valueRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(outputRawTypes[0]))
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalArrow())) {

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      resultTypes.push_back(optionalType);
    }
  }
  }
  {
    if (parseFillOpRegion(parser, *regionRegion, valueRawTypes[0], outputRawTypes[0]))
      return ::mlir::failure();
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(outputOperands, outputTypes, outputOperandsLoc, result.operands))
    return ::mlir::failure();
  result.addRegion(std::move(regionRegion));
  return ::mlir::success();
}

void FillOp::print(::mlir::OpAsmPrinter &p) {
  p << "(";
  p << value();
  p << ",";
  p << ' ';
  p << output();
  p << ")";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(value().getType());
  p << ",";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(output().getType());
  if (result()) {
  p << ' ' << "->";
  p << ' ';
  p << (result() ? ::llvm::ArrayRef<::mlir::Type>(result().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  }
  p << ' ';
  printFillOpRegion(p, *this, region(), value().getType(), output().getType());
}

} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::FillOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::FillRng2DOp definitions
//===----------------------------------------------------------------------===//

FillRng2DOpAdaptor::FillRng2DOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

FillRng2DOpAdaptor::FillRng2DOpAdaptor(FillRng2DOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange FillRng2DOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FillRng2DOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange FillRng2DOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange FillRng2DOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange FillRng2DOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr FillRng2DOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange FillRng2DOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &FillRng2DOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult FillRng2DOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    return ::mlir::success();
}













std::pair<unsigned, unsigned> FillRng2DOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range FillRng2DOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range FillRng2DOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range FillRng2DOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange FillRng2DOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange FillRng2DOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> FillRng2DOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range FillRng2DOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range FillRng2DOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &FillRng2DOp::region() {
  return (*this)->getRegion(0);
}

void FillRng2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<FillRng2DOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void FillRng2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<FillRng2DOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void FillRng2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::ParseResult FillRng2DOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<FillRng2DOp>(parser, result);
}

void FillRng2DOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult FillRng2DOp::verify() {
  if (::mlir::failed(FillRng2DOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
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





} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::FillRng2DOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::GenericOp definitions
//===----------------------------------------------------------------------===//

GenericOpAdaptor::GenericOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

GenericOpAdaptor::GenericOpAdaptor(GenericOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange GenericOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GenericOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange GenericOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange GenericOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange GenericOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr GenericOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr GenericOpAdaptor::indexing_maps() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("indexing_maps").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr GenericOpAdaptor::iterator_types() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("iterator_types").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::StringAttr GenericOpAdaptor::doc() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("doc").dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::mlir::StringAttr GenericOpAdaptor::library_call() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("library_call").dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::mlir::RegionRange GenericOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &GenericOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult GenericOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    {
  auto tblgen_indexing_maps = odsAttrs.get("indexing_maps");
  if (!tblgen_indexing_maps) return emitError(loc, "'linalg.generic' op ""requires attribute 'indexing_maps'");
    if (!(((tblgen_indexing_maps.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_indexing_maps.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return (attr.isa<::mlir::AffineMapAttr>()); })))) return emitError(loc, "'linalg.generic' op ""attribute 'indexing_maps' failed to satisfy constraint: AffineMap array attribute");
  }
  {
  auto tblgen_iterator_types = odsAttrs.get("iterator_types");
  if (!tblgen_iterator_types) return emitError(loc, "'linalg.generic' op ""requires attribute 'iterator_types'");
    if (!((tblgen_iterator_types.isa<::mlir::ArrayAttr>()))) return emitError(loc, "'linalg.generic' op ""attribute 'iterator_types' failed to satisfy constraint: array attribute");
  }
  {
  auto tblgen_doc = odsAttrs.get("doc");
  if (tblgen_doc) {
    if (!((tblgen_doc.isa<::mlir::StringAttr>()))) return emitError(loc, "'linalg.generic' op ""attribute 'doc' failed to satisfy constraint: string attribute");
  }
  }
  {
  auto tblgen_library_call = odsAttrs.get("library_call");
  if (tblgen_library_call) {
    if (!((tblgen_library_call.isa<::mlir::StringAttr>()))) return emitError(loc, "'linalg.generic' op ""attribute 'library_call' failed to satisfy constraint: string attribute");
  }
  }
  return ::mlir::success();
}





























std::pair<unsigned, unsigned> GenericOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range GenericOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range GenericOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range GenericOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange GenericOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange GenericOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> GenericOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range GenericOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range GenericOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &GenericOp::region() {
  return (*this)->getRegion(0);
}

::mlir::ArrayAttr GenericOp::indexing_mapsAttr() {
  return (*this)->getAttr(indexing_mapsAttrName()).template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr GenericOp::indexing_maps() {
  auto attr = indexing_mapsAttr();
  return attr;
}

::mlir::ArrayAttr GenericOp::iterator_typesAttr() {
  return (*this)->getAttr(iterator_typesAttrName()).template cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr GenericOp::iterator_types() {
  auto attr = iterator_typesAttr();
  return attr;
}

::mlir::StringAttr GenericOp::docAttr() {
  return (*this)->getAttr(docAttrName()).template dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > GenericOp::doc() {
  auto attr = docAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::StringAttr GenericOp::library_callAttr() {
  return (*this)->getAttr(library_callAttrName()).template dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > GenericOp::library_call() {
  auto attr = library_callAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

void GenericOp::indexing_mapsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(indexing_mapsAttrName(), attr);
}

void GenericOp::iterator_typesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(iterator_typesAttrName(), attr);
}

void GenericOp::docAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(docAttrName(), attr);
}

void GenericOp::library_callAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(library_callAttrName(), attr);
}

::mlir::Attribute GenericOp::removeDocAttr() {
  return (*this)->removeAttr(docAttrName());
}

::mlir::Attribute GenericOp::removeLibrary_callAttr() {
  return (*this)->removeAttr(library_callAttrName());
}









void GenericOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange result_tensors, ::mlir::ValueRange inputs, ::mlir::ValueRange outputs, ::mlir::ArrayAttr indexing_maps, ::mlir::ArrayAttr iterator_types, /*optional*/::mlir::StringAttr doc, /*optional*/::mlir::StringAttr library_call) {
  odsState.addOperands(inputs);
  odsState.addOperands(outputs);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({static_cast<int32_t>(inputs.size()), static_cast<int32_t>(outputs.size())}));
  odsState.addAttribute(indexing_mapsAttrName(odsState.name), indexing_maps);
  odsState.addAttribute(iterator_typesAttrName(odsState.name), iterator_types);
  if (doc) {
  odsState.addAttribute(docAttrName(odsState.name), doc);
  }
  if (library_call) {
  odsState.addAttribute(library_callAttrName(odsState.name), library_call);
  }
  (void)odsState.addRegion();
  odsState.addTypes(result_tensors);
}

void GenericOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult GenericOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseGenericOp(parser, result);
}

void GenericOp::print(::mlir::OpAsmPrinter &p) {
  return ::print(p, *this);
}

::mlir::LogicalResult GenericOp::verify() {
  if (::mlir::failed(GenericOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
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
  return ::verify(*this);
}







} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::GenericOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::MatmulI8I8I32Op definitions
//===----------------------------------------------------------------------===//

MatmulI8I8I32OpAdaptor::MatmulI8I8I32OpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

MatmulI8I8I32OpAdaptor::MatmulI8I8I32OpAdaptor(MatmulI8I8I32Op&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange MatmulI8I8I32OpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MatmulI8I8I32OpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange MatmulI8I8I32OpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange MatmulI8I8I32OpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange MatmulI8I8I32OpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr MatmulI8I8I32OpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange MatmulI8I8I32OpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &MatmulI8I8I32OpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult MatmulI8I8I32OpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    return ::mlir::success();
}













std::pair<unsigned, unsigned> MatmulI8I8I32Op::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range MatmulI8I8I32Op::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range MatmulI8I8I32Op::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range MatmulI8I8I32Op::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange MatmulI8I8I32Op::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange MatmulI8I8I32Op::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> MatmulI8I8I32Op::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range MatmulI8I8I32Op::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range MatmulI8I8I32Op::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &MatmulI8I8I32Op::region() {
  return (*this)->getRegion(0);
}

void MatmulI8I8I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          odsState.addAttributes(attributes);
          createAndFillStructuredOpRegion<MatmulI8I8I32Op>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs));
        
}

void MatmulI8I8I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputs);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          odsState.addAttributes(attributes);
          createAndFillStructuredOpRegion<MatmulI8I8I32Op>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputs));
        
}

void MatmulI8I8I32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

::mlir::ParseResult MatmulI8I8I32Op::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<MatmulI8I8I32Op>(parser, result);
}

void MatmulI8I8I32Op::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult MatmulI8I8I32Op::verify() {
  if (::mlir::failed(MatmulI8I8I32OpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
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





} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::MatmulI8I8I32Op)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::MatmulOp definitions
//===----------------------------------------------------------------------===//

MatmulOpAdaptor::MatmulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

MatmulOpAdaptor::MatmulOpAdaptor(MatmulOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange MatmulOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MatmulOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange MatmulOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange MatmulOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange MatmulOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr MatmulOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange MatmulOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &MatmulOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult MatmulOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    return ::mlir::success();
}













std::pair<unsigned, unsigned> MatmulOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range MatmulOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range MatmulOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range MatmulOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange MatmulOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange MatmulOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> MatmulOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range MatmulOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range MatmulOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &MatmulOp::region() {
  return (*this)->getRegion(0);
}

void MatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<MatmulOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void MatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<MatmulOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void MatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::ParseResult MatmulOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<MatmulOp>(parser, result);
}

void MatmulOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult MatmulOp::verify() {
  if (::mlir::failed(MatmulOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
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





} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::MatmulOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::MatvecOp definitions
//===----------------------------------------------------------------------===//

MatvecOpAdaptor::MatvecOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

MatvecOpAdaptor::MatvecOpAdaptor(MatvecOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange MatvecOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MatvecOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange MatvecOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange MatvecOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange MatvecOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr MatvecOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange MatvecOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &MatvecOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult MatvecOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    return ::mlir::success();
}













std::pair<unsigned, unsigned> MatvecOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range MatvecOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range MatvecOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range MatvecOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange MatvecOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange MatvecOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> MatvecOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range MatvecOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range MatvecOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &MatvecOp::region() {
  return (*this)->getRegion(0);
}

void MatvecOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<MatvecOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void MatvecOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<MatvecOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void MatvecOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::ParseResult MatvecOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<MatvecOp>(parser, result);
}

void MatvecOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult MatvecOp::verify() {
  if (::mlir::failed(MatvecOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
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





} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::MatvecOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Mmt4DOp definitions
//===----------------------------------------------------------------------===//

Mmt4DOpAdaptor::Mmt4DOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

Mmt4DOpAdaptor::Mmt4DOpAdaptor(Mmt4DOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange Mmt4DOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> Mmt4DOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange Mmt4DOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange Mmt4DOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange Mmt4DOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr Mmt4DOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange Mmt4DOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &Mmt4DOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult Mmt4DOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    return ::mlir::success();
}













std::pair<unsigned, unsigned> Mmt4DOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range Mmt4DOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range Mmt4DOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range Mmt4DOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange Mmt4DOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange Mmt4DOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> Mmt4DOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range Mmt4DOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range Mmt4DOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &Mmt4DOp::region() {
  return (*this)->getRegion(0);
}

void Mmt4DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<Mmt4DOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void Mmt4DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<Mmt4DOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void Mmt4DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::ParseResult Mmt4DOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<Mmt4DOp>(parser, result);
}

void Mmt4DOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult Mmt4DOp::verify() {
  if (::mlir::failed(Mmt4DOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
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





} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::Mmt4DOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingMaxOp definitions
//===----------------------------------------------------------------------===//

PoolingMaxOpAdaptor::PoolingMaxOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

PoolingMaxOpAdaptor::PoolingMaxOpAdaptor(PoolingMaxOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange PoolingMaxOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PoolingMaxOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange PoolingMaxOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PoolingMaxOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::Value PoolingMaxOpAdaptor::windowDims() {
  return *getODSOperands(1).begin();
}

::mlir::Value PoolingMaxOpAdaptor::output() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr PoolingMaxOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr PoolingMaxOpAdaptor::strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("strides").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr PoolingMaxOpAdaptor::dilations() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("dilations").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr PoolingMaxOpAdaptor::padding() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("padding").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::LogicalResult PoolingMaxOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_strides = odsAttrs.get("strides");
  if (tblgen_strides) {
    if (!(((tblgen_strides.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_strides.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); })))) return emitError(loc, "'linalg.pooling_max' op ""attribute 'strides' failed to satisfy constraint: 64-bit integer array attribute");
  }
  }
  {
  auto tblgen_dilations = odsAttrs.get("dilations");
  if (tblgen_dilations) {
    if (!(((tblgen_dilations.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_dilations.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); })))) return emitError(loc, "'linalg.pooling_max' op ""attribute 'dilations' failed to satisfy constraint: 64-bit integer array attribute");
  }
  }
  {
  auto tblgen_padding = odsAttrs.get("padding");
  if (tblgen_padding) {
    if (!(((tblgen_padding.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_padding.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) return emitError(loc, "'linalg.pooling_max' op ""attribute 'padding' failed to satisfy constraint: 64-bit signless integer elements attribute");
  }
  }
  return ::mlir::success();
}





















std::pair<unsigned, unsigned> PoolingMaxOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range PoolingMaxOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PoolingMaxOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::Value PoolingMaxOp::windowDims() {
  return *getODSOperands(1).begin();
}

::mlir::Value PoolingMaxOp::output() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange PoolingMaxOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange PoolingMaxOp::windowDimsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange PoolingMaxOp::outputMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> PoolingMaxOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range PoolingMaxOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::ArrayAttr PoolingMaxOp::stridesAttr() {
  return (*this)->getAttr(stridesAttrName()).template dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > PoolingMaxOp::strides() {
  auto attr = stridesAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr PoolingMaxOp::dilationsAttr() {
  return (*this)->getAttr(dilationsAttrName()).template dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > PoolingMaxOp::dilations() {
  auto attr = dilationsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr PoolingMaxOp::paddingAttr() {
  return (*this)->getAttr(paddingAttrName()).template dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > PoolingMaxOp::padding() {
  auto attr = paddingAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void PoolingMaxOp::stridesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void PoolingMaxOp::dilationsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

void PoolingMaxOp::paddingAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(paddingAttrName(), attr);
}

::mlir::Attribute PoolingMaxOp::removeStridesAttr() {
  return (*this)->removeAttr(stridesAttrName());
}

::mlir::Attribute PoolingMaxOp::removeDilationsAttr() {
  return (*this)->removeAttr(dilationsAttrName());
}

::mlir::Attribute PoolingMaxOp::removePaddingAttr() {
  return (*this)->removeAttr(paddingAttrName());
}

void PoolingMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value windowDims, ::mlir::Value output, /*optional*/::mlir::ArrayAttr strides, /*optional*/::mlir::ArrayAttr dilations, /*optional*/::mlir::DenseIntElementsAttr padding) {
  odsState.addOperands(input);
  odsState.addOperands(windowDims);
  odsState.addOperands(output);
  if (strides) {
  odsState.addAttribute(stridesAttrName(odsState.name), strides);
  }
  if (dilations) {
  odsState.addAttribute(dilationsAttrName(odsState.name), dilations);
  }
  if (padding) {
  odsState.addAttribute(paddingAttrName(odsState.name), padding);
  }
}

void PoolingMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value windowDims, ::mlir::Value output, /*optional*/::mlir::ArrayAttr strides, /*optional*/::mlir::ArrayAttr dilations, /*optional*/::mlir::DenseIntElementsAttr padding) {
  odsState.addOperands(input);
  odsState.addOperands(windowDims);
  odsState.addOperands(output);
  if (strides) {
  odsState.addAttribute(stridesAttrName(odsState.name), strides);
  }
  if (dilations) {
  odsState.addAttribute(dilationsAttrName(odsState.name), dilations);
  }
  if (padding) {
  odsState.addAttribute(paddingAttrName(odsState.name), padding);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PoolingMaxOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult PoolingMaxOp::verify() {
  if (::mlir::failed(PoolingMaxOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::verify(*this);
}





::mlir::ParseResult PoolingMaxOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> allOperands;
  ::mlir::SmallVector<::mlir::Type, 1> allOperandTypes;
  if (parser.parseLParen())
    return ::mlir::failure();
  ::llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(allOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(allOperandTypes))
    return ::mlir::failure();
  if (parser.resolveOperands(allOperands, allOperandTypes, allOperandLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void PoolingMaxOp::print(::mlir::OpAsmPrinter &p) {
  p << "(";
  p << getOperation()->getOperands();
  p << ")";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << getOperation()->getOperandTypes();
}

} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingMaxOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingMinOp definitions
//===----------------------------------------------------------------------===//

PoolingMinOpAdaptor::PoolingMinOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

PoolingMinOpAdaptor::PoolingMinOpAdaptor(PoolingMinOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange PoolingMinOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PoolingMinOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange PoolingMinOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PoolingMinOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::Value PoolingMinOpAdaptor::windowDims() {
  return *getODSOperands(1).begin();
}

::mlir::Value PoolingMinOpAdaptor::output() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr PoolingMinOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr PoolingMinOpAdaptor::strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("strides").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr PoolingMinOpAdaptor::dilations() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("dilations").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr PoolingMinOpAdaptor::padding() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("padding").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::LogicalResult PoolingMinOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_strides = odsAttrs.get("strides");
  if (tblgen_strides) {
    if (!(((tblgen_strides.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_strides.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); })))) return emitError(loc, "'linalg.pooling_min' op ""attribute 'strides' failed to satisfy constraint: 64-bit integer array attribute");
  }
  }
  {
  auto tblgen_dilations = odsAttrs.get("dilations");
  if (tblgen_dilations) {
    if (!(((tblgen_dilations.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_dilations.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); })))) return emitError(loc, "'linalg.pooling_min' op ""attribute 'dilations' failed to satisfy constraint: 64-bit integer array attribute");
  }
  }
  {
  auto tblgen_padding = odsAttrs.get("padding");
  if (tblgen_padding) {
    if (!(((tblgen_padding.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_padding.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) return emitError(loc, "'linalg.pooling_min' op ""attribute 'padding' failed to satisfy constraint: 64-bit signless integer elements attribute");
  }
  }
  return ::mlir::success();
}





















std::pair<unsigned, unsigned> PoolingMinOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range PoolingMinOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PoolingMinOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::Value PoolingMinOp::windowDims() {
  return *getODSOperands(1).begin();
}

::mlir::Value PoolingMinOp::output() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange PoolingMinOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange PoolingMinOp::windowDimsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange PoolingMinOp::outputMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> PoolingMinOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range PoolingMinOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::ArrayAttr PoolingMinOp::stridesAttr() {
  return (*this)->getAttr(stridesAttrName()).template dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > PoolingMinOp::strides() {
  auto attr = stridesAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr PoolingMinOp::dilationsAttr() {
  return (*this)->getAttr(dilationsAttrName()).template dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > PoolingMinOp::dilations() {
  auto attr = dilationsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr PoolingMinOp::paddingAttr() {
  return (*this)->getAttr(paddingAttrName()).template dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > PoolingMinOp::padding() {
  auto attr = paddingAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void PoolingMinOp::stridesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void PoolingMinOp::dilationsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

void PoolingMinOp::paddingAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(paddingAttrName(), attr);
}

::mlir::Attribute PoolingMinOp::removeStridesAttr() {
  return (*this)->removeAttr(stridesAttrName());
}

::mlir::Attribute PoolingMinOp::removeDilationsAttr() {
  return (*this)->removeAttr(dilationsAttrName());
}

::mlir::Attribute PoolingMinOp::removePaddingAttr() {
  return (*this)->removeAttr(paddingAttrName());
}

void PoolingMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value windowDims, ::mlir::Value output, /*optional*/::mlir::ArrayAttr strides, /*optional*/::mlir::ArrayAttr dilations, /*optional*/::mlir::DenseIntElementsAttr padding) {
  odsState.addOperands(input);
  odsState.addOperands(windowDims);
  odsState.addOperands(output);
  if (strides) {
  odsState.addAttribute(stridesAttrName(odsState.name), strides);
  }
  if (dilations) {
  odsState.addAttribute(dilationsAttrName(odsState.name), dilations);
  }
  if (padding) {
  odsState.addAttribute(paddingAttrName(odsState.name), padding);
  }
}

void PoolingMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value windowDims, ::mlir::Value output, /*optional*/::mlir::ArrayAttr strides, /*optional*/::mlir::ArrayAttr dilations, /*optional*/::mlir::DenseIntElementsAttr padding) {
  odsState.addOperands(input);
  odsState.addOperands(windowDims);
  odsState.addOperands(output);
  if (strides) {
  odsState.addAttribute(stridesAttrName(odsState.name), strides);
  }
  if (dilations) {
  odsState.addAttribute(dilationsAttrName(odsState.name), dilations);
  }
  if (padding) {
  odsState.addAttribute(paddingAttrName(odsState.name), padding);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PoolingMinOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult PoolingMinOp::verify() {
  if (::mlir::failed(PoolingMinOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::verify(*this);
}





::mlir::ParseResult PoolingMinOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> allOperands;
  ::mlir::SmallVector<::mlir::Type, 1> allOperandTypes;
  if (parser.parseLParen())
    return ::mlir::failure();
  ::llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(allOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(allOperandTypes))
    return ::mlir::failure();
  if (parser.resolveOperands(allOperands, allOperandTypes, allOperandLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void PoolingMinOp::print(::mlir::OpAsmPrinter &p) {
  p << "(";
  p << getOperation()->getOperands();
  p << ")";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << getOperation()->getOperandTypes();
}

} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingMinOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNchwMaxOp definitions
//===----------------------------------------------------------------------===//

PoolingNchwMaxOpAdaptor::PoolingNchwMaxOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

PoolingNchwMaxOpAdaptor::PoolingNchwMaxOpAdaptor(PoolingNchwMaxOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange PoolingNchwMaxOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PoolingNchwMaxOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange PoolingNchwMaxOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange PoolingNchwMaxOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange PoolingNchwMaxOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr PoolingNchwMaxOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNchwMaxOpAdaptor::strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNchwMaxOpAdaptor::dilations() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("dilations").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::RegionRange PoolingNchwMaxOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &PoolingNchwMaxOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult PoolingNchwMaxOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    {
  auto tblgen_strides = odsAttrs.get("strides");
  if (!tblgen_strides) return emitError(loc, "'linalg.pooling_nchw_max' op ""requires attribute 'strides'");
    if (!((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2}))))) return emitError(loc, "'linalg.pooling_nchw_max' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  {
  auto tblgen_dilations = odsAttrs.get("dilations");
  if (!tblgen_dilations) return emitError(loc, "'linalg.pooling_nchw_max' op ""requires attribute 'dilations'");
    if (!((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2}))))) return emitError(loc, "'linalg.pooling_nchw_max' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  return ::mlir::success();
}





















std::pair<unsigned, unsigned> PoolingNchwMaxOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range PoolingNchwMaxOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range PoolingNchwMaxOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNchwMaxOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNchwMaxOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNchwMaxOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> PoolingNchwMaxOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range PoolingNchwMaxOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range PoolingNchwMaxOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNchwMaxOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNchwMaxOp::stridesAttr() {
  return (*this)->getAttr(stridesAttrName()).template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNchwMaxOp::strides() {
  auto attr = stridesAttr();
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNchwMaxOp::dilationsAttr() {
  return (*this)->getAttr(dilationsAttrName()).template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNchwMaxOp::dilations() {
  auto attr = dilationsAttr();
  return attr;
}

void PoolingNchwMaxOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void PoolingNchwMaxOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

void PoolingNchwMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<PoolingNchwMaxOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void PoolingNchwMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<PoolingNchwMaxOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void PoolingNchwMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void PoolingNchwMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes ) {
    odsState.addOperands(inputs);
    odsState.addOperands(outputs);
    odsState.addTypes(resultTensorTypes);
    odsState.addAttribute(
      "operand_segment_sizes",
      odsBuilder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<PoolingNchwMaxOp>(
      odsBuilder,
      odsState,
      TypeRange(inputs),
      TypeRange(outputs));
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    odsState.addAttributes(attributes);
  
}

::mlir::ParseResult PoolingNchwMaxOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<PoolingNchwMaxOp>(parser, result);
}

void PoolingNchwMaxOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult PoolingNchwMaxOp::verify() {
  if (::mlir::failed(PoolingNchwMaxOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
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





} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNchwMaxOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNdhwcMaxOp definitions
//===----------------------------------------------------------------------===//

PoolingNdhwcMaxOpAdaptor::PoolingNdhwcMaxOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

PoolingNdhwcMaxOpAdaptor::PoolingNdhwcMaxOpAdaptor(PoolingNdhwcMaxOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange PoolingNdhwcMaxOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PoolingNdhwcMaxOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange PoolingNdhwcMaxOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange PoolingNdhwcMaxOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange PoolingNdhwcMaxOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr PoolingNdhwcMaxOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNdhwcMaxOpAdaptor::strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcMaxOpAdaptor::dilations() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("dilations").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::RegionRange PoolingNdhwcMaxOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &PoolingNdhwcMaxOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult PoolingNdhwcMaxOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    {
  auto tblgen_strides = odsAttrs.get("strides");
  if (!tblgen_strides) return emitError(loc, "'linalg.pooling_ndhwc_max' op ""requires attribute 'strides'");
    if (!((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({3}))))) return emitError(loc, "'linalg.pooling_ndhwc_max' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [3]");
  }
  {
  auto tblgen_dilations = odsAttrs.get("dilations");
  if (!tblgen_dilations) return emitError(loc, "'linalg.pooling_ndhwc_max' op ""requires attribute 'dilations'");
    if (!((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({3}))))) return emitError(loc, "'linalg.pooling_ndhwc_max' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [3]");
  }
  return ::mlir::success();
}





















std::pair<unsigned, unsigned> PoolingNdhwcMaxOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range PoolingNdhwcMaxOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range PoolingNdhwcMaxOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNdhwcMaxOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNdhwcMaxOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNdhwcMaxOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> PoolingNdhwcMaxOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range PoolingNdhwcMaxOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range PoolingNdhwcMaxOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNdhwcMaxOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNdhwcMaxOp::stridesAttr() {
  return (*this)->getAttr(stridesAttrName()).template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNdhwcMaxOp::strides() {
  auto attr = stridesAttr();
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcMaxOp::dilationsAttr() {
  return (*this)->getAttr(dilationsAttrName()).template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNdhwcMaxOp::dilations() {
  auto attr = dilationsAttr();
  return attr;
}

void PoolingNdhwcMaxOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void PoolingNdhwcMaxOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

void PoolingNdhwcMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<PoolingNdhwcMaxOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void PoolingNdhwcMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<PoolingNdhwcMaxOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void PoolingNdhwcMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void PoolingNdhwcMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes ) {
    odsState.addOperands(inputs);
    odsState.addOperands(outputs);
    odsState.addTypes(resultTensorTypes);
    odsState.addAttribute(
      "operand_segment_sizes",
      odsBuilder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<PoolingNdhwcMaxOp>(
      odsBuilder,
      odsState,
      TypeRange(inputs),
      TypeRange(outputs));
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    odsState.addAttributes(attributes);
  
}

::mlir::ParseResult PoolingNdhwcMaxOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<PoolingNdhwcMaxOp>(parser, result);
}

void PoolingNdhwcMaxOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult PoolingNdhwcMaxOp::verify() {
  if (::mlir::failed(PoolingNdhwcMaxOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
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





} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNdhwcMaxOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNdhwcMinOp definitions
//===----------------------------------------------------------------------===//

PoolingNdhwcMinOpAdaptor::PoolingNdhwcMinOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

PoolingNdhwcMinOpAdaptor::PoolingNdhwcMinOpAdaptor(PoolingNdhwcMinOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange PoolingNdhwcMinOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PoolingNdhwcMinOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange PoolingNdhwcMinOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange PoolingNdhwcMinOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange PoolingNdhwcMinOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr PoolingNdhwcMinOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNdhwcMinOpAdaptor::strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcMinOpAdaptor::dilations() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("dilations").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::RegionRange PoolingNdhwcMinOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &PoolingNdhwcMinOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult PoolingNdhwcMinOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    {
  auto tblgen_strides = odsAttrs.get("strides");
  if (!tblgen_strides) return emitError(loc, "'linalg.pooling_ndhwc_min' op ""requires attribute 'strides'");
    if (!((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({3}))))) return emitError(loc, "'linalg.pooling_ndhwc_min' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [3]");
  }
  {
  auto tblgen_dilations = odsAttrs.get("dilations");
  if (!tblgen_dilations) return emitError(loc, "'linalg.pooling_ndhwc_min' op ""requires attribute 'dilations'");
    if (!((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({3}))))) return emitError(loc, "'linalg.pooling_ndhwc_min' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [3]");
  }
  return ::mlir::success();
}





















std::pair<unsigned, unsigned> PoolingNdhwcMinOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range PoolingNdhwcMinOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range PoolingNdhwcMinOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNdhwcMinOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNdhwcMinOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNdhwcMinOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> PoolingNdhwcMinOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range PoolingNdhwcMinOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range PoolingNdhwcMinOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNdhwcMinOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNdhwcMinOp::stridesAttr() {
  return (*this)->getAttr(stridesAttrName()).template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNdhwcMinOp::strides() {
  auto attr = stridesAttr();
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcMinOp::dilationsAttr() {
  return (*this)->getAttr(dilationsAttrName()).template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNdhwcMinOp::dilations() {
  auto attr = dilationsAttr();
  return attr;
}

void PoolingNdhwcMinOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void PoolingNdhwcMinOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

void PoolingNdhwcMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<PoolingNdhwcMinOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void PoolingNdhwcMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<PoolingNdhwcMinOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void PoolingNdhwcMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void PoolingNdhwcMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes ) {
    odsState.addOperands(inputs);
    odsState.addOperands(outputs);
    odsState.addTypes(resultTensorTypes);
    odsState.addAttribute(
      "operand_segment_sizes",
      odsBuilder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<PoolingNdhwcMinOp>(
      odsBuilder,
      odsState,
      TypeRange(inputs),
      TypeRange(outputs));
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    odsState.addAttributes(attributes);
  
}

::mlir::ParseResult PoolingNdhwcMinOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<PoolingNdhwcMinOp>(parser, result);
}

void PoolingNdhwcMinOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult PoolingNdhwcMinOp::verify() {
  if (::mlir::failed(PoolingNdhwcMinOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
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





} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNdhwcMinOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNdhwcSumOp definitions
//===----------------------------------------------------------------------===//

PoolingNdhwcSumOpAdaptor::PoolingNdhwcSumOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

PoolingNdhwcSumOpAdaptor::PoolingNdhwcSumOpAdaptor(PoolingNdhwcSumOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange PoolingNdhwcSumOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PoolingNdhwcSumOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange PoolingNdhwcSumOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange PoolingNdhwcSumOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange PoolingNdhwcSumOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr PoolingNdhwcSumOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNdhwcSumOpAdaptor::strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcSumOpAdaptor::dilations() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("dilations").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::RegionRange PoolingNdhwcSumOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &PoolingNdhwcSumOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult PoolingNdhwcSumOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    {
  auto tblgen_strides = odsAttrs.get("strides");
  if (!tblgen_strides) return emitError(loc, "'linalg.pooling_ndhwc_sum' op ""requires attribute 'strides'");
    if (!((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({3}))))) return emitError(loc, "'linalg.pooling_ndhwc_sum' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [3]");
  }
  {
  auto tblgen_dilations = odsAttrs.get("dilations");
  if (!tblgen_dilations) return emitError(loc, "'linalg.pooling_ndhwc_sum' op ""requires attribute 'dilations'");
    if (!((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({3}))))) return emitError(loc, "'linalg.pooling_ndhwc_sum' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [3]");
  }
  return ::mlir::success();
}





















std::pair<unsigned, unsigned> PoolingNdhwcSumOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range PoolingNdhwcSumOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range PoolingNdhwcSumOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNdhwcSumOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNdhwcSumOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNdhwcSumOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> PoolingNdhwcSumOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range PoolingNdhwcSumOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range PoolingNdhwcSumOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNdhwcSumOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNdhwcSumOp::stridesAttr() {
  return (*this)->getAttr(stridesAttrName()).template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNdhwcSumOp::strides() {
  auto attr = stridesAttr();
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNdhwcSumOp::dilationsAttr() {
  return (*this)->getAttr(dilationsAttrName()).template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNdhwcSumOp::dilations() {
  auto attr = dilationsAttr();
  return attr;
}

void PoolingNdhwcSumOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void PoolingNdhwcSumOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

void PoolingNdhwcSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<PoolingNdhwcSumOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void PoolingNdhwcSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<PoolingNdhwcSumOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void PoolingNdhwcSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void PoolingNdhwcSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes ) {
    odsState.addOperands(inputs);
    odsState.addOperands(outputs);
    odsState.addTypes(resultTensorTypes);
    odsState.addAttribute(
      "operand_segment_sizes",
      odsBuilder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<PoolingNdhwcSumOp>(
      odsBuilder,
      odsState,
      TypeRange(inputs),
      TypeRange(outputs));
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    odsState.addAttributes(attributes);
  
}

::mlir::ParseResult PoolingNdhwcSumOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<PoolingNdhwcSumOp>(parser, result);
}

void PoolingNdhwcSumOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult PoolingNdhwcSumOp::verify() {
  if (::mlir::failed(PoolingNdhwcSumOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
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





} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNdhwcSumOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNhwcMaxOp definitions
//===----------------------------------------------------------------------===//

PoolingNhwcMaxOpAdaptor::PoolingNhwcMaxOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

PoolingNhwcMaxOpAdaptor::PoolingNhwcMaxOpAdaptor(PoolingNhwcMaxOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange PoolingNhwcMaxOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PoolingNhwcMaxOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange PoolingNhwcMaxOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange PoolingNhwcMaxOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange PoolingNhwcMaxOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr PoolingNhwcMaxOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxOpAdaptor::strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxOpAdaptor::dilations() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("dilations").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::RegionRange PoolingNhwcMaxOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &PoolingNhwcMaxOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult PoolingNhwcMaxOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    {
  auto tblgen_strides = odsAttrs.get("strides");
  if (!tblgen_strides) return emitError(loc, "'linalg.pooling_nhwc_max' op ""requires attribute 'strides'");
    if (!((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2}))))) return emitError(loc, "'linalg.pooling_nhwc_max' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  {
  auto tblgen_dilations = odsAttrs.get("dilations");
  if (!tblgen_dilations) return emitError(loc, "'linalg.pooling_nhwc_max' op ""requires attribute 'dilations'");
    if (!((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2}))))) return emitError(loc, "'linalg.pooling_nhwc_max' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  return ::mlir::success();
}





















std::pair<unsigned, unsigned> PoolingNhwcMaxOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range PoolingNhwcMaxOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range PoolingNhwcMaxOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNhwcMaxOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNhwcMaxOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNhwcMaxOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> PoolingNhwcMaxOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range PoolingNhwcMaxOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range PoolingNhwcMaxOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNhwcMaxOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxOp::stridesAttr() {
  return (*this)->getAttr(stridesAttrName()).template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxOp::strides() {
  auto attr = stridesAttr();
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxOp::dilationsAttr() {
  return (*this)->getAttr(dilationsAttrName()).template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcMaxOp::dilations() {
  auto attr = dilationsAttr();
  return attr;
}

void PoolingNhwcMaxOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void PoolingNhwcMaxOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

void PoolingNhwcMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<PoolingNhwcMaxOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void PoolingNhwcMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<PoolingNhwcMaxOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void PoolingNhwcMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void PoolingNhwcMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes ) {
    odsState.addOperands(inputs);
    odsState.addOperands(outputs);
    odsState.addTypes(resultTensorTypes);
    odsState.addAttribute(
      "operand_segment_sizes",
      odsBuilder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<PoolingNhwcMaxOp>(
      odsBuilder,
      odsState,
      TypeRange(inputs),
      TypeRange(outputs));
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    odsState.addAttributes(attributes);
  
}

::mlir::ParseResult PoolingNhwcMaxOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<PoolingNhwcMaxOp>(parser, result);
}

void PoolingNhwcMaxOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult PoolingNhwcMaxOp::verify() {
  if (::mlir::failed(PoolingNhwcMaxOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
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





} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNhwcMaxOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNhwcMinOp definitions
//===----------------------------------------------------------------------===//

PoolingNhwcMinOpAdaptor::PoolingNhwcMinOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

PoolingNhwcMinOpAdaptor::PoolingNhwcMinOpAdaptor(PoolingNhwcMinOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange PoolingNhwcMinOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PoolingNhwcMinOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange PoolingNhwcMinOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange PoolingNhwcMinOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange PoolingNhwcMinOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr PoolingNhwcMinOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNhwcMinOpAdaptor::strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMinOpAdaptor::dilations() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("dilations").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::RegionRange PoolingNhwcMinOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &PoolingNhwcMinOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult PoolingNhwcMinOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    {
  auto tblgen_strides = odsAttrs.get("strides");
  if (!tblgen_strides) return emitError(loc, "'linalg.pooling_nhwc_min' op ""requires attribute 'strides'");
    if (!((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2}))))) return emitError(loc, "'linalg.pooling_nhwc_min' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  {
  auto tblgen_dilations = odsAttrs.get("dilations");
  if (!tblgen_dilations) return emitError(loc, "'linalg.pooling_nhwc_min' op ""requires attribute 'dilations'");
    if (!((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2}))))) return emitError(loc, "'linalg.pooling_nhwc_min' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  return ::mlir::success();
}





















std::pair<unsigned, unsigned> PoolingNhwcMinOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range PoolingNhwcMinOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range PoolingNhwcMinOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNhwcMinOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNhwcMinOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNhwcMinOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> PoolingNhwcMinOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range PoolingNhwcMinOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range PoolingNhwcMinOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNhwcMinOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNhwcMinOp::stridesAttr() {
  return (*this)->getAttr(stridesAttrName()).template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcMinOp::strides() {
  auto attr = stridesAttr();
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcMinOp::dilationsAttr() {
  return (*this)->getAttr(dilationsAttrName()).template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcMinOp::dilations() {
  auto attr = dilationsAttr();
  return attr;
}

void PoolingNhwcMinOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void PoolingNhwcMinOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

void PoolingNhwcMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<PoolingNhwcMinOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void PoolingNhwcMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<PoolingNhwcMinOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void PoolingNhwcMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void PoolingNhwcMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes ) {
    odsState.addOperands(inputs);
    odsState.addOperands(outputs);
    odsState.addTypes(resultTensorTypes);
    odsState.addAttribute(
      "operand_segment_sizes",
      odsBuilder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<PoolingNhwcMinOp>(
      odsBuilder,
      odsState,
      TypeRange(inputs),
      TypeRange(outputs));
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    odsState.addAttributes(attributes);
  
}

::mlir::ParseResult PoolingNhwcMinOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<PoolingNhwcMinOp>(parser, result);
}

void PoolingNhwcMinOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult PoolingNhwcMinOp::verify() {
  if (::mlir::failed(PoolingNhwcMinOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
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





} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNhwcMinOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNhwcSumOp definitions
//===----------------------------------------------------------------------===//

PoolingNhwcSumOpAdaptor::PoolingNhwcSumOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

PoolingNhwcSumOpAdaptor::PoolingNhwcSumOpAdaptor(PoolingNhwcSumOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange PoolingNhwcSumOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PoolingNhwcSumOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange PoolingNhwcSumOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange PoolingNhwcSumOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange PoolingNhwcSumOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr PoolingNhwcSumOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr PoolingNhwcSumOpAdaptor::strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("strides").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcSumOpAdaptor::dilations() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("dilations").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::RegionRange PoolingNhwcSumOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &PoolingNhwcSumOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult PoolingNhwcSumOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    {
  auto tblgen_strides = odsAttrs.get("strides");
  if (!tblgen_strides) return emitError(loc, "'linalg.pooling_nhwc_sum' op ""requires attribute 'strides'");
    if (!((((tblgen_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_strides.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2}))))) return emitError(loc, "'linalg.pooling_nhwc_sum' op ""attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  {
  auto tblgen_dilations = odsAttrs.get("dilations");
  if (!tblgen_dilations) return emitError(loc, "'linalg.pooling_nhwc_sum' op ""requires attribute 'dilations'");
    if (!((((tblgen_dilations.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))) && ((tblgen_dilations.cast<::mlir::DenseIntElementsAttr>().getType().getShape() == ::mlir::ArrayRef<int64_t>({2}))))) return emitError(loc, "'linalg.pooling_nhwc_sum' op ""attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]");
  }
  return ::mlir::success();
}





















std::pair<unsigned, unsigned> PoolingNhwcSumOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range PoolingNhwcSumOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range PoolingNhwcSumOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range PoolingNhwcSumOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PoolingNhwcSumOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PoolingNhwcSumOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> PoolingNhwcSumOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range PoolingNhwcSumOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range PoolingNhwcSumOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &PoolingNhwcSumOp::region() {
  return (*this)->getRegion(0);
}

::mlir::DenseIntElementsAttr PoolingNhwcSumOp::stridesAttr() {
  return (*this)->getAttr(stridesAttrName()).template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcSumOp::strides() {
  auto attr = stridesAttr();
  return attr;
}

::mlir::DenseIntElementsAttr PoolingNhwcSumOp::dilationsAttr() {
  return (*this)->getAttr(dilationsAttrName()).template cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr PoolingNhwcSumOp::dilations() {
  auto attr = dilationsAttr();
  return attr;
}

void PoolingNhwcSumOp::stridesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void PoolingNhwcSumOp::dilationsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

void PoolingNhwcSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<PoolingNhwcSumOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void PoolingNhwcSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<PoolingNhwcSumOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void PoolingNhwcSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

void PoolingNhwcSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes ) {
    odsState.addOperands(inputs);
    odsState.addOperands(outputs);
    odsState.addTypes(resultTensorTypes);
    odsState.addAttribute(
      "operand_segment_sizes",
      odsBuilder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<PoolingNhwcSumOp>(
      odsBuilder,
      odsState,
      TypeRange(inputs),
      TypeRange(outputs));
    odsState.addAttribute("strides", strides);
odsState.addAttribute("dilations", dilations);
    odsState.addAttributes(attributes);
  
}

::mlir::ParseResult PoolingNhwcSumOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<PoolingNhwcSumOp>(parser, result);
}

void PoolingNhwcSumOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult PoolingNhwcSumOp::verify() {
  if (::mlir::failed(PoolingNhwcSumOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
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





} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNhwcSumOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingSumOp definitions
//===----------------------------------------------------------------------===//

PoolingSumOpAdaptor::PoolingSumOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

PoolingSumOpAdaptor::PoolingSumOpAdaptor(PoolingSumOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange PoolingSumOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PoolingSumOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange PoolingSumOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PoolingSumOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::Value PoolingSumOpAdaptor::windowDims() {
  return *getODSOperands(1).begin();
}

::mlir::Value PoolingSumOpAdaptor::output() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr PoolingSumOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr PoolingSumOpAdaptor::strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("strides").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr PoolingSumOpAdaptor::dilations() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("dilations").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr PoolingSumOpAdaptor::padding() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("padding").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::LogicalResult PoolingSumOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_strides = odsAttrs.get("strides");
  if (tblgen_strides) {
    if (!(((tblgen_strides.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_strides.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); })))) return emitError(loc, "'linalg.pooling_sum' op ""attribute 'strides' failed to satisfy constraint: 64-bit integer array attribute");
  }
  }
  {
  auto tblgen_dilations = odsAttrs.get("dilations");
  if (tblgen_dilations) {
    if (!(((tblgen_dilations.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_dilations.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); })))) return emitError(loc, "'linalg.pooling_sum' op ""attribute 'dilations' failed to satisfy constraint: 64-bit integer array attribute");
  }
  }
  {
  auto tblgen_padding = odsAttrs.get("padding");
  if (tblgen_padding) {
    if (!(((tblgen_padding.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_padding.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) return emitError(loc, "'linalg.pooling_sum' op ""attribute 'padding' failed to satisfy constraint: 64-bit signless integer elements attribute");
  }
  }
  return ::mlir::success();
}





















std::pair<unsigned, unsigned> PoolingSumOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range PoolingSumOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PoolingSumOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::Value PoolingSumOp::windowDims() {
  return *getODSOperands(1).begin();
}

::mlir::Value PoolingSumOp::output() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange PoolingSumOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange PoolingSumOp::windowDimsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange PoolingSumOp::outputMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> PoolingSumOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range PoolingSumOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::ArrayAttr PoolingSumOp::stridesAttr() {
  return (*this)->getAttr(stridesAttrName()).template dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > PoolingSumOp::strides() {
  auto attr = stridesAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr PoolingSumOp::dilationsAttr() {
  return (*this)->getAttr(dilationsAttrName()).template dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > PoolingSumOp::dilations() {
  auto attr = dilationsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr PoolingSumOp::paddingAttr() {
  return (*this)->getAttr(paddingAttrName()).template dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > PoolingSumOp::padding() {
  auto attr = paddingAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void PoolingSumOp::stridesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void PoolingSumOp::dilationsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(dilationsAttrName(), attr);
}

void PoolingSumOp::paddingAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(paddingAttrName(), attr);
}

::mlir::Attribute PoolingSumOp::removeStridesAttr() {
  return (*this)->removeAttr(stridesAttrName());
}

::mlir::Attribute PoolingSumOp::removeDilationsAttr() {
  return (*this)->removeAttr(dilationsAttrName());
}

::mlir::Attribute PoolingSumOp::removePaddingAttr() {
  return (*this)->removeAttr(paddingAttrName());
}

void PoolingSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value windowDims, ::mlir::Value output, /*optional*/::mlir::ArrayAttr strides, /*optional*/::mlir::ArrayAttr dilations, /*optional*/::mlir::DenseIntElementsAttr padding) {
  odsState.addOperands(input);
  odsState.addOperands(windowDims);
  odsState.addOperands(output);
  if (strides) {
  odsState.addAttribute(stridesAttrName(odsState.name), strides);
  }
  if (dilations) {
  odsState.addAttribute(dilationsAttrName(odsState.name), dilations);
  }
  if (padding) {
  odsState.addAttribute(paddingAttrName(odsState.name), padding);
  }
}

void PoolingSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value windowDims, ::mlir::Value output, /*optional*/::mlir::ArrayAttr strides, /*optional*/::mlir::ArrayAttr dilations, /*optional*/::mlir::DenseIntElementsAttr padding) {
  odsState.addOperands(input);
  odsState.addOperands(windowDims);
  odsState.addOperands(output);
  if (strides) {
  odsState.addAttribute(stridesAttrName(odsState.name), strides);
  }
  if (dilations) {
  odsState.addAttribute(dilationsAttrName(odsState.name), dilations);
  }
  if (padding) {
  odsState.addAttribute(paddingAttrName(odsState.name), padding);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PoolingSumOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult PoolingSumOp::verify() {
  if (::mlir::failed(PoolingSumOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps3(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::verify(*this);
}





::mlir::ParseResult PoolingSumOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> allOperands;
  ::mlir::SmallVector<::mlir::Type, 1> allOperandTypes;
  if (parser.parseLParen())
    return ::mlir::failure();
  ::llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(allOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(allOperandTypes))
    return ::mlir::failure();
  if (parser.resolveOperands(allOperands, allOperandTypes, allOperandLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void PoolingSumOp::print(::mlir::OpAsmPrinter &p) {
  p << "(";
  p << getOperation()->getOperands();
  p << ")";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << getOperation()->getOperandTypes();
}

} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingSumOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::QuantizedBatchMatmulOp definitions
//===----------------------------------------------------------------------===//

QuantizedBatchMatmulOpAdaptor::QuantizedBatchMatmulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

QuantizedBatchMatmulOpAdaptor::QuantizedBatchMatmulOpAdaptor(QuantizedBatchMatmulOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange QuantizedBatchMatmulOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> QuantizedBatchMatmulOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange QuantizedBatchMatmulOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange QuantizedBatchMatmulOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange QuantizedBatchMatmulOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr QuantizedBatchMatmulOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange QuantizedBatchMatmulOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &QuantizedBatchMatmulOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult QuantizedBatchMatmulOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    return ::mlir::success();
}













std::pair<unsigned, unsigned> QuantizedBatchMatmulOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range QuantizedBatchMatmulOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range QuantizedBatchMatmulOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range QuantizedBatchMatmulOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange QuantizedBatchMatmulOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange QuantizedBatchMatmulOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> QuantizedBatchMatmulOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range QuantizedBatchMatmulOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range QuantizedBatchMatmulOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &QuantizedBatchMatmulOp::region() {
  return (*this)->getRegion(0);
}

void QuantizedBatchMatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<QuantizedBatchMatmulOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void QuantizedBatchMatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<QuantizedBatchMatmulOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void QuantizedBatchMatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::ParseResult QuantizedBatchMatmulOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<QuantizedBatchMatmulOp>(parser, result);
}

void QuantizedBatchMatmulOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult QuantizedBatchMatmulOp::verify() {
  if (::mlir::failed(QuantizedBatchMatmulOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
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





} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::QuantizedBatchMatmulOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::QuantizedMatmulOp definitions
//===----------------------------------------------------------------------===//

QuantizedMatmulOpAdaptor::QuantizedMatmulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

QuantizedMatmulOpAdaptor::QuantizedMatmulOpAdaptor(QuantizedMatmulOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange QuantizedMatmulOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> QuantizedMatmulOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange QuantizedMatmulOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange QuantizedMatmulOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange QuantizedMatmulOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr QuantizedMatmulOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange QuantizedMatmulOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &QuantizedMatmulOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult QuantizedMatmulOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    return ::mlir::success();
}













std::pair<unsigned, unsigned> QuantizedMatmulOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range QuantizedMatmulOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range QuantizedMatmulOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range QuantizedMatmulOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange QuantizedMatmulOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange QuantizedMatmulOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> QuantizedMatmulOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range QuantizedMatmulOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range QuantizedMatmulOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &QuantizedMatmulOp::region() {
  return (*this)->getRegion(0);
}

void QuantizedMatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<QuantizedMatmulOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void QuantizedMatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<QuantizedMatmulOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void QuantizedMatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::ParseResult QuantizedMatmulOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<QuantizedMatmulOp>(parser, result);
}

void QuantizedMatmulOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult QuantizedMatmulOp::verify() {
  if (::mlir::failed(QuantizedMatmulOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
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





} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::QuantizedMatmulOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::SoftPlus2DOp definitions
//===----------------------------------------------------------------------===//

SoftPlus2DOpAdaptor::SoftPlus2DOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

SoftPlus2DOpAdaptor::SoftPlus2DOpAdaptor(SoftPlus2DOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange SoftPlus2DOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SoftPlus2DOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange SoftPlus2DOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange SoftPlus2DOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange SoftPlus2DOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr SoftPlus2DOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange SoftPlus2DOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &SoftPlus2DOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult SoftPlus2DOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    return ::mlir::success();
}













std::pair<unsigned, unsigned> SoftPlus2DOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range SoftPlus2DOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range SoftPlus2DOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range SoftPlus2DOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange SoftPlus2DOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange SoftPlus2DOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> SoftPlus2DOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range SoftPlus2DOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range SoftPlus2DOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &SoftPlus2DOp::region() {
  return (*this)->getRegion(0);
}

void SoftPlus2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<SoftPlus2DOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void SoftPlus2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<SoftPlus2DOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void SoftPlus2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::ParseResult SoftPlus2DOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<SoftPlus2DOp>(parser, result);
}

void SoftPlus2DOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult SoftPlus2DOp::verify() {
  if (::mlir::failed(SoftPlus2DOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
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





} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::SoftPlus2DOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::VecmatOp definitions
//===----------------------------------------------------------------------===//

VecmatOpAdaptor::VecmatOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)  : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {

}

VecmatOpAdaptor::VecmatOpAdaptor(VecmatOp&op)  : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {

}

::mlir::ValueRange VecmatOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VecmatOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange VecmatOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange VecmatOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange VecmatOpAdaptor::outputs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr VecmatOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange VecmatOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &VecmatOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult VecmatOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements, but got ") << numElements;
  }
    return ::mlir::success();
}













std::pair<unsigned, unsigned> VecmatOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range VecmatOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range VecmatOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range VecmatOp::outputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange VecmatOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange VecmatOp::outputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> VecmatOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range VecmatOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range VecmatOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &VecmatOp::region() {
  return (*this)->getRegion(0);
}

void VecmatOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        odsState.addTypes(resultTensorTypes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        odsState.addAttributes(attributes);
        createAndFillStructuredOpRegion<VecmatOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void VecmatOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(inputs);
        odsState.addOperands(outputs);
        odsState.addTypes(resultTensorTypes);
        odsState.addAttributes(attributes);
        odsState.addAttribute(
          "operand_segment_sizes",
          odsBuilder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<VecmatOp>(
          odsBuilder,
          odsState,
          TypeRange(inputs),
          TypeRange(outputs));
      
}

void VecmatOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
        odsState.addOperands(operands);
        odsState.addAttributes(attributes);
        odsState.addTypes(resultTensorTypes);
        (void)odsState.addRegion();
      
}

::mlir::ParseResult VecmatOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<VecmatOp>(parser, result);
}

void VecmatOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult VecmatOp::verify() {
  if (::mlir::failed(VecmatOpAdaptor(*this).verify((*this)->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps0(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps1(getOperation(), v.getType(), "operand", index)))
        return ::mlir::failure();
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgStructuredOps2(getOperation(), v.getType(), "result", index)))
        return ::mlir::failure();
      ++index;
    }
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





} // namespace linalg
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::VecmatOp)


#endif  // GET_OP_CLASSES

