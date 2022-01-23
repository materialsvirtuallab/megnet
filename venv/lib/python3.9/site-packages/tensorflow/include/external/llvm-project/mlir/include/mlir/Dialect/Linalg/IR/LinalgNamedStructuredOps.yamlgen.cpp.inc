
//===----------------------------------------------------------------------===//
// Implementation of MatmulOp
//===----------------------------------------------------------------------===//

ArrayAttr MatmulOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(MatmulOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));
  return exprs;
}

ArrayAttr MatmulOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2)[s0, s1, s2] -> (d2, d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0, d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned MatmulOp::getNumRegionArgs() { return 3; }

std::string MatmulOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void MatmulOp::regionBuilder(ImplicitLocOpBuilder &b, Block &block) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "MatmulOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  Value value1 = helper.cast(block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.cast(block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.applyfn__mul(value1, value2);
  Value value4 = helper.applyfn__add(block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

LogicalResult MatmulOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void MatmulOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of QuantizedMatmulOp
//===----------------------------------------------------------------------===//

ArrayAttr QuantizedMatmulOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(QuantizedMatmulOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));
  return exprs;
}

ArrayAttr QuantizedMatmulOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2)[s0, s1, s2] -> (d2, d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2)[s0, s1, s2] -> ()>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2)[s0, s1, s2] -> ()>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0, d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned QuantizedMatmulOp::getNumRegionArgs() { return 5; }

std::string QuantizedMatmulOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void QuantizedMatmulOp::regionBuilder(ImplicitLocOpBuilder &b, Block &block) {
  assert(5 > 0 && block.getNumArguments() == 5 &&
         "QuantizedMatmulOp regionBuilder expects 5 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  Value value1 = helper.cast(block.getArgument(4).getType(), block.getArgument(0));
  Value value2 = helper.cast(block.getArgument(4).getType(), block.getArgument(2));
  Value value3 = helper.applyfn__sub(value1, value2);
  Value value4 = helper.cast(block.getArgument(4).getType(), block.getArgument(1));
  Value value5 = helper.cast(block.getArgument(4).getType(), block.getArgument(3));
  Value value6 = helper.applyfn__sub(value4, value5);
  Value value7 = helper.applyfn__mul(value3, value6);
  Value value8 = helper.applyfn__add(block.getArgument(4), value7);
  yields.push_back(value8);
  helper.yieldOutputs(yields);
}

LogicalResult QuantizedMatmulOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void QuantizedMatmulOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of Mmt4DOp
//===----------------------------------------------------------------------===//

ArrayAttr Mmt4DOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(Mmt4DOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));
  exprs.push_back(getAffineSymbolExpr(3, context));
  exprs.push_back(getAffineSymbolExpr(4, context));
  exprs.push_back(getAffineSymbolExpr(5, context));
  return exprs;
}

ArrayAttr Mmt4DOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5] -> (d0, d2, d3, d5)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5] -> (d1, d2, d4, d5)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5] -> (d0, d1, d3, d4)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned Mmt4DOp::getNumRegionArgs() { return 3; }

std::string Mmt4DOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void Mmt4DOp::regionBuilder(ImplicitLocOpBuilder &b, Block &block) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "Mmt4DOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  Value value1 = helper.cast(block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.cast(block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.applyfn__mul(value1, value2);
  Value value4 = helper.applyfn__add(block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

LogicalResult Mmt4DOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void Mmt4DOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of BatchMatmulOp
//===----------------------------------------------------------------------===//

ArrayAttr BatchMatmulOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(BatchMatmulOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));
  exprs.push_back(getAffineSymbolExpr(3, context));
  return exprs;
}

ArrayAttr BatchMatmulOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0, d1, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0, d3, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0, d1, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned BatchMatmulOp::getNumRegionArgs() { return 3; }

std::string BatchMatmulOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void BatchMatmulOp::regionBuilder(ImplicitLocOpBuilder &b, Block &block) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "BatchMatmulOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  Value value1 = helper.cast(block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.cast(block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.applyfn__mul(value1, value2);
  Value value4 = helper.applyfn__add(block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

LogicalResult BatchMatmulOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void BatchMatmulOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of QuantizedBatchMatmulOp
//===----------------------------------------------------------------------===//

ArrayAttr QuantizedBatchMatmulOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(QuantizedBatchMatmulOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));
  exprs.push_back(getAffineSymbolExpr(3, context));
  return exprs;
}

ArrayAttr QuantizedBatchMatmulOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0, d1, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0, d3, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> ()>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> ()>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0, d1, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned QuantizedBatchMatmulOp::getNumRegionArgs() { return 5; }

std::string QuantizedBatchMatmulOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void QuantizedBatchMatmulOp::regionBuilder(ImplicitLocOpBuilder &b, Block &block) {
  assert(5 > 0 && block.getNumArguments() == 5 &&
         "QuantizedBatchMatmulOp regionBuilder expects 5 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  Value value1 = helper.cast(block.getArgument(4).getType(), block.getArgument(0));
  Value value2 = helper.cast(block.getArgument(4).getType(), block.getArgument(2));
  Value value3 = helper.applyfn__sub(value1, value2);
  Value value4 = helper.cast(block.getArgument(4).getType(), block.getArgument(1));
  Value value5 = helper.cast(block.getArgument(4).getType(), block.getArgument(3));
  Value value6 = helper.applyfn__sub(value4, value5);
  Value value7 = helper.applyfn__mul(value3, value6);
  Value value8 = helper.applyfn__add(block.getArgument(4), value7);
  yields.push_back(value8);
  helper.yieldOutputs(yields);
}

LogicalResult QuantizedBatchMatmulOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void QuantizedBatchMatmulOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of MatvecOp
//===----------------------------------------------------------------------===//

ArrayAttr MatvecOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(MatvecOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  return exprs;
}

ArrayAttr MatvecOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> (d0, d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> (d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> (d0)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned MatvecOp::getNumRegionArgs() { return 3; }

std::string MatvecOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void MatvecOp::regionBuilder(ImplicitLocOpBuilder &b, Block &block) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "MatvecOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  Value value1 = helper.cast(block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.cast(block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.applyfn__mul(value1, value2);
  Value value4 = helper.applyfn__add(block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

LogicalResult MatvecOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void MatvecOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of VecmatOp
//===----------------------------------------------------------------------===//

ArrayAttr VecmatOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(VecmatOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  return exprs;
}

ArrayAttr VecmatOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> (d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> (d1, d0)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> (d0)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned VecmatOp::getNumRegionArgs() { return 3; }

std::string VecmatOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void VecmatOp::regionBuilder(ImplicitLocOpBuilder &b, Block &block) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "VecmatOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  Value value1 = helper.cast(block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.cast(block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.applyfn__mul(value1, value2);
  Value value4 = helper.applyfn__add(block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

LogicalResult VecmatOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void VecmatOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of BatchMatvecOp
//===----------------------------------------------------------------------===//

ArrayAttr BatchMatvecOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(BatchMatvecOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));
  return exprs;
}

ArrayAttr BatchMatvecOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0, d1, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0, d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned BatchMatvecOp::getNumRegionArgs() { return 3; }

std::string BatchMatvecOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void BatchMatvecOp::regionBuilder(ImplicitLocOpBuilder &b, Block &block) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "BatchMatvecOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  Value value1 = helper.cast(block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.cast(block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.applyfn__mul(value1, value2);
  Value value4 = helper.applyfn__add(block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

LogicalResult BatchMatvecOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void BatchMatvecOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of DotOp
//===----------------------------------------------------------------------===//

ArrayAttr DotOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(DotOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  return exprs;
}

ArrayAttr DotOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0)[s0] -> (d0)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 1, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0)[s0] -> (d0)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 1, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0)[s0] -> ()>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 1, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned DotOp::getNumRegionArgs() { return 3; }

std::string DotOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void DotOp::regionBuilder(ImplicitLocOpBuilder &b, Block &block) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "DotOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  Value value1 = helper.cast(block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.cast(block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.applyfn__mul(value1, value2);
  Value value4 = helper.applyfn__add(block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

LogicalResult DotOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void DotOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of Conv1DOp
//===----------------------------------------------------------------------===//

ArrayAttr Conv1DOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(Conv1DOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  return exprs;
}

ArrayAttr Conv1DOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> (d0 + d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> (d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> (d0)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned Conv1DOp::getNumRegionArgs() { return 3; }

std::string Conv1DOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void Conv1DOp::regionBuilder(ImplicitLocOpBuilder &b, Block &block) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "Conv1DOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  Value value1 = helper.cast(block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.cast(block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.applyfn__mul(value1, value2);
  Value value4 = helper.applyfn__add(block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

LogicalResult Conv1DOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void Conv1DOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of Conv2DOp
//===----------------------------------------------------------------------===//

ArrayAttr Conv2DOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(Conv2DOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));
  exprs.push_back(getAffineSymbolExpr(3, context));
  return exprs;
}

ArrayAttr Conv2DOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0 + d2, d1 + d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d2, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0, d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned Conv2DOp::getNumRegionArgs() { return 3; }

std::string Conv2DOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void Conv2DOp::regionBuilder(ImplicitLocOpBuilder &b, Block &block) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "Conv2DOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  Value value1 = helper.cast(block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.cast(block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.applyfn__mul(value1, value2);
  Value value4 = helper.applyfn__add(block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

LogicalResult Conv2DOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void Conv2DOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of Conv3DOp
//===----------------------------------------------------------------------===//

ArrayAttr Conv3DOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(Conv3DOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));
  exprs.push_back(getAffineSymbolExpr(3, context));
  exprs.push_back(getAffineSymbolExpr(4, context));
  exprs.push_back(getAffineSymbolExpr(5, context));
  return exprs;
}

ArrayAttr Conv3DOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5] -> (d0 + d3, d1 + d4, d2 + d5)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5] -> (d3, d4, d5)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5] -> (d0, d1, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned Conv3DOp::getNumRegionArgs() { return 3; }

std::string Conv3DOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void Conv3DOp::regionBuilder(ImplicitLocOpBuilder &b, Block &block) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "Conv3DOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  Value value1 = helper.cast(block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.cast(block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.applyfn__mul(value1, value2);
  Value value4 = helper.applyfn__add(block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

LogicalResult Conv3DOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void Conv3DOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of Conv1DNwcWcfOp
//===----------------------------------------------------------------------===//

ArrayAttr Conv1DNwcWcfOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(Conv1DNwcWcfOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.strides().getValue<int64_t>({ 0 });
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.dilations().getValue<int64_t>({ 0 });
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));
  exprs.push_back(getAffineSymbolExpr(6, context));
  return exprs;
}

ArrayAttr Conv1DNwcWcfOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4)[s0, s1, s2, s3, s4, s5, s6] -> (d0, d1 * s2 + d3 * s4, d4)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 5, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4)[s0, s1, s2, s3, s4, s5, s6] -> (d3, d4, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 5, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4)[s0, s1, s2, s3, s4, s5, s6] -> (d0, d1, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 5, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned Conv1DNwcWcfOp::getNumRegionArgs() { return 3; }

std::string Conv1DNwcWcfOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool Conv1DNwcWcfOp::hasDynamicIndexingMaps() { return true; }
LogicalResult Conv1DNwcWcfOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError(
      "incorrect element type for indexing map required attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 1 })
    return op->emitError(
      "incorrect shape for indexing map required attribute 'strides'");
} else {
  return op->emitError(
    "missing indexing map required attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError(
      "incorrect element type for indexing map required attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 1 })
    return op->emitError(
      "incorrect shape for indexing map required attribute 'dilations'");
} else {
  return op->emitError(
    "missing indexing map required attribute 'dilations'");
}

  return success();
}

void Conv1DNwcWcfOp::regionBuilder(ImplicitLocOpBuilder &b, Block &block) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "Conv1DNwcWcfOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  Value value1 = helper.cast(block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.cast(block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.applyfn__mul(value1, value2);
  Value value4 = helper.applyfn__add(block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

LogicalResult Conv1DNwcWcfOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void Conv1DNwcWcfOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of Conv2DNhwcHwcfOp
//===----------------------------------------------------------------------===//

ArrayAttr Conv2DNhwcHwcfOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(Conv2DNhwcHwcfOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.strides().getValue<int64_t>({ 0 });
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.dilations().getValue<int64_t>({ 0 });
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.strides().getValue<int64_t>({ 1 });
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.dilations().getValue<int64_t>({ 1 });
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));
  exprs.push_back(getAffineSymbolExpr(10, context));
  return exprs;
}

ArrayAttr Conv2DNhwcHwcfOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d0, d1 * s2 + d4 * s4, d2 * s6 + d5 * s8, d6)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d4, d5, d6, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d0, d1, d2, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned Conv2DNhwcHwcfOp::getNumRegionArgs() { return 3; }

std::string Conv2DNhwcHwcfOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool Conv2DNhwcHwcfOp::hasDynamicIndexingMaps() { return true; }
LogicalResult Conv2DNhwcHwcfOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError(
      "incorrect element type for indexing map required attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError(
      "incorrect shape for indexing map required attribute 'strides'");
} else {
  return op->emitError(
    "missing indexing map required attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError(
      "incorrect element type for indexing map required attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError(
      "incorrect shape for indexing map required attribute 'dilations'");
} else {
  return op->emitError(
    "missing indexing map required attribute 'dilations'");
}

  return success();
}

void Conv2DNhwcHwcfOp::regionBuilder(ImplicitLocOpBuilder &b, Block &block) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "Conv2DNhwcHwcfOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  Value value1 = helper.cast(block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.cast(block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.applyfn__mul(value1, value2);
  Value value4 = helper.applyfn__add(block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

LogicalResult Conv2DNhwcHwcfOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void Conv2DNhwcHwcfOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of Conv2DNhwcHwcfQOp
//===----------------------------------------------------------------------===//

ArrayAttr Conv2DNhwcHwcfQOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(Conv2DNhwcHwcfQOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.strides().getValue<int64_t>({ 0 });
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.dilations().getValue<int64_t>({ 0 });
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.strides().getValue<int64_t>({ 1 });
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.dilations().getValue<int64_t>({ 1 });
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));
  exprs.push_back(getAffineSymbolExpr(10, context));
  return exprs;
}

ArrayAttr Conv2DNhwcHwcfQOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d0, d1 * s2 + d4 * s4, d2 * s6 + d5 * s8, d6)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d4, d5, d6, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> ()>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> ()>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d0, d1, d2, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned Conv2DNhwcHwcfQOp::getNumRegionArgs() { return 5; }

std::string Conv2DNhwcHwcfQOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool Conv2DNhwcHwcfQOp::hasDynamicIndexingMaps() { return true; }
LogicalResult Conv2DNhwcHwcfQOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError(
      "incorrect element type for indexing map required attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError(
      "incorrect shape for indexing map required attribute 'strides'");
} else {
  return op->emitError(
    "missing indexing map required attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError(
      "incorrect element type for indexing map required attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError(
      "incorrect shape for indexing map required attribute 'dilations'");
} else {
  return op->emitError(
    "missing indexing map required attribute 'dilations'");
}

  return success();
}

void Conv2DNhwcHwcfQOp::regionBuilder(ImplicitLocOpBuilder &b, Block &block) {
  assert(5 > 0 && block.getNumArguments() == 5 &&
         "Conv2DNhwcHwcfQOp regionBuilder expects 5 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  Value value1 = helper.cast(block.getArgument(4).getType(), block.getArgument(0));
  Value value2 = helper.cast(block.getArgument(4).getType(), block.getArgument(2));
  Value value3 = helper.applyfn__sub(value1, value2);
  Value value4 = helper.cast(block.getArgument(4).getType(), block.getArgument(1));
  Value value5 = helper.cast(block.getArgument(4).getType(), block.getArgument(3));
  Value value6 = helper.applyfn__sub(value4, value5);
  Value value7 = helper.applyfn__mul(value3, value6);
  Value value8 = helper.applyfn__add(block.getArgument(4), value7);
  yields.push_back(value8);
  helper.yieldOutputs(yields);
}

LogicalResult Conv2DNhwcHwcfQOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void Conv2DNhwcHwcfQOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of Conv2DNchwFchwOp
//===----------------------------------------------------------------------===//

ArrayAttr Conv2DNchwFchwOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(Conv2DNchwFchwOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));

int64_t cst3 = self.strides().getValue<int64_t>({ 0 });
exprs.push_back(getAffineConstantExpr(cst3, context));

  exprs.push_back(getAffineSymbolExpr(4, context));

int64_t cst5 = self.dilations().getValue<int64_t>({ 0 });
exprs.push_back(getAffineConstantExpr(cst5, context));

  exprs.push_back(getAffineSymbolExpr(6, context));

int64_t cst7 = self.strides().getValue<int64_t>({ 1 });
exprs.push_back(getAffineConstantExpr(cst7, context));

  exprs.push_back(getAffineSymbolExpr(8, context));

int64_t cst9 = self.dilations().getValue<int64_t>({ 1 });
exprs.push_back(getAffineConstantExpr(cst9, context));

  exprs.push_back(getAffineSymbolExpr(10, context));
  return exprs;
}

ArrayAttr Conv2DNchwFchwOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d0, d4, d2 * s3 + d5 * s5, d3 * s7 + d6 * s9)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d1, d4, d5, d6)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d0, d1, d2, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned Conv2DNchwFchwOp::getNumRegionArgs() { return 3; }

std::string Conv2DNchwFchwOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool Conv2DNchwFchwOp::hasDynamicIndexingMaps() { return true; }
LogicalResult Conv2DNchwFchwOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError(
      "incorrect element type for indexing map required attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError(
      "incorrect shape for indexing map required attribute 'strides'");
} else {
  return op->emitError(
    "missing indexing map required attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError(
      "incorrect element type for indexing map required attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError(
      "incorrect shape for indexing map required attribute 'dilations'");
} else {
  return op->emitError(
    "missing indexing map required attribute 'dilations'");
}

  return success();
}

void Conv2DNchwFchwOp::regionBuilder(ImplicitLocOpBuilder &b, Block &block) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "Conv2DNchwFchwOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  Value value1 = helper.cast(block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.cast(block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.applyfn__mul(value1, value2);
  Value value4 = helper.applyfn__add(block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

LogicalResult Conv2DNchwFchwOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void Conv2DNchwFchwOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of Conv3DNdhwcDhwcfOp
//===----------------------------------------------------------------------===//

ArrayAttr Conv3DNdhwcDhwcfOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(Conv3DNdhwcDhwcfOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.strides().getValue<int64_t>({ 0 });
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.dilations().getValue<int64_t>({ 0 });
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.strides().getValue<int64_t>({ 1 });
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.dilations().getValue<int64_t>({ 1 });
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));

int64_t cst10 = self.strides().getValue<int64_t>({ 2 });
exprs.push_back(getAffineConstantExpr(cst10, context));

  exprs.push_back(getAffineSymbolExpr(11, context));

int64_t cst12 = self.dilations().getValue<int64_t>({ 2 });
exprs.push_back(getAffineConstantExpr(cst12, context));

  exprs.push_back(getAffineSymbolExpr(13, context));
  exprs.push_back(getAffineSymbolExpr(14, context));
  return exprs;
}

ArrayAttr Conv3DNdhwcDhwcfOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14] -> (d0, d1 * s2 + d5 * s4, d2 * s6 + d6 * s8, d3 * s10 + d7 * s12, d8)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 9, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14] -> (d5, d6, d7, d8, d4)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 9, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14] -> (d0, d1, d2, d3, d4)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 9, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned Conv3DNdhwcDhwcfOp::getNumRegionArgs() { return 3; }

std::string Conv3DNdhwcDhwcfOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool Conv3DNdhwcDhwcfOp::hasDynamicIndexingMaps() { return true; }
LogicalResult Conv3DNdhwcDhwcfOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError(
      "incorrect element type for indexing map required attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 3 })
    return op->emitError(
      "incorrect shape for indexing map required attribute 'strides'");
} else {
  return op->emitError(
    "missing indexing map required attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError(
      "incorrect element type for indexing map required attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 3 })
    return op->emitError(
      "incorrect shape for indexing map required attribute 'dilations'");
} else {
  return op->emitError(
    "missing indexing map required attribute 'dilations'");
}

  return success();
}

void Conv3DNdhwcDhwcfOp::regionBuilder(ImplicitLocOpBuilder &b, Block &block) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "Conv3DNdhwcDhwcfOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  Value value1 = helper.cast(block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.cast(block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.applyfn__mul(value1, value2);
  Value value4 = helper.applyfn__add(block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

LogicalResult Conv3DNdhwcDhwcfOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void Conv3DNdhwcDhwcfOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of DepthwiseConv2DNhwOp
//===----------------------------------------------------------------------===//

ArrayAttr DepthwiseConv2DNhwOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(DepthwiseConv2DNhwOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.strides().getValue<int64_t>({ 0 });
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.dilations().getValue<int64_t>({ 0 });
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.strides().getValue<int64_t>({ 1 });
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.dilations().getValue<int64_t>({ 1 });
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));
  return exprs;
}

ArrayAttr DepthwiseConv2DNhwOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1 * s2 + d4 * s4, d2 * s6 + d5 * s8, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d4, d5, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1, d2, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned DepthwiseConv2DNhwOp::getNumRegionArgs() { return 3; }

std::string DepthwiseConv2DNhwOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool DepthwiseConv2DNhwOp::hasDynamicIndexingMaps() { return true; }
LogicalResult DepthwiseConv2DNhwOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError(
      "incorrect element type for indexing map required attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError(
      "incorrect shape for indexing map required attribute 'strides'");
} else {
  return op->emitError(
    "missing indexing map required attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError(
      "incorrect element type for indexing map required attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError(
      "incorrect shape for indexing map required attribute 'dilations'");
} else {
  return op->emitError(
    "missing indexing map required attribute 'dilations'");
}

  return success();
}

void DepthwiseConv2DNhwOp::regionBuilder(ImplicitLocOpBuilder &b, Block &block) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "DepthwiseConv2DNhwOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  Value value1 = helper.cast(block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.cast(block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.applyfn__mul(value1, value2);
  Value value4 = helper.applyfn__add(block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

LogicalResult DepthwiseConv2DNhwOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void DepthwiseConv2DNhwOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of DepthwiseConv2DNhwQOp
//===----------------------------------------------------------------------===//

ArrayAttr DepthwiseConv2DNhwQOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(DepthwiseConv2DNhwQOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.strides().getValue<int64_t>({ 0 });
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.dilations().getValue<int64_t>({ 0 });
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.strides().getValue<int64_t>({ 1 });
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.dilations().getValue<int64_t>({ 1 });
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));
  return exprs;
}

ArrayAttr DepthwiseConv2DNhwQOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1 * s2 + d4 * s4, d2 * s6 + d5 * s8, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d4, d5, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> ()>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> ()>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1, d2, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned DepthwiseConv2DNhwQOp::getNumRegionArgs() { return 5; }

std::string DepthwiseConv2DNhwQOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool DepthwiseConv2DNhwQOp::hasDynamicIndexingMaps() { return true; }
LogicalResult DepthwiseConv2DNhwQOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError(
      "incorrect element type for indexing map required attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError(
      "incorrect shape for indexing map required attribute 'strides'");
} else {
  return op->emitError(
    "missing indexing map required attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError(
      "incorrect element type for indexing map required attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError(
      "incorrect shape for indexing map required attribute 'dilations'");
} else {
  return op->emitError(
    "missing indexing map required attribute 'dilations'");
}

  return success();
}

void DepthwiseConv2DNhwQOp::regionBuilder(ImplicitLocOpBuilder &b, Block &block) {
  assert(5 > 0 && block.getNumArguments() == 5 &&
         "DepthwiseConv2DNhwQOp regionBuilder expects 5 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  Value value1 = helper.cast(block.getArgument(4).getType(), block.getArgument(0));
  Value value2 = helper.cast(block.getArgument(4).getType(), block.getArgument(2));
  Value value3 = helper.applyfn__sub(value1, value2);
  Value value4 = helper.cast(block.getArgument(4).getType(), block.getArgument(1));
  Value value5 = helper.cast(block.getArgument(4).getType(), block.getArgument(3));
  Value value6 = helper.applyfn__sub(value4, value5);
  Value value7 = helper.applyfn__mul(value3, value6);
  Value value8 = helper.applyfn__add(block.getArgument(4), value7);
  yields.push_back(value8);
  helper.yieldOutputs(yields);
}

LogicalResult DepthwiseConv2DNhwQOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void DepthwiseConv2DNhwQOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of DepthwiseConv2DNhwcOp
//===----------------------------------------------------------------------===//

ArrayAttr DepthwiseConv2DNhwcOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(DepthwiseConv2DNhwcOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.strides().getValue<int64_t>({ 0 });
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.dilations().getValue<int64_t>({ 0 });
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.strides().getValue<int64_t>({ 1 });
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.dilations().getValue<int64_t>({ 1 });
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));
  exprs.push_back(getAffineSymbolExpr(10, context));
  return exprs;
}

ArrayAttr DepthwiseConv2DNhwcOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d0, d1 * s2 + d5 * s4, d2 * s6 + d6 * s8, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d5, d6, d3, d4)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d0, d1, d2, d3, d4)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned DepthwiseConv2DNhwcOp::getNumRegionArgs() { return 3; }

std::string DepthwiseConv2DNhwcOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool DepthwiseConv2DNhwcOp::hasDynamicIndexingMaps() { return true; }
LogicalResult DepthwiseConv2DNhwcOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError(
      "incorrect element type for indexing map required attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError(
      "incorrect shape for indexing map required attribute 'strides'");
} else {
  return op->emitError(
    "missing indexing map required attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError(
      "incorrect element type for indexing map required attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError(
      "incorrect shape for indexing map required attribute 'dilations'");
} else {
  return op->emitError(
    "missing indexing map required attribute 'dilations'");
}

  return success();
}

void DepthwiseConv2DNhwcOp::regionBuilder(ImplicitLocOpBuilder &b, Block &block) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "DepthwiseConv2DNhwcOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  Value value1 = helper.cast(block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.cast(block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.applyfn__mul(value1, value2);
  Value value4 = helper.applyfn__add(block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

LogicalResult DepthwiseConv2DNhwcOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void DepthwiseConv2DNhwcOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of DepthwiseConv2DNhwcQOp
//===----------------------------------------------------------------------===//

ArrayAttr DepthwiseConv2DNhwcQOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(DepthwiseConv2DNhwcQOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.strides().getValue<int64_t>({ 0 });
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.dilations().getValue<int64_t>({ 0 });
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.strides().getValue<int64_t>({ 1 });
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.dilations().getValue<int64_t>({ 1 });
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));
  exprs.push_back(getAffineSymbolExpr(10, context));
  return exprs;
}

ArrayAttr DepthwiseConv2DNhwcQOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d0, d1 * s2 + d5 * s4, d2 * s6 + d6 * s8, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d5, d6, d3, d4)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> ()>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> ()>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d0, d1, d2, d3, d4)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned DepthwiseConv2DNhwcQOp::getNumRegionArgs() { return 5; }

std::string DepthwiseConv2DNhwcQOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool DepthwiseConv2DNhwcQOp::hasDynamicIndexingMaps() { return true; }
LogicalResult DepthwiseConv2DNhwcQOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError(
      "incorrect element type for indexing map required attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError(
      "incorrect shape for indexing map required attribute 'strides'");
} else {
  return op->emitError(
    "missing indexing map required attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError(
      "incorrect element type for indexing map required attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError(
      "incorrect shape for indexing map required attribute 'dilations'");
} else {
  return op->emitError(
    "missing indexing map required attribute 'dilations'");
}

  return success();
}

void DepthwiseConv2DNhwcQOp::regionBuilder(ImplicitLocOpBuilder &b, Block &block) {
  assert(5 > 0 && block.getNumArguments() == 5 &&
         "DepthwiseConv2DNhwcQOp regionBuilder expects 5 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  Value value1 = helper.cast(block.getArgument(4).getType(), block.getArgument(0));
  Value value2 = helper.cast(block.getArgument(4).getType(), block.getArgument(2));
  Value value3 = helper.applyfn__sub(value1, value2);
  Value value4 = helper.cast(block.getArgument(4).getType(), block.getArgument(1));
  Value value5 = helper.cast(block.getArgument(4).getType(), block.getArgument(3));
  Value value6 = helper.applyfn__sub(value4, value5);
  Value value7 = helper.applyfn__mul(value3, value6);
  Value value8 = helper.applyfn__add(block.getArgument(4), value7);
  yields.push_back(value8);
  helper.yieldOutputs(yields);
}

LogicalResult DepthwiseConv2DNhwcQOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void DepthwiseConv2DNhwcQOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of PoolingNhwcSumOp
//===----------------------------------------------------------------------===//

ArrayAttr PoolingNhwcSumOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName(), getParallelIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(PoolingNhwcSumOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.strides().getValue<int64_t>({ 0 });
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.dilations().getValue<int64_t>({ 0 });
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.strides().getValue<int64_t>({ 1 });
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.dilations().getValue<int64_t>({ 1 });
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));
  return exprs;
}

ArrayAttr PoolingNhwcSumOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1 * s2 + d3 * s4, d2 * s6 + d4 * s8, d5)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d3, d4)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1, d2, d5)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned PoolingNhwcSumOp::getNumRegionArgs() { return 3; }

std::string PoolingNhwcSumOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool PoolingNhwcSumOp::hasDynamicIndexingMaps() { return true; }
LogicalResult PoolingNhwcSumOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError(
      "incorrect element type for indexing map required attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError(
      "incorrect shape for indexing map required attribute 'strides'");
} else {
  return op->emitError(
    "missing indexing map required attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError(
      "incorrect element type for indexing map required attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError(
      "incorrect shape for indexing map required attribute 'dilations'");
} else {
  return op->emitError(
    "missing indexing map required attribute 'dilations'");
}

  return success();
}

void PoolingNhwcSumOp::regionBuilder(ImplicitLocOpBuilder &b, Block &block) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "PoolingNhwcSumOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  Value value1 = helper.cast(block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.applyfn__add(block.getArgument(2), value1);
  yields.push_back(value2);
  helper.yieldOutputs(yields);
}

LogicalResult PoolingNhwcSumOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void PoolingNhwcSumOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of PoolingNhwcMaxOp
//===----------------------------------------------------------------------===//

ArrayAttr PoolingNhwcMaxOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName(), getParallelIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(PoolingNhwcMaxOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.strides().getValue<int64_t>({ 0 });
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.dilations().getValue<int64_t>({ 0 });
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.strides().getValue<int64_t>({ 1 });
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.dilations().getValue<int64_t>({ 1 });
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));
  return exprs;
}

ArrayAttr PoolingNhwcMaxOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1 * s2 + d3 * s4, d2 * s6 + d4 * s8, d5)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d3, d4)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1, d2, d5)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned PoolingNhwcMaxOp::getNumRegionArgs() { return 3; }

std::string PoolingNhwcMaxOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool PoolingNhwcMaxOp::hasDynamicIndexingMaps() { return true; }
LogicalResult PoolingNhwcMaxOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError(
      "incorrect element type for indexing map required attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError(
      "incorrect shape for indexing map required attribute 'strides'");
} else {
  return op->emitError(
    "missing indexing map required attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError(
      "incorrect element type for indexing map required attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError(
      "incorrect shape for indexing map required attribute 'dilations'");
} else {
  return op->emitError(
    "missing indexing map required attribute 'dilations'");
}

  return success();
}

void PoolingNhwcMaxOp::regionBuilder(ImplicitLocOpBuilder &b, Block &block) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "PoolingNhwcMaxOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  Value value1 = helper.cast(block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.applyfn__max(block.getArgument(2), value1);
  yields.push_back(value2);
  helper.yieldOutputs(yields);
}

LogicalResult PoolingNhwcMaxOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void PoolingNhwcMaxOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of PoolingNchwMaxOp
//===----------------------------------------------------------------------===//

ArrayAttr PoolingNchwMaxOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(PoolingNchwMaxOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));

int64_t cst3 = self.strides().getValue<int64_t>({ 0 });
exprs.push_back(getAffineConstantExpr(cst3, context));

  exprs.push_back(getAffineSymbolExpr(4, context));

int64_t cst5 = self.dilations().getValue<int64_t>({ 0 });
exprs.push_back(getAffineConstantExpr(cst5, context));

  exprs.push_back(getAffineSymbolExpr(6, context));

int64_t cst7 = self.strides().getValue<int64_t>({ 1 });
exprs.push_back(getAffineConstantExpr(cst7, context));

  exprs.push_back(getAffineSymbolExpr(8, context));

int64_t cst9 = self.dilations().getValue<int64_t>({ 1 });
exprs.push_back(getAffineConstantExpr(cst9, context));

  return exprs;
}

ArrayAttr PoolingNchwMaxOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1, d2 * s3 + d4 * s5, d3 * s7 + d5 * s9)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d4, d5)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1, d2, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned PoolingNchwMaxOp::getNumRegionArgs() { return 3; }

std::string PoolingNchwMaxOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool PoolingNchwMaxOp::hasDynamicIndexingMaps() { return true; }
LogicalResult PoolingNchwMaxOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError(
      "incorrect element type for indexing map required attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError(
      "incorrect shape for indexing map required attribute 'strides'");
} else {
  return op->emitError(
    "missing indexing map required attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError(
      "incorrect element type for indexing map required attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError(
      "incorrect shape for indexing map required attribute 'dilations'");
} else {
  return op->emitError(
    "missing indexing map required attribute 'dilations'");
}

  return success();
}

void PoolingNchwMaxOp::regionBuilder(ImplicitLocOpBuilder &b, Block &block) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "PoolingNchwMaxOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  Value value1 = helper.cast(block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.applyfn__max(block.getArgument(2), value1);
  yields.push_back(value2);
  helper.yieldOutputs(yields);
}

LogicalResult PoolingNchwMaxOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void PoolingNchwMaxOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of PoolingNhwcMinOp
//===----------------------------------------------------------------------===//

ArrayAttr PoolingNhwcMinOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName(), getParallelIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(PoolingNhwcMinOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.strides().getValue<int64_t>({ 0 });
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.dilations().getValue<int64_t>({ 0 });
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.strides().getValue<int64_t>({ 1 });
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.dilations().getValue<int64_t>({ 1 });
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));
  return exprs;
}

ArrayAttr PoolingNhwcMinOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1 * s2 + d3 * s4, d2 * s6 + d4 * s8, d5)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d3, d4)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1, d2, d5)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned PoolingNhwcMinOp::getNumRegionArgs() { return 3; }

std::string PoolingNhwcMinOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool PoolingNhwcMinOp::hasDynamicIndexingMaps() { return true; }
LogicalResult PoolingNhwcMinOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError(
      "incorrect element type for indexing map required attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError(
      "incorrect shape for indexing map required attribute 'strides'");
} else {
  return op->emitError(
    "missing indexing map required attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError(
      "incorrect element type for indexing map required attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError(
      "incorrect shape for indexing map required attribute 'dilations'");
} else {
  return op->emitError(
    "missing indexing map required attribute 'dilations'");
}

  return success();
}

void PoolingNhwcMinOp::regionBuilder(ImplicitLocOpBuilder &b, Block &block) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "PoolingNhwcMinOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  Value value1 = helper.cast(block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.applyfn__min(block.getArgument(2), value1);
  yields.push_back(value2);
  helper.yieldOutputs(yields);
}

LogicalResult PoolingNhwcMinOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void PoolingNhwcMinOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of PoolingNdhwcSumOp
//===----------------------------------------------------------------------===//

ArrayAttr PoolingNdhwcSumOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName(), getParallelIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(PoolingNdhwcSumOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.strides().getValue<int64_t>({ 0 });
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.dilations().getValue<int64_t>({ 0 });
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.strides().getValue<int64_t>({ 1 });
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.dilations().getValue<int64_t>({ 1 });
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));

int64_t cst10 = self.strides().getValue<int64_t>({ 2 });
exprs.push_back(getAffineConstantExpr(cst10, context));

  exprs.push_back(getAffineSymbolExpr(11, context));

int64_t cst12 = self.dilations().getValue<int64_t>({ 2 });
exprs.push_back(getAffineConstantExpr(cst12, context));

  exprs.push_back(getAffineSymbolExpr(13, context));
  return exprs;
}

ArrayAttr PoolingNdhwcSumOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13] -> (d0, d1 * s2 + d4 * s4, d2 * s6 + d5 * s8, d3 * s10 + d6 * s12, d7)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 8, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13] -> (d4, d5, d6)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 8, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13] -> (d0, d1, d2, d3, d7)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 8, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned PoolingNdhwcSumOp::getNumRegionArgs() { return 3; }

std::string PoolingNdhwcSumOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool PoolingNdhwcSumOp::hasDynamicIndexingMaps() { return true; }
LogicalResult PoolingNdhwcSumOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError(
      "incorrect element type for indexing map required attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 3 })
    return op->emitError(
      "incorrect shape for indexing map required attribute 'strides'");
} else {
  return op->emitError(
    "missing indexing map required attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError(
      "incorrect element type for indexing map required attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 3 })
    return op->emitError(
      "incorrect shape for indexing map required attribute 'dilations'");
} else {
  return op->emitError(
    "missing indexing map required attribute 'dilations'");
}

  return success();
}

void PoolingNdhwcSumOp::regionBuilder(ImplicitLocOpBuilder &b, Block &block) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "PoolingNdhwcSumOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  Value value1 = helper.cast(block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.applyfn__add(block.getArgument(2), value1);
  yields.push_back(value2);
  helper.yieldOutputs(yields);
}

LogicalResult PoolingNdhwcSumOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void PoolingNdhwcSumOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of PoolingNdhwcMaxOp
//===----------------------------------------------------------------------===//

ArrayAttr PoolingNdhwcMaxOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName(), getParallelIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(PoolingNdhwcMaxOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.strides().getValue<int64_t>({ 0 });
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.dilations().getValue<int64_t>({ 0 });
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.strides().getValue<int64_t>({ 1 });
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.dilations().getValue<int64_t>({ 1 });
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));

int64_t cst10 = self.strides().getValue<int64_t>({ 2 });
exprs.push_back(getAffineConstantExpr(cst10, context));

  exprs.push_back(getAffineSymbolExpr(11, context));

int64_t cst12 = self.dilations().getValue<int64_t>({ 2 });
exprs.push_back(getAffineConstantExpr(cst12, context));

  exprs.push_back(getAffineSymbolExpr(13, context));
  return exprs;
}

ArrayAttr PoolingNdhwcMaxOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13] -> (d0, d1 * s2 + d4 * s4, d2 * s6 + d5 * s8, d3 * s10 + d6 * s12, d7)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 8, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13] -> (d4, d5, d6)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 8, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13] -> (d0, d1, d2, d3, d7)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 8, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned PoolingNdhwcMaxOp::getNumRegionArgs() { return 3; }

std::string PoolingNdhwcMaxOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool PoolingNdhwcMaxOp::hasDynamicIndexingMaps() { return true; }
LogicalResult PoolingNdhwcMaxOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError(
      "incorrect element type for indexing map required attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 3 })
    return op->emitError(
      "incorrect shape for indexing map required attribute 'strides'");
} else {
  return op->emitError(
    "missing indexing map required attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError(
      "incorrect element type for indexing map required attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 3 })
    return op->emitError(
      "incorrect shape for indexing map required attribute 'dilations'");
} else {
  return op->emitError(
    "missing indexing map required attribute 'dilations'");
}

  return success();
}

void PoolingNdhwcMaxOp::regionBuilder(ImplicitLocOpBuilder &b, Block &block) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "PoolingNdhwcMaxOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  Value value1 = helper.cast(block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.applyfn__max(block.getArgument(2), value1);
  yields.push_back(value2);
  helper.yieldOutputs(yields);
}

LogicalResult PoolingNdhwcMaxOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void PoolingNdhwcMaxOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of PoolingNdhwcMinOp
//===----------------------------------------------------------------------===//

ArrayAttr PoolingNdhwcMinOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName(), getReductionIteratorTypeName(), getParallelIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(PoolingNdhwcMinOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.strides().getValue<int64_t>({ 0 });
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.dilations().getValue<int64_t>({ 0 });
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.strides().getValue<int64_t>({ 1 });
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.dilations().getValue<int64_t>({ 1 });
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));

int64_t cst10 = self.strides().getValue<int64_t>({ 2 });
exprs.push_back(getAffineConstantExpr(cst10, context));

  exprs.push_back(getAffineSymbolExpr(11, context));

int64_t cst12 = self.dilations().getValue<int64_t>({ 2 });
exprs.push_back(getAffineConstantExpr(cst12, context));

  exprs.push_back(getAffineSymbolExpr(13, context));
  return exprs;
}

ArrayAttr PoolingNdhwcMinOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13] -> (d0, d1 * s2 + d4 * s4, d2 * s6 + d5 * s8, d3 * s10 + d6 * s12, d7)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 8, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13] -> (d4, d5, d6)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 8, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13] -> (d0, d1, d2, d3, d7)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 8, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned PoolingNdhwcMinOp::getNumRegionArgs() { return 3; }

std::string PoolingNdhwcMinOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool PoolingNdhwcMinOp::hasDynamicIndexingMaps() { return true; }
LogicalResult PoolingNdhwcMinOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError(
      "incorrect element type for indexing map required attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 3 })
    return op->emitError(
      "incorrect shape for indexing map required attribute 'strides'");
} else {
  return op->emitError(
    "missing indexing map required attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError(
      "incorrect element type for indexing map required attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 3 })
    return op->emitError(
      "incorrect shape for indexing map required attribute 'dilations'");
} else {
  return op->emitError(
    "missing indexing map required attribute 'dilations'");
}

  return success();
}

void PoolingNdhwcMinOp::regionBuilder(ImplicitLocOpBuilder &b, Block &block) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "PoolingNdhwcMinOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  Value value1 = helper.cast(block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.applyfn__min(block.getArgument(2), value1);
  yields.push_back(value2);
  helper.yieldOutputs(yields);
}

LogicalResult PoolingNdhwcMinOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void PoolingNdhwcMinOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of FillRng2DOp
//===----------------------------------------------------------------------===//

ArrayAttr FillRng2DOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(FillRng2DOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  return exprs;
}

ArrayAttr FillRng2DOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> ()>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> ()>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> ()>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> (d0, d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned FillRng2DOp::getNumRegionArgs() { return 4; }

std::string FillRng2DOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void FillRng2DOp::regionBuilder(ImplicitLocOpBuilder &b, Block &block) {
  assert(4 > 0 && block.getNumArguments() == 4 &&
         "FillRng2DOp regionBuilder expects 4 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  Value value1 = helper.constant("2147483647 : i64");
  Value value2 = helper.cast(helper.getFloat64Type(), value1);
  Value value3 = helper.index(1);
  Value value4 = helper.cast(helper.getIntegerType(32), value3);
  Value value5 = helper.index(0);
  Value value6 = helper.cast(helper.getIntegerType(32), value5);
  Value value7 = helper.applyfn__add(value6, block.getArgument(2));
  Value value8 = helper.constant("1103515245 : i64");
  Value value9 = helper.cast(helper.getIntegerType(32), value8);
  Value value10 = helper.applyfn__mul(value7, value9);
  Value value11 = helper.constant("12345 : i64");
  Value value12 = helper.cast(helper.getIntegerType(32), value11);
  Value value13 = helper.applyfn__add(value10, value12);
  Value value14 = helper.applyfn__add(value4, value13);
  Value value15 = helper.constant("1103515245 : i64");
  Value value16 = helper.cast(helper.getIntegerType(32), value15);
  Value value17 = helper.applyfn__mul(value14, value16);
  Value value18 = helper.constant("12345 : i64");
  Value value19 = helper.cast(helper.getIntegerType(32), value18);
  Value value20 = helper.applyfn__add(value17, value19);
  Value value21 = helper.cast(helper.getFloat64Type(), value20);
  Value value22 = helper.applyfn__add(value2, value21);
  Value value23 = helper.applyfn__sub(block.getArgument(1), block.getArgument(0));
  Value value24 = helper.constant("2.3283063999999999E-10 : f64");
  Value value25 = helper.cast(helper.getFloat64Type(), value24);
  Value value26 = helper.applyfn__mul(value23, value25);
  Value value27 = helper.applyfn__mul(value22, value26);
  Value value28 = helper.applyfn__add(value27, block.getArgument(0));
  Value value29 = helper.cast(block.getArgument(3).getType(), value28);
  yields.push_back(value29);
  helper.yieldOutputs(yields);
}

LogicalResult FillRng2DOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void FillRng2DOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}

//===----------------------------------------------------------------------===//
// Implementation of SoftPlus2DOp
//===----------------------------------------------------------------------===//

ArrayAttr SoftPlus2DOp::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{ getParallelIteratorTypeName(), getParallelIteratorTypeName() });
}

static SmallVector<AffineExpr> getSymbolBindings(SoftPlus2DOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  return exprs;
}

ArrayAttr SoftPlus2DOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> (d0, d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> (d0, d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned SoftPlus2DOp::getNumRegionArgs() { return 2; }

std::string SoftPlus2DOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void SoftPlus2DOp::regionBuilder(ImplicitLocOpBuilder &b, Block &block) {
  assert(2 > 0 && block.getNumArguments() == 2 &&
         "SoftPlus2DOp regionBuilder expects 2 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  Value value1 = helper.constant("1.000000e+00 : f64");
  Value value2 = helper.cast(block.getArgument(1).getType(), value1);
  Value value3 = helper.cast(block.getArgument(1).getType(), block.getArgument(0));
  Value value4 = helper.applyfn__exp(value3);
  Value value5 = helper.applyfn__add(value2, value4);
  Value value6 = helper.applyfn__log(value5);
  yields.push_back(value6);
  helper.yieldOutputs(yields);
}

LogicalResult SoftPlus2DOp::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void SoftPlus2DOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
}
