
    ArrayAttr MatmulI8I8I32Op::iterator_types() {
      return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef, 8>{ getParallelIteratorTypeName(), getParallelIteratorTypeName(), getReductionIteratorTypeName() });
    }
  // This is temporary until we transition out of manually specified ops that
  // should be auto-generated with linalg-ods-gen.
  ArrayAttr MatmulI8I8I32Op::indexing_maps() {
    MLIRContext *context = getContext();
    AffineExpr d0, d1, d2;
    bindDims(context, d0, d1, d2);
    
	auto s0 = getAffineSymbolExpr(0, context); (void)s0;
	auto s1 = getAffineSymbolExpr(1, context); (void)s1;
	auto s2 = getAffineSymbolExpr(2, context); (void)s2;
	auto map0 = AffineMap::get(3, 3, {d0, d2}, context);
	map0 = map0.replaceDimsAndSymbols({}, { s0, s1, s2 }, 3, 0);
	map0 = simplifyAffineMap(map0);
	auto map1 = AffineMap::get(3, 3, {d2, d1}, context);
	map1 = map1.replaceDimsAndSymbols({}, { s0, s1, s2 }, 3, 0);
	map1 = simplifyAffineMap(map1);
	auto map2 = AffineMap::get(3, 3, {d0, d1}, context);
	map2 = map2.replaceDimsAndSymbols({}, { s0, s1, s2 }, 3, 0);
	map2 = simplifyAffineMap(map2);
    return Builder(context).getAffineMapArrayAttr({ map0, map1, map2 });
  }
  void MatmulI8I8I32Op::regionBuilder(ImplicitLocOpBuilder &b, Block &block) {
    auto args = block.getArguments();
    Value _0(args[0]), _1(args[1]), _2(args[2]);
    
    Value _4 = b.create<SignExtendIOp>(b.getI32Type(), _0);
    Value _5 = b.create<SignExtendIOp>(b.getI32Type(), _1);
    Value _6 = b.create<MulIOp>(_4, _5);
    Value _7 = b.create<AddIOp>(_2, _6);
    b.create<linalg::YieldOp>(ValueRange{ _7 });
  }
    LogicalResult MatmulI8I8I32Op::fold(ArrayRef<Attribute>,
                            SmallVectorImpl<OpFoldResult> &) {
      return foldMemRefCast(*this);
    }
    void MatmulI8I8I32Op::getEffects(SmallVectorImpl<
        SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      SmallVector<Value> inputBuffers = getInputBufferOperands();
      SmallVector<Value> outputBuffers = getOutputBufferOperands();
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), inputBuffers, outputBuffers);
    }
