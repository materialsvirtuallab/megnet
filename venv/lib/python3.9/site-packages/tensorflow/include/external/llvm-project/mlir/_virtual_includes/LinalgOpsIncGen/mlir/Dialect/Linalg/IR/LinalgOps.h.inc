/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Declarations                                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#if defined(GET_OP_CLASSES) || defined(GET_OP_FWD_DEFINES)
#undef GET_OP_FWD_DEFINES
namespace mlir {
namespace linalg {
class IndexOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class InitTensorOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class PadTensorOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class RangeOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class TensorCollapseShapeOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class TensorExpandShapeOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class TiledLoopOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class YieldOp;
} // namespace linalg
} // namespace mlir
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::IndexOp declarations
//===----------------------------------------------------------------------===//

class IndexOpAdaptor {
public:
  IndexOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  IndexOpAdaptor(IndexOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr dim();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class IndexOp : public ::mlir::Op<IndexOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::IndexType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = IndexOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dim")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier dimAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier dimAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.index");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value result();
  ::mlir::IntegerAttr dimAttr();
  uint64_t dim();
  void dimAttr(::mlir::IntegerAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::IntegerAttr dim);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::IntegerAttr dim);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr dim);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, uint64_t dim);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, uint64_t dim);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint64_t dim);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 1 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace linalg
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::IndexOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::InitTensorOp declarations
//===----------------------------------------------------------------------===//

class InitTensorOpAdaptor {
public:
  InitTensorOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  InitTensorOpAdaptor(InitTensorOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange sizes();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr static_sizes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class InitTensorOp : public ::mlir::Op<InitTensorOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = InitTensorOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("static_sizes")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier static_sizesAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier static_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.init_tensor");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range sizes();
  ::mlir::MutableOperandRange sizesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value result();
  ::mlir::ArrayAttr static_sizesAttr();
  ::mlir::ArrayAttr static_sizes();
  void static_sizesAttr(::mlir::ArrayAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange shape, ArrayRef<int64_t> staticShape, Type elementType);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange shape, Type elementType);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ArrayRef<int64_t> staticShape, Type elementType);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ArrayRef<OpFoldResult> sizes, Type elementType, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::ValueRange sizes, ::mlir::ArrayAttr static_sizes);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange sizes, ::mlir::ArrayAttr static_sizes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::LogicalResult reifyResultShapes(::mlir::OpBuilder &builder, ::mlir::ReifiedRankedShapedTypeDims &reifiedReturnShapes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    static StringRef getStaticSizesAttrName() {
      return "static_sizes";
    }

    RankedTensorType getType() {
      return getResult().getType().cast<RankedTensorType>(); }

    // Infer the shape of the result tensor given the static shapes
    // and element type of the result tensor.
    static Type inferResultType(ArrayRef<int64_t> staticSizes, Type elementType);

    // Return true if the size of the tensor is dynamic at `idx`
    bool isDynamicSize(unsigned idx) {
      APInt v = *(static_sizes().getAsValueRange<IntegerAttr>().begin() + idx);
      return ShapedType::isDynamic(v.getSExtValue());
    }

    // Assert that the size of the result tensor is static at `idx`
    // and return the shape.
    int64_t getStaticSize(unsigned idx) {
      assert(!isDynamicSize(idx) && "expected static size");
      APInt v = *(static_sizes().
          template getAsValueRange<IntegerAttr>().begin() + idx);
        return v.getSExtValue();
    }

    // Return the argument position that contains the dynamic size of
    // the tensor at dimension `idx`. Asserts that the shape is
    // dynamic at that `idx`.
    unsigned getIndexOfDynamicSize(unsigned idx) {
      assert(isDynamicSize(idx) && "expected dynamic size");
      return std::count_if(
          static_sizes().getValue().begin(),
          static_sizes().getValue().begin() + idx,
          [&](Attribute attr) {
            return ShapedType::isDynamic(attr.cast<IntegerAttr>().getInt());
          });
    }

    // Return the Value of the dynamic size of the tensor at dimension
    // `idx`. Asserts that the shape is dynamic at that `idx.
    Value getDynamicSize(unsigned idx) {
      return getOperand(getIndexOfDynamicSize(idx));
    }
  

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 1 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace linalg
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::InitTensorOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PadTensorOp declarations
//===----------------------------------------------------------------------===//

class PadTensorOpAdaptor {
public:
  PadTensorOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  PadTensorOpAdaptor(PadTensorOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value source();
  ::mlir::ValueRange low();
  ::mlir::ValueRange high();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr static_low();
  ::mlir::ArrayAttr static_high();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class PadTensorOp : public ::mlir::Op<PadTensorOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::TilingInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = PadTensorOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("static_low"), ::llvm::StringRef("static_high"), ::llvm::StringRef("operand_segment_sizes")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier static_lowAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier static_lowAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier static_highAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier static_highAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.pad_tensor");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value source();
  ::mlir::Operation::operand_range low();
  ::mlir::Operation::operand_range high();
  ::mlir::MutableOperandRange sourceMutable();
  ::mlir::MutableOperandRange lowMutable();
  ::mlir::MutableOperandRange highMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value result();
  ::mlir::Region &region();
  ::mlir::ArrayAttr static_lowAttr();
  ::mlir::ArrayAttr static_low();
  ::mlir::ArrayAttr static_highAttr();
  ::mlir::ArrayAttr static_high();
  void static_lowAttr(::mlir::ArrayAttr attr);
  void static_highAttr(::mlir::ArrayAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, ArrayRef<int64_t> staticLow, ArrayRef<int64_t> staticHigh, ValueRange low, ValueRange high, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, ValueRange low, ValueRange high, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, Value source, ArrayRef<OpFoldResult> low, ArrayRef<OpFoldResult> high, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::ValueRange low, ::mlir::ValueRange high, ::mlir::ArrayAttr static_low, ::mlir::ArrayAttr static_high);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::ValueRange low, ::mlir::ValueRange high, ::mlir::ArrayAttr static_low, ::mlir::ArrayAttr static_high);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::OpFoldResult fold(::llvm::ArrayRef<::mlir::Attribute> operands);
  ::mlir::LogicalResult reifyResultShapes(::mlir::OpBuilder &builder, ::mlir::ReifiedRankedShapedTypeDims &reifiedReturnShapes);
  SmallVector<Value> getDestinationOperands(OpBuilder &b);
  SmallVector<StringRef> getLoopIteratorTypes();
  SmallVector<Range> getLoopBounds(OpBuilder &b);
  Operation *getTiledImplementation(OpBuilder &b, ValueRange  dest, ArrayRef<OpFoldResult>  offsets, ArrayRef<OpFoldResult>  sizes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    static StringRef getStaticLowAttrName() {
      return "static_low";
    }

    static StringRef getStaticHighAttrName() {
      return "static_high";
    }

    RankedTensorType getSourceType() {
      return source().getType().cast<RankedTensorType>();
    }
    RankedTensorType getResultType() {
      return getResult().getType().cast<RankedTensorType>();
    }

    // Infer the shape of the result tensor given the type of the source tensor
    // and paddings.
    static RankedTensorType inferResultType(RankedTensorType sourceType,
                                ArrayRef<int64_t> staticLow,
                                ArrayRef<int64_t> staticHigh);

    // Return a PadTensorOp that pads `source` to `type` size where the static
    // sizes are assumed to be greater than the dynamic sizes. The op performs
    // "high" padding (i.e. it adds trailing padding values until the desired
    // size is met).
    static linalg::PadTensorOp createPadHighOp(
        Type type, Value source, Value pad, Location loc, OpBuilder & builder);

    // Return a PadTensorOp that pads `source to `type` size with `pad` value.
    // I.e., a block will be created and the `pad` value will be yielded
    // directly. If the type passed is nullptr, it is inferred.
    static linalg::PadTensorOp createPadScalarOp(
        Type type, Value source, Value pad, ArrayRef<OpFoldResult> low,
        ArrayRef<OpFoldResult> high, Location loc, OpBuilder & builder);

    // Return the pad value if it is a constant. Return null value otherwise.
    Value getConstantPaddingValue();

    // Return a vector of all the static or dynamic values (low/high padding) of
    // the op.
    inline SmallVector<OpFoldResult> getMixedPadImpl(ArrayAttr staticAttrs,
                                                     ValueRange values) {
      SmallVector<OpFoldResult> res;
      unsigned numDynamic = 0;
      unsigned count = staticAttrs.size();
      for (unsigned idx = 0; idx < count; ++idx) {
        if (ShapedType::isDynamic(staticAttrs[idx].cast<IntegerAttr>().getInt()))
          res.push_back(values[numDynamic++]);
        else
          res.push_back(staticAttrs[idx]);
      }
      return res;
    }
    SmallVector<OpFoldResult> getMixedLowPad() {
      return getMixedPadImpl(static_low(), low());
    }
    SmallVector<OpFoldResult> getMixedHighPad() {
      return getMixedPadImpl(static_high(), high());
    }
    // Return true if low padding is guaranteed to be 0.
    bool hasZeroLowPad() {
      return llvm::all_of(getMixedLowPad(), [](OpFoldResult ofr) {
        return getConstantIntValue(ofr) == static_cast<int64_t>(0);
      });
    }
    // Return true if high padding is guaranteed to be 0.
    bool hasZeroHighPad() {
      return llvm::all_of(getMixedHighPad(), [](OpFoldResult ofr) {
        return getConstantIntValue(ofr) == static_cast<int64_t>(0);
      });
    }
  

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 3 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace linalg
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::PadTensorOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::RangeOp declarations
//===----------------------------------------------------------------------===//

class RangeOpAdaptor {
public:
  RangeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  RangeOpAdaptor(RangeOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value min();
  ::mlir::Value max();
  ::mlir::Value step();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class RangeOp : public ::mlir::Op<RangeOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = RangeOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.range");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value min();
  ::mlir::Value max();
  ::mlir::Value step();
  ::mlir::MutableOperandRange minMutable();
  ::mlir::MutableOperandRange maxMutable();
  ::mlir::MutableOperandRange stepMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value min, Value max, Value step);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value min, ::mlir::Value max, ::mlir::Value step);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value min, ::mlir::Value max, ::mlir::Value step);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
};
} // namespace linalg
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::RangeOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::TensorCollapseShapeOp declarations
//===----------------------------------------------------------------------===//

class TensorCollapseShapeOpAdaptor {
public:
  TensorCollapseShapeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  TensorCollapseShapeOpAdaptor(TensorCollapseShapeOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value src();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr reassociation();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class TensorCollapseShapeOp : public ::mlir::Op<TensorCollapseShapeOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::OneOperand, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TensorCollapseShapeOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("reassociation")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier reassociationAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier reassociationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.tensor_collapse_shape");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value src();
  ::mlir::MutableOperandRange srcMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value result();
  ::mlir::ArrayAttr reassociationAttr();
  ::mlir::ArrayAttr reassociation();
  void reassociationAttr(::mlir::ArrayAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value src, ArrayRef<ReassociationIndices> reassociation, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value src, ArrayRef<ReassociationExprs> reassociation, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, Value src, ArrayRef<ReassociationIndices> reassociation, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, Value src, ArrayRef<ReassociationExprs> reassociation, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value src, ::mlir::ArrayAttr reassociation);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::ArrayAttr reassociation);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::OpFoldResult fold(::llvm::ArrayRef<::mlir::Attribute> operands);
  ::mlir::LogicalResult reifyResultShapes(::mlir::OpBuilder &builder, ::mlir::ReifiedRankedShapedTypeDims &reifiedReturnShapes);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    static StringRef getReassociationAttrName() { return "reassociation"; }
    SmallVector<AffineMap, 4> getReassociationMaps();
    SmallVector<ReassociationExprs, 4> getReassociationExprs();
    SmallVector<ReassociationIndices, 4> getReassociationIndices() {
      SmallVector<ReassociationIndices, 4> reassociationIndices;
      for (auto attr : reassociation())
        reassociationIndices.push_back(llvm::to_vector<2>(
            llvm::map_range(attr.cast<ArrayAttr>(), [&](Attribute indexAttr) {
              return indexAttr.cast<IntegerAttr>().getInt();
            })));
      return reassociationIndices;
    };
  
    RankedTensorType getSrcType() {
      return src().getType().cast<RankedTensorType>();
    }
    RankedTensorType getResultType() {
      return result().getType().cast<RankedTensorType>();
    }
  

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 1 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace linalg
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::TensorCollapseShapeOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::TensorExpandShapeOp declarations
//===----------------------------------------------------------------------===//

class TensorExpandShapeOpAdaptor {
public:
  TensorExpandShapeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  TensorExpandShapeOpAdaptor(TensorExpandShapeOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value src();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr reassociation();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class TensorExpandShapeOp : public ::mlir::Op<TensorExpandShapeOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::OneOperand, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TensorExpandShapeOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("reassociation")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier reassociationAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier reassociationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.tensor_expand_shape");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value src();
  ::mlir::MutableOperandRange srcMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value result();
  ::mlir::ArrayAttr reassociationAttr();
  ::mlir::ArrayAttr reassociation();
  void reassociationAttr(::mlir::ArrayAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value src, ArrayRef<ReassociationIndices> reassociation, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value src, ArrayRef<ReassociationExprs> reassociation, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, Value src, ArrayRef<ReassociationIndices> reassociation, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, Value src, ArrayRef<ReassociationExprs> reassociation, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value src, ::mlir::ArrayAttr reassociation);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::ArrayAttr reassociation);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::OpFoldResult fold(::llvm::ArrayRef<::mlir::Attribute> operands);
  ::mlir::LogicalResult reifyResultShapes(::mlir::OpBuilder &builder, ::mlir::ReifiedRankedShapedTypeDims &reifiedReturnShapes);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    static StringRef getReassociationAttrName() { return "reassociation"; }
    SmallVector<AffineMap, 4> getReassociationMaps();
    SmallVector<ReassociationExprs, 4> getReassociationExprs();
    SmallVector<ReassociationIndices, 4> getReassociationIndices() {
      SmallVector<ReassociationIndices, 4> reassociationIndices;
      for (auto attr : reassociation())
        reassociationIndices.push_back(llvm::to_vector<2>(
            llvm::map_range(attr.cast<ArrayAttr>(), [&](Attribute indexAttr) {
              return indexAttr.cast<IntegerAttr>().getInt();
            })));
      return reassociationIndices;
    };
  
    RankedTensorType getSrcType() {
      return src().getType().cast<RankedTensorType>();
    }
    RankedTensorType getResultType() {
      return result().getType().cast<RankedTensorType>();
    }
  

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 1 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace linalg
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::TensorExpandShapeOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::TiledLoopOp declarations
//===----------------------------------------------------------------------===//

class TiledLoopOpAdaptor {
public:
  TiledLoopOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  TiledLoopOpAdaptor(TiledLoopOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange lowerBound();
  ::mlir::ValueRange upperBound();
  ::mlir::ValueRange step();
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr iterator_types();
  ::mlir::ArrayAttr distribution_types();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class TiledLoopOp : public ::mlir::Op<TiledLoopOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::LoopLikeOpInterface::Trait, ::mlir::OpTrait::HasRecursiveSideEffects, ::mlir::OpTrait::SingleBlockImplicitTerminator<linalg::YieldOp>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TiledLoopOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("iterator_types"), ::llvm::StringRef("distribution_types"), ::llvm::StringRef("operand_segment_sizes")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier iterator_typesAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier iterator_typesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier distribution_typesAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier distribution_typesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.tiled_loop");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range lowerBound();
  ::mlir::Operation::operand_range upperBound();
  ::mlir::Operation::operand_range step();
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange lowerBoundMutable();
  ::mlir::MutableOperandRange upperBoundMutable();
  ::mlir::MutableOperandRange stepMutable();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range results();
  ::mlir::Region &region();
  ::mlir::ArrayAttr iterator_typesAttr();
  ::mlir::ArrayAttr iterator_types();
  ::mlir::ArrayAttr distribution_typesAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > distribution_types();
  void iterator_typesAttr(::mlir::ArrayAttr attr);
  void distribution_typesAttr(::mlir::ArrayAttr attr);
  ::mlir::Attribute removeDistribution_typesAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange lowerBounds, ValueRange upperBounds, ValueRange steps, ValueRange inputs, ValueRange outputs, ArrayAttr iteratorTypes, Optional<ArrayAttr> distributionTypes, function_ref<void (OpBuilder &, Location, /*ivs=*/ValueRange,/*inputs=*/ValueRange, /*outputs=*/ValueRange)> bodyBuilderFn = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange lowerBounds, ValueRange upperBounds, ValueRange steps, ValueRange inputs, ValueRange outputs, ArrayAttr iteratorTypes, function_ref<void (OpBuilder &, Location, /*ivs=*/ValueRange,/*inputs=*/ValueRange, /*outputs=*/ValueRange)> bodyBuilderFn = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, ::mlir::ValueRange inputs, ::mlir::ValueRange outputs, ::mlir::ArrayAttr iterator_types, /*optional*/::mlir::ArrayAttr distribution_types);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  bool isDefinedOutsideOfLoop(::mlir::Value  value);
  ::mlir::Region &getLoopBody();
  ::mlir::LogicalResult moveOutOfLoop(::mlir::ArrayRef<::mlir::Operation *> ops);

    /// Number of loops
    unsigned getNumLoops() { return step().size(); }

    /// Number of input operands
    unsigned getNumInputs() { return inputs().size(); }

    /// Number of output operands
    unsigned getNumOutputs() { return outputs().size(); }

    /// Number of operands controlling the loop: lbs, ubs, steps
    unsigned getNumControlOperands() { return 3 * getNumLoops(); }

    ValueRange getInductionVars() {
      return getBody()->getArguments().take_front(getNumLoops());
    }
    ValueRange getRegionInputArgs() {
      return getBody()->getArguments().slice(getNumLoops(), inputs().size());
    }
    ValueRange getRegionOutputArgs() {
      return getBody()->getArguments().take_back(outputs().size());
    }

    void setDistributionTypes(Builder& b, ArrayRef<StringRef> types) {
      assert(types.size() == getNumLoops() &&
             "expected distribution type for every dimension");
      distribution_typesAttr(b.getStrArrayAttr(types));
    }

    void setLowerBounds(ValueRange lowerBounds) {
      unsigned numLoops = getNumLoops();
      assert(lowerBounds.size() == numLoops &&
             "expected lower bounds for every loop dimension");
      for (unsigned i = 0; i < numLoops; ++i)
        setOperand(i, lowerBounds[i]);
    }

    void setUpperBounds(ValueRange upperBounds) {
      unsigned numLoops = getNumLoops();
      assert(upperBounds.size() == numLoops &&
             "expected upper bounds for every loop dimension");
      for (unsigned i = 0, pos = numLoops; i < numLoops; ++i, ++pos)
        setOperand(pos, upperBounds[i]);
    }

    void setSteps(ValueRange steps) {
      unsigned numLoops = getNumLoops();
      assert(steps.size() == numLoops &&
             "expected upper bounds for every loop dimension");
      for (unsigned i = 0, pos = 2 * numLoops; i < numLoops; ++i, ++pos)
        setOperand(pos, steps[i]);
    }

    /// Block argument that corresponds to the `input` or `output` operand.
    BlockArgument getTiedBlockArgument(OpOperand& operand) {
      auto operandIndex = operand.getOperandNumber();
      assert(
          operandIndex >= getNumControlOperands() &&
          operandIndex < getNumOperands() &&
          "tied block arg is defined only for `input` and `output` arguments");
      return getBody()->getArgument(operandIndex - 2 * getNumLoops());
    }

   /// Result that corresponds to the `outputs` argument of tensor type.
   OpResult getTiedOpResult(OpOperand& opOperand) {
      // No result can correspond to a memref argument.
      if (opOperand.get().getType().isa<MemRefType>()) return OpResult();

      // Check whether the operand index is in bounds of `outputs()` arg.
      int operandIndex = opOperand.getOperandNumber();
      int outputIndexStart =
          getNumControlOperands() + inputs().size();
      int outputIndexEnd = outputIndexStart + outputs().size();
      if (operandIndex < outputIndexStart || operandIndex >= outputIndexEnd)
        return OpResult();

      // Count tensor arguments in `outputs` to compute the result index.
      int tensorId = -1;
      for (int i = outputIndexStart; i <= operandIndex; ++i)
        tensorId += getOperand(i).getType().isa<RankedTensorType>();
      return getOperation()->getResult(tensorId);
    }

    /// Append `operand` to the `input` arguments.
    OpOperand& appendInputOperand(OpBuilder& builder, Value operand) {
      int numLoops = getNumLoops();
      int numInputs = getNumInputs();
      int numOutputs = getNumOutputs();

      getOperation()->insertOperands(getNumControlOperands() + numInputs,
                                     operand);
      getBody()->insertArgument(numLoops + numInputs, operand.getType());
      getOperation()->setAttr(
          TiledLoopOp::getOperandSegmentSizeAttr(),
          builder.getI32VectorAttr(
              {numLoops, numLoops, numLoops, numInputs + 1, numOutputs}));
      return getOperation()->getOpOperand(getNumControlOperands() + numInputs);
    }

    /// Append `operand` to the `output` arguments.
    OpOperand& appendOutputOperand(OpBuilder& builder, Value operand) {
      int numLoops = getNumLoops();
      int numInputs = getNumInputs();
      int numOutputs = getNumOutputs();

      getOperation()->insertOperands(
          getNumControlOperands() + numInputs + numOutputs, operand);
      getBody()->insertArgument(numLoops + numInputs + numOutputs,
                                operand.getType());
      getOperation()->setAttr(
          TiledLoopOp::getOperandSegmentSizeAttr(),
          builder.getI32VectorAttr(
              {numLoops, numLoops, numLoops, numInputs, numOutputs + 1}));
      return getOperation()->getOpOperand(getNumControlOperands() + numInputs +
                                          numOutputs);
    }

    /// Erase `operand` from the `input` or `output` arguments.
    void eraseOperand(OpBuilder& builder, OpOperand& operand) {
      int numInputs = getNumInputs();
      int numLoops = getNumLoops();
      int numOutputs = getNumOutputs();
      int numControlOperands = getNumControlOperands();

      int operandIndex = operand.getOperandNumber();
      assert(operandIndex >= numControlOperands &&
             operandIndex < static_cast<int>(getNumOperands()) &&
             "Can erase only `input` or `output` operand");

      if (operandIndex >= numControlOperands + numInputs)
        --numOutputs;
      else
        --numInputs;

      getOperation()->eraseOperand(operandIndex);
      getBody()->eraseArgument(operandIndex - 2 * numLoops);
      getOperation()->setAttr(
          TiledLoopOp::getOperandSegmentSizeAttr(),
          builder.getI32VectorAttr(
              {numLoops, numLoops, numLoops, numInputs, numOutputs}));
    }

    OpOperand* findInputOperand(Value value) {
      OperandRange::iterator it = llvm::find(inputs(), value);
      if (it == inputs().end()) return nullptr;
      return it.getBase();
    }

    OpOperand* findOutputOperand(Value value) {
      OperandRange::iterator it = llvm::find(outputs(), value);
      if (it == outputs().end()) return nullptr;
      return it.getBase();
    }

    /// Return whether the op has only MemRef input and outputs.
    bool hasBufferSemantics() {
      Operation* op = this->getOperation();
      return op->getNumResults() == 0 &&
             llvm::all_of(op->getOpOperands(), [&](OpOperand & operand) {
               return !operand.get().getType().template isa<ShapedType>() ||
                      operand.get().getType().template isa<MemRefType>();
             });
    }

    /// Return whether the loop dimension is parallel or not.
    bool isParallelDimension(unsigned dim) {
      StringAttr attr = this->iterator_types()[dim].cast<StringAttr>();
      return attr.getValue() == getParallelIteratorTypeName();
    }
  

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 3 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace linalg
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::TiledLoopOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::YieldOp declarations
//===----------------------------------------------------------------------===//

class YieldOpAdaptor {
public:
  YieldOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  YieldOpAdaptor(YieldOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange values();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class YieldOp : public ::mlir::Op<YieldOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::ReturnLike, ::mlir::OpTrait::IsTerminator> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = YieldOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.yield");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range values();
  ::mlir::MutableOperandRange valuesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange values);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
};
} // namespace linalg
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::YieldOp)


#endif  // GET_OP_CLASSES

