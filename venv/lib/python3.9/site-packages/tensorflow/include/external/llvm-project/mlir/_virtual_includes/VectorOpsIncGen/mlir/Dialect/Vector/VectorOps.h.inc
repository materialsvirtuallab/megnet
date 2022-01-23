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
namespace vector {
class BitCastOp;
} // namespace vector
} // namespace mlir
namespace mlir {
namespace vector {
class BroadcastOp;
} // namespace vector
} // namespace mlir
namespace mlir {
namespace vector {
class CompressStoreOp;
} // namespace vector
} // namespace mlir
namespace mlir {
namespace vector {
class ConstantMaskOp;
} // namespace vector
} // namespace mlir
namespace mlir {
namespace vector {
class ContractionOp;
} // namespace vector
} // namespace mlir
namespace mlir {
namespace vector {
class CreateMaskOp;
} // namespace vector
} // namespace mlir
namespace mlir {
namespace vector {
class ExpandLoadOp;
} // namespace vector
} // namespace mlir
namespace mlir {
namespace vector {
class ExtractElementOp;
} // namespace vector
} // namespace mlir
namespace mlir {
namespace vector {
class ExtractMapOp;
} // namespace vector
} // namespace mlir
namespace mlir {
namespace vector {
class ExtractOp;
} // namespace vector
} // namespace mlir
namespace mlir {
namespace vector {
class ExtractStridedSliceOp;
} // namespace vector
} // namespace mlir
namespace mlir {
namespace vector {
class FMAOp;
} // namespace vector
} // namespace mlir
namespace mlir {
namespace vector {
class FlatTransposeOp;
} // namespace vector
} // namespace mlir
namespace mlir {
namespace vector {
class GatherOp;
} // namespace vector
} // namespace mlir
namespace mlir {
namespace vector {
class InsertElementOp;
} // namespace vector
} // namespace mlir
namespace mlir {
namespace vector {
class InsertMapOp;
} // namespace vector
} // namespace mlir
namespace mlir {
namespace vector {
class InsertOp;
} // namespace vector
} // namespace mlir
namespace mlir {
namespace vector {
class InsertStridedSliceOp;
} // namespace vector
} // namespace mlir
namespace mlir {
namespace vector {
class LoadOp;
} // namespace vector
} // namespace mlir
namespace mlir {
namespace vector {
class MaskedLoadOp;
} // namespace vector
} // namespace mlir
namespace mlir {
namespace vector {
class MaskedStoreOp;
} // namespace vector
} // namespace mlir
namespace mlir {
namespace vector {
class MatmulOp;
} // namespace vector
} // namespace mlir
namespace mlir {
namespace vector {
class MultiDimReductionOp;
} // namespace vector
} // namespace mlir
namespace mlir {
namespace vector {
class OuterProductOp;
} // namespace vector
} // namespace mlir
namespace mlir {
namespace vector {
class PrintOp;
} // namespace vector
} // namespace mlir
namespace mlir {
namespace vector {
class ReductionOp;
} // namespace vector
} // namespace mlir
namespace mlir {
namespace vector {
class ReshapeOp;
} // namespace vector
} // namespace mlir
namespace mlir {
namespace vector {
class ScatterOp;
} // namespace vector
} // namespace mlir
namespace mlir {
namespace vector {
class ShapeCastOp;
} // namespace vector
} // namespace mlir
namespace mlir {
namespace vector {
class ShuffleOp;
} // namespace vector
} // namespace mlir
namespace mlir {
namespace vector {
class StoreOp;
} // namespace vector
} // namespace mlir
namespace mlir {
namespace vector {
class TransferReadOp;
} // namespace vector
} // namespace mlir
namespace mlir {
namespace vector {
class TransferWriteOp;
} // namespace vector
} // namespace mlir
namespace mlir {
namespace vector {
class TransposeOp;
} // namespace vector
} // namespace mlir
namespace mlir {
namespace vector {
class TypeCastOp;
} // namespace vector
} // namespace mlir
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::BitCastOp declarations
//===----------------------------------------------------------------------===//

class BitCastOpAdaptor {
public:
  BitCastOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  BitCastOpAdaptor(BitCastOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value source();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class BitCastOp : public ::mlir::Op<BitCastOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::VectorType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::OneOperand, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = BitCastOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("vector.bitcast");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value source();
  ::mlir::MutableOperandRange sourceMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value result();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  ::mlir::OpFoldResult fold(::llvm::ArrayRef<::mlir::Attribute> operands);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    VectorType getSourceVectorType() {
      return source().getType().cast<VectorType>();
    }
    VectorType getResultVectorType() {
      return getResult().getType().cast<VectorType>();
    }
  
};
} // namespace vector
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::vector::BitCastOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::BroadcastOp declarations
//===----------------------------------------------------------------------===//

class BroadcastOpAdaptor {
public:
  BroadcastOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  BroadcastOpAdaptor(BroadcastOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value source();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class BroadcastOp : public ::mlir::Op<BroadcastOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::VectorType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::OneOperand, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = BroadcastOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("vector.broadcast");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value source();
  ::mlir::MutableOperandRange sourceMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value vector();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type vector, ::mlir::Value source);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::OpFoldResult fold(::llvm::ArrayRef<::mlir::Attribute> operands);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    Type getSourceType() { return source().getType(); }
    VectorType getVectorType() {
      return vector().getType().cast<VectorType>();
    }
  
};
} // namespace vector
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::vector::BroadcastOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::CompressStoreOp declarations
//===----------------------------------------------------------------------===//

class CompressStoreOpAdaptor {
public:
  CompressStoreOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  CompressStoreOpAdaptor(CompressStoreOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value base();
  ::mlir::ValueRange indices();
  ::mlir::Value mask();
  ::mlir::Value valueToStore();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class CompressStoreOp : public ::mlir::Op<CompressStoreOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<3>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CompressStoreOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("vector.compressstore");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value base();
  ::mlir::Operation::operand_range indices();
  ::mlir::Value mask();
  ::mlir::Value valueToStore();
  ::mlir::MutableOperandRange baseMutable();
  ::mlir::MutableOperandRange indicesMutable();
  ::mlir::MutableOperandRange maskMutable();
  ::mlir::MutableOperandRange valueToStoreMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value base, ::mlir::ValueRange indices, ::mlir::Value mask, ::mlir::Value valueToStore);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value base, ::mlir::ValueRange indices, ::mlir::Value mask, ::mlir::Value valueToStore);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    MemRefType getMemRefType() {
      return base().getType().cast<MemRefType>();
    }
    VectorType getMaskVectorType() {
      return mask().getType().cast<VectorType>();
    }
    VectorType getVectorType() {
      return valueToStore().getType().cast<VectorType>();
    }
  
};
} // namespace vector
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::vector::CompressStoreOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::ConstantMaskOp declarations
//===----------------------------------------------------------------------===//

class ConstantMaskOpAdaptor {
public:
  ConstantMaskOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ConstantMaskOpAdaptor(ConstantMaskOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr mask_dim_sizes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ConstantMaskOp : public ::mlir::Op<ConstantMaskOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::VectorType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConstantMaskOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("mask_dim_sizes")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier mask_dim_sizesAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier mask_dim_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("vector.constant_mask");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::ArrayAttr mask_dim_sizesAttr();
  ::mlir::ArrayAttr mask_dim_sizes();
  void mask_dim_sizesAttr(::mlir::ArrayAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::ArrayAttr mask_dim_sizes);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ArrayAttr mask_dim_sizes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    static StringRef getMaskDimSizesAttrName() { return "mask_dim_sizes"; }
  

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 1 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace vector
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::vector::ConstantMaskOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::ContractionOp declarations
//===----------------------------------------------------------------------===//

class ContractionOpAdaptor {
public:
  ContractionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ContractionOpAdaptor(ContractionOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value lhs();
  ::mlir::Value rhs();
  ::mlir::Value acc();
  ::mlir::ValueRange masks();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr indexing_maps();
  ::mlir::ArrayAttr iterator_types();
  ::mlir::vector::CombiningKindAttr kind();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ContractionOp : public ::mlir::Op<ContractionOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<3>::Impl, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::VectorUnrollOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ContractionOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("iterator_types"), ::llvm::StringRef("kind")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier indexing_mapsAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier indexing_mapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier iterator_typesAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier iterator_typesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier kindAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier kindAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("vector.contract");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value lhs();
  ::mlir::Value rhs();
  ::mlir::Value acc();
  ::mlir::Operation::operand_range masks();
  ::mlir::MutableOperandRange lhsMutable();
  ::mlir::MutableOperandRange rhsMutable();
  ::mlir::MutableOperandRange accMutable();
  ::mlir::MutableOperandRange masksMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::ArrayAttr indexing_mapsAttr();
  ::mlir::ArrayAttr indexing_maps();
  ::mlir::ArrayAttr iterator_typesAttr();
  ::mlir::ArrayAttr iterator_types();
  ::mlir::vector::CombiningKindAttr kindAttr();
  ::mlir::vector::CombiningKind kind();
  void indexing_mapsAttr(::mlir::ArrayAttr attr);
  void iterator_typesAttr(::mlir::ArrayAttr attr);
  void kindAttr(::mlir::vector::CombiningKindAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value lhs, Value rhs, Value acc, ArrayAttr indexingMaps, ArrayAttr iteratorTypes);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value lhs, Value rhs, Value acc, ArrayRef<ArrayRef<AffineExpr>> indexingExprs, ArrayRef<StringRef> iteratorTypes);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value acc, ::mlir::ValueRange masks, ::mlir::ArrayAttr indexing_maps, ::mlir::ArrayAttr iterator_types, ::mlir::vector::CombiningKindAttr kind);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value acc, ::mlir::ValueRange masks, ::mlir::ArrayAttr indexing_maps, ::mlir::ArrayAttr iterator_types, ::mlir::vector::CombiningKindAttr kind);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value acc, ::mlir::ValueRange masks, ::mlir::ArrayAttr indexing_maps, ::mlir::ArrayAttr iterator_types, ::mlir::vector::CombiningKind kind = CombiningKind::ADD);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value acc, ::mlir::ValueRange masks, ::mlir::ArrayAttr indexing_maps, ::mlir::ArrayAttr iterator_types, ::mlir::vector::CombiningKind kind = CombiningKind::ADD);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::Optional<::mlir::SmallVector<int64_t, 4>> getShapeForUnroll();
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    VectorType getLhsType() {
      return lhs().getType().cast<VectorType>();
    }
    VectorType getRhsType() {
      return rhs().getType().cast<VectorType>();
    }
    Type getAccType() { return acc().getType(); }
    VectorType getLHSVectorMaskType() {
      if (llvm::size(masks()) != 2) return VectorType();
      return getOperand(3).getType().cast<VectorType>();
    }
    VectorType getRHSVectorMaskType() {
      if (llvm::size(masks()) != 2) return VectorType();
      return getOperand(4).getType().cast<VectorType>();
    }
    Type getResultType() { return getResult().getType(); }
    ArrayRef<StringRef> getTraitAttrNames();
    SmallVector<AffineMap, 4> getIndexingMaps();
    static unsigned getAccOperandIndex() { return 2; }

    // Returns the bounds of each dimension in the iteration space spanned
    // by the iterator types of this operation.
    void getIterationBounds(SmallVectorImpl<int64_t> &iterationBounds);

    // Returns a list of index maps, where there is a list entry for each
    // op indexing map attribute (i.e. one for each input and output, with
    // the output listed last). Each index map, maps from this operations
    // iteration space, to vector dimensions of the maps input/output.
    void getIterationIndexMap(
      std::vector<DenseMap<int64_t, int64_t>> &iterationIndexMap);

    std::vector<std::pair<int64_t, int64_t>> getContractingDimMap();
    std::vector<std::pair<int64_t, int64_t>> getBatchDimMap();

    static constexpr StringRef getKindAttrName() { return "kind"; }

    static CombiningKind getDefaultKind() {
      return CombiningKind::ADD;
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
} // namespace vector
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::vector::ContractionOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::CreateMaskOp declarations
//===----------------------------------------------------------------------===//

class CreateMaskOpAdaptor {
public:
  CreateMaskOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  CreateMaskOpAdaptor(CreateMaskOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange operands();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class CreateMaskOp : public ::mlir::Op<CreateMaskOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::VectorType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CreateMaskOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("vector.create_mask");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range operands();
  ::mlir::MutableOperandRange operandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::ValueRange operands);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
};
} // namespace vector
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::vector::CreateMaskOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::ExpandLoadOp declarations
//===----------------------------------------------------------------------===//

class ExpandLoadOpAdaptor {
public:
  ExpandLoadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ExpandLoadOpAdaptor(ExpandLoadOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value base();
  ::mlir::ValueRange indices();
  ::mlir::Value mask();
  ::mlir::Value pass_thru();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ExpandLoadOp : public ::mlir::Op<ExpandLoadOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::VectorType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<3>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExpandLoadOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("vector.expandload");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value base();
  ::mlir::Operation::operand_range indices();
  ::mlir::Value mask();
  ::mlir::Value pass_thru();
  ::mlir::MutableOperandRange baseMutable();
  ::mlir::MutableOperandRange indicesMutable();
  ::mlir::MutableOperandRange maskMutable();
  ::mlir::MutableOperandRange pass_thruMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value result();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value base, ::mlir::ValueRange indices, ::mlir::Value mask, ::mlir::Value pass_thru);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value base, ::mlir::ValueRange indices, ::mlir::Value mask, ::mlir::Value pass_thru);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    MemRefType getMemRefType() {
      return base().getType().cast<MemRefType>();
    }
    VectorType getMaskVectorType() {
      return mask().getType().cast<VectorType>();
    }
    VectorType getPassThruVectorType() {
      return pass_thru().getType().cast<VectorType>();
    }
    VectorType getVectorType() {
      return result().getType().cast<VectorType>();
    }
  
};
} // namespace vector
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::vector::ExpandLoadOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::ExtractElementOp declarations
//===----------------------------------------------------------------------===//

class ExtractElementOpAdaptor {
public:
  ExtractElementOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ExtractElementOpAdaptor(ExtractElementOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value vector();
  ::mlir::Value position();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ExtractElementOp : public ::mlir::Op<ExtractElementOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExtractElementOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("vector.extractelement");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value vector();
  ::mlir::Value position();
  ::mlir::MutableOperandRange vectorMutable();
  ::mlir::MutableOperandRange positionMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value result();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, int64_t position);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, Value position);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value vector, ::mlir::Value position);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value vector, ::mlir::Value position);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    VectorType getVectorType() {
      return vector().getType().cast<VectorType>();
    }
  
};
} // namespace vector
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::vector::ExtractElementOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::ExtractMapOp declarations
//===----------------------------------------------------------------------===//

class ExtractMapOpAdaptor {
public:
  ExtractMapOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ExtractMapOpAdaptor(ExtractMapOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value vector();
  ::mlir::ValueRange ids();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ExtractMapOp : public ::mlir::Op<ExtractMapOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::VectorType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExtractMapOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("vector.extract_map");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value vector();
  ::mlir::Operation::operand_range ids();
  ::mlir::MutableOperandRange vectorMutable();
  ::mlir::MutableOperandRange idsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value vector, ValueRange ids, ArrayRef<int64_t> multiplicity, AffineMap map);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value vector, ::mlir::ValueRange ids);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value vector, ::mlir::ValueRange ids);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  ::mlir::OpFoldResult fold(::llvm::ArrayRef<::mlir::Attribute> operands);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    VectorType getSourceVectorType() {
      return vector().getType().cast<VectorType>();
    }
    VectorType getResultType() {
      return getResult().getType().cast<VectorType>();
    }
    void getMultiplicity(SmallVectorImpl<int64_t> &multiplicity);
    AffineMap map();
  
};
} // namespace vector
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::vector::ExtractMapOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::ExtractOp declarations
//===----------------------------------------------------------------------===//

class ExtractOpAdaptor {
public:
  ExtractOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ExtractOpAdaptor(ExtractOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value vector();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr position();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ExtractOp : public ::mlir::Op<ExtractOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::OneOperand, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExtractOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("position")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier positionAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier positionAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("vector.extract");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value vector();
  ::mlir::MutableOperandRange vectorMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::ArrayAttr positionAttr();
  ::mlir::ArrayAttr position();
  void positionAttr(::mlir::ArrayAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, ArrayRef<int64_t> position);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, ValueRange position);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value vector, ::mlir::ArrayAttr position);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value vector, ::mlir::ArrayAttr position);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::OpFoldResult fold(::llvm::ArrayRef<::mlir::Attribute> operands);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    static StringRef getPositionAttrName() { return "position"; }
    VectorType getVectorType() {
      return vector().getType().cast<VectorType>();
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
} // namespace vector
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::vector::ExtractOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::ExtractStridedSliceOp declarations
//===----------------------------------------------------------------------===//

class ExtractStridedSliceOpAdaptor {
public:
  ExtractStridedSliceOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ExtractStridedSliceOpAdaptor(ExtractStridedSliceOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value vector();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr offsets();
  ::mlir::ArrayAttr sizes();
  ::mlir::ArrayAttr strides();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ExtractStridedSliceOp : public ::mlir::Op<ExtractStridedSliceOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::VectorType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::OneOperand, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExtractStridedSliceOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("offsets"), ::llvm::StringRef("sizes"), ::llvm::StringRef("strides")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier offsetsAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier offsetsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier sizesAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier stridesAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("vector.extract_strided_slice");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value vector();
  ::mlir::MutableOperandRange vectorMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::ArrayAttr offsetsAttr();
  ::mlir::ArrayAttr offsets();
  ::mlir::ArrayAttr sizesAttr();
  ::mlir::ArrayAttr sizes();
  ::mlir::ArrayAttr stridesAttr();
  ::mlir::ArrayAttr strides();
  void offsetsAttr(::mlir::ArrayAttr attr);
  void sizesAttr(::mlir::ArrayAttr attr);
  void stridesAttr(::mlir::ArrayAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, ArrayRef<int64_t> offsets, ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value vector, ::mlir::ArrayAttr offsets, ::mlir::ArrayAttr sizes, ::mlir::ArrayAttr strides);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value vector, ::mlir::ArrayAttr offsets, ::mlir::ArrayAttr sizes, ::mlir::ArrayAttr strides);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::OpFoldResult fold(::llvm::ArrayRef<::mlir::Attribute> operands);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    static StringRef getOffsetsAttrName() { return "offsets"; }
    static StringRef getSizesAttrName() { return "sizes"; }
    static StringRef getStridesAttrName() { return "strides"; }
    VectorType getVectorType(){ return vector().getType().cast<VectorType>(); }
    void getOffsets(SmallVectorImpl<int64_t> &results);
  

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 3 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace vector
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::vector::ExtractStridedSliceOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::FMAOp declarations
//===----------------------------------------------------------------------===//

class FMAOpAdaptor {
public:
  FMAOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  FMAOpAdaptor(FMAOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value lhs();
  ::mlir::Value rhs();
  ::mlir::Value acc();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class FMAOp : public ::mlir::Op<FMAOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::VectorType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::VectorUnrollOpInterface::Trait, ::mlir::OpTrait::Elementwise, ::mlir::OpTrait::Scalarizable, ::mlir::OpTrait::Vectorizable, ::mlir::OpTrait::Tensorizable> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = FMAOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("vector.fma");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value lhs();
  ::mlir::Value rhs();
  ::mlir::Value acc();
  ::mlir::MutableOperandRange lhsMutable();
  ::mlir::MutableOperandRange rhsMutable();
  ::mlir::MutableOperandRange accMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value result();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value lhs, Value rhs, Value acc);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value acc);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value acc);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  ::mlir::Optional<::mlir::SmallVector<int64_t, 4>> getShapeForUnroll();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    VectorType getVectorType() { return lhs().getType().cast<VectorType>(); }
  
};
} // namespace vector
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::vector::FMAOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::FlatTransposeOp declarations
//===----------------------------------------------------------------------===//

class FlatTransposeOpAdaptor {
public:
  FlatTransposeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  FlatTransposeOpAdaptor(FlatTransposeOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value matrix();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr rows();
  ::mlir::IntegerAttr columns();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class FlatTransposeOp : public ::mlir::Op<FlatTransposeOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::VectorType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::OneOperand, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = FlatTransposeOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("rows"), ::llvm::StringRef("columns")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier rowsAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier rowsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier columnsAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier columnsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("vector.flat_transpose");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value matrix();
  ::mlir::MutableOperandRange matrixMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  ::mlir::IntegerAttr rowsAttr();
  uint32_t rows();
  ::mlir::IntegerAttr columnsAttr();
  uint32_t columns();
  void rowsAttr(::mlir::IntegerAttr attr);
  void columnsAttr(::mlir::IntegerAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value matrix, ::mlir::IntegerAttr rows, ::mlir::IntegerAttr columns);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value matrix, ::mlir::IntegerAttr rows, ::mlir::IntegerAttr columns);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value matrix, uint32_t rows, uint32_t columns);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value matrix, uint32_t rows, uint32_t columns);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 2 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace vector
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::vector::FlatTransposeOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::GatherOp declarations
//===----------------------------------------------------------------------===//

class GatherOpAdaptor {
public:
  GatherOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  GatherOpAdaptor(GatherOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value base();
  ::mlir::ValueRange indices();
  ::mlir::Value index_vec();
  ::mlir::Value mask();
  ::mlir::Value pass_thru();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class GatherOp : public ::mlir::Op<GatherOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::VectorType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<4>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GatherOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("vector.gather");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value base();
  ::mlir::Operation::operand_range indices();
  ::mlir::Value index_vec();
  ::mlir::Value mask();
  ::mlir::Value pass_thru();
  ::mlir::MutableOperandRange baseMutable();
  ::mlir::MutableOperandRange indicesMutable();
  ::mlir::MutableOperandRange index_vecMutable();
  ::mlir::MutableOperandRange maskMutable();
  ::mlir::MutableOperandRange pass_thruMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value result();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value base, ::mlir::ValueRange indices, ::mlir::Value index_vec, ::mlir::Value mask, ::mlir::Value pass_thru);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value base, ::mlir::ValueRange indices, ::mlir::Value index_vec, ::mlir::Value mask, ::mlir::Value pass_thru);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    MemRefType getMemRefType() {
      return base().getType().cast<MemRefType>();
    }
    VectorType getIndexVectorType() {
      return index_vec().getType().cast<VectorType>();
    }
    VectorType getMaskVectorType() {
      return mask().getType().cast<VectorType>();
    }
    VectorType getPassThruVectorType() {
      return pass_thru().getType().cast<VectorType>();
    }
    VectorType getVectorType() {
      return result().getType().cast<VectorType>();
    }
  
};
} // namespace vector
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::vector::GatherOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::InsertElementOp declarations
//===----------------------------------------------------------------------===//

class InsertElementOpAdaptor {
public:
  InsertElementOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  InsertElementOpAdaptor(InsertElementOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value source();
  ::mlir::Value dest();
  ::mlir::Value position();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class InsertElementOp : public ::mlir::Op<InsertElementOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::VectorType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = InsertElementOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("vector.insertelement");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value source();
  ::mlir::Value dest();
  ::mlir::Value position();
  ::mlir::MutableOperandRange sourceMutable();
  ::mlir::MutableOperandRange destMutable();
  ::mlir::MutableOperandRange positionMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value result();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, Value dest, int64_t position);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, Value dest, Value position);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::Value dest, ::mlir::Value position);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value dest, ::mlir::Value position);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    Type getSourceType() { return source().getType(); }
    VectorType getDestVectorType() {
      return dest().getType().cast<VectorType>();
    }
  
};
} // namespace vector
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::vector::InsertElementOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::InsertMapOp declarations
//===----------------------------------------------------------------------===//

class InsertMapOpAdaptor {
public:
  InsertMapOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  InsertMapOpAdaptor(InsertMapOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value vector();
  ::mlir::Value dest();
  ::mlir::ValueRange ids();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class InsertMapOp : public ::mlir::Op<InsertMapOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::VectorType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = InsertMapOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("vector.insert_map");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value vector();
  ::mlir::Value dest();
  ::mlir::Operation::operand_range ids();
  ::mlir::MutableOperandRange vectorMutable();
  ::mlir::MutableOperandRange destMutable();
  ::mlir::MutableOperandRange idsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value result();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value vector, Value dest, ValueRange ids);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value vector, ::mlir::Value dest, ::mlir::ValueRange ids);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value vector, ::mlir::Value dest, ::mlir::ValueRange ids);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    VectorType getSourceVectorType() {
      return vector().getType().cast<VectorType>();
    }
    VectorType getResultType() {
      return getResult().getType().cast<VectorType>();
    }
    // Return a map indicating the dimension mapping to the given ids.
    AffineMap map();
  
};
} // namespace vector
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::vector::InsertMapOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::InsertOp declarations
//===----------------------------------------------------------------------===//

class InsertOpAdaptor {
public:
  InsertOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  InsertOpAdaptor(InsertOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value source();
  ::mlir::Value dest();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr position();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class InsertOp : public ::mlir::Op<InsertOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::VectorType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = InsertOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("position")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier positionAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier positionAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("vector.insert");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value source();
  ::mlir::Value dest();
  ::mlir::MutableOperandRange sourceMutable();
  ::mlir::MutableOperandRange destMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  ::mlir::ArrayAttr positionAttr();
  ::mlir::ArrayAttr position();
  void positionAttr(::mlir::ArrayAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, Value dest, ArrayRef<int64_t> position);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, Value dest, ValueRange position);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value source, ::mlir::Value dest, ::mlir::ArrayAttr position);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value dest, ::mlir::ArrayAttr position);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::OpFoldResult fold(::llvm::ArrayRef<::mlir::Attribute> operands);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    static StringRef getPositionAttrName() { return "position"; }
    Type getSourceType() { return source().getType(); }
    VectorType getDestVectorType() {
      return dest().getType().cast<VectorType>();
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
} // namespace vector
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::vector::InsertOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::InsertStridedSliceOp declarations
//===----------------------------------------------------------------------===//

class InsertStridedSliceOpAdaptor {
public:
  InsertStridedSliceOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  InsertStridedSliceOpAdaptor(InsertStridedSliceOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value source();
  ::mlir::Value dest();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr offsets();
  ::mlir::ArrayAttr strides();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class InsertStridedSliceOp : public ::mlir::Op<InsertStridedSliceOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::VectorType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = InsertStridedSliceOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("offsets"), ::llvm::StringRef("strides")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier offsetsAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier offsetsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier stridesAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("vector.insert_strided_slice");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value source();
  ::mlir::Value dest();
  ::mlir::MutableOperandRange sourceMutable();
  ::mlir::MutableOperandRange destMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  ::mlir::ArrayAttr offsetsAttr();
  ::mlir::ArrayAttr offsets();
  ::mlir::ArrayAttr stridesAttr();
  ::mlir::ArrayAttr strides();
  void offsetsAttr(::mlir::ArrayAttr attr);
  void stridesAttr(::mlir::ArrayAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, Value dest, ArrayRef<int64_t> offsets, ArrayRef<int64_t> strides);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value source, ::mlir::Value dest, ::mlir::ArrayAttr offsets, ::mlir::ArrayAttr strides);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value dest, ::mlir::ArrayAttr offsets, ::mlir::ArrayAttr strides);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    static StringRef getOffsetsAttrName() { return "offsets"; }
    static StringRef getStridesAttrName() { return "strides"; }
    VectorType getSourceVectorType() {
      return source().getType().cast<VectorType>();
    }
    VectorType getDestVectorType() {
      return dest().getType().cast<VectorType>();
    }
  

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 2 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace vector
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::vector::InsertStridedSliceOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::LoadOp declarations
//===----------------------------------------------------------------------===//

class LoadOpAdaptor {
public:
  LoadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  LoadOpAdaptor(LoadOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value base();
  ::mlir::ValueRange indices();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class LoadOp : public ::mlir::Op<LoadOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::VectorType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LoadOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("vector.load");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value base();
  ::mlir::Operation::operand_range indices();
  ::mlir::MutableOperandRange baseMutable();
  ::mlir::MutableOperandRange indicesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value result();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value base, ::mlir::ValueRange indices);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value base, ::mlir::ValueRange indices);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  ::mlir::OpFoldResult fold(::llvm::ArrayRef<::mlir::Attribute> operands);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    MemRefType getMemRefType() {
      return base().getType().cast<MemRefType>();
    }

    VectorType getVectorType() {
      return result().getType().cast<VectorType>();
    }
  
};
} // namespace vector
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::vector::LoadOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::MaskedLoadOp declarations
//===----------------------------------------------------------------------===//

class MaskedLoadOpAdaptor {
public:
  MaskedLoadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  MaskedLoadOpAdaptor(MaskedLoadOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value base();
  ::mlir::ValueRange indices();
  ::mlir::Value mask();
  ::mlir::Value pass_thru();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class MaskedLoadOp : public ::mlir::Op<MaskedLoadOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::VectorType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<3>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = MaskedLoadOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("vector.maskedload");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value base();
  ::mlir::Operation::operand_range indices();
  ::mlir::Value mask();
  ::mlir::Value pass_thru();
  ::mlir::MutableOperandRange baseMutable();
  ::mlir::MutableOperandRange indicesMutable();
  ::mlir::MutableOperandRange maskMutable();
  ::mlir::MutableOperandRange pass_thruMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value result();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value base, ::mlir::ValueRange indices, ::mlir::Value mask, ::mlir::Value pass_thru);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value base, ::mlir::ValueRange indices, ::mlir::Value mask, ::mlir::Value pass_thru);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::OpFoldResult fold(::llvm::ArrayRef<::mlir::Attribute> operands);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    MemRefType getMemRefType() {
      return base().getType().cast<MemRefType>();
    }
    VectorType getMaskVectorType() {
      return mask().getType().cast<VectorType>();
    }
    VectorType getPassThruVectorType() {
      return pass_thru().getType().cast<VectorType>();
    }
    VectorType getVectorType() {
      return result().getType().cast<VectorType>();
    }
  
};
} // namespace vector
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::vector::MaskedLoadOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::MaskedStoreOp declarations
//===----------------------------------------------------------------------===//

class MaskedStoreOpAdaptor {
public:
  MaskedStoreOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  MaskedStoreOpAdaptor(MaskedStoreOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value base();
  ::mlir::ValueRange indices();
  ::mlir::Value mask();
  ::mlir::Value valueToStore();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class MaskedStoreOp : public ::mlir::Op<MaskedStoreOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<3>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = MaskedStoreOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("vector.maskedstore");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value base();
  ::mlir::Operation::operand_range indices();
  ::mlir::Value mask();
  ::mlir::Value valueToStore();
  ::mlir::MutableOperandRange baseMutable();
  ::mlir::MutableOperandRange indicesMutable();
  ::mlir::MutableOperandRange maskMutable();
  ::mlir::MutableOperandRange valueToStoreMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value base, ::mlir::ValueRange indices, ::mlir::Value mask, ::mlir::Value valueToStore);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value base, ::mlir::ValueRange indices, ::mlir::Value mask, ::mlir::Value valueToStore);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    MemRefType getMemRefType() {
      return base().getType().cast<MemRefType>();
    }
    VectorType getMaskVectorType() {
      return mask().getType().cast<VectorType>();
    }
    VectorType getVectorType() {
      return valueToStore().getType().cast<VectorType>();
    }
  
};
} // namespace vector
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::vector::MaskedStoreOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::MatmulOp declarations
//===----------------------------------------------------------------------===//

class MatmulOpAdaptor {
public:
  MatmulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  MatmulOpAdaptor(MatmulOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value lhs();
  ::mlir::Value rhs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr lhs_rows();
  ::mlir::IntegerAttr lhs_columns();
  ::mlir::IntegerAttr rhs_columns();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class MatmulOp : public ::mlir::Op<MatmulOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::VectorType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = MatmulOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("lhs_rows"), ::llvm::StringRef("lhs_columns"), ::llvm::StringRef("rhs_columns")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier lhs_rowsAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier lhs_rowsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier lhs_columnsAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier lhs_columnsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier rhs_columnsAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier rhs_columnsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("vector.matrix_multiply");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value lhs();
  ::mlir::Value rhs();
  ::mlir::MutableOperandRange lhsMutable();
  ::mlir::MutableOperandRange rhsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  ::mlir::IntegerAttr lhs_rowsAttr();
  uint32_t lhs_rows();
  ::mlir::IntegerAttr lhs_columnsAttr();
  uint32_t lhs_columns();
  ::mlir::IntegerAttr rhs_columnsAttr();
  uint32_t rhs_columns();
  void lhs_rowsAttr(::mlir::IntegerAttr attr);
  void lhs_columnsAttr(::mlir::IntegerAttr attr);
  void rhs_columnsAttr(::mlir::IntegerAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value lhs, Value rhs, unsigned lhsRows, unsigned lhsColumns, unsigned rhsColumns);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::IntegerAttr lhs_rows, ::mlir::IntegerAttr lhs_columns, ::mlir::IntegerAttr rhs_columns);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::IntegerAttr lhs_rows, ::mlir::IntegerAttr lhs_columns, ::mlir::IntegerAttr rhs_columns);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, uint32_t lhs_rows, uint32_t lhs_columns, uint32_t rhs_columns);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, uint32_t lhs_rows, uint32_t lhs_columns, uint32_t rhs_columns);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 3 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace vector
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::vector::MatmulOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::MultiDimReductionOp declarations
//===----------------------------------------------------------------------===//

class MultiDimReductionOpAdaptor {
public:
  MultiDimReductionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  MultiDimReductionOpAdaptor(MultiDimReductionOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value source();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::vector::CombiningKindAttr kind();
  ::mlir::ArrayAttr reduction_dims();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class MultiDimReductionOp : public ::mlir::Op<MultiDimReductionOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::OneOperand, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = MultiDimReductionOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("kind"), ::llvm::StringRef("reduction_dims")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier kindAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier kindAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier reduction_dimsAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier reduction_dimsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("vector.multi_reduction");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value source();
  ::mlir::MutableOperandRange sourceMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value dest();
  ::mlir::vector::CombiningKindAttr kindAttr();
  ::mlir::vector::CombiningKind kind();
  ::mlir::ArrayAttr reduction_dimsAttr();
  ::mlir::ArrayAttr reduction_dims();
  void kindAttr(::mlir::vector::CombiningKindAttr attr);
  void reduction_dimsAttr(::mlir::ArrayAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, ArrayRef<bool> reductionMask, CombiningKind kind);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dest, ::mlir::vector::CombiningKindAttr kind, ::mlir::Value source, ::mlir::ArrayAttr reduction_dims);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::vector::CombiningKindAttr kind, ::mlir::Value source, ::mlir::ArrayAttr reduction_dims);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dest, ::mlir::vector::CombiningKind kind, ::mlir::Value source, ::mlir::ArrayAttr reduction_dims);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::vector::CombiningKind kind, ::mlir::Value source, ::mlir::ArrayAttr reduction_dims);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    static StringRef getKindAttrName() { return "kind"; }
    static StringRef getReductionDimsAttrName() { return "reduction_dims"; }

    VectorType getSourceVectorType() {
      return source().getType().cast<VectorType>();
    }
    VectorType getDestVectorType() {
      return dest().getType().cast<VectorType>();
    }

    SmallVector<bool> getReductionMask() {
      SmallVector<bool> res(getSourceVectorType().getRank(), false);
      for (auto ia : reduction_dims().getAsRange<IntegerAttr>())
        res[ia.getInt()] = true;
      return res;
    }
    static SmallVector<bool> getReductionMask(
        ArrayRef<int64_t> reductionDims, unsigned sourceRank) {
      SmallVector<bool> res(sourceRank, false);
      for (auto idx : reductionDims)
        res[idx] = true;
      return res;
    }

    static SmallVector<int64_t> inferDestShape(
      ArrayRef<int64_t> shape, ArrayRef<bool> reducedDimsMask) {
      assert(shape.size() == reducedDimsMask.size() && 
             "shape and maks of different sizes");
      SmallVector<int64_t> res;
      for (auto it : llvm::zip(reducedDimsMask, shape))
        if (!std::get<0>(it))
          res.push_back(std::get<1>(it));
      return res;
    }
  

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 2 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace vector
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::vector::MultiDimReductionOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::OuterProductOp declarations
//===----------------------------------------------------------------------===//

class OuterProductOpAdaptor {
public:
  OuterProductOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  OuterProductOpAdaptor(OuterProductOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value lhs();
  ::mlir::Value rhs();
  ::mlir::ValueRange acc();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::vector::CombiningKindAttr kind();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class OuterProductOp : public ::mlir::Op<OuterProductOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::VectorType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = OuterProductOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("kind")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier kindAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier kindAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("vector.outerproduct");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value lhs();
  ::mlir::Value rhs();
  ::mlir::Operation::operand_range acc();
  ::mlir::MutableOperandRange lhsMutable();
  ::mlir::MutableOperandRange rhsMutable();
  ::mlir::MutableOperandRange accMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::vector::CombiningKindAttr kindAttr();
  ::mlir::vector::CombiningKind kind();
  void kindAttr(::mlir::vector::CombiningKindAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value lhs, Value rhs, Value acc);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::ValueRange acc, ::mlir::vector::CombiningKindAttr kind);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::ValueRange acc, ::mlir::vector::CombiningKindAttr kind);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::ValueRange acc, ::mlir::vector::CombiningKind kind = CombiningKind::ADD);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::ValueRange acc, ::mlir::vector::CombiningKind kind = CombiningKind::ADD);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    VectorType getOperandVectorTypeLHS() {
      return lhs().getType().cast<VectorType>();
    }
    Type getOperandTypeRHS() {
      return rhs().getType();
    }
    VectorType getOperandVectorTypeACC() {
      return (llvm::size(acc()) == 0)
        ? VectorType()
        : (*acc().begin()).getType().cast<VectorType>();
    }
    VectorType getVectorType() {
      return getResult().getType().cast<VectorType>();
    }
    static constexpr StringRef getKindAttrName() {
      return "kind";
    }
    static CombiningKind getDefaultKind() {
      return CombiningKind::ADD;
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
} // namespace vector
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::vector::OuterProductOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::PrintOp declarations
//===----------------------------------------------------------------------===//

class PrintOpAdaptor {
public:
  PrintOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  PrintOpAdaptor(PrintOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value source();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class PrintOp : public ::mlir::Op<PrintOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::OneOperand> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = PrintOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("vector.print");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value source();
  ::mlir::MutableOperandRange sourceMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value source);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);

    Type getPrintType() {
      return source().getType();
    }
  
};
} // namespace vector
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::vector::PrintOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::ReductionOp declarations
//===----------------------------------------------------------------------===//

class ReductionOpAdaptor {
public:
  ReductionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ReductionOpAdaptor(ReductionOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value vector();
  ::mlir::ValueRange acc();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr kind();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ReductionOp : public ::mlir::Op<ReductionOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ReductionOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("kind")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier kindAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier kindAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("vector.reduction");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value vector();
  ::mlir::Operation::operand_range acc();
  ::mlir::MutableOperandRange vectorMutable();
  ::mlir::MutableOperandRange accMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value dest();
  ::mlir::StringAttr kindAttr();
  ::llvm::StringRef kind();
  void kindAttr(::mlir::StringAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dest, ::mlir::StringAttr kind, ::mlir::Value vector, ::mlir::ValueRange acc);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr kind, ::mlir::Value vector, ::mlir::ValueRange acc);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dest, ::llvm::StringRef kind, ::mlir::Value vector, ::mlir::ValueRange acc);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef kind, ::mlir::Value vector, ::mlir::ValueRange acc);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    VectorType getVectorType() {
      return vector().getType().cast<VectorType>();
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
} // namespace vector
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::vector::ReductionOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::ReshapeOp declarations
//===----------------------------------------------------------------------===//

class ReshapeOpAdaptor {
public:
  ReshapeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  ReshapeOpAdaptor(ReshapeOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value vector();
  ::mlir::ValueRange input_shape();
  ::mlir::ValueRange output_shape();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr fixed_vector_sizes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ReshapeOp : public ::mlir::Op<ReshapeOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::VectorType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ReshapeOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("fixed_vector_sizes"), ::llvm::StringRef("operand_segment_sizes")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier fixed_vector_sizesAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier fixed_vector_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("vector.reshape");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value vector();
  ::mlir::Operation::operand_range input_shape();
  ::mlir::Operation::operand_range output_shape();
  ::mlir::MutableOperandRange vectorMutable();
  ::mlir::MutableOperandRange input_shapeMutable();
  ::mlir::MutableOperandRange output_shapeMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value result();
  ::mlir::ArrayAttr fixed_vector_sizesAttr();
  ::mlir::ArrayAttr fixed_vector_sizes();
  void fixed_vector_sizesAttr(::mlir::ArrayAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value vector, ::mlir::ValueRange input_shape, ::mlir::ValueRange output_shape, ::mlir::ArrayAttr fixed_vector_sizes);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value vector, ::mlir::ValueRange input_shape, ::mlir::ValueRange output_shape, ::mlir::ArrayAttr fixed_vector_sizes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    VectorType getInputVectorType() {
      return vector().getType().cast<VectorType>();
    }
    VectorType getOutputVectorType() {
      return getResult().getType().cast<VectorType>();
    }

    /// Returns as integer value the number of input shape operands.
    int64_t getNumInputShapeSizes() { return input_shape().size(); }

    /// Returns as integer value the number of output shape operands.
    int64_t getNumOutputShapeSizes() { return output_shape().size(); }

    void getFixedVectorSizes(SmallVectorImpl<int64_t> &results);

    static StringRef getFixedVectorSizesAttrName() {
      return "fixed_vector_sizes";
    }
    static StringRef getInputShapeAttrName() { return "input_shape"; }
    static StringRef getOutputShapeAttrName() { return "output_shape"; }
  

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 2 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace vector
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::vector::ReshapeOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::ScatterOp declarations
//===----------------------------------------------------------------------===//

class ScatterOpAdaptor {
public:
  ScatterOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ScatterOpAdaptor(ScatterOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value base();
  ::mlir::ValueRange indices();
  ::mlir::Value index_vec();
  ::mlir::Value mask();
  ::mlir::Value valueToStore();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ScatterOp : public ::mlir::Op<ScatterOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<4>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ScatterOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("vector.scatter");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value base();
  ::mlir::Operation::operand_range indices();
  ::mlir::Value index_vec();
  ::mlir::Value mask();
  ::mlir::Value valueToStore();
  ::mlir::MutableOperandRange baseMutable();
  ::mlir::MutableOperandRange indicesMutable();
  ::mlir::MutableOperandRange index_vecMutable();
  ::mlir::MutableOperandRange maskMutable();
  ::mlir::MutableOperandRange valueToStoreMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value base, ::mlir::ValueRange indices, ::mlir::Value index_vec, ::mlir::Value mask, ::mlir::Value valueToStore);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value base, ::mlir::ValueRange indices, ::mlir::Value index_vec, ::mlir::Value mask, ::mlir::Value valueToStore);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    MemRefType getMemRefType() {
      return base().getType().cast<MemRefType>();
    }
    VectorType getIndexVectorType() {
      return index_vec().getType().cast<VectorType>();
    }
    VectorType getMaskVectorType() {
      return mask().getType().cast<VectorType>();
    }
    VectorType getVectorType() {
      return valueToStore().getType().cast<VectorType>();
    }
  
};
} // namespace vector
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::vector::ScatterOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::ShapeCastOp declarations
//===----------------------------------------------------------------------===//

class ShapeCastOpAdaptor {
public:
  ShapeCastOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ShapeCastOpAdaptor(ShapeCastOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value source();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ShapeCastOp : public ::mlir::Op<ShapeCastOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::VectorType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::OneOperand, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ShapeCastOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("vector.shape_cast");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value source();
  ::mlir::MutableOperandRange sourceMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value result();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::OpFoldResult fold(::llvm::ArrayRef<::mlir::Attribute> operands);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    VectorType getSourceVectorType() {
      return source().getType().cast<VectorType>();
    }
    VectorType getResultVectorType() {
      return getResult().getType().cast<VectorType>();
    }
  
};
} // namespace vector
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::vector::ShapeCastOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::ShuffleOp declarations
//===----------------------------------------------------------------------===//

class ShuffleOpAdaptor {
public:
  ShuffleOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ShuffleOpAdaptor(ShuffleOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value v1();
  ::mlir::Value v2();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr mask();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ShuffleOp : public ::mlir::Op<ShuffleOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::VectorType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ShuffleOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("mask")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier maskAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier maskAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("vector.shuffle");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value v1();
  ::mlir::Value v2();
  ::mlir::MutableOperandRange v1Mutable();
  ::mlir::MutableOperandRange v2Mutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value vector();
  ::mlir::ArrayAttr maskAttr();
  ::mlir::ArrayAttr mask();
  void maskAttr(::mlir::ArrayAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value v1, Value v2, ArrayRef<int64_t> odsArg2);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type vector, ::mlir::Value v1, ::mlir::Value v2, ::mlir::ArrayAttr mask);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value v1, ::mlir::Value v2, ::mlir::ArrayAttr mask);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    static StringRef getMaskAttrName() { return "mask"; }
    VectorType getV1VectorType() {
      return v1().getType().cast<VectorType>();
    }
    VectorType getV2VectorType() {
      return v2().getType().cast<VectorType>();
    }
    VectorType getVectorType() {
      return vector().getType().cast<VectorType>();
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
} // namespace vector
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::vector::ShuffleOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::StoreOp declarations
//===----------------------------------------------------------------------===//

class StoreOpAdaptor {
public:
  StoreOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  StoreOpAdaptor(StoreOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value valueToStore();
  ::mlir::Value base();
  ::mlir::ValueRange indices();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class StoreOp : public ::mlir::Op<StoreOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = StoreOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("vector.store");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value valueToStore();
  ::mlir::Value base();
  ::mlir::Operation::operand_range indices();
  ::mlir::MutableOperandRange valueToStoreMutable();
  ::mlir::MutableOperandRange baseMutable();
  ::mlir::MutableOperandRange indicesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value valueToStore, ::mlir::Value base, ::mlir::ValueRange indices);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value valueToStore, ::mlir::Value base, ::mlir::ValueRange indices);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    MemRefType getMemRefType() {
      return base().getType().cast<MemRefType>();
    }

    VectorType getVectorType() {
      return valueToStore().getType().cast<VectorType>();
    }
  
};
} // namespace vector
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::vector::StoreOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::TransferReadOp declarations
//===----------------------------------------------------------------------===//

class TransferReadOpAdaptor {
public:
  TransferReadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  TransferReadOpAdaptor(TransferReadOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value source();
  ::mlir::ValueRange indices();
  ::mlir::Value padding();
  ::mlir::Value mask();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::AffineMapAttr permutation_map();
  ::mlir::ArrayAttr in_bounds();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class TransferReadOp : public ::mlir::Op<TransferReadOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::VectorType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::VectorTransferOpInterface::Trait, ::mlir::VectorUnrollOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::AttrSizedOperandSegments> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TransferReadOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("permutation_map"), ::llvm::StringRef("in_bounds"), ::llvm::StringRef("operand_segment_sizes")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier permutation_mapAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier permutation_mapAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier in_boundsAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier in_boundsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("vector.transfer_read");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value source();
  ::mlir::Operation::operand_range indices();
  ::mlir::Value padding();
  ::mlir::Value mask();
  ::mlir::MutableOperandRange sourceMutable();
  ::mlir::MutableOperandRange indicesMutable();
  ::mlir::MutableOperandRange paddingMutable();
  ::mlir::MutableOperandRange maskMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value vector();
  ::mlir::AffineMapAttr permutation_mapAttr();
  ::mlir::AffineMap permutation_map();
  ::mlir::ArrayAttr in_boundsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > in_bounds();
  void permutation_mapAttr(::mlir::AffineMapAttr attr);
  void in_boundsAttr(::mlir::ArrayAttr attr);
  ::mlir::Attribute removeIn_boundsAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, VectorType vector, Value source, ValueRange indices, AffineMap permutationMap, ArrayRef<bool> inBounds = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, VectorType vector, Value source, ValueRange indices, Value padding, ArrayRef<bool> inBounds = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, VectorType vector, Value source, ValueRange indices, ArrayRef<bool> inBounds = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type vector, Value source, ValueRange indices, AffineMapAttr permutationMap, Value padding, ArrayAttr inBounds);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type vector, Value source, ValueRange indices, AffineMap permutationMap, Value padding, ArrayAttr inBounds);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type vector, ::mlir::Value source, ::mlir::ValueRange indices, ::mlir::AffineMapAttr permutation_map, ::mlir::Value padding, /*optional*/::mlir::Value mask, /*optional*/::mlir::ArrayAttr in_bounds);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::ValueRange indices, ::mlir::AffineMapAttr permutation_map, ::mlir::Value padding, /*optional*/::mlir::Value mask, /*optional*/::mlir::ArrayAttr in_bounds);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type vector, ::mlir::Value source, ::mlir::ValueRange indices, ::mlir::AffineMap permutation_map, ::mlir::Value padding, /*optional*/::mlir::Value mask, /*optional*/::mlir::ArrayAttr in_bounds);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::ValueRange indices, ::mlir::AffineMap permutation_map, ::mlir::Value padding, /*optional*/::mlir::Value mask, /*optional*/::mlir::ArrayAttr in_bounds);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  ::mlir::OpFoldResult fold(::llvm::ArrayRef<::mlir::Attribute> operands);
  ::mlir::Optional<::mlir::SmallVector<int64_t, 4>> getShapeForUnroll();
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 3 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace vector
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::vector::TransferReadOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::TransferWriteOp declarations
//===----------------------------------------------------------------------===//

class TransferWriteOpAdaptor {
public:
  TransferWriteOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  TransferWriteOpAdaptor(TransferWriteOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value vector();
  ::mlir::Value source();
  ::mlir::ValueRange indices();
  ::mlir::Value mask();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::AffineMapAttr permutation_map();
  ::mlir::ArrayAttr in_bounds();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class TransferWriteOp : public ::mlir::Op<TransferWriteOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::VectorTransferOpInterface::Trait, ::mlir::VectorUnrollOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::AttrSizedOperandSegments> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TransferWriteOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("permutation_map"), ::llvm::StringRef("in_bounds"), ::llvm::StringRef("operand_segment_sizes")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier permutation_mapAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier permutation_mapAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier in_boundsAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier in_boundsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("vector.transfer_write");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value vector();
  ::mlir::Value source();
  ::mlir::Operation::operand_range indices();
  ::mlir::Value mask();
  ::mlir::MutableOperandRange vectorMutable();
  ::mlir::MutableOperandRange sourceMutable();
  ::mlir::MutableOperandRange indicesMutable();
  ::mlir::MutableOperandRange maskMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value result();
  ::mlir::AffineMapAttr permutation_mapAttr();
  ::mlir::AffineMap permutation_map();
  ::mlir::ArrayAttr in_boundsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > in_bounds();
  void permutation_mapAttr(::mlir::AffineMapAttr attr);
  void in_boundsAttr(::mlir::ArrayAttr attr);
  ::mlir::Attribute removeIn_boundsAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value vector, Value source, ValueRange indices, ArrayRef<bool> inBounds = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value vector, Value source, ValueRange indices, AffineMap permutationMap);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value vector, Value source, ValueRange indices, AffineMapAttr permutationMap, ArrayAttr inBounds);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value vector, Value source, ValueRange indices, AffineMap permutationMap, Value mask, ArrayAttr inBounds);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value vector, Value source, ValueRange indices, AffineMap permutationMap, ArrayAttr inBounds);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type result, ::mlir::Value vector, ::mlir::Value source, ::mlir::ValueRange indices, ::mlir::AffineMapAttr permutation_map, /*optional*/::mlir::Value mask, /*optional*/::mlir::ArrayAttr in_bounds);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value vector, ::mlir::Value source, ::mlir::ValueRange indices, ::mlir::AffineMapAttr permutation_map, /*optional*/::mlir::Value mask, /*optional*/::mlir::ArrayAttr in_bounds);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type result, ::mlir::Value vector, ::mlir::Value source, ::mlir::ValueRange indices, ::mlir::AffineMap permutation_map, /*optional*/::mlir::Value mask, /*optional*/::mlir::ArrayAttr in_bounds);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value vector, ::mlir::Value source, ::mlir::ValueRange indices, ::mlir::AffineMap permutation_map, /*optional*/::mlir::Value mask, /*optional*/::mlir::ArrayAttr in_bounds);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  ::mlir::Optional<::mlir::SmallVector<int64_t, 4>> getShapeForUnroll();
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 3 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace vector
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::vector::TransferWriteOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::TransposeOp declarations
//===----------------------------------------------------------------------===//

class TransposeOpAdaptor {
public:
  TransposeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  TransposeOpAdaptor(TransposeOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value vector();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr transp();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class TransposeOp : public ::mlir::Op<TransposeOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::VectorType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::OneOperand, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TransposeOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("transp")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier transpAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier transpAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("vector.transpose");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value vector();
  ::mlir::MutableOperandRange vectorMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value result();
  ::mlir::ArrayAttr transpAttr();
  ::mlir::ArrayAttr transp();
  void transpAttr(::mlir::ArrayAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value vector, ArrayRef<int64_t> transp);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value vector, ::mlir::ArrayAttr transp);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value vector, ::mlir::ArrayAttr transp);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::OpFoldResult fold(::llvm::ArrayRef<::mlir::Attribute> operands);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    VectorType getVectorType() {
      return vector().getType().cast<VectorType>();
    }
    VectorType getResultType() {
      return result().getType().cast<VectorType>();
    }
    void getTransp(SmallVectorImpl<int64_t> &results);
    static StringRef getTranspAttrName() { return "transp"; }
  

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 1 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace vector
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::vector::TransposeOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::TypeCastOp declarations
//===----------------------------------------------------------------------===//

class TypeCastOpAdaptor {
public:
  TypeCastOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  TypeCastOpAdaptor(TypeCastOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value memref();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class TypeCastOp : public ::mlir::Op<TypeCastOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::MemRefType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::OneOperand, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::ViewLikeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TypeCastOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("vector.type_cast");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value memref();
  ::mlir::MutableOperandRange memrefMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value result();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value memref);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    MemRefType getMemRefType() {
      return memref().getType().cast<MemRefType>();
    }
    MemRefType getResultMemRefType() {
      return getResult().getType().cast<MemRefType>();
    }
    // Implement ViewLikeOpInterface.
    Value getViewSource() { return memref(); }
  
};
} // namespace vector
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::vector::TypeCastOp)


#endif  // GET_OP_CLASSES

