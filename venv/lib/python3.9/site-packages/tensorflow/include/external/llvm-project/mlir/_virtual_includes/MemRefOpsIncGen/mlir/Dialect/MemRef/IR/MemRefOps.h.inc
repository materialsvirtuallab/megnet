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
namespace memref {
class AssumeAlignmentOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class CloneOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class CopyOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class LoadOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class AllocOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class AllocaOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class AllocaScopeOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class AllocaScopeReturnOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class BufferCastOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class CastOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class CollapseShapeOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class DeallocOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class DimOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class ExpandShapeOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class GetGlobalOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class GlobalOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class PrefetchOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class ReinterpretCastOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class ReshapeOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class StoreOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class TransposeOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class ViewOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class SubViewOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class TensorLoadOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class TensorStoreOp;
} // namespace memref
} // namespace mlir
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::AssumeAlignmentOp declarations
//===----------------------------------------------------------------------===//

class AssumeAlignmentOpAdaptor {
public:
  AssumeAlignmentOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  AssumeAlignmentOpAdaptor(AssumeAlignmentOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value memref();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr alignment();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class AssumeAlignmentOp : public ::mlir::Op<AssumeAlignmentOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::OneOperand> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AssumeAlignmentOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("alignment")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier alignmentAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier alignmentAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.assume_alignment");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value memref();
  ::mlir::MutableOperandRange memrefMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::IntegerAttr alignmentAttr();
  uint32_t alignment();
  void alignmentAttr(::mlir::IntegerAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value memref, ::mlir::IntegerAttr alignment);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::IntegerAttr alignment);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value memref, uint32_t alignment);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, uint32_t alignment);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 1 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace memref
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::AssumeAlignmentOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::CloneOp declarations
//===----------------------------------------------------------------------===//

class CloneOpAdaptor {
public:
  CloneOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  CloneOpAdaptor(CloneOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class CloneOp : public ::mlir::Op<CloneOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::OneOperand, ::mlir::CopyOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CloneOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.clone");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::MutableOperandRange inputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value output();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value value);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::OpFoldResult fold(::llvm::ArrayRef<::mlir::Attribute> operands);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);

    Value getSource() { return input(); }
    Value getTarget() { return output(); }
  
};
} // namespace memref
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::CloneOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::CopyOp declarations
//===----------------------------------------------------------------------===//

class CopyOpAdaptor {
public:
  CopyOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  CopyOpAdaptor(CopyOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value source();
  ::mlir::Value target();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class CopyOp : public ::mlir::Op<CopyOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::CopyOpInterface::Trait, ::mlir::OpTrait::SameOperandsElementType, ::mlir::OpTrait::SameOperandsShape, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CopyOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.copy");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value source();
  ::mlir::Value target();
  ::mlir::MutableOperandRange sourceMutable();
  ::mlir::MutableOperandRange targetMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value source, ::mlir::Value target);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value target);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    Value getSource() { return source();}
    Value getTarget() { return target(); }
  
};
} // namespace memref
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::CopyOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::LoadOp declarations
//===----------------------------------------------------------------------===//

class LoadOpAdaptor {
public:
  LoadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  LoadOpAdaptor(LoadOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value memref();
  ::mlir::ValueRange indices();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class LoadOp : public ::mlir::Op<LoadOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::MemRefsNormalizable, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LoadOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.load");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value memref();
  ::mlir::Operation::operand_range indices();
  ::mlir::MutableOperandRange memrefMutable();
  ::mlir::MutableOperandRange indicesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value result();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value memref, ValueRange indices = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value memref, ::mlir::ValueRange indices);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::ValueRange indices);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::OpFoldResult fold(::llvm::ArrayRef<::mlir::Attribute> operands);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    Value getMemRef() { return getOperand(0); }
    void setMemRef(Value value) { setOperand(0, value); }
    MemRefType getMemRefType() {
      return getMemRef().getType().cast<MemRefType>();
    }

    operand_range getIndices() { return {operand_begin() + 1, operand_end()}; }
  
};
} // namespace memref
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::LoadOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::AllocOp declarations
//===----------------------------------------------------------------------===//

class AllocOpAdaptor {
public:
  AllocOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  AllocOpAdaptor(AllocOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange dynamicSizes();
  ::mlir::ValueRange symbolOperands();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr alignment();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class AllocOp : public ::mlir::Op<AllocOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::MemRefType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AllocOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("alignment"), ::llvm::StringRef("operand_segment_sizes")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier alignmentAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier alignmentAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.alloc");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range dynamicSizes();
  ::mlir::Operation::operand_range symbolOperands();
  ::mlir::MutableOperandRange dynamicSizesMutable();
  ::mlir::MutableOperandRange symbolOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value memref();
  ::mlir::IntegerAttr alignmentAttr();
  ::llvm::Optional<uint64_t> alignment();
  void alignmentAttr(::mlir::IntegerAttr attr);
  ::mlir::Attribute removeAlignmentAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, MemRefType memrefType, IntegerAttr alignment = IntegerAttr());
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, MemRefType memrefType, ValueRange dynamicSizes, IntegerAttr alignment = IntegerAttr());
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, MemRefType memrefType, ValueRange dynamicSizes, ValueRange symbolOperands, IntegerAttr alignment = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type memref, ::mlir::ValueRange dynamicSizes, ::mlir::ValueRange symbolOperands, /*optional*/::mlir::IntegerAttr alignment);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange dynamicSizes, ::mlir::ValueRange symbolOperands, /*optional*/::mlir::IntegerAttr alignment);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    static StringRef getAlignmentAttrName() { return "alignment"; }

    MemRefType getType() { return getResult().getType().cast<MemRefType>(); }

    /// Returns the dynamic sizes for this alloc operation if specified.
    operand_range getDynamicSizes() { return dynamicSizes(); }
  

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 2 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace memref
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::AllocOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::AllocaOp declarations
//===----------------------------------------------------------------------===//

class AllocaOpAdaptor {
public:
  AllocaOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  AllocaOpAdaptor(AllocaOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange dynamicSizes();
  ::mlir::ValueRange symbolOperands();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr alignment();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class AllocaOp : public ::mlir::Op<AllocaOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::MemRefType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AllocaOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("alignment"), ::llvm::StringRef("operand_segment_sizes")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier alignmentAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier alignmentAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.alloca");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range dynamicSizes();
  ::mlir::Operation::operand_range symbolOperands();
  ::mlir::MutableOperandRange dynamicSizesMutable();
  ::mlir::MutableOperandRange symbolOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value memref();
  ::mlir::IntegerAttr alignmentAttr();
  ::llvm::Optional<uint64_t> alignment();
  void alignmentAttr(::mlir::IntegerAttr attr);
  ::mlir::Attribute removeAlignmentAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, MemRefType memrefType, IntegerAttr alignment = IntegerAttr());
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, MemRefType memrefType, ValueRange dynamicSizes, IntegerAttr alignment = IntegerAttr());
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, MemRefType memrefType, ValueRange dynamicSizes, ValueRange symbolOperands, IntegerAttr alignment = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type memref, ::mlir::ValueRange dynamicSizes, ::mlir::ValueRange symbolOperands, /*optional*/::mlir::IntegerAttr alignment);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange dynamicSizes, ::mlir::ValueRange symbolOperands, /*optional*/::mlir::IntegerAttr alignment);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    static StringRef getAlignmentAttrName() { return "alignment"; }

    MemRefType getType() { return getResult().getType().cast<MemRefType>(); }

    /// Returns the dynamic sizes for this alloc operation if specified.
    operand_range getDynamicSizes() { return dynamicSizes(); }
  

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 2 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace memref
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::AllocaOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::AllocaScopeOp declarations
//===----------------------------------------------------------------------===//

class AllocaScopeOpAdaptor {
public:
  AllocaScopeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  AllocaScopeOpAdaptor(AllocaScopeOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &bodyRegion();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class AllocaScopeOp : public ::mlir::Op<AllocaScopeOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::AutomaticAllocationScope, ::mlir::RegionBranchOpInterface::Trait, ::mlir::OpTrait::SingleBlockImplicitTerminator<AllocaScopeReturnOp>::Impl, ::mlir::OpTrait::HasRecursiveSideEffects, ::mlir::OpTrait::NoRegionArguments> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AllocaScopeOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.alloca_scope");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range results();
  ::mlir::Region &bodyRegion();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  void getSuccessorRegions(::mlir::Optional<unsigned> index, ::mlir::ArrayRef<::mlir::Attribute> operands, ::mlir::SmallVectorImpl<::mlir::RegionSuccessor> &regions);
};
} // namespace memref
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::AllocaScopeOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::AllocaScopeReturnOp declarations
//===----------------------------------------------------------------------===//

class AllocaScopeReturnOpAdaptor {
public:
  AllocaScopeReturnOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  AllocaScopeReturnOpAdaptor(AllocaScopeReturnOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange results();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class AllocaScopeReturnOp : public ::mlir::Op<AllocaScopeReturnOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::HasParent<AllocaScopeOp>::Impl, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::ReturnLike, ::mlir::OpTrait::IsTerminator> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AllocaScopeReturnOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.alloca_scope.return");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range results();
  ::mlir::MutableOperandRange resultsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange results);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
};
} // namespace memref
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::AllocaScopeReturnOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::BufferCastOp declarations
//===----------------------------------------------------------------------===//

class BufferCastOpAdaptor {
public:
  BufferCastOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  BufferCastOpAdaptor(BufferCastOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value tensor();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class BufferCastOp : public ::mlir::Op<BufferCastOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::SameOperandsAndResultShape, ::mlir::OpTrait::SameOperandsAndResultElementType, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = BufferCastOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.buffer_cast");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value tensor();
  ::mlir::MutableOperandRange tensorMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value memref();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type memref, ::mlir::Value tensor);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value tensor);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::OpFoldResult fold(::llvm::ArrayRef<::mlir::Attribute> operands);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
};
} // namespace memref
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::BufferCastOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::CastOp declarations
//===----------------------------------------------------------------------===//

class CastOpAdaptor {
public:
  CastOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  CastOpAdaptor(CastOp&op);
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
class CastOp : public ::mlir::Op<CastOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::OneOperand, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SameOperandsAndResultShape, ::mlir::CastOpInterface::Trait, ::mlir::ViewLikeOpInterface::Trait, ::mlir::OpTrait::MemRefsNormalizable> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CastOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.cast");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value source();
  ::mlir::MutableOperandRange sourceMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value dest();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, Type destType);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dest, ::mlir::Value source);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  ::mlir::OpFoldResult fold(::llvm::ArrayRef<::mlir::Attribute> operands);
  static bool areCastCompatible(::mlir::TypeRange inputs, ::mlir::TypeRange outputs);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    /// Fold the given CastOp into consumer op.
    static bool canFoldIntoConsumerOp(CastOp castOp);

    Value getViewSource() { return source(); }
  
};
} // namespace memref
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::CastOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::CollapseShapeOp declarations
//===----------------------------------------------------------------------===//

class CollapseShapeOpAdaptor {
public:
  CollapseShapeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  CollapseShapeOpAdaptor(CollapseShapeOp&op);
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
class CollapseShapeOp : public ::mlir::Op<CollapseShapeOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::OneOperand, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::ViewLikeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CollapseShapeOpAdaptor;
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
    return ::llvm::StringLiteral("memref.collapse_shape");
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
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

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
    MemRefType getSrcType() { return src().getType().cast<MemRefType>(); }
    MemRefType getResultType() { return result().getType().cast<MemRefType>(); }
    Value getViewSource() { return src(); }
  

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 1 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace memref
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::CollapseShapeOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::DeallocOp declarations
//===----------------------------------------------------------------------===//

class DeallocOpAdaptor {
public:
  DeallocOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  DeallocOpAdaptor(DeallocOp&op);
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
class DeallocOp : public ::mlir::Op<DeallocOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::MemRefsNormalizable, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = DeallocOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.dealloc");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value memref();
  ::mlir::MutableOperandRange memrefMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value memref);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
};
} // namespace memref
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::DeallocOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::DimOp declarations
//===----------------------------------------------------------------------===//

class DimOpAdaptor {
public:
  DimOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  DimOpAdaptor(DimOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value source();
  ::mlir::Value index();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class DimOp : public ::mlir::Op<DimOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::IndexType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::MemRefsNormalizable> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = DimOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.dim");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value source();
  ::mlir::Value index();
  ::mlir::MutableOperandRange sourceMutable();
  ::mlir::MutableOperandRange indexMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value result();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, int64_t index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, Value index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::Value index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value index);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::OpFoldResult fold(::llvm::ArrayRef<::mlir::Attribute> operands);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    /// Helper function to get the index as a simple integer if it is constant.
    Optional<int64_t> getConstantIndex();
  
};
} // namespace memref
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::DimOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::ExpandShapeOp declarations
//===----------------------------------------------------------------------===//

class ExpandShapeOpAdaptor {
public:
  ExpandShapeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ExpandShapeOpAdaptor(ExpandShapeOp&op);
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
class ExpandShapeOp : public ::mlir::Op<ExpandShapeOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::OneOperand, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::ViewLikeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExpandShapeOpAdaptor;
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
    return ::llvm::StringLiteral("memref.expand_shape");
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
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

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
    MemRefType getSrcType() { return src().getType().cast<MemRefType>(); }
    MemRefType getResultType() { return result().getType().cast<MemRefType>(); }
    Value getViewSource() { return src(); }
  

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 1 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace memref
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::ExpandShapeOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::GetGlobalOp declarations
//===----------------------------------------------------------------------===//

class GetGlobalOpAdaptor {
public:
  GetGlobalOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  GetGlobalOpAdaptor(GetGlobalOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::FlatSymbolRefAttr name();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class GetGlobalOp : public ::mlir::Op<GetGlobalOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::MemRefType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::SymbolUserOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GetGlobalOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("name")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier nameAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier nameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.get_global");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value result();
  ::mlir::FlatSymbolRefAttr nameAttr();
  ::llvm::StringRef name();
  void nameAttr(::mlir::FlatSymbolRefAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::FlatSymbolRefAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::FlatSymbolRefAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::llvm::StringRef name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef name);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult verifySymbolUses(::mlir::SymbolTableCollection &symbolTable);
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
} // namespace memref
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::GetGlobalOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::GlobalOp declarations
//===----------------------------------------------------------------------===//

class GlobalOpAdaptor {
public:
  GlobalOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  GlobalOpAdaptor(GlobalOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr sym_name();
  ::mlir::StringAttr sym_visibility();
  ::mlir::TypeAttr type();
  ::mlir::Attribute initial_value();
  ::mlir::UnitAttr constant();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class GlobalOp : public ::mlir::Op<GlobalOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::SymbolOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GlobalOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("sym_name"), ::llvm::StringRef("sym_visibility"), ::llvm::StringRef("type"), ::llvm::StringRef("initial_value"), ::llvm::StringRef("constant")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier sym_nameAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier sym_nameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier sym_visibilityAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier sym_visibilityAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier typeAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier typeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  ::mlir::Identifier initial_valueAttrName() {
    return getAttributeNameForIndex(3);
  }
  static ::mlir::Identifier initial_valueAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }
  ::mlir::Identifier constantAttrName() {
    return getAttributeNameForIndex(4);
  }
  static ::mlir::Identifier constantAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.global");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::StringAttr sym_nameAttr();
  ::llvm::StringRef sym_name();
  ::mlir::StringAttr sym_visibilityAttr();
  ::llvm::Optional< ::llvm::StringRef > sym_visibility();
  ::mlir::TypeAttr typeAttr();
  ::mlir::MemRefType type();
  ::mlir::Attribute initial_valueAttr();
  ::llvm::Optional<::mlir::Attribute> initial_value();
  ::mlir::UnitAttr constantAttr();
  bool constant();
  void sym_nameAttr(::mlir::StringAttr attr);
  void sym_visibilityAttr(::mlir::StringAttr attr);
  void typeAttr(::mlir::TypeAttr attr);
  void initial_valueAttr(::mlir::Attribute attr);
  void constantAttr(::mlir::UnitAttr attr);
  ::mlir::Attribute removeSym_visibilityAttr();
  ::mlir::Attribute removeInitial_valueAttr();
  ::mlir::Attribute removeConstantAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr sym_name, /*optional*/::mlir::StringAttr sym_visibility, ::mlir::TypeAttr type, /*optional*/::mlir::Attribute initial_value, /*optional*/::mlir::UnitAttr constant);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr sym_name, /*optional*/::mlir::StringAttr sym_visibility, ::mlir::TypeAttr type, /*optional*/::mlir::Attribute initial_value, /*optional*/::mlir::UnitAttr constant);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef sym_name, /*optional*/::mlir::StringAttr sym_visibility, ::mlir::MemRefType type, /*optional*/::mlir::Attribute initial_value, /*optional*/bool constant);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef sym_name, /*optional*/::mlir::StringAttr sym_visibility, ::mlir::MemRefType type, /*optional*/::mlir::Attribute initial_value, /*optional*/bool constant);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);

     bool isExternal() { return !initial_value(); }
     bool isUninitialized() {
       return !isExternal() && initial_value().getValue().isa<UnitAttr>();
     }
  

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 5 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace memref
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::GlobalOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::PrefetchOp declarations
//===----------------------------------------------------------------------===//

class PrefetchOpAdaptor {
public:
  PrefetchOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  PrefetchOpAdaptor(PrefetchOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value memref();
  ::mlir::ValueRange indices();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::BoolAttr isWrite();
  ::mlir::IntegerAttr localityHint();
  ::mlir::BoolAttr isDataCache();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class PrefetchOp : public ::mlir::Op<PrefetchOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<1>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = PrefetchOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("isWrite"), ::llvm::StringRef("localityHint"), ::llvm::StringRef("isDataCache")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier isWriteAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier isWriteAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier localityHintAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier localityHintAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier isDataCacheAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier isDataCacheAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.prefetch");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value memref();
  ::mlir::Operation::operand_range indices();
  ::mlir::MutableOperandRange memrefMutable();
  ::mlir::MutableOperandRange indicesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::BoolAttr isWriteAttr();
  bool isWrite();
  ::mlir::IntegerAttr localityHintAttr();
  uint32_t localityHint();
  ::mlir::BoolAttr isDataCacheAttr();
  bool isDataCache();
  void isWriteAttr(::mlir::BoolAttr attr);
  void localityHintAttr(::mlir::IntegerAttr attr);
  void isDataCacheAttr(::mlir::BoolAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::BoolAttr isWrite, ::mlir::IntegerAttr localityHint, ::mlir::BoolAttr isDataCache);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::BoolAttr isWrite, ::mlir::IntegerAttr localityHint, ::mlir::BoolAttr isDataCache);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value memref, ::mlir::ValueRange indices, bool isWrite, uint32_t localityHint, bool isDataCache);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::ValueRange indices, bool isWrite, uint32_t localityHint, bool isDataCache);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);

    MemRefType getMemRefType() {
      return memref().getType().cast<MemRefType>();
    }
    static StringRef getLocalityHintAttrName() { return "localityHint"; }
    static StringRef getIsWriteAttrName() { return "isWrite"; }
    static StringRef getIsDataCacheAttrName() { return "isDataCache"; }
  

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 3 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace memref
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::PrefetchOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::ReinterpretCastOp declarations
//===----------------------------------------------------------------------===//

class ReinterpretCastOpAdaptor {
public:
  ReinterpretCastOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  ReinterpretCastOpAdaptor(ReinterpretCastOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value source();
  ::mlir::ValueRange offsets();
  ::mlir::ValueRange sizes();
  ::mlir::ValueRange strides();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr static_offsets();
  ::mlir::ArrayAttr static_sizes();
  ::mlir::ArrayAttr static_strides();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ReinterpretCastOp : public ::mlir::Op<ReinterpretCastOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::MemRefType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::ViewLikeOpInterface::Trait, ::mlir::OffsetSizeAndStrideOpInterface::Trait, ::mlir::OpTrait::MemRefsNormalizable> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ReinterpretCastOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("static_offsets"), ::llvm::StringRef("static_sizes"), ::llvm::StringRef("static_strides"), ::llvm::StringRef("operand_segment_sizes")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier static_offsetsAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier static_offsetsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier static_sizesAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier static_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier static_stridesAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier static_stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  ::mlir::Identifier operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(3);
  }
  static ::mlir::Identifier operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.reinterpret_cast");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value source();
  ::mlir::Operation::operand_range offsets();
  ::mlir::Operation::operand_range sizes();
  ::mlir::Operation::operand_range strides();
  ::mlir::MutableOperandRange sourceMutable();
  ::mlir::MutableOperandRange offsetsMutable();
  ::mlir::MutableOperandRange sizesMutable();
  ::mlir::MutableOperandRange stridesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value result();
  ::mlir::ArrayAttr static_offsetsAttr();
  ::mlir::ArrayAttr static_offsets();
  ::mlir::ArrayAttr static_sizesAttr();
  ::mlir::ArrayAttr static_sizes();
  ::mlir::ArrayAttr static_stridesAttr();
  ::mlir::ArrayAttr static_strides();
  void static_offsetsAttr(::mlir::ArrayAttr attr);
  void static_sizesAttr(::mlir::ArrayAttr attr);
  void static_stridesAttr(::mlir::ArrayAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, MemRefType resultType, Value source, OpFoldResult offset, ArrayRef<OpFoldResult> sizes, ArrayRef<OpFoldResult> strides, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, MemRefType resultType, Value source, int64_t offset, ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, MemRefType resultType, Value source, Value offset, ValueRange sizes, ValueRange strides, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::ArrayAttr static_offsets, ::mlir::ArrayAttr static_sizes, ::mlir::ArrayAttr static_strides);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::ArrayAttr static_offsets, ::mlir::ArrayAttr static_sizes, ::mlir::ArrayAttr static_strides);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    /// Returns the dynamic sizes for this subview operation if specified.
    operand_range getDynamicSizes() { return sizes(); }

    /// Return the list of Range (i.e. offset, size, stride). Each
    /// Range entry contains either the dynamic value or a ConstantIndexOp
    /// constructed with `b` at location `loc`.
    SmallVector<Range, 8> getOrCreateRanges(OpBuilder &b, Location loc) {
      return mlir::getOrCreateRanges(*this, b, loc);
    }
  
    // The result of the op is always a ranked memref.
    MemRefType getType() { return getResult().getType().cast<MemRefType>(); }
    Value getViewSource() { return source(); }

    /// Return the rank of the source ShapedType.
    unsigned getResultRank() {
      return getResult().getType().cast<ShapedType>().getRank();
    }

    /// Return the expected rank of each of the`static_offsets`, `static_sizes`
    /// and `static_strides` attributes.
    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned resultRank = getResult().getType().cast<ShapedType>().getRank();
      return {1, resultRank, resultRank};
    }

    /// Return the number of leading operands before the `offsets`, `sizes` and
    /// and `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 1; }
  

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 4 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace memref
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::ReinterpretCastOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::ReshapeOp declarations
//===----------------------------------------------------------------------===//

class ReshapeOpAdaptor {
public:
  ReshapeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ReshapeOpAdaptor(ReshapeOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value source();
  ::mlir::Value shape();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ReshapeOp : public ::mlir::Op<ReshapeOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::ViewLikeOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ReshapeOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.reshape");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value source();
  ::mlir::Value shape();
  ::mlir::MutableOperandRange sourceMutable();
  ::mlir::MutableOperandRange shapeMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value result();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, MemRefType resultType, Value operand, Value shape);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::Value shape);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value shape);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    MemRefType getType() { return getResult().getType().cast<MemRefType>(); }
    Value getViewSource() { return source(); }
  
};
} // namespace memref
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::ReshapeOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::StoreOp declarations
//===----------------------------------------------------------------------===//

class StoreOpAdaptor {
public:
  StoreOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  StoreOpAdaptor(StoreOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value value();
  ::mlir::Value memref();
  ::mlir::ValueRange indices();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class StoreOp : public ::mlir::Op<StoreOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::MemRefsNormalizable, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = StoreOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.store");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value value();
  ::mlir::Value memref();
  ::mlir::Operation::operand_range indices();
  ::mlir::MutableOperandRange valueMutable();
  ::mlir::MutableOperandRange memrefMutable();
  ::mlir::MutableOperandRange indicesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value valueToStore, Value memref);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

      Value getValueToStore() { return getOperand(0); }

      Value getMemRef() { return getOperand(1); }
      void setMemRef(Value value) { setOperand(1, value); }
      MemRefType getMemRefType() {
        return getMemRef().getType().cast<MemRefType>();
      }

      operand_range getIndices() {
        return {operand_begin() + 2, operand_end()};
      }
  
};
} // namespace memref
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::StoreOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::TransposeOp declarations
//===----------------------------------------------------------------------===//

class TransposeOpAdaptor {
public:
  TransposeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  TransposeOpAdaptor(TransposeOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value in();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::AffineMapAttr permutation();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class TransposeOp : public ::mlir::Op<TransposeOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::OneOperand, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TransposeOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("permutation")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier permutationAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier permutationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.transpose");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value in();
  ::mlir::MutableOperandRange inMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::AffineMapAttr permutationAttr();
  ::mlir::AffineMap permutation();
  void permutationAttr(::mlir::AffineMapAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value in, AffineMapAttr permutation, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value in, ::mlir::AffineMapAttr permutation);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value in, ::mlir::AffineMapAttr permutation);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value in, ::mlir::AffineMap permutation);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value in, ::mlir::AffineMap permutation);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  ::mlir::OpFoldResult fold(::llvm::ArrayRef<::mlir::Attribute> operands);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    static StringRef getPermutationAttrName() { return "permutation"; }
    ShapedType getShapedType() { return in().getType().cast<ShapedType>(); }
  

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 1 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace memref
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::TransposeOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::ViewOp declarations
//===----------------------------------------------------------------------===//

class ViewOpAdaptor {
public:
  ViewOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ViewOpAdaptor(ViewOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value source();
  ::mlir::Value byte_shift();
  ::mlir::ValueRange sizes();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ViewOp : public ::mlir::Op<ViewOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::MemRefType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::ViewLikeOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ViewOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.view");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value source();
  ::mlir::Value byte_shift();
  ::mlir::Operation::operand_range sizes();
  ::mlir::MutableOperandRange sourceMutable();
  ::mlir::MutableOperandRange byte_shiftMutable();
  ::mlir::MutableOperandRange sizesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value source, ::mlir::Value byte_shift, ::mlir::ValueRange sizes);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value byte_shift, ::mlir::ValueRange sizes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::Value getViewSource();
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    /// The result of a view is always a memref.
    MemRefType getType() { return getResult().getType().cast<MemRefType>(); }

    /// Returns the dynamic sizes for this view operation. This is redundant
    /// with `sizes` but needed in template implementations. More specifically:
    /// ```
    /// template <typename AnyMemRefDefOp>
    /// bool isMemRefSizeValidSymbol(AnyMemRefDefOp memrefDefOp, unsigned index,
    ///                              Region *region)
    /// ```
    operand_range getDynamicSizes() {
      return {sizes().begin(), sizes().end()};
    }
  
};
} // namespace memref
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::ViewOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::SubViewOp declarations
//===----------------------------------------------------------------------===//

class SubViewOpAdaptor {
public:
  SubViewOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  SubViewOpAdaptor(SubViewOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value source();
  ::mlir::ValueRange offsets();
  ::mlir::ValueRange sizes();
  ::mlir::ValueRange strides();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr static_offsets();
  ::mlir::ArrayAttr static_sizes();
  ::mlir::ArrayAttr static_strides();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class SubViewOp : public ::mlir::Op<SubViewOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::MemRefType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::ViewLikeOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OffsetSizeAndStrideOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SubViewOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("static_offsets"), ::llvm::StringRef("static_sizes"), ::llvm::StringRef("static_strides"), ::llvm::StringRef("operand_segment_sizes")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier static_offsetsAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier static_offsetsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier static_sizesAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier static_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier static_stridesAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier static_stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  ::mlir::Identifier operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(3);
  }
  static ::mlir::Identifier operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.subview");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value source();
  ::mlir::Operation::operand_range offsets();
  ::mlir::Operation::operand_range sizes();
  ::mlir::Operation::operand_range strides();
  ::mlir::MutableOperandRange sourceMutable();
  ::mlir::MutableOperandRange offsetsMutable();
  ::mlir::MutableOperandRange sizesMutable();
  ::mlir::MutableOperandRange stridesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value result();
  ::mlir::ArrayAttr static_offsetsAttr();
  ::mlir::ArrayAttr static_offsets();
  ::mlir::ArrayAttr static_sizesAttr();
  ::mlir::ArrayAttr static_sizes();
  ::mlir::ArrayAttr static_stridesAttr();
  ::mlir::ArrayAttr static_strides();
  void static_offsetsAttr(::mlir::ArrayAttr attr);
  void static_sizesAttr(::mlir::ArrayAttr attr);
  void static_stridesAttr(::mlir::ArrayAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes, ArrayRef<OpFoldResult> strides, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, MemRefType resultType, Value source, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes, ArrayRef<OpFoldResult> strides, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, ArrayRef<int64_t> offsets, ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, MemRefType resultType, Value source, ArrayRef<int64_t> offsets, ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, ValueRange offsets, ValueRange sizes, ValueRange strides, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, MemRefType resultType, Value source, ValueRange offsets, ValueRange sizes, ValueRange strides, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::ArrayAttr static_offsets, ::mlir::ArrayAttr static_sizes, ::mlir::ArrayAttr static_strides);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::ArrayAttr static_offsets, ::mlir::ArrayAttr static_sizes, ::mlir::ArrayAttr static_strides);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::OpFoldResult fold(::llvm::ArrayRef<::mlir::Attribute> operands);
  ::mlir::Value getViewSource();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    /// Returns the dynamic sizes for this subview operation if specified.
    operand_range getDynamicSizes() { return sizes(); }

    /// Return the list of Range (i.e. offset, size, stride). Each
    /// Range entry contains either the dynamic value or a ConstantIndexOp
    /// constructed with `b` at location `loc`.
    SmallVector<Range, 8> getOrCreateRanges(OpBuilder &b, Location loc) {
      return mlir::getOrCreateRanges(*this, b, loc);
    }
  
    /// Returns the type of the base memref operand.
    MemRefType getSourceType() {
      return source().getType().cast<MemRefType>();
    }

    /// The result of a subview is always a memref.
    MemRefType getType() { return getResult().getType().cast<MemRefType>(); }

    /// A subview result type can be fully inferred from the source type and the
    /// static representation of offsets, sizes and strides. Special sentinels
    /// encode the dynamic case.
    static Type inferResultType(MemRefType sourceMemRefType,
                                ArrayRef<int64_t> staticOffsets,
                                ArrayRef<int64_t> staticSizes,
                                ArrayRef<int64_t> staticStrides);
    static Type inferResultType(MemRefType sourceMemRefType,
                                ArrayRef<OpFoldResult> staticOffsets,
                                ArrayRef<OpFoldResult> staticSizes,
                                ArrayRef<OpFoldResult> staticStrides);
    static Type inferRankReducedResultType(unsigned resultRank,
                                           MemRefType sourceMemRefType,
                                           ArrayRef<int64_t> staticOffsets,
                                           ArrayRef<int64_t> staticSizes,
                                           ArrayRef<int64_t> staticStrides);
    static Type inferRankReducedResultType(unsigned resultRank,
                                           MemRefType sourceMemRefType,
                                           ArrayRef<OpFoldResult> staticOffsets,
                                           ArrayRef<OpFoldResult> staticSizes,
                                           ArrayRef<OpFoldResult> staticStrides);

    /// Return the expected rank of each of the`static_offsets`, `static_sizes`
    /// and `static_strides` attributes.
    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned rank = getSourceType().getRank();
      return {rank, rank, rank};
    }

    /// Return the number of leading operands before the `offsets`, `sizes` and
    /// and `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 1; }

    /// Return the dimensions of the source type that are dropped when
    /// the result is rank-reduced.
    llvm::SmallDenseSet<unsigned> getDroppedDims();
  

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 4 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace memref
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::SubViewOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::TensorLoadOp declarations
//===----------------------------------------------------------------------===//

class TensorLoadOpAdaptor {
public:
  TensorLoadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  TensorLoadOpAdaptor(TensorLoadOp&op);
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
class TensorLoadOp : public ::mlir::Op<TensorLoadOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::SameOperandsAndResultShape, ::mlir::OpTrait::SameOperandsAndResultElementType, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TensorLoadOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.tensor_load");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value memref();
  ::mlir::MutableOperandRange memrefMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value result();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value memref);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value memref);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::OpFoldResult fold(::llvm::ArrayRef<::mlir::Attribute> operands);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    /// The result of a tensor_load is always a tensor.
    TensorType getType() {
      Type resultType = getResult().getType();
      if (resultType.isa<TensorType>())
        return resultType.cast<TensorType>();
      return {};
    }
  
};
} // namespace memref
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::TensorLoadOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::TensorStoreOp declarations
//===----------------------------------------------------------------------===//

class TensorStoreOpAdaptor {
public:
  TensorStoreOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  TensorStoreOpAdaptor(TensorStoreOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value tensor();
  ::mlir::Value memref();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class TensorStoreOp : public ::mlir::Op<TensorStoreOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::SameOperandsShape, ::mlir::OpTrait::SameOperandsElementType, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TensorStoreOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.tensor_store");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value tensor();
  ::mlir::Value memref();
  ::mlir::MutableOperandRange tensorMutable();
  ::mlir::MutableOperandRange memrefMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value tensor, ::mlir::Value memref);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value tensor, ::mlir::Value memref);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
};
} // namespace memref
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::TensorStoreOp)


#endif  // GET_OP_CLASSES

