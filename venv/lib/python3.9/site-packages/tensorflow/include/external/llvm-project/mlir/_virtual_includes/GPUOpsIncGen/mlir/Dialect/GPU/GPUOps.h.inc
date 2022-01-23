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
namespace gpu {
class AllReduceOp;
} // namespace gpu
} // namespace mlir
namespace mlir {
namespace gpu {
class AllocOp;
} // namespace gpu
} // namespace mlir
namespace mlir {
namespace gpu {
class BarrierOp;
} // namespace gpu
} // namespace mlir
namespace mlir {
namespace gpu {
class BlockDimOp;
} // namespace gpu
} // namespace mlir
namespace mlir {
namespace gpu {
class BlockIdOp;
} // namespace gpu
} // namespace mlir
namespace mlir {
namespace gpu {
class DeallocOp;
} // namespace gpu
} // namespace mlir
namespace mlir {
namespace gpu {
class GPUFuncOp;
} // namespace gpu
} // namespace mlir
namespace mlir {
namespace gpu {
class GPUModuleOp;
} // namespace gpu
} // namespace mlir
namespace mlir {
namespace gpu {
class GridDimOp;
} // namespace gpu
} // namespace mlir
namespace mlir {
namespace gpu {
class HostRegisterOp;
} // namespace gpu
} // namespace mlir
namespace mlir {
namespace gpu {
class LaunchFuncOp;
} // namespace gpu
} // namespace mlir
namespace mlir {
namespace gpu {
class LaunchOp;
} // namespace gpu
} // namespace mlir
namespace mlir {
namespace gpu {
class MemcpyOp;
} // namespace gpu
} // namespace mlir
namespace mlir {
namespace gpu {
class MemsetOp;
} // namespace gpu
} // namespace mlir
namespace mlir {
namespace gpu {
class ModuleEndOp;
} // namespace gpu
} // namespace mlir
namespace mlir {
namespace gpu {
class NumSubgroupsOp;
} // namespace gpu
} // namespace mlir
namespace mlir {
namespace gpu {
class ReturnOp;
} // namespace gpu
} // namespace mlir
namespace mlir {
namespace gpu {
class ShuffleOp;
} // namespace gpu
} // namespace mlir
namespace mlir {
namespace gpu {
class SubgroupIdOp;
} // namespace gpu
} // namespace mlir
namespace mlir {
namespace gpu {
class SubgroupMmaComputeOp;
} // namespace gpu
} // namespace mlir
namespace mlir {
namespace gpu {
class SubgroupMmaConstantMatrixOp;
} // namespace gpu
} // namespace mlir
namespace mlir {
namespace gpu {
class SubgroupMmaLoadMatrixOp;
} // namespace gpu
} // namespace mlir
namespace mlir {
namespace gpu {
class SubgroupMmaStoreMatrixOp;
} // namespace gpu
} // namespace mlir
namespace mlir {
namespace gpu {
class SubgroupSizeOp;
} // namespace gpu
} // namespace mlir
namespace mlir {
namespace gpu {
class TerminatorOp;
} // namespace gpu
} // namespace mlir
namespace mlir {
namespace gpu {
class ThreadIdOp;
} // namespace gpu
} // namespace mlir
namespace mlir {
namespace gpu {
class WaitOp;
} // namespace gpu
} // namespace mlir
namespace mlir {
namespace gpu {
class YieldOp;
} // namespace gpu
} // namespace mlir
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::AllReduceOp declarations
//===----------------------------------------------------------------------===//

class AllReduceOpAdaptor {
public:
  AllReduceOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  AllReduceOpAdaptor(AllReduceOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value value();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr op();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &body();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class AllReduceOp : public ::mlir::Op<AllReduceOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::SameOperandsAndResultType, ::mlir::OpTrait::IsIsolatedFromAbove> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AllReduceOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("op")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier opAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier opAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gpu.all_reduce");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value value();
  ::mlir::MutableOperandRange valueMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &body();
  ::mlir::StringAttr opAttr();
  ::llvm::Optional< ::llvm::StringRef > op();
  void opAttr(::mlir::StringAttr attr);
  ::mlir::Attribute removeOpAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value value, /*optional*/::mlir::StringAttr op);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, /*optional*/::mlir::StringAttr op);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, /*optional*/::mlir::StringAttr op);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 1 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::AllReduceOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::AllocOp declarations
//===----------------------------------------------------------------------===//

class AllocOpAdaptor {
public:
  AllocOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  AllocOpAdaptor(AllocOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange asyncDependencies();
  ::mlir::ValueRange dynamicSizes();
  ::mlir::ValueRange symbolOperands();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class AllocOp : public ::mlir::Op<AllocOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::AtLeastNResults<1>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::gpu::AsyncOpInterface::Trait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpAsmOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AllocOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("operand_segment_sizes")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gpu.alloc");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range asyncDependencies();
  ::mlir::Operation::operand_range dynamicSizes();
  ::mlir::Operation::operand_range symbolOperands();
  ::mlir::MutableOperandRange asyncDependenciesMutable();
  ::mlir::MutableOperandRange dynamicSizesMutable();
  ::mlir::MutableOperandRange symbolOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value memref();
  ::mlir::Value asyncToken();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type memref, /*optional*/::mlir::Type asyncToken, ::mlir::ValueRange asyncDependencies, ::mlir::ValueRange dynamicSizes, ::mlir::ValueRange symbolOperands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange asyncDependencies, ::mlir::ValueRange dynamicSizes, ::mlir::ValueRange symbolOperands);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    MemRefType getType() { return memref().getType().cast<MemRefType>(); }
  

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 1 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::AllocOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::BarrierOp declarations
//===----------------------------------------------------------------------===//

class BarrierOpAdaptor {
public:
  BarrierOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  BarrierOpAdaptor(BarrierOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class BarrierOp : public ::mlir::Op<BarrierOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = BarrierOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gpu.barrier");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
};
} // namespace gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::BarrierOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::BlockDimOp declarations
//===----------------------------------------------------------------------===//

class BlockDimOpAdaptor {
public:
  BlockDimOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  BlockDimOpAdaptor(BlockDimOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr dimension();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class BlockDimOp : public ::mlir::Op<BlockDimOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::IndexType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = BlockDimOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dimension")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier dimensionAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier dimensionAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gpu.block_dim");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::StringAttr dimensionAttr();
  ::llvm::StringRef dimension();
  void dimensionAttr(::mlir::StringAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::StringAttr dimension);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr dimension);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::llvm::StringRef dimension);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef dimension);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
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
} // namespace gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::BlockDimOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::BlockIdOp declarations
//===----------------------------------------------------------------------===//

class BlockIdOpAdaptor {
public:
  BlockIdOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  BlockIdOpAdaptor(BlockIdOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr dimension();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class BlockIdOp : public ::mlir::Op<BlockIdOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::IndexType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = BlockIdOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dimension")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier dimensionAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier dimensionAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gpu.block_id");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::StringAttr dimensionAttr();
  ::llvm::StringRef dimension();
  void dimensionAttr(::mlir::StringAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::StringAttr dimension);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr dimension);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::llvm::StringRef dimension);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef dimension);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
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
} // namespace gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::BlockIdOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::DeallocOp declarations
//===----------------------------------------------------------------------===//

class DeallocOpAdaptor {
public:
  DeallocOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  DeallocOpAdaptor(DeallocOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange asyncDependencies();
  ::mlir::Value memref();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class DeallocOp : public ::mlir::Op<DeallocOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::gpu::AsyncOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = DeallocOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gpu.dealloc");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range asyncDependencies();
  ::mlir::Value memref();
  ::mlir::MutableOperandRange asyncDependenciesMutable();
  ::mlir::MutableOperandRange memrefMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value asyncToken();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type asyncToken, ::mlir::ValueRange asyncDependencies, ::mlir::Value memref);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange asyncDependencies, ::mlir::Value memref);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
};
} // namespace gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::DeallocOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::GPUFuncOp declarations
//===----------------------------------------------------------------------===//

class GPUFuncOpAdaptor {
public:
  GPUFuncOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  GPUFuncOpAdaptor(GPUFuncOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &body();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class GPUFuncOp : public ::mlir::Op<GPUFuncOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::HasParent<GPUModuleOp>::Impl, ::mlir::OpTrait::AutomaticAllocationScope, ::mlir::OpTrait::FunctionLike, ::mlir::OpTrait::IsIsolatedFromAbove, ::mlir::SymbolOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GPUFuncOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gpu.func");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &body();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, StringRef name, FunctionType type, TypeRange workgroupAttributions = {}, TypeRange privateAttributions = {}, ArrayRef<NamedAttribute> attrs = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();

    /// Returns `true` if the GPU function defined by this Op is a kernel, i.e.
    /// it is intended to be launched from host.
    bool isKernel() {
      return (*this)->getAttrOfType<UnitAttr>(
          GPUDialect::getKernelFuncAttrName()) != nullptr;
    }

    /// Returns the number of buffers located in the workgroup memory.
    unsigned getNumWorkgroupAttributions() {
      return (*this)->getAttrOfType<IntegerAttr>(
          getNumWorkgroupAttributionsAttrName()).getInt();
    }

    /// Returns a list of block arguments that correspond to buffers located in
    /// the workgroup memory
    ArrayRef<BlockArgument> getWorkgroupAttributions() {
      auto begin =
          std::next(getBody().args_begin(), getType().getNumInputs());
      auto end = std::next(begin, getNumWorkgroupAttributions());
      return {begin, end};
    }

    /// Adds a new block argument that corresponds to buffers located in
    /// workgroup memory.
    BlockArgument addWorkgroupAttribution(Type type);

    /// Returns the number of buffers located in the private memory.
    unsigned getNumPrivateAttributions() {
      return getBody().getNumArguments() - getType().getNumInputs() -
          getNumWorkgroupAttributions();
    }

    /// Returns a list of block arguments that correspond to buffers located in
    /// the private memory.
    ArrayRef<BlockArgument> getPrivateAttributions() {
      // Buffers on the private memory always come after buffers on the workgroup
      // memory.
      auto begin =
          std::next(getBody().args_begin(),
                    getType().getNumInputs() + getNumWorkgroupAttributions());
      return {begin, getBody().args_end()};
    }

    /// Adds a new block argument that corresponds to buffers located in
    /// private memory.
    BlockArgument addPrivateAttribution(Type type);

    /// Returns the name of the attribute containing the number of buffers
    /// located in the workgroup memory.
    static StringRef getNumWorkgroupAttributionsAttrName() {
      return "workgroup_attributions";
    }

    // FunctionLike trait needs access to the functions below.
    friend class OpTrait::FunctionLike<GPUFuncOp>;

    /// Hooks for the input/output type enumeration in FunctionLike .
    unsigned getNumFuncArguments() { return getType().getNumInputs(); }
    unsigned getNumFuncResults() { return getType().getNumResults(); }

    /// Returns the keywords used in the custom syntax for this Op.
    static StringRef getWorkgroupKeyword() { return "workgroup"; }
    static StringRef getPrivateKeyword() { return "private"; }
    static StringRef getKernelKeyword() { return "kernel"; }

    /// Hook for FunctionLike verifier.
    LogicalResult verifyType();

    /// Verifies the body of the function.
    LogicalResult verifyBody();
  
};
} // namespace gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::GPUFuncOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::GPUModuleOp declarations
//===----------------------------------------------------------------------===//

class GPUModuleOpAdaptor {
public:
  GPUModuleOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  GPUModuleOpAdaptor(GPUModuleOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &body();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class GPUModuleOp : public ::mlir::Op<GPUModuleOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::DataLayoutOpInterface::Trait, ::mlir::HasDefaultDLTIDataLayout, ::mlir::OpTrait::IsIsolatedFromAbove, ::mlir::OpTrait::SymbolTable, ::mlir::SymbolOpInterface::Trait, ::mlir::OpTrait::SingleBlockImplicitTerminator<ModuleEndOp>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GPUModuleOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gpu.module");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &body();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, StringRef name);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
};
} // namespace gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::GPUModuleOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::GridDimOp declarations
//===----------------------------------------------------------------------===//

class GridDimOpAdaptor {
public:
  GridDimOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  GridDimOpAdaptor(GridDimOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr dimension();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class GridDimOp : public ::mlir::Op<GridDimOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::IndexType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GridDimOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dimension")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier dimensionAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier dimensionAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gpu.grid_dim");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::StringAttr dimensionAttr();
  ::llvm::StringRef dimension();
  void dimensionAttr(::mlir::StringAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::StringAttr dimension);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr dimension);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::llvm::StringRef dimension);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef dimension);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
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
} // namespace gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::GridDimOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::HostRegisterOp declarations
//===----------------------------------------------------------------------===//

class HostRegisterOpAdaptor {
public:
  HostRegisterOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  HostRegisterOpAdaptor(HostRegisterOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value value();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class HostRegisterOp : public ::mlir::Op<HostRegisterOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::OneOperand> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = HostRegisterOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gpu.host_register");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value value();
  ::mlir::MutableOperandRange valueMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
};
} // namespace gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::HostRegisterOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::LaunchFuncOp declarations
//===----------------------------------------------------------------------===//

class LaunchFuncOpAdaptor {
public:
  LaunchFuncOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  LaunchFuncOpAdaptor(LaunchFuncOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange asyncDependencies();
  ::mlir::Value gridSizeX();
  ::mlir::Value gridSizeY();
  ::mlir::Value gridSizeZ();
  ::mlir::Value blockSizeX();
  ::mlir::Value blockSizeY();
  ::mlir::Value blockSizeZ();
  ::mlir::ValueRange operands();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::SymbolRefAttr kernel();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class LaunchFuncOp : public ::mlir::Op<LaunchFuncOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<6>::Impl, ::mlir::gpu::AsyncOpInterface::Trait, ::mlir::OpTrait::AttrSizedOperandSegments> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LaunchFuncOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("kernel"), ::llvm::StringRef("operand_segment_sizes")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier kernelAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier kernelAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gpu.launch_func");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range asyncDependencies();
  ::mlir::Value gridSizeX();
  ::mlir::Value gridSizeY();
  ::mlir::Value gridSizeZ();
  ::mlir::Value blockSizeX();
  ::mlir::Value blockSizeY();
  ::mlir::Value blockSizeZ();
  ::mlir::Operation::operand_range operands();
  ::mlir::MutableOperandRange asyncDependenciesMutable();
  ::mlir::MutableOperandRange gridSizeXMutable();
  ::mlir::MutableOperandRange gridSizeYMutable();
  ::mlir::MutableOperandRange gridSizeZMutable();
  ::mlir::MutableOperandRange blockSizeXMutable();
  ::mlir::MutableOperandRange blockSizeYMutable();
  ::mlir::MutableOperandRange blockSizeZMutable();
  ::mlir::MutableOperandRange operandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value asyncToken();
  ::mlir::SymbolRefAttr kernelAttr();
  ::mlir::SymbolRefAttr kernel();
  void kernelAttr(::mlir::SymbolRefAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, GPUFuncOp kernelFunc, KernelDim3 gridSize, KernelDim3 blockSize, ValueRange kernelOperands);
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);

    /// The number of operands passed to the kernel function.
    unsigned getNumKernelOperands();

    /// The name of the kernel's containing module.
    StringAttr getKernelModuleName();

    /// The name of the kernel.
    StringAttr getKernelName();

    /// The i-th operand passed to the kernel function.
    Value getKernelOperand(unsigned i);

    /// Get the SSA values passed as operands to specify the grid size.
    KernelDim3 getGridSizeOperandValues();

    /// Get the SSA values passed as operands to specify the block size.
    KernelDim3 getBlockSizeOperandValues();

    /// The number of launch configuration operands, placed at the leading
    /// positions of the operand list.
    static constexpr unsigned kNumConfigOperands = 6;

    // This needs to quietly verify if attributes with names defined below are
    // present since it is run before the verifier of this op.
    friend LogicalResult GPUDialect::verifyOperationAttribute(Operation *,
                                                              NamedAttribute);

    /// The name of the symbol reference attribute specifying the kernel to launch.
    static StringRef getKernelAttrName() { return "kernel"; }
  

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 2 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::LaunchFuncOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::LaunchOp declarations
//===----------------------------------------------------------------------===//

class LaunchOpAdaptor {
public:
  LaunchOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  LaunchOpAdaptor(LaunchOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value gridSizeX();
  ::mlir::Value gridSizeY();
  ::mlir::Value gridSizeZ();
  ::mlir::Value blockSizeX();
  ::mlir::Value blockSizeY();
  ::mlir::Value blockSizeZ();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &body();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class LaunchOp : public ::mlir::Op<LaunchOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<6>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LaunchOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gpu.launch");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value gridSizeX();
  ::mlir::Value gridSizeY();
  ::mlir::Value gridSizeZ();
  ::mlir::Value blockSizeX();
  ::mlir::Value blockSizeY();
  ::mlir::Value blockSizeZ();
  ::mlir::MutableOperandRange gridSizeXMutable();
  ::mlir::MutableOperandRange gridSizeYMutable();
  ::mlir::MutableOperandRange gridSizeZMutable();
  ::mlir::MutableOperandRange blockSizeXMutable();
  ::mlir::MutableOperandRange blockSizeYMutable();
  ::mlir::MutableOperandRange blockSizeZMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &body();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value gridSizeX, Value gridSizeY, Value gridSizeZ, Value blockSizeX, Value blockSizeY, Value blockSizeZ);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);

    /// Get the SSA values corresponding to kernel block identifiers.
    KernelDim3 getBlockIds();
    /// Get the SSA values corresponding to kernel thread identifiers.
    KernelDim3 getThreadIds();
    /// Get the SSA values corresponding to kernel grid size.
    KernelDim3 getGridSize();
    /// Get the SSA values corresponding to kernel block size.
    KernelDim3 getBlockSize();

    /// Get the SSA values passed as operands to specify the grid size.
    KernelDim3 getGridSizeOperandValues();
    /// Get the SSA values passed as operands to specify the block size.
    KernelDim3 getBlockSizeOperandValues();

    static StringRef getBlocksKeyword() { return "blocks"; }
    static StringRef getThreadsKeyword() { return "threads"; }

    /// The number of launch configuration operands, placed at the leading
    /// positions of the operand list.
    static constexpr unsigned kNumConfigOperands = 6;

    /// The number of region attributes containing the launch configuration,
    /// placed in the leading positions of the argument list.
    static constexpr unsigned kNumConfigRegionAttributes = 12;
  
};
} // namespace gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::LaunchOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::MemcpyOp declarations
//===----------------------------------------------------------------------===//

class MemcpyOpAdaptor {
public:
  MemcpyOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  MemcpyOpAdaptor(MemcpyOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange asyncDependencies();
  ::mlir::Value dst();
  ::mlir::Value src();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class MemcpyOp : public ::mlir::Op<MemcpyOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::gpu::AsyncOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = MemcpyOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gpu.memcpy");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range asyncDependencies();
  ::mlir::Value dst();
  ::mlir::Value src();
  ::mlir::MutableOperandRange asyncDependenciesMutable();
  ::mlir::MutableOperandRange dstMutable();
  ::mlir::MutableOperandRange srcMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value asyncToken();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type asyncToken, ::mlir::ValueRange asyncDependencies, ::mlir::Value dst, ::mlir::Value src);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange asyncDependencies, ::mlir::Value dst, ::mlir::Value src);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
};
} // namespace gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::MemcpyOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::MemsetOp declarations
//===----------------------------------------------------------------------===//

class MemsetOpAdaptor {
public:
  MemsetOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  MemsetOpAdaptor(MemsetOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange asyncDependencies();
  ::mlir::Value dst();
  ::mlir::Value value();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class MemsetOp : public ::mlir::Op<MemsetOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::gpu::AsyncOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = MemsetOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gpu.memset");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range asyncDependencies();
  ::mlir::Value dst();
  ::mlir::Value value();
  ::mlir::MutableOperandRange asyncDependenciesMutable();
  ::mlir::MutableOperandRange dstMutable();
  ::mlir::MutableOperandRange valueMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value asyncToken();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type asyncToken, ::mlir::ValueRange asyncDependencies, ::mlir::Value dst, ::mlir::Value value);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange asyncDependencies, ::mlir::Value dst, ::mlir::Value value);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
};
} // namespace gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::MemsetOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::ModuleEndOp declarations
//===----------------------------------------------------------------------===//

class ModuleEndOpAdaptor {
public:
  ModuleEndOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ModuleEndOpAdaptor(ModuleEndOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ModuleEndOp : public ::mlir::Op<ModuleEndOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::IsTerminator, ::mlir::OpTrait::HasParent<GPUModuleOp>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ModuleEndOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gpu.module_end");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
};
} // namespace gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::ModuleEndOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::NumSubgroupsOp declarations
//===----------------------------------------------------------------------===//

class NumSubgroupsOpAdaptor {
public:
  NumSubgroupsOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  NumSubgroupsOpAdaptor(NumSubgroupsOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class NumSubgroupsOp : public ::mlir::Op<NumSubgroupsOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::IndexType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = NumSubgroupsOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gpu.num_subgroups");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value result();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
};
} // namespace gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::NumSubgroupsOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::ReturnOp declarations
//===----------------------------------------------------------------------===//

class ReturnOpAdaptor {
public:
  ReturnOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ReturnOpAdaptor(ReturnOp&op);
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
class ReturnOp : public ::mlir::Op<ReturnOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::HasParent<GPUFuncOp>::Impl, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::IsTerminator> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ReturnOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gpu.return");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range operands();
  ::mlir::MutableOperandRange operandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
};
} // namespace gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::ReturnOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::ShuffleOp declarations
//===----------------------------------------------------------------------===//

class ShuffleOpAdaptor {
public:
  ShuffleOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ShuffleOpAdaptor(ShuffleOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value value();
  ::mlir::Value offset();
  ::mlir::Value width();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr mode();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ShuffleOp : public ::mlir::Op<ShuffleOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::NResults<2>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpAsmOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ShuffleOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("mode")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier modeAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier modeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gpu.shuffle");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value value();
  ::mlir::Value offset();
  ::mlir::Value width();
  ::mlir::MutableOperandRange valueMutable();
  ::mlir::MutableOperandRange offsetMutable();
  ::mlir::MutableOperandRange widthMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value result();
  ::mlir::Value valid();
  ::mlir::StringAttr modeAttr();
  ::llvm::StringRef mode();
  void modeAttr(::mlir::StringAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Type valid, ::mlir::Value value, ::mlir::Value offset, ::mlir::Value width, ::mlir::StringAttr mode);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value offset, ::mlir::Value width, ::mlir::StringAttr mode);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Type valid, ::mlir::Value value, ::mlir::Value offset, ::mlir::Value width, ::llvm::StringRef mode);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value offset, ::mlir::Value width, ::llvm::StringRef mode);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
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
} // namespace gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::ShuffleOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::SubgroupIdOp declarations
//===----------------------------------------------------------------------===//

class SubgroupIdOpAdaptor {
public:
  SubgroupIdOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  SubgroupIdOpAdaptor(SubgroupIdOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class SubgroupIdOp : public ::mlir::Op<SubgroupIdOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::IndexType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SubgroupIdOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gpu.subgroup_id");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value result();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
};
} // namespace gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::SubgroupIdOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::SubgroupMmaComputeOp declarations
//===----------------------------------------------------------------------===//

class SubgroupMmaComputeOpAdaptor {
public:
  SubgroupMmaComputeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  SubgroupMmaComputeOpAdaptor(SubgroupMmaComputeOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value opA();
  ::mlir::Value opB();
  ::mlir::Value opC();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class SubgroupMmaComputeOp : public ::mlir::Op<SubgroupMmaComputeOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SubgroupMmaComputeOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gpu.subgroup_mma_compute");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value opA();
  ::mlir::Value opB();
  ::mlir::Value opC();
  ::mlir::MutableOperandRange opAMutable();
  ::mlir::MutableOperandRange opBMutable();
  ::mlir::MutableOperandRange opCMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value opA, ::mlir::Value opB, ::mlir::Value opC);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value opA, ::mlir::Value opB, ::mlir::Value opC);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
};
} // namespace gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::SubgroupMmaComputeOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::SubgroupMmaConstantMatrixOp declarations
//===----------------------------------------------------------------------===//

class SubgroupMmaConstantMatrixOpAdaptor {
public:
  SubgroupMmaConstantMatrixOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  SubgroupMmaConstantMatrixOpAdaptor(SubgroupMmaConstantMatrixOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value value();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class SubgroupMmaConstantMatrixOp : public ::mlir::Op<SubgroupMmaConstantMatrixOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::OneOperand, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SubgroupMmaConstantMatrixOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gpu.subgroup_mma_constant_matrix");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value value();
  ::mlir::MutableOperandRange valueMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value value);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    gpu::MMAMatrixType getType() {
      return res().getType().cast<gpu::MMAMatrixType>();
    }
  
};
} // namespace gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::SubgroupMmaConstantMatrixOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::SubgroupMmaLoadMatrixOp declarations
//===----------------------------------------------------------------------===//

class SubgroupMmaLoadMatrixOpAdaptor {
public:
  SubgroupMmaLoadMatrixOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  SubgroupMmaLoadMatrixOpAdaptor(SubgroupMmaLoadMatrixOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value srcMemref();
  ::mlir::ValueRange indices();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr leadDimension();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class SubgroupMmaLoadMatrixOp : public ::mlir::Op<SubgroupMmaLoadMatrixOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SubgroupMmaLoadMatrixOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("leadDimension")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier leadDimensionAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier leadDimensionAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gpu.subgroup_mma_load_matrix");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value srcMemref();
  ::mlir::Operation::operand_range indices();
  ::mlir::MutableOperandRange srcMemrefMutable();
  ::mlir::MutableOperandRange indicesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  ::mlir::IntegerAttr leadDimensionAttr();
  ::llvm::APInt leadDimension();
  void leadDimensionAttr(::mlir::IntegerAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value srcMemref, ::mlir::ValueRange indices, ::mlir::IntegerAttr leadDimension);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value srcMemref, ::mlir::ValueRange indices, ::mlir::IntegerAttr leadDimension);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value srcMemref, ::mlir::ValueRange indices, ::llvm::APInt leadDimension);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value srcMemref, ::mlir::ValueRange indices, ::llvm::APInt leadDimension);
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
  assert(index < 1 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::SubgroupMmaLoadMatrixOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::SubgroupMmaStoreMatrixOp declarations
//===----------------------------------------------------------------------===//

class SubgroupMmaStoreMatrixOpAdaptor {
public:
  SubgroupMmaStoreMatrixOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  SubgroupMmaStoreMatrixOpAdaptor(SubgroupMmaStoreMatrixOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value src();
  ::mlir::Value dstMemref();
  ::mlir::ValueRange indices();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr leadDimension();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class SubgroupMmaStoreMatrixOp : public ::mlir::Op<SubgroupMmaStoreMatrixOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SubgroupMmaStoreMatrixOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("leadDimension")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier leadDimensionAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier leadDimensionAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gpu.subgroup_mma_store_matrix");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value src();
  ::mlir::Value dstMemref();
  ::mlir::Operation::operand_range indices();
  ::mlir::MutableOperandRange srcMutable();
  ::mlir::MutableOperandRange dstMemrefMutable();
  ::mlir::MutableOperandRange indicesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::IntegerAttr leadDimensionAttr();
  ::llvm::APInt leadDimension();
  void leadDimensionAttr(::mlir::IntegerAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value src, ::mlir::Value dstMemref, ::mlir::ValueRange indices, ::mlir::IntegerAttr leadDimension);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value dstMemref, ::mlir::ValueRange indices, ::mlir::IntegerAttr leadDimension);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value src, ::mlir::Value dstMemref, ::mlir::ValueRange indices, ::llvm::APInt leadDimension);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value dstMemref, ::mlir::ValueRange indices, ::llvm::APInt leadDimension);
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
  assert(index < 1 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::SubgroupMmaStoreMatrixOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::SubgroupSizeOp declarations
//===----------------------------------------------------------------------===//

class SubgroupSizeOpAdaptor {
public:
  SubgroupSizeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  SubgroupSizeOpAdaptor(SubgroupSizeOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class SubgroupSizeOp : public ::mlir::Op<SubgroupSizeOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::IndexType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SubgroupSizeOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gpu.subgroup_size");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value result();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
};
} // namespace gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::SubgroupSizeOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::TerminatorOp declarations
//===----------------------------------------------------------------------===//

class TerminatorOpAdaptor {
public:
  TerminatorOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  TerminatorOpAdaptor(TerminatorOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class TerminatorOp : public ::mlir::Op<TerminatorOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::HasParent<LaunchOp>::Impl, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::IsTerminator> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TerminatorOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gpu.terminator");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
};
} // namespace gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::TerminatorOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::ThreadIdOp declarations
//===----------------------------------------------------------------------===//

class ThreadIdOpAdaptor {
public:
  ThreadIdOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ThreadIdOpAdaptor(ThreadIdOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr dimension();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ThreadIdOp : public ::mlir::Op<ThreadIdOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::IndexType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ThreadIdOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dimension")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier dimensionAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier dimensionAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gpu.thread_id");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::StringAttr dimensionAttr();
  ::llvm::StringRef dimension();
  void dimensionAttr(::mlir::StringAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::StringAttr dimension);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr dimension);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::llvm::StringRef dimension);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef dimension);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
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
} // namespace gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::ThreadIdOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::WaitOp declarations
//===----------------------------------------------------------------------===//

class WaitOpAdaptor {
public:
  WaitOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  WaitOpAdaptor(WaitOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange asyncDependencies();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class WaitOp : public ::mlir::Op<WaitOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::gpu::AsyncOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = WaitOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gpu.wait");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range asyncDependencies();
  ::mlir::MutableOperandRange asyncDependenciesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value asyncToken();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type asyncToken, ::mlir::ValueRange asyncDependencies);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
};
} // namespace gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::WaitOp)

namespace mlir {
namespace gpu {

//===----------------------------------------------------------------------===//
// ::mlir::gpu::YieldOp declarations
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
class YieldOp : public ::mlir::Op<YieldOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::IsTerminator> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = YieldOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gpu.yield");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range values();
  ::mlir::MutableOperandRange valuesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange values);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
};
} // namespace gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::YieldOp)


#endif  // GET_OP_CLASSES

