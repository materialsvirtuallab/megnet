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
class BatchMatmulOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class BatchMatvecOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class Conv1DNwcWcfOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class Conv1DOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class Conv2DNchwFchwOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class Conv2DNhwcHwcfOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class Conv2DNhwcHwcfQOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class Conv2DOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class Conv3DNdhwcDhwcfOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class Conv3DOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class ConvOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class CopyOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class DepthwiseConv2DNhwOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class DepthwiseConv2DNhwQOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class DepthwiseConv2DNhwcOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class DepthwiseConv2DNhwcQOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class DotOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class FillOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class FillRng2DOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class GenericOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class MatmulI8I8I32Op;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class MatmulOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class MatvecOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class Mmt4DOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class PoolingMaxOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class PoolingMinOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class PoolingNchwMaxOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class PoolingNdhwcMaxOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class PoolingNdhwcMinOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class PoolingNdhwcSumOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class PoolingNhwcMaxOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class PoolingNhwcMinOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class PoolingNhwcSumOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class PoolingSumOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class QuantizedBatchMatmulOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class QuantizedMatmulOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class SoftPlus2DOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class VecmatOp;
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
// ::mlir::linalg::BatchMatmulOp declarations
//===----------------------------------------------------------------------===//

class BatchMatmulOpAdaptor {
public:
  BatchMatmulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  BatchMatmulOpAdaptor(BatchMatmulOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class BatchMatmulOp : public ::mlir::Op<BatchMatmulOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::linalg::LinalgOp::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::linalg::ContractionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = BatchMatmulOpAdaptor;
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
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.batch_matmul");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      Operation *op = this->getOperation();
      if(op->getNumRegions() == 0 || op->getRegion(0).empty())
        return false;
      return !op->getRegion(0).front().getOps<IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return cast<LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }
  
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    

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
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::BatchMatmulOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::BatchMatvecOp declarations
//===----------------------------------------------------------------------===//

class BatchMatvecOpAdaptor {
public:
  BatchMatvecOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  BatchMatvecOpAdaptor(BatchMatvecOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class BatchMatvecOp : public ::mlir::Op<BatchMatvecOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::linalg::LinalgOp::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::linalg::ContractionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = BatchMatvecOpAdaptor;
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
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.batch_matvec");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      Operation *op = this->getOperation();
      if(op->getNumRegions() == 0 || op->getRegion(0).empty())
        return false;
      return !op->getRegion(0).front().getOps<IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return cast<LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }
  
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    

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
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::BatchMatvecOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Conv1DNwcWcfOp declarations
//===----------------------------------------------------------------------===//

class Conv1DNwcWcfOpAdaptor {
public:
  Conv1DNwcWcfOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  Conv1DNwcWcfOpAdaptor(Conv1DNwcWcfOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilations();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class Conv1DNwcWcfOp : public ::mlir::Op<Conv1DNwcWcfOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::linalg::LinalgOp::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::linalg::ConvolutionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = Conv1DNwcWcfOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("strides"), ::llvm::StringRef("dilations"), ::llvm::StringRef("operand_segment_sizes")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier stridesAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier dilationsAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier dilationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.conv_1d_nwc_wcf");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  void stridesAttr(::mlir::DenseIntElementsAttr attr);
  void dilationsAttr(::mlir::DenseIntElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      Operation *op = this->getOperation();
      if(op->getNumRegions() == 0 || op->getRegion(0).empty())
        return false;
      return !op->getRegion(0).front().getOps<IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return cast<LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }
  
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();
    
    

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
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::Conv1DNwcWcfOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Conv1DOp declarations
//===----------------------------------------------------------------------===//

class Conv1DOpAdaptor {
public:
  Conv1DOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  Conv1DOpAdaptor(Conv1DOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class Conv1DOp : public ::mlir::Op<Conv1DOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::linalg::LinalgOp::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::linalg::ConvolutionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = Conv1DOpAdaptor;
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
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.conv_1d");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      Operation *op = this->getOperation();
      if(op->getNumRegions() == 0 || op->getRegion(0).empty())
        return false;
      return !op->getRegion(0).front().getOps<IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return cast<LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }
  
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    

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
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::Conv1DOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Conv2DNchwFchwOp declarations
//===----------------------------------------------------------------------===//

class Conv2DNchwFchwOpAdaptor {
public:
  Conv2DNchwFchwOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  Conv2DNchwFchwOpAdaptor(Conv2DNchwFchwOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilations();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class Conv2DNchwFchwOp : public ::mlir::Op<Conv2DNchwFchwOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::linalg::LinalgOp::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::linalg::ConvolutionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = Conv2DNchwFchwOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("strides"), ::llvm::StringRef("dilations"), ::llvm::StringRef("operand_segment_sizes")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier stridesAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier dilationsAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier dilationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.conv_2d_nchw_fchw");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  void stridesAttr(::mlir::DenseIntElementsAttr attr);
  void dilationsAttr(::mlir::DenseIntElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      Operation *op = this->getOperation();
      if(op->getNumRegions() == 0 || op->getRegion(0).empty())
        return false;
      return !op->getRegion(0).front().getOps<IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return cast<LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }
  
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();
    
    

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
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::Conv2DNchwFchwOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Conv2DNhwcHwcfOp declarations
//===----------------------------------------------------------------------===//

class Conv2DNhwcHwcfOpAdaptor {
public:
  Conv2DNhwcHwcfOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  Conv2DNhwcHwcfOpAdaptor(Conv2DNhwcHwcfOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilations();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class Conv2DNhwcHwcfOp : public ::mlir::Op<Conv2DNhwcHwcfOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::linalg::LinalgOp::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::linalg::ConvolutionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = Conv2DNhwcHwcfOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("strides"), ::llvm::StringRef("dilations"), ::llvm::StringRef("operand_segment_sizes")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier stridesAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier dilationsAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier dilationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.conv_2d_nhwc_hwcf");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  void stridesAttr(::mlir::DenseIntElementsAttr attr);
  void dilationsAttr(::mlir::DenseIntElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      Operation *op = this->getOperation();
      if(op->getNumRegions() == 0 || op->getRegion(0).empty())
        return false;
      return !op->getRegion(0).front().getOps<IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return cast<LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }
  
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();
    
    

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
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::Conv2DNhwcHwcfOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Conv2DNhwcHwcfQOp declarations
//===----------------------------------------------------------------------===//

class Conv2DNhwcHwcfQOpAdaptor {
public:
  Conv2DNhwcHwcfQOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  Conv2DNhwcHwcfQOpAdaptor(Conv2DNhwcHwcfQOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilations();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class Conv2DNhwcHwcfQOp : public ::mlir::Op<Conv2DNhwcHwcfQOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::linalg::LinalgOp::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::linalg::ConvolutionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = Conv2DNhwcHwcfQOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("strides"), ::llvm::StringRef("dilations"), ::llvm::StringRef("operand_segment_sizes")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier stridesAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier dilationsAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier dilationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.conv_2d_nhwc_hwcf_q");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  void stridesAttr(::mlir::DenseIntElementsAttr attr);
  void dilationsAttr(::mlir::DenseIntElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      Operation *op = this->getOperation();
      if(op->getNumRegions() == 0 || op->getRegion(0).empty())
        return false;
      return !op->getRegion(0).front().getOps<IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return cast<LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }
  
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();
    
    

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
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::Conv2DNhwcHwcfQOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Conv2DOp declarations
//===----------------------------------------------------------------------===//

class Conv2DOpAdaptor {
public:
  Conv2DOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  Conv2DOpAdaptor(Conv2DOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class Conv2DOp : public ::mlir::Op<Conv2DOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::linalg::LinalgOp::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::linalg::ConvolutionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = Conv2DOpAdaptor;
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
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.conv_2d");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      Operation *op = this->getOperation();
      if(op->getNumRegions() == 0 || op->getRegion(0).empty())
        return false;
      return !op->getRegion(0).front().getOps<IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return cast<LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }
  
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    

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
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::Conv2DOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Conv3DNdhwcDhwcfOp declarations
//===----------------------------------------------------------------------===//

class Conv3DNdhwcDhwcfOpAdaptor {
public:
  Conv3DNdhwcDhwcfOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  Conv3DNdhwcDhwcfOpAdaptor(Conv3DNdhwcDhwcfOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilations();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class Conv3DNdhwcDhwcfOp : public ::mlir::Op<Conv3DNdhwcDhwcfOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::linalg::LinalgOp::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::linalg::ConvolutionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = Conv3DNdhwcDhwcfOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("strides"), ::llvm::StringRef("dilations"), ::llvm::StringRef("operand_segment_sizes")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier stridesAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier dilationsAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier dilationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.conv_3d_ndhwc_dhwcf");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  void stridesAttr(::mlir::DenseIntElementsAttr attr);
  void dilationsAttr(::mlir::DenseIntElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      Operation *op = this->getOperation();
      if(op->getNumRegions() == 0 || op->getRegion(0).empty())
        return false;
      return !op->getRegion(0).front().getOps<IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return cast<LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }
  
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();
    
    

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
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::Conv3DNdhwcDhwcfOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Conv3DOp declarations
//===----------------------------------------------------------------------===//

class Conv3DOpAdaptor {
public:
  Conv3DOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  Conv3DOpAdaptor(Conv3DOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class Conv3DOp : public ::mlir::Op<Conv3DOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::linalg::LinalgOp::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::linalg::ConvolutionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = Conv3DOpAdaptor;
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
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.conv_3d");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      Operation *op = this->getOperation();
      if(op->getNumRegions() == 0 || op->getRegion(0).empty())
        return false;
      return !op->getRegion(0).front().getOps<IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return cast<LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }
  
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    

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
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::Conv3DOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::ConvOp declarations
//===----------------------------------------------------------------------===//

class ConvOpAdaptor {
public:
  ConvOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ConvOpAdaptor(ConvOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value filter();
  ::mlir::Value input();
  ::mlir::Value output();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr strides();
  ::mlir::ArrayAttr dilations();
  ::mlir::DenseIntElementsAttr padding();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ConvOp : public ::mlir::Op<ConvOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::linalg::LinalgOp::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConvOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("strides"), ::llvm::StringRef("dilations"), ::llvm::StringRef("padding")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier stridesAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier dilationsAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier dilationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier paddingAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier paddingAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.conv");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value filter();
  ::mlir::Value input();
  ::mlir::Value output();
  ::mlir::MutableOperandRange filterMutable();
  ::mlir::MutableOperandRange inputMutable();
  ::mlir::MutableOperandRange outputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::ArrayAttr stridesAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > strides();
  ::mlir::ArrayAttr dilationsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > dilations();
  ::mlir::DenseIntElementsAttr paddingAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > padding();
  void stridesAttr(::mlir::ArrayAttr attr);
  void dilationsAttr(::mlir::ArrayAttr attr);
  void paddingAttr(::mlir::DenseIntElementsAttr attr);
  ::mlir::Attribute removeStridesAttr();
  ::mlir::Attribute removeDilationsAttr();
  ::mlir::Attribute removePaddingAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value filter, ::mlir::Value input, ::mlir::Value output, /*optional*/::mlir::ArrayAttr strides, /*optional*/::mlir::ArrayAttr dilations, /*optional*/::mlir::DenseIntElementsAttr padding);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value filter, ::mlir::Value input, ::mlir::Value output, /*optional*/::mlir::ArrayAttr strides, /*optional*/::mlir::ArrayAttr dilations, /*optional*/::mlir::DenseIntElementsAttr padding);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);

    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      Operation *op = this->getOperation();
      if(op->getNumRegions() == 0 || op->getRegion(0).empty())
        return false;
      return !op->getRegion(0).front().getOps<IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return cast<LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }
  
    std::string getLibraryCallName() {
      return generateLibraryCallName(getOperation());
    }
  
    int64_t getStride(unsigned i) {
      assert(i < getNumWindowLoops());
      if (!strides().hasValue()) return 1;
      return strides()->getValue()[i]
        .cast<IntegerAttr>().getValue().getSExtValue();
    }

    int64_t getDilation(unsigned i) {
      assert(i < getNumWindowLoops());
      if (!dilations().hasValue()) return 1;
      return dilations()->getValue()[i]
        .cast<IntegerAttr>().getValue().getSExtValue();
    }

    int64_t getLowPad(unsigned i) {
      assert(i < getNumWindowLoops());
      if (!padding().hasValue()) return 0;
      return padding().getValue().getValue<int64_t>({i, 0});
    }

    int64_t getHighPad(unsigned i) {
      assert(i < getNumWindowLoops());
      if (!padding().hasValue()) return 0;
      return padding().getValue().getValue<int64_t>({i, 1});
    }

    static std::function<void(ImplicitLocOpBuilder &b, Block &block)>
    getRegionBuilder() {
      return nullptr;
    }
  
    ValueRange inputs() { return getOperands().slice(0, 2); }
    ValueRange outputs() { return getOperands().take_back(); }

    // TODO: extend to support more than 1 dimensions and potentially grouping
    // too.
    unsigned getNumBatchDimensions() { return 1; }

    unsigned getNumInputFeatureDimensions() { return 1; }

    unsigned getNumOutputFeatureDimensions() { return 1; }

    unsigned getNumSpatialDimensions() {
      return getRank(getOutputOperand(0)) - getNumBatchDimensions() -
             getNumOutputFeatureDimensions();
    }

    ArrayAttr iterator_types() {
      // Outer parallel loops are always the number of output dimensions; i.e.
      // [b, xs, q] in the TF notation above.
      int64_t nPar = getRank(getOutputOperand(0));
      unsigned nRed = getNumInputFeatureDimensions();
      // Window loops are a special kind of reduction that is never tiled or
      // parallelized across; i.e. [zs] in the TF notation above whose number
      // match `xs` (i.e. 1 window loop per "image" dimension).
      // This may evolve in the future.
      // Conditionally check nPar is large enough for cases of ill-formed op:
      // this avoids overflows before hitting the verifier.
      assert(nPar > getNumBatchDimensions() + getNumInputFeatureDimensions() &&
             "expected at least one window dimension (i.e. memref ranks greater "
             "than 2). See 'func @conv_rank_limit' in "
             "mlir/test/Dialect/Linalg/invalid.mlir");
      unsigned nWin =
        nPar - getNumBatchDimensions() - getNumInputFeatureDimensions();
      SmallVector<StringRef, 8> iters(nPar, getParallelIteratorTypeName());
      iters.reserve(nPar + nRed + nWin);
      iters.append(nRed, getReductionIteratorTypeName());
      iters.append(nWin, getWindowIteratorTypeName());
      return Builder(getContext()).getStrArrayAttr(iters);
    }

    //   F(z0, ..., zN-1, q, k) *
    //     I(b, x0 + z0 - pad_low_0, ..., xN-1 + zN-1 - pad_low_N-1, q)
    //   ->  O(b, x0, ..., xN-1, k)
    // for N equal to `nWindow`. If there is no padding attribute, it will be
    // ignored.
    ArrayAttr indexing_maps() {
      MLIRContext *context = getContext();
      auto nWin = getNumWindowLoops();
      assert(nWin > 0 && "expected at least one window dimension (i.e. memref "
                         "ranks greater than 2)");
      unsigned idx = 0;
      // In the following, AffineDimExprs are indexed in loop order:
      //   [ b, xs, k,           q,                     zs]
      //    parallels     non-window reductions     windows
      //
      // Parallel dims are exactly the dimensions indexing `output`:
      //     output[b, x[0], ..., x[N-1], k]; i.e.
      //  * batch dimensions (bs with #bs = 1 for now)
      //  * "image" dimensions (xs with #xs = #zs = output_rank - #bs - #ks)
      //  * output filter dimensions (ks with #ks = 1 for now)
      auto bs = makeAffineDimExprs(getNumBatchDimensions(), idx, context);
      auto xs = makeAffineDimExprs(nWin, idx, context);
      auto ks = makeAffineDimExprs(
        getNumOutputFeatureDimensions(), idx, context);
      // Non-window reduction dim: sum_{z[0], ..., z[N-1], q}
      auto qs = makeAffineDimExprs(
        getNumInputFeatureDimensions(), idx, context);
      // Window reduction dims: sum_{z[0], ..., z[N-1], q}
      auto zs = makeAffineDimExprs(nWin, idx, context);
      // Construct the weighedSum expression.
      auto ws = weightedPoolingInputIndex(*this, xs, zs);
      return Builder(getContext()).getAffineMapArrayAttr({
        // filter[z[0], ..., z[N-1], q, k]
        AffineMap::get(idx, 0, concat(concat(zs, qs), ks), context),
        // input[b,
        //       x[0]*s[0] + d[0]*z[0] - pad_low[0],
        //       ...
        //       x[N-1]*s[N-1] + d[N-1]*z[N-1] - pad_low[N-1],
        //       q]
        AffineMap::get(idx, 0, concat(concat(bs, ws), qs), context),
        // output[b, x[0], ..., x[N-1], k]
        AffineMap::get(idx, 0, concat(concat(bs, xs), ks), context)});
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
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::ConvOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::CopyOp declarations
//===----------------------------------------------------------------------===//

class CopyOpAdaptor {
public:
  CopyOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  CopyOpAdaptor(CopyOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value output();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::AffineMapAttr inputPermutation();
  ::mlir::AffineMapAttr outputPermutation();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class CopyOp : public ::mlir::Op<CopyOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::linalg::LinalgOp::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::CopyOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CopyOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("inputPermutation"), ::llvm::StringRef("outputPermutation")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier inputPermutationAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier inputPermutationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier outputPermutationAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier outputPermutationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.copy");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value output();
  ::mlir::MutableOperandRange inputMutable();
  ::mlir::MutableOperandRange outputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &region();
  ::mlir::AffineMapAttr inputPermutationAttr();
  ::llvm::Optional< ::mlir::AffineMap > inputPermutation();
  ::mlir::AffineMapAttr outputPermutationAttr();
  ::llvm::Optional< ::mlir::AffineMap > outputPermutation();
  void inputPermutationAttr(::mlir::AffineMapAttr attr);
  void outputPermutationAttr(::mlir::AffineMapAttr attr);
  ::mlir::Attribute removeInputPermutationAttr();
  ::mlir::Attribute removeOutputPermutationAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value input, Value output, AffineMap inputPermutation = AffineMap(), AffineMap outputPermutation = AffineMap(), ArrayRef<NamedAttribute> attrs = {});
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);

    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      Operation *op = this->getOperation();
      if(op->getNumRegions() == 0 || op->getRegion(0).empty())
        return false;
      return !op->getRegion(0).front().getOps<IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return cast<LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }
  
    std::string getLibraryCallName() {
      return generateLibraryCallName(getOperation());
    }
  
    ValueRange inputs() { return getOperands().take_front(); }
    ValueRange outputs() { return getOperands().take_back(); }

    // Rank-polymorphic.
    //   filling_value -> O(ivs) with parallel iterators.
    ArrayAttr iterator_types() {
      int64_t nPar = getRank(getInputOperand(0));
      return Builder(getContext()).getStrArrayAttr(
        SmallVector<StringRef, 8>(nPar, getParallelIteratorTypeName()));
    }

    // I(input_perm(ivs)) -> O(output_perm(ivs))
    ArrayAttr indexing_maps() {
      MLIRContext *context = getContext();
      auto maybeInputMap = inputPermutation();
      auto maybeOutputMap = outputPermutation();
      int64_t inputRank = getRank(getInputOperand(0));
      int64_t outputRank = getRank(getOutputOperand(0));
      return Builder(getContext()).getAffineMapArrayAttr({
          extractOrIdentityMap(maybeInputMap, inputRank, context),
          extractOrIdentityMap(maybeOutputMap, outputRank, context)});
    }

    Value getSource() { return input();}
    Value getTarget() { return output(); }

    static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
    static std::function<void(ImplicitLocOpBuilder &b, Block &block)>
    getRegionBuilder() {
      return &regionBuilder;
    }
    static unsigned getNumRegionArgs() { return 2; }
  

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 2 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace linalg
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::CopyOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::DepthwiseConv2DNhwOp declarations
//===----------------------------------------------------------------------===//

class DepthwiseConv2DNhwOpAdaptor {
public:
  DepthwiseConv2DNhwOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  DepthwiseConv2DNhwOpAdaptor(DepthwiseConv2DNhwOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilations();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class DepthwiseConv2DNhwOp : public ::mlir::Op<DepthwiseConv2DNhwOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::linalg::LinalgOp::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::linalg::ConvolutionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = DepthwiseConv2DNhwOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("strides"), ::llvm::StringRef("dilations"), ::llvm::StringRef("operand_segment_sizes")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier stridesAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier dilationsAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier dilationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.depthwise_conv2D_nhw");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  void stridesAttr(::mlir::DenseIntElementsAttr attr);
  void dilationsAttr(::mlir::DenseIntElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      Operation *op = this->getOperation();
      if(op->getNumRegions() == 0 || op->getRegion(0).empty())
        return false;
      return !op->getRegion(0).front().getOps<IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return cast<LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }
  
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();
    
    

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
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::DepthwiseConv2DNhwOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::DepthwiseConv2DNhwQOp declarations
//===----------------------------------------------------------------------===//

class DepthwiseConv2DNhwQOpAdaptor {
public:
  DepthwiseConv2DNhwQOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  DepthwiseConv2DNhwQOpAdaptor(DepthwiseConv2DNhwQOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilations();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class DepthwiseConv2DNhwQOp : public ::mlir::Op<DepthwiseConv2DNhwQOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::linalg::LinalgOp::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::linalg::ConvolutionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = DepthwiseConv2DNhwQOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("strides"), ::llvm::StringRef("dilations"), ::llvm::StringRef("operand_segment_sizes")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier stridesAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier dilationsAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier dilationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.depthwise_conv2D_nhw_q");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  void stridesAttr(::mlir::DenseIntElementsAttr attr);
  void dilationsAttr(::mlir::DenseIntElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      Operation *op = this->getOperation();
      if(op->getNumRegions() == 0 || op->getRegion(0).empty())
        return false;
      return !op->getRegion(0).front().getOps<IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return cast<LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }
  
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();
    
    

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
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::DepthwiseConv2DNhwQOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::DepthwiseConv2DNhwcOp declarations
//===----------------------------------------------------------------------===//

class DepthwiseConv2DNhwcOpAdaptor {
public:
  DepthwiseConv2DNhwcOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  DepthwiseConv2DNhwcOpAdaptor(DepthwiseConv2DNhwcOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilations();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class DepthwiseConv2DNhwcOp : public ::mlir::Op<DepthwiseConv2DNhwcOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::linalg::LinalgOp::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::linalg::ConvolutionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = DepthwiseConv2DNhwcOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("strides"), ::llvm::StringRef("dilations"), ::llvm::StringRef("operand_segment_sizes")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier stridesAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier dilationsAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier dilationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.depthwise_conv2D_nhwc");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  void stridesAttr(::mlir::DenseIntElementsAttr attr);
  void dilationsAttr(::mlir::DenseIntElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      Operation *op = this->getOperation();
      if(op->getNumRegions() == 0 || op->getRegion(0).empty())
        return false;
      return !op->getRegion(0).front().getOps<IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return cast<LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }
  
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();
    
    

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
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::DepthwiseConv2DNhwcOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::DepthwiseConv2DNhwcQOp declarations
//===----------------------------------------------------------------------===//

class DepthwiseConv2DNhwcQOpAdaptor {
public:
  DepthwiseConv2DNhwcQOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  DepthwiseConv2DNhwcQOpAdaptor(DepthwiseConv2DNhwcQOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilations();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class DepthwiseConv2DNhwcQOp : public ::mlir::Op<DepthwiseConv2DNhwcQOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::linalg::LinalgOp::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::linalg::ConvolutionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = DepthwiseConv2DNhwcQOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("strides"), ::llvm::StringRef("dilations"), ::llvm::StringRef("operand_segment_sizes")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier stridesAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier dilationsAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier dilationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.depthwise_conv2D_nhwc_q");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  void stridesAttr(::mlir::DenseIntElementsAttr attr);
  void dilationsAttr(::mlir::DenseIntElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      Operation *op = this->getOperation();
      if(op->getNumRegions() == 0 || op->getRegion(0).empty())
        return false;
      return !op->getRegion(0).front().getOps<IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return cast<LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }
  
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();
    
    

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
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::DepthwiseConv2DNhwcQOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::DotOp declarations
//===----------------------------------------------------------------------===//

class DotOpAdaptor {
public:
  DotOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  DotOpAdaptor(DotOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class DotOp : public ::mlir::Op<DotOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::linalg::LinalgOp::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::linalg::ContractionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = DotOpAdaptor;
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
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.dot");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      Operation *op = this->getOperation();
      if(op->getNumRegions() == 0 || op->getRegion(0).empty())
        return false;
      return !op->getRegion(0).front().getOps<IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return cast<LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }
  
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    

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
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::DotOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::FillOp declarations
//===----------------------------------------------------------------------===//

class FillOpAdaptor {
public:
  FillOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  FillOpAdaptor(FillOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value value();
  ::mlir::Value output();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class FillOp : public ::mlir::Op<FillOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::linalg::LinalgOp::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = FillOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.fill");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value value();
  ::mlir::Value output();
  ::mlir::MutableOperandRange valueMutable();
  ::mlir::MutableOperandRange outputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value result();
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value value, Value output);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type result, ::mlir::Value value, ::mlir::Value output);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value output);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);

    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      Operation *op = this->getOperation();
      if(op->getNumRegions() == 0 || op->getRegion(0).empty())
        return false;
      return !op->getRegion(0).front().getOps<IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return cast<LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }
  
    std::string getLibraryCallName() {
      return generateLibraryCallName(getOperation());
    }
  
    ValueRange inputs() { return getOperands().take_front(); }
    ValueRange outputs() { return getOperands().take_back(); }

    // Rank-polymorphic.
    //   filling_value -> O(ivs) with parallel iterators.
    ArrayAttr iterator_types() {
      int64_t nPar = getRank(getOutputOperand(0));
      return Builder(getContext()).getStrArrayAttr(
        SmallVector<StringRef, 8>(nPar, getParallelIteratorTypeName()));
    }

    ArrayAttr indexing_maps() {
      MLIRContext *context = getContext();
      // filling_value -> O(ivs)
      return Builder(getContext()).getAffineMapArrayAttr({
          AffineMap::get(getNumParallelLoops(), 0, {}, getContext()),
          extractOrIdentityMap(llvm::None, getNumParallelLoops(), context)});
    }

    static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
    static std::function<void(ImplicitLocOpBuilder &b, Block &block)>
    getRegionBuilder() {
      return &regionBuilder;
    }
    static unsigned getNumRegionArgs() { return 2; }
  
};
} // namespace linalg
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::FillOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::FillRng2DOp declarations
//===----------------------------------------------------------------------===//

class FillRng2DOpAdaptor {
public:
  FillRng2DOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  FillRng2DOpAdaptor(FillRng2DOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class FillRng2DOp : public ::mlir::Op<FillRng2DOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::linalg::LinalgOp::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = FillRng2DOpAdaptor;
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
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.fill_rng_2d");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      Operation *op = this->getOperation();
      if(op->getNumRegions() == 0 || op->getRegion(0).empty())
        return false;
      return !op->getRegion(0).front().getOps<IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return cast<LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }
  
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    

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
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::FillRng2DOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::GenericOp declarations
//===----------------------------------------------------------------------===//

class GenericOpAdaptor {
public:
  GenericOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  GenericOpAdaptor(GenericOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr indexing_maps();
  ::mlir::ArrayAttr iterator_types();
  ::mlir::StringAttr doc();
  ::mlir::StringAttr library_call();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class GenericOp : public ::mlir::Op<GenericOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::linalg::LinalgOp::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GenericOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("iterator_types"), ::llvm::StringRef("doc"), ::llvm::StringRef("library_call"), ::llvm::StringRef("operand_segment_sizes")};
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
  ::mlir::Identifier docAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier docAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  ::mlir::Identifier library_callAttrName() {
    return getAttributeNameForIndex(3);
  }
  static ::mlir::Identifier library_callAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }
  ::mlir::Identifier operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(4);
  }
  static ::mlir::Identifier operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.generic");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  ::mlir::ArrayAttr indexing_mapsAttr();
  ::mlir::ArrayAttr indexing_maps();
  ::mlir::ArrayAttr iterator_typesAttr();
  ::mlir::ArrayAttr iterator_types();
  ::mlir::StringAttr docAttr();
  ::llvm::Optional< ::llvm::StringRef > doc();
  ::mlir::StringAttr library_callAttr();
  ::llvm::Optional< ::llvm::StringRef > library_call();
  void indexing_mapsAttr(::mlir::ArrayAttr attr);
  void iterator_typesAttr(::mlir::ArrayAttr attr);
  void docAttr(::mlir::StringAttr attr);
  void library_callAttr(::mlir::StringAttr attr);
  ::mlir::Attribute removeDocAttr();
  ::mlir::Attribute removeLibrary_callAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<AffineMap> indexingMaps, ArrayRef<StringRef> iteratorTypes, StringRef doc, StringRef libraryCall, function_ref<void(OpBuilder &, Location, ValueRange)> odsArg7 = nullptr, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputBuffers, ArrayRef<AffineMap> indexingMaps, ArrayRef<StringRef> iteratorTypes, StringRef doc, StringRef libraryCall, function_ref<void(OpBuilder &, Location, ValueRange)> odsArg6 = nullptr, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<AffineMap> indexingMaps, ArrayRef<StringRef> iteratorTypes, function_ref<void(OpBuilder &, Location, ValueRange)> odsArg5 = nullptr, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputBuffers, ArrayRef<AffineMap> indexingMaps, ArrayRef<StringRef> iteratorTypes, function_ref<void(OpBuilder &, Location, ValueRange)> odsArg4 = nullptr, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange result_tensors, ::mlir::ValueRange inputs, ::mlir::ValueRange outputs, ::mlir::ArrayAttr indexing_maps, ::mlir::ArrayAttr iterator_types, /*optional*/::mlir::StringAttr doc, /*optional*/::mlir::StringAttr library_call);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      Operation *op = this->getOperation();
      if(op->getNumRegions() == 0 || op->getRegion(0).empty())
        return false;
      return !op->getRegion(0).front().getOps<IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return cast<LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }
  
    SmallVector<StringRef, 8> linalgTraitAttrNames() {
      return SmallVector<StringRef, 8>{
        getDocAttrName(),
        getIndexingMapsAttrName(), getLibraryCallAttrName(),
        getIteratorTypesAttrName(),
      };
    }
    std::string getLibraryCallName() {
      return library_call().hasValue() ?
        library_call()->str() : "op_has_no_registered_library_name";
    }

    static std::function<void(ImplicitLocOpBuilder &b, Block &block)>
    getRegionBuilder() {
      return nullptr;
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
} // namespace linalg
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::GenericOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::MatmulI8I8I32Op declarations
//===----------------------------------------------------------------------===//

class MatmulI8I8I32OpAdaptor {
public:
  MatmulI8I8I32OpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  MatmulI8I8I32OpAdaptor(MatmulI8I8I32Op&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class MatmulI8I8I32Op : public ::mlir::Op<MatmulI8I8I32Op, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::linalg::LinalgOp::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::linalg::ContractionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = MatmulI8I8I32OpAdaptor;
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
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.matmul_i8_i8_i32");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      Operation *op = this->getOperation();
      if(op->getNumRegions() == 0 || op->getRegion(0).empty())
        return false;
      return !op->getRegion(0).front().getOps<IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return cast<LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }
  
        // Auto-generated.
        ArrayAttr iterator_types();
        ArrayAttr indexing_maps();
        static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
        static std::function<void(ImplicitLocOpBuilder &b, Block &)>
        getRegionBuilder() {
          return regionBuilder;
        }

        // Generic methods.
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
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
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::MatmulI8I8I32Op)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::MatmulOp declarations
//===----------------------------------------------------------------------===//

class MatmulOpAdaptor {
public:
  MatmulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  MatmulOpAdaptor(MatmulOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class MatmulOp : public ::mlir::Op<MatmulOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::linalg::LinalgOp::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::linalg::ContractionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = MatmulOpAdaptor;
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
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.matmul");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      Operation *op = this->getOperation();
      if(op->getNumRegions() == 0 || op->getRegion(0).empty())
        return false;
      return !op->getRegion(0).front().getOps<IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return cast<LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }
  
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    

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
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::MatmulOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::MatvecOp declarations
//===----------------------------------------------------------------------===//

class MatvecOpAdaptor {
public:
  MatvecOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  MatvecOpAdaptor(MatvecOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class MatvecOp : public ::mlir::Op<MatvecOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::linalg::LinalgOp::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::linalg::ContractionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = MatvecOpAdaptor;
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
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.matvec");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      Operation *op = this->getOperation();
      if(op->getNumRegions() == 0 || op->getRegion(0).empty())
        return false;
      return !op->getRegion(0).front().getOps<IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return cast<LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }
  
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    

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
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::MatvecOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Mmt4DOp declarations
//===----------------------------------------------------------------------===//

class Mmt4DOpAdaptor {
public:
  Mmt4DOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  Mmt4DOpAdaptor(Mmt4DOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class Mmt4DOp : public ::mlir::Op<Mmt4DOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::linalg::LinalgOp::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::linalg::ContractionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = Mmt4DOpAdaptor;
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
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.mmt4d");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      Operation *op = this->getOperation();
      if(op->getNumRegions() == 0 || op->getRegion(0).empty())
        return false;
      return !op->getRegion(0).front().getOps<IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return cast<LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }
  
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    

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
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::Mmt4DOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingMaxOp declarations
//===----------------------------------------------------------------------===//

class PoolingMaxOpAdaptor {
public:
  PoolingMaxOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  PoolingMaxOpAdaptor(PoolingMaxOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value windowDims();
  ::mlir::Value output();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr strides();
  ::mlir::ArrayAttr dilations();
  ::mlir::DenseIntElementsAttr padding();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class PoolingMaxOp : public ::mlir::Op<PoolingMaxOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::linalg::LinalgOp::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = PoolingMaxOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("strides"), ::llvm::StringRef("dilations"), ::llvm::StringRef("padding")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier stridesAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier dilationsAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier dilationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier paddingAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier paddingAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.pooling_max");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value windowDims();
  ::mlir::Value output();
  ::mlir::MutableOperandRange inputMutable();
  ::mlir::MutableOperandRange windowDimsMutable();
  ::mlir::MutableOperandRange outputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::ArrayAttr stridesAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > strides();
  ::mlir::ArrayAttr dilationsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > dilations();
  ::mlir::DenseIntElementsAttr paddingAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > padding();
  void stridesAttr(::mlir::ArrayAttr attr);
  void dilationsAttr(::mlir::ArrayAttr attr);
  void paddingAttr(::mlir::DenseIntElementsAttr attr);
  ::mlir::Attribute removeStridesAttr();
  ::mlir::Attribute removeDilationsAttr();
  ::mlir::Attribute removePaddingAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value windowDims, ::mlir::Value output, /*optional*/::mlir::ArrayAttr strides, /*optional*/::mlir::ArrayAttr dilations, /*optional*/::mlir::DenseIntElementsAttr padding);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value windowDims, ::mlir::Value output, /*optional*/::mlir::ArrayAttr strides, /*optional*/::mlir::ArrayAttr dilations, /*optional*/::mlir::DenseIntElementsAttr padding);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);

    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      Operation *op = this->getOperation();
      if(op->getNumRegions() == 0 || op->getRegion(0).empty())
        return false;
      return !op->getRegion(0).front().getOps<IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return cast<LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }
  
    std::string getLibraryCallName() {
      return generateLibraryCallName(getOperation());
    }
  
    int64_t getStride(unsigned i) {
      assert(i < getNumWindowLoops());
      if (!strides().hasValue()) return 1;
      return strides()->getValue()[i]
        .cast<IntegerAttr>().getValue().getSExtValue();
    }

    int64_t getDilation(unsigned i) {
      assert(i < getNumWindowLoops());
      if (!dilations().hasValue()) return 1;
      return dilations()->getValue()[i]
        .cast<IntegerAttr>().getValue().getSExtValue();
    }

    int64_t getLowPad(unsigned i) {
      assert(i < getNumWindowLoops());
      if (!padding().hasValue()) return 0;
      return padding().getValue().getValue<int64_t>({i, 0});
    }

    int64_t getHighPad(unsigned i) {
      assert(i < getNumWindowLoops());
      if (!padding().hasValue()) return 0;
      return padding().getValue().getValue<int64_t>({i, 1});
    }

    static std::function<void(ImplicitLocOpBuilder &b, Block &block)>
    getRegionBuilder() {
      return nullptr;
    }
  
    ValueRange inputs() { return getOperands().slice(0, 2); }
    ValueRange outputs() { return getOperands().take_back(); }

    ArrayAttr iterator_types() {
      // Outer parallel loops are always the number of output dimensions.
      int64_t nPar = getRank(getOutputOperand(0));
      // The window loops has the same number loops with output dimensions.
      unsigned nWin = nPar;
      SmallVector<StringRef, 8> iters(nPar, getParallelIteratorTypeName());
      iters.reserve(nPar + nWin);
      iters.append(nWin, getWindowIteratorTypeName());
      return Builder(getContext()).getStrArrayAttr(iters);
    }

    ArrayAttr indexing_maps() {
      MLIRContext *context = getContext();
      auto nPar = getNumParallelLoops();
      auto nWin = getNumWindowLoops();
      assert(nWin > 0 && "expected at least one window dimension");
      unsigned idx = 0;
      auto outputDims = makeAffineDimExprs(nPar, idx, context);
      auto windowDims = makeAffineDimExprs(nWin, idx, context);
      // Construct the weighedSum expression.
      auto inputDims =
          weightedPoolingInputIndex(*this, outputDims, windowDims);
      return Builder(getContext()).getAffineMapArrayAttr({
        // input
        AffineMap::get(idx, 0, inputDims, context),
        // windowDims
        AffineMap::get(idx, 0, windowDims, context),
        // output
        AffineMap::get(idx, 0, outputDims, context)});
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
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingMaxOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingMinOp declarations
//===----------------------------------------------------------------------===//

class PoolingMinOpAdaptor {
public:
  PoolingMinOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  PoolingMinOpAdaptor(PoolingMinOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value windowDims();
  ::mlir::Value output();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr strides();
  ::mlir::ArrayAttr dilations();
  ::mlir::DenseIntElementsAttr padding();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class PoolingMinOp : public ::mlir::Op<PoolingMinOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::linalg::LinalgOp::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = PoolingMinOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("strides"), ::llvm::StringRef("dilations"), ::llvm::StringRef("padding")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier stridesAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier dilationsAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier dilationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier paddingAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier paddingAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.pooling_min");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value windowDims();
  ::mlir::Value output();
  ::mlir::MutableOperandRange inputMutable();
  ::mlir::MutableOperandRange windowDimsMutable();
  ::mlir::MutableOperandRange outputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::ArrayAttr stridesAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > strides();
  ::mlir::ArrayAttr dilationsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > dilations();
  ::mlir::DenseIntElementsAttr paddingAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > padding();
  void stridesAttr(::mlir::ArrayAttr attr);
  void dilationsAttr(::mlir::ArrayAttr attr);
  void paddingAttr(::mlir::DenseIntElementsAttr attr);
  ::mlir::Attribute removeStridesAttr();
  ::mlir::Attribute removeDilationsAttr();
  ::mlir::Attribute removePaddingAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value windowDims, ::mlir::Value output, /*optional*/::mlir::ArrayAttr strides, /*optional*/::mlir::ArrayAttr dilations, /*optional*/::mlir::DenseIntElementsAttr padding);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value windowDims, ::mlir::Value output, /*optional*/::mlir::ArrayAttr strides, /*optional*/::mlir::ArrayAttr dilations, /*optional*/::mlir::DenseIntElementsAttr padding);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);

    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      Operation *op = this->getOperation();
      if(op->getNumRegions() == 0 || op->getRegion(0).empty())
        return false;
      return !op->getRegion(0).front().getOps<IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return cast<LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }
  
    std::string getLibraryCallName() {
      return generateLibraryCallName(getOperation());
    }
  
    int64_t getStride(unsigned i) {
      assert(i < getNumWindowLoops());
      if (!strides().hasValue()) return 1;
      return strides()->getValue()[i]
        .cast<IntegerAttr>().getValue().getSExtValue();
    }

    int64_t getDilation(unsigned i) {
      assert(i < getNumWindowLoops());
      if (!dilations().hasValue()) return 1;
      return dilations()->getValue()[i]
        .cast<IntegerAttr>().getValue().getSExtValue();
    }

    int64_t getLowPad(unsigned i) {
      assert(i < getNumWindowLoops());
      if (!padding().hasValue()) return 0;
      return padding().getValue().getValue<int64_t>({i, 0});
    }

    int64_t getHighPad(unsigned i) {
      assert(i < getNumWindowLoops());
      if (!padding().hasValue()) return 0;
      return padding().getValue().getValue<int64_t>({i, 1});
    }

    static std::function<void(ImplicitLocOpBuilder &b, Block &block)>
    getRegionBuilder() {
      return nullptr;
    }
  
    ValueRange inputs() { return getOperands().slice(0, 2); }
    ValueRange outputs() { return getOperands().take_back(); }

    ArrayAttr iterator_types() {
      // Outer parallel loops are always the number of output dimensions.
      int64_t nPar = getRank(getOutputOperand(0));
      // The window loops has the same number loops with output dimensions.
      unsigned nWin = nPar;
      SmallVector<StringRef, 8> iters(nPar, getParallelIteratorTypeName());
      iters.reserve(nPar + nWin);
      iters.append(nWin, getWindowIteratorTypeName());
      return Builder(getContext()).getStrArrayAttr(iters);
    }

    ArrayAttr indexing_maps() {
      MLIRContext *context = getContext();
      auto nPar = getNumParallelLoops();
      auto nWin = getNumWindowLoops();
      assert(nWin > 0 && "expected at least one window dimension");
      unsigned idx = 0;
      auto outputDims = makeAffineDimExprs(nPar, idx, context);
      auto windowDims = makeAffineDimExprs(nWin, idx, context);
      // Construct the weighedSum expression.
      auto inputDims =
          weightedPoolingInputIndex(*this, outputDims, windowDims);
      return Builder(getContext()).getAffineMapArrayAttr({
        // input
        AffineMap::get(idx, 0, inputDims, context),
        // windowDims
        AffineMap::get(idx, 0, windowDims, context),
        // output
        AffineMap::get(idx, 0, outputDims, context)});
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
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingMinOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNchwMaxOp declarations
//===----------------------------------------------------------------------===//

class PoolingNchwMaxOpAdaptor {
public:
  PoolingNchwMaxOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  PoolingNchwMaxOpAdaptor(PoolingNchwMaxOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilations();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class PoolingNchwMaxOp : public ::mlir::Op<PoolingNchwMaxOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::linalg::LinalgOp::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::linalg::ConvolutionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = PoolingNchwMaxOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("strides"), ::llvm::StringRef("dilations"), ::llvm::StringRef("operand_segment_sizes")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier stridesAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier dilationsAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier dilationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.pooling_nchw_max");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  void stridesAttr(::mlir::DenseIntElementsAttr attr);
  void dilationsAttr(::mlir::DenseIntElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      Operation *op = this->getOperation();
      if(op->getNumRegions() == 0 || op->getRegion(0).empty())
        return false;
      return !op->getRegion(0).front().getOps<IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return cast<LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }
  
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();
    
    

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
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNchwMaxOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNdhwcMaxOp declarations
//===----------------------------------------------------------------------===//

class PoolingNdhwcMaxOpAdaptor {
public:
  PoolingNdhwcMaxOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  PoolingNdhwcMaxOpAdaptor(PoolingNdhwcMaxOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilations();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class PoolingNdhwcMaxOp : public ::mlir::Op<PoolingNdhwcMaxOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::linalg::LinalgOp::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::linalg::ConvolutionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = PoolingNdhwcMaxOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("strides"), ::llvm::StringRef("dilations"), ::llvm::StringRef("operand_segment_sizes")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier stridesAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier dilationsAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier dilationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.pooling_ndhwc_max");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  void stridesAttr(::mlir::DenseIntElementsAttr attr);
  void dilationsAttr(::mlir::DenseIntElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      Operation *op = this->getOperation();
      if(op->getNumRegions() == 0 || op->getRegion(0).empty())
        return false;
      return !op->getRegion(0).front().getOps<IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return cast<LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }
  
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();
    
    

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
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNdhwcMaxOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNdhwcMinOp declarations
//===----------------------------------------------------------------------===//

class PoolingNdhwcMinOpAdaptor {
public:
  PoolingNdhwcMinOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  PoolingNdhwcMinOpAdaptor(PoolingNdhwcMinOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilations();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class PoolingNdhwcMinOp : public ::mlir::Op<PoolingNdhwcMinOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::linalg::LinalgOp::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::linalg::ConvolutionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = PoolingNdhwcMinOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("strides"), ::llvm::StringRef("dilations"), ::llvm::StringRef("operand_segment_sizes")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier stridesAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier dilationsAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier dilationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.pooling_ndhwc_min");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  void stridesAttr(::mlir::DenseIntElementsAttr attr);
  void dilationsAttr(::mlir::DenseIntElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      Operation *op = this->getOperation();
      if(op->getNumRegions() == 0 || op->getRegion(0).empty())
        return false;
      return !op->getRegion(0).front().getOps<IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return cast<LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }
  
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();
    
    

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
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNdhwcMinOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNdhwcSumOp declarations
//===----------------------------------------------------------------------===//

class PoolingNdhwcSumOpAdaptor {
public:
  PoolingNdhwcSumOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  PoolingNdhwcSumOpAdaptor(PoolingNdhwcSumOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilations();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class PoolingNdhwcSumOp : public ::mlir::Op<PoolingNdhwcSumOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::linalg::LinalgOp::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::linalg::ConvolutionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = PoolingNdhwcSumOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("strides"), ::llvm::StringRef("dilations"), ::llvm::StringRef("operand_segment_sizes")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier stridesAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier dilationsAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier dilationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.pooling_ndhwc_sum");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  void stridesAttr(::mlir::DenseIntElementsAttr attr);
  void dilationsAttr(::mlir::DenseIntElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      Operation *op = this->getOperation();
      if(op->getNumRegions() == 0 || op->getRegion(0).empty())
        return false;
      return !op->getRegion(0).front().getOps<IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return cast<LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }
  
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();
    
    

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
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNdhwcSumOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNhwcMaxOp declarations
//===----------------------------------------------------------------------===//

class PoolingNhwcMaxOpAdaptor {
public:
  PoolingNhwcMaxOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  PoolingNhwcMaxOpAdaptor(PoolingNhwcMaxOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilations();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class PoolingNhwcMaxOp : public ::mlir::Op<PoolingNhwcMaxOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::linalg::LinalgOp::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::linalg::ConvolutionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = PoolingNhwcMaxOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("strides"), ::llvm::StringRef("dilations"), ::llvm::StringRef("operand_segment_sizes")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier stridesAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier dilationsAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier dilationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.pooling_nhwc_max");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  void stridesAttr(::mlir::DenseIntElementsAttr attr);
  void dilationsAttr(::mlir::DenseIntElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      Operation *op = this->getOperation();
      if(op->getNumRegions() == 0 || op->getRegion(0).empty())
        return false;
      return !op->getRegion(0).front().getOps<IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return cast<LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }
  
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();
    
    

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
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNhwcMaxOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNhwcMinOp declarations
//===----------------------------------------------------------------------===//

class PoolingNhwcMinOpAdaptor {
public:
  PoolingNhwcMinOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  PoolingNhwcMinOpAdaptor(PoolingNhwcMinOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilations();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class PoolingNhwcMinOp : public ::mlir::Op<PoolingNhwcMinOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::linalg::LinalgOp::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::linalg::ConvolutionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = PoolingNhwcMinOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("strides"), ::llvm::StringRef("dilations"), ::llvm::StringRef("operand_segment_sizes")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier stridesAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier dilationsAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier dilationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.pooling_nhwc_min");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  void stridesAttr(::mlir::DenseIntElementsAttr attr);
  void dilationsAttr(::mlir::DenseIntElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      Operation *op = this->getOperation();
      if(op->getNumRegions() == 0 || op->getRegion(0).empty())
        return false;
      return !op->getRegion(0).front().getOps<IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return cast<LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }
  
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();
    
    

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
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNhwcMinOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNhwcSumOp declarations
//===----------------------------------------------------------------------===//

class PoolingNhwcSumOpAdaptor {
public:
  PoolingNhwcSumOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  PoolingNhwcSumOpAdaptor(PoolingNhwcSumOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilations();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class PoolingNhwcSumOp : public ::mlir::Op<PoolingNhwcSumOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::linalg::LinalgOp::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::linalg::ConvolutionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = PoolingNhwcSumOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("strides"), ::llvm::StringRef("dilations"), ::llvm::StringRef("operand_segment_sizes")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier stridesAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier dilationsAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier dilationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.pooling_nhwc_sum");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  void stridesAttr(::mlir::DenseIntElementsAttr attr);
  void dilationsAttr(::mlir::DenseIntElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      Operation *op = this->getOperation();
      if(op->getNumRegions() == 0 || op->getRegion(0).empty())
        return false;
      return !op->getRegion(0).front().getOps<IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return cast<LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }
  
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();
    
    

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
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNhwcSumOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingSumOp declarations
//===----------------------------------------------------------------------===//

class PoolingSumOpAdaptor {
public:
  PoolingSumOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  PoolingSumOpAdaptor(PoolingSumOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value windowDims();
  ::mlir::Value output();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr strides();
  ::mlir::ArrayAttr dilations();
  ::mlir::DenseIntElementsAttr padding();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class PoolingSumOp : public ::mlir::Op<PoolingSumOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::linalg::LinalgOp::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = PoolingSumOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("strides"), ::llvm::StringRef("dilations"), ::llvm::StringRef("padding")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier stridesAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier dilationsAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier dilationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier paddingAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier paddingAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.pooling_sum");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value windowDims();
  ::mlir::Value output();
  ::mlir::MutableOperandRange inputMutable();
  ::mlir::MutableOperandRange windowDimsMutable();
  ::mlir::MutableOperandRange outputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::ArrayAttr stridesAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > strides();
  ::mlir::ArrayAttr dilationsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > dilations();
  ::mlir::DenseIntElementsAttr paddingAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > padding();
  void stridesAttr(::mlir::ArrayAttr attr);
  void dilationsAttr(::mlir::ArrayAttr attr);
  void paddingAttr(::mlir::DenseIntElementsAttr attr);
  ::mlir::Attribute removeStridesAttr();
  ::mlir::Attribute removeDilationsAttr();
  ::mlir::Attribute removePaddingAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value windowDims, ::mlir::Value output, /*optional*/::mlir::ArrayAttr strides, /*optional*/::mlir::ArrayAttr dilations, /*optional*/::mlir::DenseIntElementsAttr padding);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value windowDims, ::mlir::Value output, /*optional*/::mlir::ArrayAttr strides, /*optional*/::mlir::ArrayAttr dilations, /*optional*/::mlir::DenseIntElementsAttr padding);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);

    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      Operation *op = this->getOperation();
      if(op->getNumRegions() == 0 || op->getRegion(0).empty())
        return false;
      return !op->getRegion(0).front().getOps<IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return cast<LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }
  
    std::string getLibraryCallName() {
      return generateLibraryCallName(getOperation());
    }
  
    int64_t getStride(unsigned i) {
      assert(i < getNumWindowLoops());
      if (!strides().hasValue()) return 1;
      return strides()->getValue()[i]
        .cast<IntegerAttr>().getValue().getSExtValue();
    }

    int64_t getDilation(unsigned i) {
      assert(i < getNumWindowLoops());
      if (!dilations().hasValue()) return 1;
      return dilations()->getValue()[i]
        .cast<IntegerAttr>().getValue().getSExtValue();
    }

    int64_t getLowPad(unsigned i) {
      assert(i < getNumWindowLoops());
      if (!padding().hasValue()) return 0;
      return padding().getValue().getValue<int64_t>({i, 0});
    }

    int64_t getHighPad(unsigned i) {
      assert(i < getNumWindowLoops());
      if (!padding().hasValue()) return 0;
      return padding().getValue().getValue<int64_t>({i, 1});
    }

    static std::function<void(ImplicitLocOpBuilder &b, Block &block)>
    getRegionBuilder() {
      return nullptr;
    }
  
    ValueRange inputs() { return getOperands().slice(0, 2); }
    ValueRange outputs() { return getOperands().take_back(); }

    ArrayAttr iterator_types() {
      // Outer parallel loops are always the number of output dimensions.
      int64_t nPar = getRank(getOutputOperand(0));
      // The window loops has the same number loops with output dimensions.
      unsigned nWin = nPar;
      SmallVector<StringRef, 8> iters(nPar, getParallelIteratorTypeName());
      iters.reserve(nPar + nWin);
      iters.append(nWin, getWindowIteratorTypeName());
      return Builder(getContext()).getStrArrayAttr(iters);
    }

    ArrayAttr indexing_maps() {
      MLIRContext *context = getContext();
      auto nPar = getNumParallelLoops();
      auto nWin = getNumWindowLoops();
      assert(nWin > 0 && "expected at least one window dimension");
      unsigned idx = 0;
      auto outputDims = makeAffineDimExprs(nPar, idx, context);
      auto windowDims = makeAffineDimExprs(nWin, idx, context);
      // Construct the weighedSum expression.
      auto inputDims =
          weightedPoolingInputIndex(*this, outputDims, windowDims);
      return Builder(getContext()).getAffineMapArrayAttr({
        // input
        AffineMap::get(idx, 0, inputDims, context),
        // windowDims
        AffineMap::get(idx, 0, windowDims, context),
        // output
        AffineMap::get(idx, 0, outputDims, context)});
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
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingSumOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::QuantizedBatchMatmulOp declarations
//===----------------------------------------------------------------------===//

class QuantizedBatchMatmulOpAdaptor {
public:
  QuantizedBatchMatmulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  QuantizedBatchMatmulOpAdaptor(QuantizedBatchMatmulOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class QuantizedBatchMatmulOp : public ::mlir::Op<QuantizedBatchMatmulOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::linalg::LinalgOp::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = QuantizedBatchMatmulOpAdaptor;
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
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.quantized_batch_matmul");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      Operation *op = this->getOperation();
      if(op->getNumRegions() == 0 || op->getRegion(0).empty())
        return false;
      return !op->getRegion(0).front().getOps<IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return cast<LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }
  
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    

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
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::QuantizedBatchMatmulOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::QuantizedMatmulOp declarations
//===----------------------------------------------------------------------===//

class QuantizedMatmulOpAdaptor {
public:
  QuantizedMatmulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  QuantizedMatmulOpAdaptor(QuantizedMatmulOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class QuantizedMatmulOp : public ::mlir::Op<QuantizedMatmulOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::linalg::LinalgOp::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = QuantizedMatmulOpAdaptor;
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
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.quantized_matmul");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      Operation *op = this->getOperation();
      if(op->getNumRegions() == 0 || op->getRegion(0).empty())
        return false;
      return !op->getRegion(0).front().getOps<IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return cast<LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }
  
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    

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
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::QuantizedMatmulOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::SoftPlus2DOp declarations
//===----------------------------------------------------------------------===//

class SoftPlus2DOpAdaptor {
public:
  SoftPlus2DOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  SoftPlus2DOpAdaptor(SoftPlus2DOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class SoftPlus2DOp : public ::mlir::Op<SoftPlus2DOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::linalg::LinalgOp::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SoftPlus2DOpAdaptor;
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
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.soft_plus_2d");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      Operation *op = this->getOperation();
      if(op->getNumRegions() == 0 || op->getRegion(0).empty())
        return false;
      return !op->getRegion(0).front().getOps<IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return cast<LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }
  
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    

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
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::SoftPlus2DOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::VecmatOp declarations
//===----------------------------------------------------------------------===//

class VecmatOpAdaptor {
public:
  VecmatOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  VecmatOpAdaptor(VecmatOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class VecmatOp : public ::mlir::Op<VecmatOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::linalg::LinalgOp::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::linalg::ContractionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = VecmatOpAdaptor;
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
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.vecmat");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      Operation *op = this->getOperation();
      if(op->getNumRegions() == 0 || op->getRegion(0).empty())
        return false;
      return !op->getRegion(0).front().getOps<IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return cast<LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }
  
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    

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
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::VecmatOp)


#endif  // GET_OP_CLASSES

