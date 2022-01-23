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
namespace lmhlo_gpu {
class AllReduceDoneOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class AllReduceStartOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class BatchNormGradOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class BatchNormInferenceOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class BatchNormTrainingOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class CholeskyOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class ConvBackwardFilterOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class ConvBackwardInputOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class ConvForwardFusedOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class ConvForwardFusedSideInputOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class ConvForwardOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class GEMMOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class GEMM_BiasOp;
} // namespace lmhlo_gpu
} // namespace mlir
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::AllReduceDoneOp declarations
//===----------------------------------------------------------------------===//

class AllReduceDoneOpAdaptor {
public:
  AllReduceDoneOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  AllReduceDoneOpAdaptor(AllReduceDoneOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value token();
  ::mlir::ValueRange operands();
  ::mlir::ValueRange results();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class AllReduceDoneOp : public ::mlir::Op<AllReduceDoneOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AllReduceDoneOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.all_reduce_done");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value token();
  ::mlir::Operation::operand_range operands();
  ::mlir::Operation::operand_range results();
  ::mlir::MutableOperandRange tokenMutable();
  ::mlir::MutableOperandRange operandsMutable();
  ::mlir::MutableOperandRange resultsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value token, ::mlir::ValueRange operands, ::mlir::ValueRange results);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value token, ::mlir::ValueRange operands, ::mlir::ValueRange results);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
};
} // namespace lmhlo_gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::AllReduceDoneOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::AllReduceStartOp declarations
//===----------------------------------------------------------------------===//

class AllReduceStartOpAdaptor {
public:
  AllReduceStartOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  AllReduceStartOpAdaptor(AllReduceStartOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange operands();
  ::mlir::ValueRange results();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr replica_groups();
  ::mlir::BoolAttr constrain_layout();
  ::mlir::mhlo::ChannelHandle channel_id();
  ::mlir::BoolAttr use_global_device_ids();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &computation();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class AllReduceStartOp : public ::mlir::Op<AllReduceStartOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SameOperandsElementType> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AllReduceStartOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("replica_groups"), ::llvm::StringRef("constrain_layout"), ::llvm::StringRef("channel_id"), ::llvm::StringRef("use_global_device_ids")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier replica_groupsAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier replica_groupsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier constrain_layoutAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier constrain_layoutAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier channel_idAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier channel_idAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  ::mlir::Identifier use_global_device_idsAttrName() {
    return getAttributeNameForIndex(3);
  }
  static ::mlir::Identifier use_global_device_idsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.all_reduce_start");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range operands();
  ::mlir::Operation::operand_range results();
  ::mlir::MutableOperandRange operandsMutable();
  ::mlir::MutableOperandRange resultsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value token();
  ::mlir::Region &computation();
  ::mlir::DenseIntElementsAttr replica_groupsAttr();
  ::mlir::DenseIntElementsAttr replica_groups();
  ::mlir::BoolAttr constrain_layoutAttr();
  bool constrain_layout();
  ::mlir::mhlo::ChannelHandle channel_idAttr();
  ::llvm::Optional<::mlir::mhlo::ChannelHandle> channel_id();
  ::mlir::BoolAttr use_global_device_idsAttr();
  bool use_global_device_ids();
  void replica_groupsAttr(::mlir::DenseIntElementsAttr attr);
  void constrain_layoutAttr(::mlir::BoolAttr attr);
  void channel_idAttr(::mlir::mhlo::ChannelHandle attr);
  void use_global_device_idsAttr(::mlir::BoolAttr attr);
  ::mlir::Attribute removeChannel_idAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type token, ::mlir::ValueRange operands, ::mlir::ValueRange results, ::mlir::DenseIntElementsAttr replica_groups, ::mlir::BoolAttr constrain_layout, /*optional*/::mlir::mhlo::ChannelHandle channel_id, ::mlir::BoolAttr use_global_device_ids);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::mlir::ValueRange results, ::mlir::DenseIntElementsAttr replica_groups, ::mlir::BoolAttr constrain_layout, /*optional*/::mlir::mhlo::ChannelHandle channel_id, ::mlir::BoolAttr use_global_device_ids);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type token, ::mlir::ValueRange operands, ::mlir::ValueRange results, ::mlir::DenseIntElementsAttr replica_groups, bool constrain_layout, /*optional*/::mlir::mhlo::ChannelHandle channel_id, bool use_global_device_ids = false);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::mlir::ValueRange results, ::mlir::DenseIntElementsAttr replica_groups, bool constrain_layout, /*optional*/::mlir::mhlo::ChannelHandle channel_id, bool use_global_device_ids = false);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 4 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace lmhlo_gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::AllReduceStartOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::BatchNormGradOp declarations
//===----------------------------------------------------------------------===//

class BatchNormGradOpAdaptor {
public:
  BatchNormGradOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  BatchNormGradOpAdaptor(BatchNormGradOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value operand();
  ::mlir::Value scale();
  ::mlir::Value mean();
  ::mlir::Value stddev();
  ::mlir::Value grad_output();
  ::mlir::Value grad_operand();
  ::mlir::Value grad_scale();
  ::mlir::Value grad_offset();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::FloatAttr epsilon();
  ::mlir::IntegerAttr feature_index();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class BatchNormGradOp : public ::mlir::Op<BatchNormGradOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<8>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = BatchNormGradOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("epsilon"), ::llvm::StringRef("feature_index")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier epsilonAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier epsilonAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier feature_indexAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier feature_indexAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.batch_norm_grad");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value operand();
  ::mlir::Value scale();
  ::mlir::Value mean();
  ::mlir::Value stddev();
  ::mlir::Value grad_output();
  ::mlir::Value grad_operand();
  ::mlir::Value grad_scale();
  ::mlir::Value grad_offset();
  ::mlir::MutableOperandRange operandMutable();
  ::mlir::MutableOperandRange scaleMutable();
  ::mlir::MutableOperandRange meanMutable();
  ::mlir::MutableOperandRange stddevMutable();
  ::mlir::MutableOperandRange grad_outputMutable();
  ::mlir::MutableOperandRange grad_operandMutable();
  ::mlir::MutableOperandRange grad_scaleMutable();
  ::mlir::MutableOperandRange grad_offsetMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::FloatAttr epsilonAttr();
  ::llvm::APFloat epsilon();
  ::mlir::IntegerAttr feature_indexAttr();
  uint64_t feature_index();
  void epsilonAttr(::mlir::FloatAttr attr);
  void feature_indexAttr(::mlir::IntegerAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand, ::mlir::Value scale, ::mlir::Value mean, ::mlir::Value stddev, ::mlir::Value grad_output, ::mlir::Value grad_operand, ::mlir::Value grad_scale, ::mlir::Value grad_offset, ::mlir::FloatAttr epsilon, ::mlir::IntegerAttr feature_index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand, ::mlir::Value scale, ::mlir::Value mean, ::mlir::Value stddev, ::mlir::Value grad_output, ::mlir::Value grad_operand, ::mlir::Value grad_scale, ::mlir::Value grad_offset, ::mlir::FloatAttr epsilon, ::mlir::IntegerAttr feature_index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand, ::mlir::Value scale, ::mlir::Value mean, ::mlir::Value stddev, ::mlir::Value grad_output, ::mlir::Value grad_operand, ::mlir::Value grad_scale, ::mlir::Value grad_offset, ::llvm::APFloat epsilon, uint64_t feature_index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand, ::mlir::Value scale, ::mlir::Value mean, ::mlir::Value stddev, ::mlir::Value grad_output, ::mlir::Value grad_operand, ::mlir::Value grad_scale, ::mlir::Value grad_offset, ::llvm::APFloat epsilon, uint64_t feature_index);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
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
} // namespace lmhlo_gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::BatchNormGradOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::BatchNormInferenceOp declarations
//===----------------------------------------------------------------------===//

class BatchNormInferenceOpAdaptor {
public:
  BatchNormInferenceOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  BatchNormInferenceOpAdaptor(BatchNormInferenceOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value operand();
  ::mlir::Value scale();
  ::mlir::Value offset();
  ::mlir::Value mean();
  ::mlir::Value stddev();
  ::mlir::Value output();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::FloatAttr epsilon();
  ::mlir::IntegerAttr feature_index();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class BatchNormInferenceOp : public ::mlir::Op<BatchNormInferenceOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<6>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = BatchNormInferenceOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("epsilon"), ::llvm::StringRef("feature_index")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier epsilonAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier epsilonAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier feature_indexAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier feature_indexAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.batch_norm_inference");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value operand();
  ::mlir::Value scale();
  ::mlir::Value offset();
  ::mlir::Value mean();
  ::mlir::Value stddev();
  ::mlir::Value output();
  ::mlir::MutableOperandRange operandMutable();
  ::mlir::MutableOperandRange scaleMutable();
  ::mlir::MutableOperandRange offsetMutable();
  ::mlir::MutableOperandRange meanMutable();
  ::mlir::MutableOperandRange stddevMutable();
  ::mlir::MutableOperandRange outputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::FloatAttr epsilonAttr();
  ::llvm::APFloat epsilon();
  ::mlir::IntegerAttr feature_indexAttr();
  uint64_t feature_index();
  void epsilonAttr(::mlir::FloatAttr attr);
  void feature_indexAttr(::mlir::IntegerAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand, ::mlir::Value scale, ::mlir::Value offset, ::mlir::Value mean, ::mlir::Value stddev, ::mlir::Value output, ::mlir::FloatAttr epsilon, ::mlir::IntegerAttr feature_index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand, ::mlir::Value scale, ::mlir::Value offset, ::mlir::Value mean, ::mlir::Value stddev, ::mlir::Value output, ::mlir::FloatAttr epsilon, ::mlir::IntegerAttr feature_index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand, ::mlir::Value scale, ::mlir::Value offset, ::mlir::Value mean, ::mlir::Value stddev, ::mlir::Value output, ::llvm::APFloat epsilon, uint64_t feature_index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand, ::mlir::Value scale, ::mlir::Value offset, ::mlir::Value mean, ::mlir::Value stddev, ::mlir::Value output, ::llvm::APFloat epsilon, uint64_t feature_index);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
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
} // namespace lmhlo_gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::BatchNormInferenceOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::BatchNormTrainingOp declarations
//===----------------------------------------------------------------------===//

class BatchNormTrainingOpAdaptor {
public:
  BatchNormTrainingOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  BatchNormTrainingOpAdaptor(BatchNormTrainingOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value operand();
  ::mlir::Value scale();
  ::mlir::Value offset();
  ::mlir::Value output();
  ::mlir::Value batch_mean();
  ::mlir::Value batch_stddev();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::FloatAttr epsilon();
  ::mlir::IntegerAttr feature_index();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class BatchNormTrainingOp : public ::mlir::Op<BatchNormTrainingOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<6>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = BatchNormTrainingOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("epsilon"), ::llvm::StringRef("feature_index")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier epsilonAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier epsilonAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier feature_indexAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier feature_indexAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.batch_norm_training");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value operand();
  ::mlir::Value scale();
  ::mlir::Value offset();
  ::mlir::Value output();
  ::mlir::Value batch_mean();
  ::mlir::Value batch_stddev();
  ::mlir::MutableOperandRange operandMutable();
  ::mlir::MutableOperandRange scaleMutable();
  ::mlir::MutableOperandRange offsetMutable();
  ::mlir::MutableOperandRange outputMutable();
  ::mlir::MutableOperandRange batch_meanMutable();
  ::mlir::MutableOperandRange batch_stddevMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::FloatAttr epsilonAttr();
  ::llvm::APFloat epsilon();
  ::mlir::IntegerAttr feature_indexAttr();
  uint64_t feature_index();
  void epsilonAttr(::mlir::FloatAttr attr);
  void feature_indexAttr(::mlir::IntegerAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand, ::mlir::Value scale, ::mlir::Value offset, ::mlir::Value output, ::mlir::Value batch_mean, ::mlir::Value batch_stddev, ::mlir::FloatAttr epsilon, ::mlir::IntegerAttr feature_index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand, ::mlir::Value scale, ::mlir::Value offset, ::mlir::Value output, ::mlir::Value batch_mean, ::mlir::Value batch_stddev, ::mlir::FloatAttr epsilon, ::mlir::IntegerAttr feature_index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand, ::mlir::Value scale, ::mlir::Value offset, ::mlir::Value output, ::mlir::Value batch_mean, ::mlir::Value batch_stddev, ::llvm::APFloat epsilon, uint64_t feature_index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand, ::mlir::Value scale, ::mlir::Value offset, ::mlir::Value output, ::mlir::Value batch_mean, ::mlir::Value batch_stddev, ::llvm::APFloat epsilon, uint64_t feature_index);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
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
} // namespace lmhlo_gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::BatchNormTrainingOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::CholeskyOp declarations
//===----------------------------------------------------------------------===//

class CholeskyOpAdaptor {
public:
  CholeskyOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  CholeskyOpAdaptor(CholeskyOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value output();
  ::mlir::Value scratch();
  ::mlir::Value info();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::BoolAttr is_lower();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class CholeskyOp : public ::mlir::Op<CholeskyOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<4>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CholeskyOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("is_lower")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier is_lowerAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier is_lowerAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.cholesky");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value output();
  ::mlir::Value scratch();
  ::mlir::Value info();
  ::mlir::MutableOperandRange inputMutable();
  ::mlir::MutableOperandRange outputMutable();
  ::mlir::MutableOperandRange scratchMutable();
  ::mlir::MutableOperandRange infoMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::BoolAttr is_lowerAttr();
  bool is_lower();
  void is_lowerAttr(::mlir::BoolAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value output, ::mlir::Value scratch, ::mlir::Value info, ::mlir::BoolAttr is_lower);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value output, ::mlir::Value scratch, ::mlir::Value info, ::mlir::BoolAttr is_lower);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value output, ::mlir::Value scratch, ::mlir::Value info, bool is_lower);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value output, ::mlir::Value scratch, ::mlir::Value info, bool is_lower);
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
} // namespace lmhlo_gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::CholeskyOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::ConvBackwardFilterOp declarations
//===----------------------------------------------------------------------===//

class ConvBackwardFilterOpAdaptor {
public:
  ConvBackwardFilterOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ConvBackwardFilterOpAdaptor(ConvBackwardFilterOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value d_output();
  ::mlir::Value d_filter();
  ::mlir::Value scratch();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr window_strides();
  ::mlir::DenseIntElementsAttr padding();
  ::mlir::DenseIntElementsAttr lhs_dilation();
  ::mlir::DenseIntElementsAttr rhs_dilation();
  ::mlir::DenseElementsAttr window_reversal();
  ::mlir::mhlo::ConvDimensionNumbers dimension_numbers();
  ::mlir::IntegerAttr feature_group_count();
  ::mlir::IntegerAttr batch_group_count();
  ::mlir::ArrayAttr precision_config();
  ::mlir::FloatAttr result_scale();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ConvBackwardFilterOp : public ::mlir::Op<ConvBackwardFilterOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<4>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConvBackwardFilterOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("window_strides"), ::llvm::StringRef("padding"), ::llvm::StringRef("lhs_dilation"), ::llvm::StringRef("rhs_dilation"), ::llvm::StringRef("window_reversal"), ::llvm::StringRef("dimension_numbers"), ::llvm::StringRef("feature_group_count"), ::llvm::StringRef("batch_group_count"), ::llvm::StringRef("precision_config"), ::llvm::StringRef("result_scale"), ::llvm::StringRef("backend_config")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier window_stridesAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier window_stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier paddingAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier paddingAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier lhs_dilationAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier lhs_dilationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  ::mlir::Identifier rhs_dilationAttrName() {
    return getAttributeNameForIndex(3);
  }
  static ::mlir::Identifier rhs_dilationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }
  ::mlir::Identifier window_reversalAttrName() {
    return getAttributeNameForIndex(4);
  }
  static ::mlir::Identifier window_reversalAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }
  ::mlir::Identifier dimension_numbersAttrName() {
    return getAttributeNameForIndex(5);
  }
  static ::mlir::Identifier dimension_numbersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }
  ::mlir::Identifier feature_group_countAttrName() {
    return getAttributeNameForIndex(6);
  }
  static ::mlir::Identifier feature_group_countAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }
  ::mlir::Identifier batch_group_countAttrName() {
    return getAttributeNameForIndex(7);
  }
  static ::mlir::Identifier batch_group_countAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 7);
  }
  ::mlir::Identifier precision_configAttrName() {
    return getAttributeNameForIndex(8);
  }
  static ::mlir::Identifier precision_configAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 8);
  }
  ::mlir::Identifier result_scaleAttrName() {
    return getAttributeNameForIndex(9);
  }
  static ::mlir::Identifier result_scaleAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 9);
  }
  ::mlir::Identifier backend_configAttrName() {
    return getAttributeNameForIndex(10);
  }
  static ::mlir::Identifier backend_configAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 10);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.conv_backwardfilter");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value d_output();
  ::mlir::Value d_filter();
  ::mlir::Value scratch();
  ::mlir::MutableOperandRange inputMutable();
  ::mlir::MutableOperandRange d_outputMutable();
  ::mlir::MutableOperandRange d_filterMutable();
  ::mlir::MutableOperandRange scratchMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::DenseIntElementsAttr window_stridesAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > window_strides();
  ::mlir::DenseIntElementsAttr paddingAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > padding();
  ::mlir::DenseIntElementsAttr lhs_dilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > lhs_dilation();
  ::mlir::DenseIntElementsAttr rhs_dilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > rhs_dilation();
  ::mlir::DenseElementsAttr window_reversalAttr();
  ::llvm::Optional< ::mlir::DenseElementsAttr > window_reversal();
  ::mlir::mhlo::ConvDimensionNumbers dimension_numbersAttr();
  ::mlir::mhlo::ConvDimensionNumbers dimension_numbers();
  ::mlir::IntegerAttr feature_group_countAttr();
  uint64_t feature_group_count();
  ::mlir::IntegerAttr batch_group_countAttr();
  uint64_t batch_group_count();
  ::mlir::ArrayAttr precision_configAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > precision_config();
  ::mlir::FloatAttr result_scaleAttr();
  ::llvm::APFloat result_scale();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_configAttr();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config();
  void window_stridesAttr(::mlir::DenseIntElementsAttr attr);
  void paddingAttr(::mlir::DenseIntElementsAttr attr);
  void lhs_dilationAttr(::mlir::DenseIntElementsAttr attr);
  void rhs_dilationAttr(::mlir::DenseIntElementsAttr attr);
  void window_reversalAttr(::mlir::DenseElementsAttr attr);
  void dimension_numbersAttr(::mlir::mhlo::ConvDimensionNumbers attr);
  void feature_group_countAttr(::mlir::IntegerAttr attr);
  void batch_group_countAttr(::mlir::IntegerAttr attr);
  void precision_configAttr(::mlir::ArrayAttr attr);
  void result_scaleAttr(::mlir::FloatAttr attr);
  void backend_configAttr(::mlir::lmhlo_gpu::ConvolutionBackendConfig attr);
  ::mlir::Attribute removeWindow_stridesAttr();
  ::mlir::Attribute removePaddingAttr();
  ::mlir::Attribute removeLhs_dilationAttr();
  ::mlir::Attribute removeRhs_dilationAttr();
  ::mlir::Attribute removeWindow_reversalAttr();
  ::mlir::Attribute removePrecision_configAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value d_output, ::mlir::Value d_filter, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value d_output, ::mlir::Value d_filter, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value d_output, ::mlir::Value d_filter, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value d_output, ::mlir::Value d_filter, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
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
  assert(index < 11 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace lmhlo_gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::ConvBackwardFilterOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::ConvBackwardInputOp declarations
//===----------------------------------------------------------------------===//

class ConvBackwardInputOpAdaptor {
public:
  ConvBackwardInputOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ConvBackwardInputOpAdaptor(ConvBackwardInputOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value d_output();
  ::mlir::Value filter();
  ::mlir::Value d_input();
  ::mlir::Value scratch();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr window_strides();
  ::mlir::DenseIntElementsAttr padding();
  ::mlir::DenseIntElementsAttr lhs_dilation();
  ::mlir::DenseIntElementsAttr rhs_dilation();
  ::mlir::DenseElementsAttr window_reversal();
  ::mlir::mhlo::ConvDimensionNumbers dimension_numbers();
  ::mlir::IntegerAttr feature_group_count();
  ::mlir::IntegerAttr batch_group_count();
  ::mlir::ArrayAttr precision_config();
  ::mlir::FloatAttr result_scale();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ConvBackwardInputOp : public ::mlir::Op<ConvBackwardInputOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<4>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConvBackwardInputOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("window_strides"), ::llvm::StringRef("padding"), ::llvm::StringRef("lhs_dilation"), ::llvm::StringRef("rhs_dilation"), ::llvm::StringRef("window_reversal"), ::llvm::StringRef("dimension_numbers"), ::llvm::StringRef("feature_group_count"), ::llvm::StringRef("batch_group_count"), ::llvm::StringRef("precision_config"), ::llvm::StringRef("result_scale"), ::llvm::StringRef("backend_config")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier window_stridesAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier window_stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier paddingAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier paddingAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier lhs_dilationAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier lhs_dilationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  ::mlir::Identifier rhs_dilationAttrName() {
    return getAttributeNameForIndex(3);
  }
  static ::mlir::Identifier rhs_dilationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }
  ::mlir::Identifier window_reversalAttrName() {
    return getAttributeNameForIndex(4);
  }
  static ::mlir::Identifier window_reversalAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }
  ::mlir::Identifier dimension_numbersAttrName() {
    return getAttributeNameForIndex(5);
  }
  static ::mlir::Identifier dimension_numbersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }
  ::mlir::Identifier feature_group_countAttrName() {
    return getAttributeNameForIndex(6);
  }
  static ::mlir::Identifier feature_group_countAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }
  ::mlir::Identifier batch_group_countAttrName() {
    return getAttributeNameForIndex(7);
  }
  static ::mlir::Identifier batch_group_countAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 7);
  }
  ::mlir::Identifier precision_configAttrName() {
    return getAttributeNameForIndex(8);
  }
  static ::mlir::Identifier precision_configAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 8);
  }
  ::mlir::Identifier result_scaleAttrName() {
    return getAttributeNameForIndex(9);
  }
  static ::mlir::Identifier result_scaleAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 9);
  }
  ::mlir::Identifier backend_configAttrName() {
    return getAttributeNameForIndex(10);
  }
  static ::mlir::Identifier backend_configAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 10);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.conv_backwardinput");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value d_output();
  ::mlir::Value filter();
  ::mlir::Value d_input();
  ::mlir::Value scratch();
  ::mlir::MutableOperandRange d_outputMutable();
  ::mlir::MutableOperandRange filterMutable();
  ::mlir::MutableOperandRange d_inputMutable();
  ::mlir::MutableOperandRange scratchMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::DenseIntElementsAttr window_stridesAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > window_strides();
  ::mlir::DenseIntElementsAttr paddingAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > padding();
  ::mlir::DenseIntElementsAttr lhs_dilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > lhs_dilation();
  ::mlir::DenseIntElementsAttr rhs_dilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > rhs_dilation();
  ::mlir::DenseElementsAttr window_reversalAttr();
  ::llvm::Optional< ::mlir::DenseElementsAttr > window_reversal();
  ::mlir::mhlo::ConvDimensionNumbers dimension_numbersAttr();
  ::mlir::mhlo::ConvDimensionNumbers dimension_numbers();
  ::mlir::IntegerAttr feature_group_countAttr();
  uint64_t feature_group_count();
  ::mlir::IntegerAttr batch_group_countAttr();
  uint64_t batch_group_count();
  ::mlir::ArrayAttr precision_configAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > precision_config();
  ::mlir::FloatAttr result_scaleAttr();
  ::llvm::APFloat result_scale();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_configAttr();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config();
  void window_stridesAttr(::mlir::DenseIntElementsAttr attr);
  void paddingAttr(::mlir::DenseIntElementsAttr attr);
  void lhs_dilationAttr(::mlir::DenseIntElementsAttr attr);
  void rhs_dilationAttr(::mlir::DenseIntElementsAttr attr);
  void window_reversalAttr(::mlir::DenseElementsAttr attr);
  void dimension_numbersAttr(::mlir::mhlo::ConvDimensionNumbers attr);
  void feature_group_countAttr(::mlir::IntegerAttr attr);
  void batch_group_countAttr(::mlir::IntegerAttr attr);
  void precision_configAttr(::mlir::ArrayAttr attr);
  void result_scaleAttr(::mlir::FloatAttr attr);
  void backend_configAttr(::mlir::lmhlo_gpu::ConvolutionBackendConfig attr);
  ::mlir::Attribute removeWindow_stridesAttr();
  ::mlir::Attribute removePaddingAttr();
  ::mlir::Attribute removeLhs_dilationAttr();
  ::mlir::Attribute removeRhs_dilationAttr();
  ::mlir::Attribute removeWindow_reversalAttr();
  ::mlir::Attribute removePrecision_configAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value d_output, ::mlir::Value filter, ::mlir::Value d_input, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value d_output, ::mlir::Value filter, ::mlir::Value d_input, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value d_output, ::mlir::Value filter, ::mlir::Value d_input, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value d_output, ::mlir::Value filter, ::mlir::Value d_input, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
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
  assert(index < 11 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace lmhlo_gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::ConvBackwardInputOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::ConvForwardFusedOp declarations
//===----------------------------------------------------------------------===//

class ConvForwardFusedOpAdaptor {
public:
  ConvForwardFusedOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ConvForwardFusedOpAdaptor(ConvForwardFusedOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value filter();
  ::mlir::Value bias();
  ::mlir::Value output();
  ::mlir::Value scratch();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr window_strides();
  ::mlir::DenseIntElementsAttr padding();
  ::mlir::DenseIntElementsAttr lhs_dilation();
  ::mlir::DenseIntElementsAttr rhs_dilation();
  ::mlir::DenseElementsAttr window_reversal();
  ::mlir::mhlo::ConvDimensionNumbers dimension_numbers();
  ::mlir::IntegerAttr feature_group_count();
  ::mlir::IntegerAttr batch_group_count();
  ::mlir::ArrayAttr precision_config();
  ::mlir::FloatAttr result_scale();
  ::mlir::StringAttr activation_mode();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ConvForwardFusedOp : public ::mlir::Op<ConvForwardFusedOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<5>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConvForwardFusedOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("window_strides"), ::llvm::StringRef("padding"), ::llvm::StringRef("lhs_dilation"), ::llvm::StringRef("rhs_dilation"), ::llvm::StringRef("window_reversal"), ::llvm::StringRef("dimension_numbers"), ::llvm::StringRef("feature_group_count"), ::llvm::StringRef("batch_group_count"), ::llvm::StringRef("precision_config"), ::llvm::StringRef("result_scale"), ::llvm::StringRef("activation_mode"), ::llvm::StringRef("backend_config")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier window_stridesAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier window_stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier paddingAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier paddingAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier lhs_dilationAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier lhs_dilationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  ::mlir::Identifier rhs_dilationAttrName() {
    return getAttributeNameForIndex(3);
  }
  static ::mlir::Identifier rhs_dilationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }
  ::mlir::Identifier window_reversalAttrName() {
    return getAttributeNameForIndex(4);
  }
  static ::mlir::Identifier window_reversalAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }
  ::mlir::Identifier dimension_numbersAttrName() {
    return getAttributeNameForIndex(5);
  }
  static ::mlir::Identifier dimension_numbersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }
  ::mlir::Identifier feature_group_countAttrName() {
    return getAttributeNameForIndex(6);
  }
  static ::mlir::Identifier feature_group_countAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }
  ::mlir::Identifier batch_group_countAttrName() {
    return getAttributeNameForIndex(7);
  }
  static ::mlir::Identifier batch_group_countAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 7);
  }
  ::mlir::Identifier precision_configAttrName() {
    return getAttributeNameForIndex(8);
  }
  static ::mlir::Identifier precision_configAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 8);
  }
  ::mlir::Identifier result_scaleAttrName() {
    return getAttributeNameForIndex(9);
  }
  static ::mlir::Identifier result_scaleAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 9);
  }
  ::mlir::Identifier activation_modeAttrName() {
    return getAttributeNameForIndex(10);
  }
  static ::mlir::Identifier activation_modeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 10);
  }
  ::mlir::Identifier backend_configAttrName() {
    return getAttributeNameForIndex(11);
  }
  static ::mlir::Identifier backend_configAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 11);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.conv_forward_fused");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value filter();
  ::mlir::Value bias();
  ::mlir::Value output();
  ::mlir::Value scratch();
  ::mlir::MutableOperandRange inputMutable();
  ::mlir::MutableOperandRange filterMutable();
  ::mlir::MutableOperandRange biasMutable();
  ::mlir::MutableOperandRange outputMutable();
  ::mlir::MutableOperandRange scratchMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::DenseIntElementsAttr window_stridesAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > window_strides();
  ::mlir::DenseIntElementsAttr paddingAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > padding();
  ::mlir::DenseIntElementsAttr lhs_dilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > lhs_dilation();
  ::mlir::DenseIntElementsAttr rhs_dilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > rhs_dilation();
  ::mlir::DenseElementsAttr window_reversalAttr();
  ::llvm::Optional< ::mlir::DenseElementsAttr > window_reversal();
  ::mlir::mhlo::ConvDimensionNumbers dimension_numbersAttr();
  ::mlir::mhlo::ConvDimensionNumbers dimension_numbers();
  ::mlir::IntegerAttr feature_group_countAttr();
  uint64_t feature_group_count();
  ::mlir::IntegerAttr batch_group_countAttr();
  uint64_t batch_group_count();
  ::mlir::ArrayAttr precision_configAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > precision_config();
  ::mlir::FloatAttr result_scaleAttr();
  ::llvm::APFloat result_scale();
  ::mlir::StringAttr activation_modeAttr();
  ::llvm::StringRef activation_mode();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_configAttr();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config();
  void window_stridesAttr(::mlir::DenseIntElementsAttr attr);
  void paddingAttr(::mlir::DenseIntElementsAttr attr);
  void lhs_dilationAttr(::mlir::DenseIntElementsAttr attr);
  void rhs_dilationAttr(::mlir::DenseIntElementsAttr attr);
  void window_reversalAttr(::mlir::DenseElementsAttr attr);
  void dimension_numbersAttr(::mlir::mhlo::ConvDimensionNumbers attr);
  void feature_group_countAttr(::mlir::IntegerAttr attr);
  void batch_group_countAttr(::mlir::IntegerAttr attr);
  void precision_configAttr(::mlir::ArrayAttr attr);
  void result_scaleAttr(::mlir::FloatAttr attr);
  void activation_modeAttr(::mlir::StringAttr attr);
  void backend_configAttr(::mlir::lmhlo_gpu::ConvolutionBackendConfig attr);
  ::mlir::Attribute removeWindow_stridesAttr();
  ::mlir::Attribute removePaddingAttr();
  ::mlir::Attribute removeLhs_dilationAttr();
  ::mlir::Attribute removeRhs_dilationAttr();
  ::mlir::Attribute removeWindow_reversalAttr();
  ::mlir::Attribute removePrecision_configAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::StringAttr activation_mode, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::StringAttr activation_mode, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::llvm::StringRef activation_mode, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::llvm::StringRef activation_mode, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
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
  assert(index < 12 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace lmhlo_gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::ConvForwardFusedOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::ConvForwardFusedSideInputOp declarations
//===----------------------------------------------------------------------===//

class ConvForwardFusedSideInputOpAdaptor {
public:
  ConvForwardFusedSideInputOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ConvForwardFusedSideInputOpAdaptor(ConvForwardFusedSideInputOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value filter();
  ::mlir::Value bias();
  ::mlir::Value side_input();
  ::mlir::Value output();
  ::mlir::Value scratch();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr window_strides();
  ::mlir::DenseIntElementsAttr padding();
  ::mlir::DenseIntElementsAttr lhs_dilation();
  ::mlir::DenseIntElementsAttr rhs_dilation();
  ::mlir::DenseElementsAttr window_reversal();
  ::mlir::mhlo::ConvDimensionNumbers dimension_numbers();
  ::mlir::IntegerAttr feature_group_count();
  ::mlir::IntegerAttr batch_group_count();
  ::mlir::ArrayAttr precision_config();
  ::mlir::FloatAttr result_scale();
  ::mlir::StringAttr activation_mode();
  ::mlir::FloatAttr side_input_scale();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ConvForwardFusedSideInputOp : public ::mlir::Op<ConvForwardFusedSideInputOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<6>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConvForwardFusedSideInputOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("window_strides"), ::llvm::StringRef("padding"), ::llvm::StringRef("lhs_dilation"), ::llvm::StringRef("rhs_dilation"), ::llvm::StringRef("window_reversal"), ::llvm::StringRef("dimension_numbers"), ::llvm::StringRef("feature_group_count"), ::llvm::StringRef("batch_group_count"), ::llvm::StringRef("precision_config"), ::llvm::StringRef("result_scale"), ::llvm::StringRef("activation_mode"), ::llvm::StringRef("side_input_scale"), ::llvm::StringRef("backend_config")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier window_stridesAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier window_stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier paddingAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier paddingAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier lhs_dilationAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier lhs_dilationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  ::mlir::Identifier rhs_dilationAttrName() {
    return getAttributeNameForIndex(3);
  }
  static ::mlir::Identifier rhs_dilationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }
  ::mlir::Identifier window_reversalAttrName() {
    return getAttributeNameForIndex(4);
  }
  static ::mlir::Identifier window_reversalAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }
  ::mlir::Identifier dimension_numbersAttrName() {
    return getAttributeNameForIndex(5);
  }
  static ::mlir::Identifier dimension_numbersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }
  ::mlir::Identifier feature_group_countAttrName() {
    return getAttributeNameForIndex(6);
  }
  static ::mlir::Identifier feature_group_countAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }
  ::mlir::Identifier batch_group_countAttrName() {
    return getAttributeNameForIndex(7);
  }
  static ::mlir::Identifier batch_group_countAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 7);
  }
  ::mlir::Identifier precision_configAttrName() {
    return getAttributeNameForIndex(8);
  }
  static ::mlir::Identifier precision_configAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 8);
  }
  ::mlir::Identifier result_scaleAttrName() {
    return getAttributeNameForIndex(9);
  }
  static ::mlir::Identifier result_scaleAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 9);
  }
  ::mlir::Identifier activation_modeAttrName() {
    return getAttributeNameForIndex(10);
  }
  static ::mlir::Identifier activation_modeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 10);
  }
  ::mlir::Identifier side_input_scaleAttrName() {
    return getAttributeNameForIndex(11);
  }
  static ::mlir::Identifier side_input_scaleAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 11);
  }
  ::mlir::Identifier backend_configAttrName() {
    return getAttributeNameForIndex(12);
  }
  static ::mlir::Identifier backend_configAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 12);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.conv_forward_fused_with_side_input");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value filter();
  ::mlir::Value bias();
  ::mlir::Value side_input();
  ::mlir::Value output();
  ::mlir::Value scratch();
  ::mlir::MutableOperandRange inputMutable();
  ::mlir::MutableOperandRange filterMutable();
  ::mlir::MutableOperandRange biasMutable();
  ::mlir::MutableOperandRange side_inputMutable();
  ::mlir::MutableOperandRange outputMutable();
  ::mlir::MutableOperandRange scratchMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::DenseIntElementsAttr window_stridesAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > window_strides();
  ::mlir::DenseIntElementsAttr paddingAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > padding();
  ::mlir::DenseIntElementsAttr lhs_dilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > lhs_dilation();
  ::mlir::DenseIntElementsAttr rhs_dilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > rhs_dilation();
  ::mlir::DenseElementsAttr window_reversalAttr();
  ::llvm::Optional< ::mlir::DenseElementsAttr > window_reversal();
  ::mlir::mhlo::ConvDimensionNumbers dimension_numbersAttr();
  ::mlir::mhlo::ConvDimensionNumbers dimension_numbers();
  ::mlir::IntegerAttr feature_group_countAttr();
  uint64_t feature_group_count();
  ::mlir::IntegerAttr batch_group_countAttr();
  uint64_t batch_group_count();
  ::mlir::ArrayAttr precision_configAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > precision_config();
  ::mlir::FloatAttr result_scaleAttr();
  ::llvm::APFloat result_scale();
  ::mlir::StringAttr activation_modeAttr();
  ::llvm::StringRef activation_mode();
  ::mlir::FloatAttr side_input_scaleAttr();
  ::llvm::APFloat side_input_scale();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_configAttr();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config();
  void window_stridesAttr(::mlir::DenseIntElementsAttr attr);
  void paddingAttr(::mlir::DenseIntElementsAttr attr);
  void lhs_dilationAttr(::mlir::DenseIntElementsAttr attr);
  void rhs_dilationAttr(::mlir::DenseIntElementsAttr attr);
  void window_reversalAttr(::mlir::DenseElementsAttr attr);
  void dimension_numbersAttr(::mlir::mhlo::ConvDimensionNumbers attr);
  void feature_group_countAttr(::mlir::IntegerAttr attr);
  void batch_group_countAttr(::mlir::IntegerAttr attr);
  void precision_configAttr(::mlir::ArrayAttr attr);
  void result_scaleAttr(::mlir::FloatAttr attr);
  void activation_modeAttr(::mlir::StringAttr attr);
  void side_input_scaleAttr(::mlir::FloatAttr attr);
  void backend_configAttr(::mlir::lmhlo_gpu::ConvolutionBackendConfig attr);
  ::mlir::Attribute removeWindow_stridesAttr();
  ::mlir::Attribute removePaddingAttr();
  ::mlir::Attribute removeLhs_dilationAttr();
  ::mlir::Attribute removeRhs_dilationAttr();
  ::mlir::Attribute removeWindow_reversalAttr();
  ::mlir::Attribute removePrecision_configAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::Value side_input, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::StringAttr activation_mode, ::mlir::FloatAttr side_input_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::Value side_input, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::StringAttr activation_mode, ::mlir::FloatAttr side_input_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::Value side_input, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::llvm::StringRef activation_mode, ::llvm::APFloat side_input_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::Value side_input, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::llvm::StringRef activation_mode, ::llvm::APFloat side_input_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
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
  assert(index < 13 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace lmhlo_gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::ConvForwardFusedSideInputOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::ConvForwardOp declarations
//===----------------------------------------------------------------------===//

class ConvForwardOpAdaptor {
public:
  ConvForwardOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ConvForwardOpAdaptor(ConvForwardOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value filter();
  ::mlir::Value output();
  ::mlir::Value scratch();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr window_strides();
  ::mlir::DenseIntElementsAttr padding();
  ::mlir::DenseIntElementsAttr lhs_dilation();
  ::mlir::DenseIntElementsAttr rhs_dilation();
  ::mlir::DenseElementsAttr window_reversal();
  ::mlir::mhlo::ConvDimensionNumbers dimension_numbers();
  ::mlir::IntegerAttr feature_group_count();
  ::mlir::IntegerAttr batch_group_count();
  ::mlir::ArrayAttr precision_config();
  ::mlir::FloatAttr result_scale();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ConvForwardOp : public ::mlir::Op<ConvForwardOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<4>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConvForwardOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("window_strides"), ::llvm::StringRef("padding"), ::llvm::StringRef("lhs_dilation"), ::llvm::StringRef("rhs_dilation"), ::llvm::StringRef("window_reversal"), ::llvm::StringRef("dimension_numbers"), ::llvm::StringRef("feature_group_count"), ::llvm::StringRef("batch_group_count"), ::llvm::StringRef("precision_config"), ::llvm::StringRef("result_scale"), ::llvm::StringRef("backend_config")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier window_stridesAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier window_stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier paddingAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier paddingAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier lhs_dilationAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier lhs_dilationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  ::mlir::Identifier rhs_dilationAttrName() {
    return getAttributeNameForIndex(3);
  }
  static ::mlir::Identifier rhs_dilationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }
  ::mlir::Identifier window_reversalAttrName() {
    return getAttributeNameForIndex(4);
  }
  static ::mlir::Identifier window_reversalAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }
  ::mlir::Identifier dimension_numbersAttrName() {
    return getAttributeNameForIndex(5);
  }
  static ::mlir::Identifier dimension_numbersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }
  ::mlir::Identifier feature_group_countAttrName() {
    return getAttributeNameForIndex(6);
  }
  static ::mlir::Identifier feature_group_countAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }
  ::mlir::Identifier batch_group_countAttrName() {
    return getAttributeNameForIndex(7);
  }
  static ::mlir::Identifier batch_group_countAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 7);
  }
  ::mlir::Identifier precision_configAttrName() {
    return getAttributeNameForIndex(8);
  }
  static ::mlir::Identifier precision_configAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 8);
  }
  ::mlir::Identifier result_scaleAttrName() {
    return getAttributeNameForIndex(9);
  }
  static ::mlir::Identifier result_scaleAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 9);
  }
  ::mlir::Identifier backend_configAttrName() {
    return getAttributeNameForIndex(10);
  }
  static ::mlir::Identifier backend_configAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 10);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.conv_forward");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value filter();
  ::mlir::Value output();
  ::mlir::Value scratch();
  ::mlir::MutableOperandRange inputMutable();
  ::mlir::MutableOperandRange filterMutable();
  ::mlir::MutableOperandRange outputMutable();
  ::mlir::MutableOperandRange scratchMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::DenseIntElementsAttr window_stridesAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > window_strides();
  ::mlir::DenseIntElementsAttr paddingAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > padding();
  ::mlir::DenseIntElementsAttr lhs_dilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > lhs_dilation();
  ::mlir::DenseIntElementsAttr rhs_dilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > rhs_dilation();
  ::mlir::DenseElementsAttr window_reversalAttr();
  ::llvm::Optional< ::mlir::DenseElementsAttr > window_reversal();
  ::mlir::mhlo::ConvDimensionNumbers dimension_numbersAttr();
  ::mlir::mhlo::ConvDimensionNumbers dimension_numbers();
  ::mlir::IntegerAttr feature_group_countAttr();
  uint64_t feature_group_count();
  ::mlir::IntegerAttr batch_group_countAttr();
  uint64_t batch_group_count();
  ::mlir::ArrayAttr precision_configAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > precision_config();
  ::mlir::FloatAttr result_scaleAttr();
  ::llvm::APFloat result_scale();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_configAttr();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config();
  void window_stridesAttr(::mlir::DenseIntElementsAttr attr);
  void paddingAttr(::mlir::DenseIntElementsAttr attr);
  void lhs_dilationAttr(::mlir::DenseIntElementsAttr attr);
  void rhs_dilationAttr(::mlir::DenseIntElementsAttr attr);
  void window_reversalAttr(::mlir::DenseElementsAttr attr);
  void dimension_numbersAttr(::mlir::mhlo::ConvDimensionNumbers attr);
  void feature_group_countAttr(::mlir::IntegerAttr attr);
  void batch_group_countAttr(::mlir::IntegerAttr attr);
  void precision_configAttr(::mlir::ArrayAttr attr);
  void result_scaleAttr(::mlir::FloatAttr attr);
  void backend_configAttr(::mlir::lmhlo_gpu::ConvolutionBackendConfig attr);
  ::mlir::Attribute removeWindow_stridesAttr();
  ::mlir::Attribute removePaddingAttr();
  ::mlir::Attribute removeLhs_dilationAttr();
  ::mlir::Attribute removeRhs_dilationAttr();
  ::mlir::Attribute removeWindow_reversalAttr();
  ::mlir::Attribute removePrecision_configAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
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
  assert(index < 11 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace lmhlo_gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::ConvForwardOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::GEMMOp declarations
//===----------------------------------------------------------------------===//

class GEMMOpAdaptor {
public:
  GEMMOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  GEMMOpAdaptor(GEMMOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value lhs();
  ::mlir::Value rhs();
  ::mlir::Value output();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::mhlo::DotDimensionNumbers dot_dimension_numbers();
  ::mlir::FloatAttr alpha_real();
  ::mlir::FloatAttr alpha_imag();
  ::mlir::IntegerAttr batch_size();
  ::mlir::IntegerAttr lhs_stride();
  ::mlir::IntegerAttr rhs_stride();
  ::mlir::IntegerAttr algorithm();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class GEMMOp : public ::mlir::Op<GEMMOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GEMMOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dot_dimension_numbers"), ::llvm::StringRef("alpha_real"), ::llvm::StringRef("alpha_imag"), ::llvm::StringRef("batch_size"), ::llvm::StringRef("lhs_stride"), ::llvm::StringRef("rhs_stride"), ::llvm::StringRef("algorithm")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier dot_dimension_numbersAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier dot_dimension_numbersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier alpha_realAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier alpha_realAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier alpha_imagAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier alpha_imagAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  ::mlir::Identifier batch_sizeAttrName() {
    return getAttributeNameForIndex(3);
  }
  static ::mlir::Identifier batch_sizeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }
  ::mlir::Identifier lhs_strideAttrName() {
    return getAttributeNameForIndex(4);
  }
  static ::mlir::Identifier lhs_strideAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }
  ::mlir::Identifier rhs_strideAttrName() {
    return getAttributeNameForIndex(5);
  }
  static ::mlir::Identifier rhs_strideAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }
  ::mlir::Identifier algorithmAttrName() {
    return getAttributeNameForIndex(6);
  }
  static ::mlir::Identifier algorithmAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.gemm");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value lhs();
  ::mlir::Value rhs();
  ::mlir::Value output();
  ::mlir::MutableOperandRange lhsMutable();
  ::mlir::MutableOperandRange rhsMutable();
  ::mlir::MutableOperandRange outputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::mhlo::DotDimensionNumbers dot_dimension_numbersAttr();
  ::mlir::mhlo::DotDimensionNumbers dot_dimension_numbers();
  ::mlir::FloatAttr alpha_realAttr();
  ::llvm::APFloat alpha_real();
  ::mlir::FloatAttr alpha_imagAttr();
  ::llvm::APFloat alpha_imag();
  ::mlir::IntegerAttr batch_sizeAttr();
  uint64_t batch_size();
  ::mlir::IntegerAttr lhs_strideAttr();
  uint64_t lhs_stride();
  ::mlir::IntegerAttr rhs_strideAttr();
  uint64_t rhs_stride();
  ::mlir::IntegerAttr algorithmAttr();
  ::llvm::Optional<uint64_t> algorithm();
  void dot_dimension_numbersAttr(::mlir::mhlo::DotDimensionNumbers attr);
  void alpha_realAttr(::mlir::FloatAttr attr);
  void alpha_imagAttr(::mlir::FloatAttr attr);
  void batch_sizeAttr(::mlir::IntegerAttr attr);
  void lhs_strideAttr(::mlir::IntegerAttr attr);
  void rhs_strideAttr(::mlir::IntegerAttr attr);
  void algorithmAttr(::mlir::IntegerAttr attr);
  ::mlir::Attribute removeAlgorithmAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value output, ::mlir::mhlo::DotDimensionNumbers dot_dimension_numbers, ::mlir::FloatAttr alpha_real, ::mlir::FloatAttr alpha_imag, ::mlir::IntegerAttr batch_size, ::mlir::IntegerAttr lhs_stride, ::mlir::IntegerAttr rhs_stride, /*optional*/::mlir::IntegerAttr algorithm);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value output, ::mlir::mhlo::DotDimensionNumbers dot_dimension_numbers, ::mlir::FloatAttr alpha_real, ::mlir::FloatAttr alpha_imag, ::mlir::IntegerAttr batch_size, ::mlir::IntegerAttr lhs_stride, ::mlir::IntegerAttr rhs_stride, /*optional*/::mlir::IntegerAttr algorithm);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value output, ::mlir::mhlo::DotDimensionNumbers dot_dimension_numbers, ::llvm::APFloat alpha_real, ::llvm::APFloat alpha_imag, uint64_t batch_size, uint64_t lhs_stride, uint64_t rhs_stride, /*optional*/::mlir::IntegerAttr algorithm);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value output, ::mlir::mhlo::DotDimensionNumbers dot_dimension_numbers, ::llvm::APFloat alpha_real, ::llvm::APFloat alpha_imag, uint64_t batch_size, uint64_t lhs_stride, uint64_t rhs_stride, /*optional*/::mlir::IntegerAttr algorithm);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 7 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace lmhlo_gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::GEMMOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::GEMM_BiasOp declarations
//===----------------------------------------------------------------------===//

class GEMM_BiasOpAdaptor {
public:
  GEMM_BiasOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  GEMM_BiasOpAdaptor(GEMM_BiasOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value lhs();
  ::mlir::Value rhs();
  ::mlir::Value bias();
  ::mlir::Value output();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::mhlo::DotDimensionNumbers dot_dimension_numbers();
  ::mlir::FloatAttr alpha_real();
  ::mlir::FloatAttr alpha_imag();
  ::mlir::FloatAttr beta();
  ::mlir::IntegerAttr batch_size();
  ::mlir::IntegerAttr lhs_stride();
  ::mlir::IntegerAttr rhs_stride();
  ::mlir::IntegerAttr algorithm();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class GEMM_BiasOp : public ::mlir::Op<GEMM_BiasOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<4>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GEMM_BiasOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dot_dimension_numbers"), ::llvm::StringRef("alpha_real"), ::llvm::StringRef("alpha_imag"), ::llvm::StringRef("beta"), ::llvm::StringRef("batch_size"), ::llvm::StringRef("lhs_stride"), ::llvm::StringRef("rhs_stride"), ::llvm::StringRef("algorithm")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier dot_dimension_numbersAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier dot_dimension_numbersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier alpha_realAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier alpha_realAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier alpha_imagAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier alpha_imagAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  ::mlir::Identifier betaAttrName() {
    return getAttributeNameForIndex(3);
  }
  static ::mlir::Identifier betaAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }
  ::mlir::Identifier batch_sizeAttrName() {
    return getAttributeNameForIndex(4);
  }
  static ::mlir::Identifier batch_sizeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }
  ::mlir::Identifier lhs_strideAttrName() {
    return getAttributeNameForIndex(5);
  }
  static ::mlir::Identifier lhs_strideAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }
  ::mlir::Identifier rhs_strideAttrName() {
    return getAttributeNameForIndex(6);
  }
  static ::mlir::Identifier rhs_strideAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }
  ::mlir::Identifier algorithmAttrName() {
    return getAttributeNameForIndex(7);
  }
  static ::mlir::Identifier algorithmAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 7);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.gemm_bias");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value lhs();
  ::mlir::Value rhs();
  ::mlir::Value bias();
  ::mlir::Value output();
  ::mlir::MutableOperandRange lhsMutable();
  ::mlir::MutableOperandRange rhsMutable();
  ::mlir::MutableOperandRange biasMutable();
  ::mlir::MutableOperandRange outputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::mhlo::DotDimensionNumbers dot_dimension_numbersAttr();
  ::mlir::mhlo::DotDimensionNumbers dot_dimension_numbers();
  ::mlir::FloatAttr alpha_realAttr();
  ::llvm::APFloat alpha_real();
  ::mlir::FloatAttr alpha_imagAttr();
  ::llvm::APFloat alpha_imag();
  ::mlir::FloatAttr betaAttr();
  ::llvm::APFloat beta();
  ::mlir::IntegerAttr batch_sizeAttr();
  uint64_t batch_size();
  ::mlir::IntegerAttr lhs_strideAttr();
  uint64_t lhs_stride();
  ::mlir::IntegerAttr rhs_strideAttr();
  uint64_t rhs_stride();
  ::mlir::IntegerAttr algorithmAttr();
  ::llvm::Optional<uint64_t> algorithm();
  void dot_dimension_numbersAttr(::mlir::mhlo::DotDimensionNumbers attr);
  void alpha_realAttr(::mlir::FloatAttr attr);
  void alpha_imagAttr(::mlir::FloatAttr attr);
  void betaAttr(::mlir::FloatAttr attr);
  void batch_sizeAttr(::mlir::IntegerAttr attr);
  void lhs_strideAttr(::mlir::IntegerAttr attr);
  void rhs_strideAttr(::mlir::IntegerAttr attr);
  void algorithmAttr(::mlir::IntegerAttr attr);
  ::mlir::Attribute removeAlgorithmAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value bias, ::mlir::Value output, ::mlir::mhlo::DotDimensionNumbers dot_dimension_numbers, ::mlir::FloatAttr alpha_real, ::mlir::FloatAttr alpha_imag, ::mlir::FloatAttr beta, ::mlir::IntegerAttr batch_size, ::mlir::IntegerAttr lhs_stride, ::mlir::IntegerAttr rhs_stride, /*optional*/::mlir::IntegerAttr algorithm);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value bias, ::mlir::Value output, ::mlir::mhlo::DotDimensionNumbers dot_dimension_numbers, ::mlir::FloatAttr alpha_real, ::mlir::FloatAttr alpha_imag, ::mlir::FloatAttr beta, ::mlir::IntegerAttr batch_size, ::mlir::IntegerAttr lhs_stride, ::mlir::IntegerAttr rhs_stride, /*optional*/::mlir::IntegerAttr algorithm);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value bias, ::mlir::Value output, ::mlir::mhlo::DotDimensionNumbers dot_dimension_numbers, ::llvm::APFloat alpha_real, ::llvm::APFloat alpha_imag, ::llvm::APFloat beta, uint64_t batch_size, uint64_t lhs_stride, uint64_t rhs_stride, /*optional*/::mlir::IntegerAttr algorithm);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value bias, ::mlir::Value output, ::mlir::mhlo::DotDimensionNumbers dot_dimension_numbers, ::llvm::APFloat alpha_real, ::llvm::APFloat alpha_imag, ::llvm::APFloat beta, uint64_t batch_size, uint64_t lhs_stride, uint64_t rhs_stride, /*optional*/::mlir::IntegerAttr algorithm);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 8 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace lmhlo_gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::GEMM_BiasOp)


#endif  // GET_OP_CLASSES

