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
namespace TF {
class _ArrayToListOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _EagerConstOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _FusedBatchNormExOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _FusedConv2DOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _FusedMatMulOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _HostRecvOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _HostSendOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _InternalTestNonResourceValueSideEffects_;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _ListToArrayOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _RecvTPUEmbeddingActivationsOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _RecvTPUEmbeddingDeduplicationDataOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _SendTPUEmbeddingGradientsOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _TPUCompileMlirOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _TPUCompileMlirPlaceholderProgramKeyOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _TPUDeviceOrdinalPlaceholderOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _UnaryOpsCompositionOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _XlaHostComputeMlirOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _XlaRecvAtHostOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _XlaRecvAtHostV2Op;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _XlaSendFromHostOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _XlaSendFromHostV2Op;
} // namespace TF
} // namespace mlir
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_ArrayToListOp declarations
//===----------------------------------------------------------------------===//

class _ArrayToListOpAdaptor {
public:
  _ArrayToListOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  _ArrayToListOpAdaptor(_ArrayToListOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange input();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class _ArrayToListOp : public ::mlir::Op<_ArrayToListOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::DerivedAttributeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = _ArrayToListOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("N"), ::llvm::StringRef("T"), ::llvm::StringRef("out_types")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier NAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier NAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier TAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier TAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier out_typesAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier out_typesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf._ArrayToList");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range input();
  ::mlir::MutableOperandRange inputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range output();
  size_t N();
  ::mlir::Type T();
  mlir::ResultElementTypeRange out_types();
  static bool isDerivedAttribute(::llvm::StringRef name);
  ::mlir::DictionaryAttr materializeDerivedAttributes();
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
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
} // namespace TF
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::TF::_ArrayToListOp)

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_EagerConstOp declarations
//===----------------------------------------------------------------------===//

class _EagerConstOpAdaptor {
public:
  _EagerConstOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  _EagerConstOpAdaptor(_EagerConstOp&op);
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
class _EagerConstOp : public ::mlir::Op<_EagerConstOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::OneOperand, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::DerivedAttributeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = _EagerConstOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("T")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier TAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier TAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf._EagerConst");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::MutableOperandRange inputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value output();
  ::mlir::Type T();
  static bool isDerivedAttribute(::llvm::StringRef name);
  ::mlir::DictionaryAttr materializeDerivedAttributes();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input);
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
} // namespace TF
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::TF::_EagerConstOp)

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_FusedBatchNormExOp declarations
//===----------------------------------------------------------------------===//

class _FusedBatchNormExOpAdaptor {
public:
  _FusedBatchNormExOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  _FusedBatchNormExOpAdaptor(_FusedBatchNormExOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value x();
  ::mlir::Value scale();
  ::mlir::Value offset();
  ::mlir::Value mean();
  ::mlir::Value variance();
  ::mlir::ValueRange side_input();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::FloatAttr epsilon();
  ::mlir::FloatAttr exponential_avg_factor();
  ::mlir::StringAttr activation_mode();
  ::mlir::StringAttr data_format();
  ::mlir::BoolAttr is_training();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class _FusedBatchNormExOp : public ::mlir::Op<_FusedBatchNormExOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::NResults<6>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<5>::Impl, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpAsmOpInterface::Trait, ::mlir::DerivedAttributeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = _FusedBatchNormExOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("epsilon"), ::llvm::StringRef("exponential_avg_factor"), ::llvm::StringRef("activation_mode"), ::llvm::StringRef("data_format"), ::llvm::StringRef("is_training"), ::llvm::StringRef("num_side_inputs"), ::llvm::StringRef("T"), ::llvm::StringRef("U")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier epsilonAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier epsilonAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier exponential_avg_factorAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier exponential_avg_factorAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier activation_modeAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier activation_modeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  ::mlir::Identifier data_formatAttrName() {
    return getAttributeNameForIndex(3);
  }
  static ::mlir::Identifier data_formatAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }
  ::mlir::Identifier is_trainingAttrName() {
    return getAttributeNameForIndex(4);
  }
  static ::mlir::Identifier is_trainingAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }
  ::mlir::Identifier num_side_inputsAttrName() {
    return getAttributeNameForIndex(5);
  }
  static ::mlir::Identifier num_side_inputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }
  ::mlir::Identifier TAttrName() {
    return getAttributeNameForIndex(6);
  }
  static ::mlir::Identifier TAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }
  ::mlir::Identifier UAttrName() {
    return getAttributeNameForIndex(7);
  }
  static ::mlir::Identifier UAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 7);
  }
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf._FusedBatchNormEx");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value x();
  ::mlir::Value scale();
  ::mlir::Value offset();
  ::mlir::Value mean();
  ::mlir::Value variance();
  ::mlir::Operation::operand_range side_input();
  ::mlir::MutableOperandRange xMutable();
  ::mlir::MutableOperandRange scaleMutable();
  ::mlir::MutableOperandRange offsetMutable();
  ::mlir::MutableOperandRange meanMutable();
  ::mlir::MutableOperandRange varianceMutable();
  ::mlir::MutableOperandRange side_inputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value y();
  ::mlir::Value batch_mean();
  ::mlir::Value batch_variance();
  ::mlir::Value reserve_space_1();
  ::mlir::Value reserve_space_2();
  ::mlir::Value reserve_space_3();
  ::mlir::FloatAttr epsilonAttr();
  ::llvm::APFloat epsilon();
  ::mlir::FloatAttr exponential_avg_factorAttr();
  ::llvm::APFloat exponential_avg_factor();
  ::mlir::StringAttr activation_modeAttr();
  ::llvm::StringRef activation_mode();
  ::mlir::StringAttr data_formatAttr();
  ::llvm::StringRef data_format();
  ::mlir::BoolAttr is_trainingAttr();
  bool is_training();
  size_t num_side_inputs();
  ::mlir::Type T();
  ::mlir::Type U();
  static bool isDerivedAttribute(::llvm::StringRef name);
  ::mlir::DictionaryAttr materializeDerivedAttributes();
  void epsilonAttr(::mlir::FloatAttr attr);
  void exponential_avg_factorAttr(::mlir::FloatAttr attr);
  void activation_modeAttr(::mlir::StringAttr attr);
  void data_formatAttr(::mlir::StringAttr attr);
  void is_trainingAttr(::mlir::BoolAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type y, ::mlir::Type batch_mean, ::mlir::Type batch_variance, ::mlir::Type reserve_space_1, ::mlir::Type reserve_space_2, ::mlir::Type reserve_space_3, ::mlir::Value x, ::mlir::Value scale, ::mlir::Value offset, ::mlir::Value mean, ::mlir::Value variance, ::mlir::ValueRange side_input, ::mlir::FloatAttr epsilon, ::mlir::FloatAttr exponential_avg_factor, ::mlir::StringAttr activation_mode, ::mlir::StringAttr data_format, ::mlir::BoolAttr is_training);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value x, ::mlir::Value scale, ::mlir::Value offset, ::mlir::Value mean, ::mlir::Value variance, ::mlir::ValueRange side_input, ::mlir::FloatAttr epsilon, ::mlir::FloatAttr exponential_avg_factor, ::mlir::StringAttr activation_mode, ::mlir::StringAttr data_format, ::mlir::BoolAttr is_training);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type y, ::mlir::Type batch_mean, ::mlir::Type batch_variance, ::mlir::Type reserve_space_1, ::mlir::Type reserve_space_2, ::mlir::Type reserve_space_3, ::mlir::Value x, ::mlir::Value scale, ::mlir::Value offset, ::mlir::Value mean, ::mlir::Value variance, ::mlir::ValueRange side_input, ::llvm::APFloat epsilon, ::llvm::APFloat exponential_avg_factor, ::llvm::StringRef activation_mode = "Identity", ::llvm::StringRef data_format = "NHWC", bool is_training = true);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value x, ::mlir::Value scale, ::mlir::Value offset, ::mlir::Value mean, ::mlir::Value variance, ::mlir::ValueRange side_input, ::llvm::APFloat epsilon, ::llvm::APFloat exponential_avg_factor, ::llvm::StringRef activation_mode = "Identity", ::llvm::StringRef data_format = "NHWC", bool is_training = true);
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
} // namespace TF
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::TF::_FusedBatchNormExOp)

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_FusedConv2DOp declarations
//===----------------------------------------------------------------------===//

class _FusedConv2DOpAdaptor {
public:
  _FusedConv2DOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  _FusedConv2DOpAdaptor(_FusedConv2DOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value filter();
  ::mlir::ValueRange args();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr strides();
  ::mlir::StringAttr padding();
  ::mlir::ArrayAttr explicit_paddings();
  ::mlir::StringAttr data_format();
  ::mlir::ArrayAttr dilations();
  ::mlir::BoolAttr use_cudnn_on_gpu();
  ::mlir::ArrayAttr fused_ops();
  ::mlir::FloatAttr epsilon();
  ::mlir::FloatAttr leakyrelu_alpha();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class _FusedConv2DOp : public ::mlir::Op<_FusedConv2DOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::DerivedAttributeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = _FusedConv2DOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("strides"), ::llvm::StringRef("padding"), ::llvm::StringRef("explicit_paddings"), ::llvm::StringRef("data_format"), ::llvm::StringRef("dilations"), ::llvm::StringRef("use_cudnn_on_gpu"), ::llvm::StringRef("fused_ops"), ::llvm::StringRef("epsilon"), ::llvm::StringRef("leakyrelu_alpha"), ::llvm::StringRef("num_args"), ::llvm::StringRef("T")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier stridesAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier paddingAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier paddingAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier explicit_paddingsAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier explicit_paddingsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  ::mlir::Identifier data_formatAttrName() {
    return getAttributeNameForIndex(3);
  }
  static ::mlir::Identifier data_formatAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }
  ::mlir::Identifier dilationsAttrName() {
    return getAttributeNameForIndex(4);
  }
  static ::mlir::Identifier dilationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }
  ::mlir::Identifier use_cudnn_on_gpuAttrName() {
    return getAttributeNameForIndex(5);
  }
  static ::mlir::Identifier use_cudnn_on_gpuAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }
  ::mlir::Identifier fused_opsAttrName() {
    return getAttributeNameForIndex(6);
  }
  static ::mlir::Identifier fused_opsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }
  ::mlir::Identifier epsilonAttrName() {
    return getAttributeNameForIndex(7);
  }
  static ::mlir::Identifier epsilonAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 7);
  }
  ::mlir::Identifier leakyrelu_alphaAttrName() {
    return getAttributeNameForIndex(8);
  }
  static ::mlir::Identifier leakyrelu_alphaAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 8);
  }
  ::mlir::Identifier num_argsAttrName() {
    return getAttributeNameForIndex(9);
  }
  static ::mlir::Identifier num_argsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 9);
  }
  ::mlir::Identifier TAttrName() {
    return getAttributeNameForIndex(10);
  }
  static ::mlir::Identifier TAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 10);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf._FusedConv2D");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value filter();
  ::mlir::Operation::operand_range args();
  ::mlir::MutableOperandRange inputMutable();
  ::mlir::MutableOperandRange filterMutable();
  ::mlir::MutableOperandRange argsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value output();
  ::mlir::ArrayAttr stridesAttr();
  ::mlir::ArrayAttr strides();
  ::mlir::StringAttr paddingAttr();
  ::llvm::StringRef padding();
  ::mlir::ArrayAttr explicit_paddingsAttr();
  ::mlir::ArrayAttr explicit_paddings();
  ::mlir::StringAttr data_formatAttr();
  ::llvm::StringRef data_format();
  ::mlir::ArrayAttr dilationsAttr();
  ::mlir::ArrayAttr dilations();
  ::mlir::BoolAttr use_cudnn_on_gpuAttr();
  bool use_cudnn_on_gpu();
  ::mlir::ArrayAttr fused_opsAttr();
  ::mlir::ArrayAttr fused_ops();
  ::mlir::FloatAttr epsilonAttr();
  ::llvm::APFloat epsilon();
  ::mlir::FloatAttr leakyrelu_alphaAttr();
  ::llvm::APFloat leakyrelu_alpha();
  size_t num_args();
  ::mlir::Type T();
  static bool isDerivedAttribute(::llvm::StringRef name);
  ::mlir::DictionaryAttr materializeDerivedAttributes();
  void stridesAttr(::mlir::ArrayAttr attr);
  void paddingAttr(::mlir::StringAttr attr);
  void explicit_paddingsAttr(::mlir::ArrayAttr attr);
  void data_formatAttr(::mlir::StringAttr attr);
  void dilationsAttr(::mlir::ArrayAttr attr);
  void use_cudnn_on_gpuAttr(::mlir::BoolAttr attr);
  void fused_opsAttr(::mlir::ArrayAttr attr);
  void epsilonAttr(::mlir::FloatAttr attr);
  void leakyrelu_alphaAttr(::mlir::FloatAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::Value filter, ::mlir::ValueRange args, ::mlir::ArrayAttr strides, ::mlir::StringAttr padding, ::mlir::ArrayAttr explicit_paddings, ::mlir::StringAttr data_format, ::mlir::ArrayAttr dilations, ::mlir::BoolAttr use_cudnn_on_gpu, ::mlir::ArrayAttr fused_ops, ::mlir::FloatAttr epsilon, ::mlir::FloatAttr leakyrelu_alpha);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value filter, ::mlir::ValueRange args, ::mlir::ArrayAttr strides, ::mlir::StringAttr padding, ::mlir::ArrayAttr explicit_paddings, ::mlir::StringAttr data_format, ::mlir::ArrayAttr dilations, ::mlir::BoolAttr use_cudnn_on_gpu, ::mlir::ArrayAttr fused_ops, ::mlir::FloatAttr epsilon, ::mlir::FloatAttr leakyrelu_alpha);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::Value filter, ::mlir::ValueRange args, ::mlir::ArrayAttr strides, ::llvm::StringRef padding, ::mlir::ArrayAttr explicit_paddings, ::llvm::StringRef data_format, ::mlir::ArrayAttr dilations, bool use_cudnn_on_gpu, ::mlir::ArrayAttr fused_ops, ::llvm::APFloat epsilon, ::llvm::APFloat leakyrelu_alpha);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value filter, ::mlir::ValueRange args, ::mlir::ArrayAttr strides, ::llvm::StringRef padding, ::mlir::ArrayAttr explicit_paddings, ::llvm::StringRef data_format, ::mlir::ArrayAttr dilations, bool use_cudnn_on_gpu, ::mlir::ArrayAttr fused_ops, ::llvm::APFloat epsilon, ::llvm::APFloat leakyrelu_alpha);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
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
} // namespace TF
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::TF::_FusedConv2DOp)

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_FusedMatMulOp declarations
//===----------------------------------------------------------------------===//

class _FusedMatMulOpAdaptor {
public:
  _FusedMatMulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  _FusedMatMulOpAdaptor(_FusedMatMulOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value a();
  ::mlir::Value b();
  ::mlir::ValueRange args();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::BoolAttr transpose_a();
  ::mlir::BoolAttr transpose_b();
  ::mlir::ArrayAttr fused_ops();
  ::mlir::FloatAttr epsilon();
  ::mlir::FloatAttr leakyrelu_alpha();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class _FusedMatMulOp : public ::mlir::Op<_FusedMatMulOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::TF::SameOperandsAndResultElementTypeResolveRef, ::mlir::DerivedAttributeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = _FusedMatMulOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("transpose_a"), ::llvm::StringRef("transpose_b"), ::llvm::StringRef("fused_ops"), ::llvm::StringRef("epsilon"), ::llvm::StringRef("leakyrelu_alpha"), ::llvm::StringRef("num_args"), ::llvm::StringRef("T")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier transpose_aAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier transpose_aAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier transpose_bAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier transpose_bAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier fused_opsAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier fused_opsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  ::mlir::Identifier epsilonAttrName() {
    return getAttributeNameForIndex(3);
  }
  static ::mlir::Identifier epsilonAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }
  ::mlir::Identifier leakyrelu_alphaAttrName() {
    return getAttributeNameForIndex(4);
  }
  static ::mlir::Identifier leakyrelu_alphaAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }
  ::mlir::Identifier num_argsAttrName() {
    return getAttributeNameForIndex(5);
  }
  static ::mlir::Identifier num_argsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }
  ::mlir::Identifier TAttrName() {
    return getAttributeNameForIndex(6);
  }
  static ::mlir::Identifier TAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf._FusedMatMul");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value a();
  ::mlir::Value b();
  ::mlir::Operation::operand_range args();
  ::mlir::MutableOperandRange aMutable();
  ::mlir::MutableOperandRange bMutable();
  ::mlir::MutableOperandRange argsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value product();
  ::mlir::BoolAttr transpose_aAttr();
  bool transpose_a();
  ::mlir::BoolAttr transpose_bAttr();
  bool transpose_b();
  ::mlir::ArrayAttr fused_opsAttr();
  ::mlir::ArrayAttr fused_ops();
  ::mlir::FloatAttr epsilonAttr();
  ::llvm::APFloat epsilon();
  ::mlir::FloatAttr leakyrelu_alphaAttr();
  ::llvm::APFloat leakyrelu_alpha();
  size_t num_args();
  ::mlir::Type T();
  static bool isDerivedAttribute(::llvm::StringRef name);
  ::mlir::DictionaryAttr materializeDerivedAttributes();
  void transpose_aAttr(::mlir::BoolAttr attr);
  void transpose_bAttr(::mlir::BoolAttr attr);
  void fused_opsAttr(::mlir::ArrayAttr attr);
  void epsilonAttr(::mlir::FloatAttr attr);
  void leakyrelu_alphaAttr(::mlir::FloatAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type product, ::mlir::Value a, ::mlir::Value b, ::mlir::ValueRange args, ::mlir::BoolAttr transpose_a, ::mlir::BoolAttr transpose_b, ::mlir::ArrayAttr fused_ops, ::mlir::FloatAttr epsilon, ::mlir::FloatAttr leakyrelu_alpha);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b, ::mlir::ValueRange args, ::mlir::BoolAttr transpose_a, ::mlir::BoolAttr transpose_b, ::mlir::ArrayAttr fused_ops, ::mlir::FloatAttr epsilon, ::mlir::FloatAttr leakyrelu_alpha);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type product, ::mlir::Value a, ::mlir::Value b, ::mlir::ValueRange args, bool transpose_a, bool transpose_b, ::mlir::ArrayAttr fused_ops, ::llvm::APFloat epsilon, ::llvm::APFloat leakyrelu_alpha);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b, ::mlir::ValueRange args, bool transpose_a, bool transpose_b, ::mlir::ArrayAttr fused_ops, ::llvm::APFloat epsilon, ::llvm::APFloat leakyrelu_alpha);
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
} // namespace TF
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::TF::_FusedMatMulOp)

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_HostRecvOp declarations
//===----------------------------------------------------------------------===//

class _HostRecvOpAdaptor {
public:
  _HostRecvOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  _HostRecvOpAdaptor(_HostRecvOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr tensor_name();
  ::mlir::StringAttr send_device();
  ::mlir::IntegerAttr send_device_incarnation();
  ::mlir::StringAttr recv_device();
  ::mlir::BoolAttr client_terminated();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class _HostRecvOp : public ::mlir::Op<_HostRecvOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::DerivedAttributeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = _HostRecvOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("tensor_name"), ::llvm::StringRef("send_device"), ::llvm::StringRef("send_device_incarnation"), ::llvm::StringRef("recv_device"), ::llvm::StringRef("client_terminated"), ::llvm::StringRef("tensor_type")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier tensor_nameAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier tensor_nameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier send_deviceAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier send_deviceAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier send_device_incarnationAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier send_device_incarnationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  ::mlir::Identifier recv_deviceAttrName() {
    return getAttributeNameForIndex(3);
  }
  static ::mlir::Identifier recv_deviceAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }
  ::mlir::Identifier client_terminatedAttrName() {
    return getAttributeNameForIndex(4);
  }
  static ::mlir::Identifier client_terminatedAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }
  ::mlir::Identifier tensor_typeAttrName() {
    return getAttributeNameForIndex(5);
  }
  static ::mlir::Identifier tensor_typeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf._HostRecv");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value tensor();
  ::mlir::StringAttr tensor_nameAttr();
  ::llvm::StringRef tensor_name();
  ::mlir::StringAttr send_deviceAttr();
  ::llvm::StringRef send_device();
  ::mlir::IntegerAttr send_device_incarnationAttr();
  uint64_t send_device_incarnation();
  ::mlir::StringAttr recv_deviceAttr();
  ::llvm::StringRef recv_device();
  ::mlir::BoolAttr client_terminatedAttr();
  bool client_terminated();
  ::mlir::Type tensor_type();
  static bool isDerivedAttribute(::llvm::StringRef name);
  ::mlir::DictionaryAttr materializeDerivedAttributes();
  void tensor_nameAttr(::mlir::StringAttr attr);
  void send_deviceAttr(::mlir::StringAttr attr);
  void send_device_incarnationAttr(::mlir::IntegerAttr attr);
  void recv_deviceAttr(::mlir::StringAttr attr);
  void client_terminatedAttr(::mlir::BoolAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type tensor, ::mlir::StringAttr tensor_name, ::mlir::StringAttr send_device, ::mlir::IntegerAttr send_device_incarnation, ::mlir::StringAttr recv_device, ::mlir::BoolAttr client_terminated);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr tensor_name, ::mlir::StringAttr send_device, ::mlir::IntegerAttr send_device_incarnation, ::mlir::StringAttr recv_device, ::mlir::BoolAttr client_terminated);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type tensor, ::llvm::StringRef tensor_name, ::llvm::StringRef send_device, uint64_t send_device_incarnation, ::llvm::StringRef recv_device, bool client_terminated = false);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef tensor_name, ::llvm::StringRef send_device, uint64_t send_device_incarnation, ::llvm::StringRef recv_device, bool client_terminated = false);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 6 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace TF
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::TF::_HostRecvOp)

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_HostSendOp declarations
//===----------------------------------------------------------------------===//

class _HostSendOpAdaptor {
public:
  _HostSendOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  _HostSendOpAdaptor(_HostSendOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value tensor();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr tensor_name();
  ::mlir::StringAttr send_device();
  ::mlir::IntegerAttr send_device_incarnation();
  ::mlir::StringAttr recv_device();
  ::mlir::BoolAttr client_terminated();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class _HostSendOp : public ::mlir::Op<_HostSendOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::OneOperand, ::mlir::DerivedAttributeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = _HostSendOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("tensor_name"), ::llvm::StringRef("send_device"), ::llvm::StringRef("send_device_incarnation"), ::llvm::StringRef("recv_device"), ::llvm::StringRef("client_terminated"), ::llvm::StringRef("T")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier tensor_nameAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier tensor_nameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier send_deviceAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier send_deviceAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier send_device_incarnationAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier send_device_incarnationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  ::mlir::Identifier recv_deviceAttrName() {
    return getAttributeNameForIndex(3);
  }
  static ::mlir::Identifier recv_deviceAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }
  ::mlir::Identifier client_terminatedAttrName() {
    return getAttributeNameForIndex(4);
  }
  static ::mlir::Identifier client_terminatedAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }
  ::mlir::Identifier TAttrName() {
    return getAttributeNameForIndex(5);
  }
  static ::mlir::Identifier TAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf._HostSend");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value tensor();
  ::mlir::MutableOperandRange tensorMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::StringAttr tensor_nameAttr();
  ::llvm::StringRef tensor_name();
  ::mlir::StringAttr send_deviceAttr();
  ::llvm::StringRef send_device();
  ::mlir::IntegerAttr send_device_incarnationAttr();
  uint64_t send_device_incarnation();
  ::mlir::StringAttr recv_deviceAttr();
  ::llvm::StringRef recv_device();
  ::mlir::BoolAttr client_terminatedAttr();
  bool client_terminated();
  ::mlir::Type T();
  static bool isDerivedAttribute(::llvm::StringRef name);
  ::mlir::DictionaryAttr materializeDerivedAttributes();
  void tensor_nameAttr(::mlir::StringAttr attr);
  void send_deviceAttr(::mlir::StringAttr attr);
  void send_device_incarnationAttr(::mlir::IntegerAttr attr);
  void recv_deviceAttr(::mlir::StringAttr attr);
  void client_terminatedAttr(::mlir::BoolAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value tensor, ::mlir::StringAttr tensor_name, ::mlir::StringAttr send_device, ::mlir::IntegerAttr send_device_incarnation, ::mlir::StringAttr recv_device, ::mlir::BoolAttr client_terminated);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value tensor, ::mlir::StringAttr tensor_name, ::mlir::StringAttr send_device, ::mlir::IntegerAttr send_device_incarnation, ::mlir::StringAttr recv_device, ::mlir::BoolAttr client_terminated);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value tensor, ::llvm::StringRef tensor_name, ::llvm::StringRef send_device, uint64_t send_device_incarnation, ::llvm::StringRef recv_device, bool client_terminated = false);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value tensor, ::llvm::StringRef tensor_name, ::llvm::StringRef send_device, uint64_t send_device_incarnation, ::llvm::StringRef recv_device, bool client_terminated = false);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 6 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace TF
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::TF::_HostSendOp)

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_InternalTestNonResourceValueSideEffects_ declarations
//===----------------------------------------------------------------------===//

class _InternalTestNonResourceValueSideEffects_Adaptor {
public:
  _InternalTestNonResourceValueSideEffects_Adaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  _InternalTestNonResourceValueSideEffects_Adaptor(_InternalTestNonResourceValueSideEffects_&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value key();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class _InternalTestNonResourceValueSideEffects_ : public ::mlir::Op<_InternalTestNonResourceValueSideEffects_, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::OneOperand, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = _InternalTestNonResourceValueSideEffects_Adaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf._InternalTestNonResourceValueSideEffects_");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value key();
  ::mlir::MutableOperandRange keyMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value key);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value key);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
};
} // namespace TF
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::TF::_InternalTestNonResourceValueSideEffects_)

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_ListToArrayOp declarations
//===----------------------------------------------------------------------===//

class _ListToArrayOpAdaptor {
public:
  _ListToArrayOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  _ListToArrayOpAdaptor(_ListToArrayOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange input();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class _ListToArrayOp : public ::mlir::Op<_ListToArrayOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::DerivedAttributeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = _ListToArrayOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("Tin"), ::llvm::StringRef("N"), ::llvm::StringRef("T")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier TinAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier TinAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier NAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier NAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier TAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier TAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf._ListToArray");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range input();
  ::mlir::MutableOperandRange inputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range output();
  mlir::OperandElementTypeRange Tin();
  size_t N();
  ::mlir::Type T();
  static bool isDerivedAttribute(::llvm::StringRef name);
  ::mlir::DictionaryAttr materializeDerivedAttributes();
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
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
} // namespace TF
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::TF::_ListToArrayOp)

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_RecvTPUEmbeddingActivationsOp declarations
//===----------------------------------------------------------------------===//

class _RecvTPUEmbeddingActivationsOpAdaptor {
public:
  _RecvTPUEmbeddingActivationsOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  _RecvTPUEmbeddingActivationsOpAdaptor(_RecvTPUEmbeddingActivationsOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value deduplication_data();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr config();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class _RecvTPUEmbeddingActivationsOp : public ::mlir::Op<_RecvTPUEmbeddingActivationsOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::OneOperand, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::DerivedAttributeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = _RecvTPUEmbeddingActivationsOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("config"), ::llvm::StringRef("num_tables")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier configAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier configAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier num_tablesAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier num_tablesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf._RecvTPUEmbeddingActivations");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value deduplication_data();
  ::mlir::MutableOperandRange deduplication_dataMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range outputs();
  ::mlir::StringAttr configAttr();
  ::llvm::StringRef config();
  size_t num_tables();
  static bool isDerivedAttribute(::llvm::StringRef name);
  ::mlir::DictionaryAttr materializeDerivedAttributes();
  void configAttr(::mlir::StringAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outputs, ::mlir::Value deduplication_data, ::mlir::StringAttr config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outputs, ::mlir::Value deduplication_data, ::llvm::StringRef config);
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
} // namespace TF
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::TF::_RecvTPUEmbeddingActivationsOp)

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_RecvTPUEmbeddingDeduplicationDataOp declarations
//===----------------------------------------------------------------------===//

class _RecvTPUEmbeddingDeduplicationDataOpAdaptor {
public:
  _RecvTPUEmbeddingDeduplicationDataOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  _RecvTPUEmbeddingDeduplicationDataOpAdaptor(_RecvTPUEmbeddingDeduplicationDataOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr config();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class _RecvTPUEmbeddingDeduplicationDataOp : public ::mlir::Op<_RecvTPUEmbeddingDeduplicationDataOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = _RecvTPUEmbeddingDeduplicationDataOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("config")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier configAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier configAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf._RecvTPUEmbeddingDeduplicationData");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value output();
  ::mlir::StringAttr configAttr();
  ::llvm::StringRef config();
  void configAttr(::mlir::StringAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::StringAttr config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::llvm::StringRef config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef config);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
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
} // namespace TF
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::TF::_RecvTPUEmbeddingDeduplicationDataOp)

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_SendTPUEmbeddingGradientsOp declarations
//===----------------------------------------------------------------------===//

class _SendTPUEmbeddingGradientsOpAdaptor {
public:
  _SendTPUEmbeddingGradientsOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  _SendTPUEmbeddingGradientsOpAdaptor(_SendTPUEmbeddingGradientsOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange gradients();
  ::mlir::ValueRange learning_rates();
  ::mlir::Value deduplication_data();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr config();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class _SendTPUEmbeddingGradientsOp : public ::mlir::Op<_SendTPUEmbeddingGradientsOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::DerivedAttributeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = _SendTPUEmbeddingGradientsOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("config"), ::llvm::StringRef("NumLearningRateTags"), ::llvm::StringRef("NumTables"), ::llvm::StringRef("operand_segment_sizes")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier configAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier configAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier NumLearningRateTagsAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier NumLearningRateTagsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier NumTablesAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier NumTablesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  ::mlir::Identifier operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(3);
  }
  static ::mlir::Identifier operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf._SendTPUEmbeddingGradients");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range gradients();
  ::mlir::Operation::operand_range learning_rates();
  ::mlir::Value deduplication_data();
  ::mlir::MutableOperandRange gradientsMutable();
  ::mlir::MutableOperandRange learning_ratesMutable();
  ::mlir::MutableOperandRange deduplication_dataMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::StringAttr configAttr();
  ::llvm::StringRef config();
  size_t NumLearningRateTags();
  size_t NumTables();
  static bool isDerivedAttribute(::llvm::StringRef name);
  ::mlir::DictionaryAttr materializeDerivedAttributes();
  void configAttr(::mlir::StringAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange gradients, ::mlir::ValueRange learning_rates, ::mlir::Value deduplication_data, ::mlir::StringAttr config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange gradients, ::mlir::ValueRange learning_rates, ::mlir::Value deduplication_data, ::mlir::StringAttr config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange gradients, ::mlir::ValueRange learning_rates, ::mlir::Value deduplication_data, ::llvm::StringRef config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange gradients, ::mlir::ValueRange learning_rates, ::mlir::Value deduplication_data, ::llvm::StringRef config);
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
} // namespace TF
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::TF::_SendTPUEmbeddingGradientsOp)

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_TPUCompileMlirOp declarations
//===----------------------------------------------------------------------===//

class _TPUCompileMlirOpAdaptor {
public:
  _TPUCompileMlirOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  _TPUCompileMlirOpAdaptor(_TPUCompileMlirOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange dynamic_shapes();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr mlir_module();
  ::mlir::StringAttr metadata();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class _TPUCompileMlirOp : public ::mlir::Op<_TPUCompileMlirOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::AtLeastNResults<1>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpAsmOpInterface::Trait, ::mlir::DerivedAttributeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = _TPUCompileMlirOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("mlir_module"), ::llvm::StringRef("metadata"), ::llvm::StringRef("NumDynamicShapes"), ::llvm::StringRef("num_computations")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier mlir_moduleAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier mlir_moduleAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier metadataAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier metadataAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier NumDynamicShapesAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier NumDynamicShapesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  ::mlir::Identifier num_computationsAttrName() {
    return getAttributeNameForIndex(3);
  }
  static ::mlir::Identifier num_computationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf._TPUCompileMlir");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range dynamic_shapes();
  ::mlir::MutableOperandRange dynamic_shapesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value compilation_status();
  ::mlir::Operation::result_range program();
  ::mlir::StringAttr mlir_moduleAttr();
  ::llvm::StringRef mlir_module();
  ::mlir::StringAttr metadataAttr();
  ::llvm::StringRef metadata();
  size_t NumDynamicShapes();
  size_t num_computations();
  static bool isDerivedAttribute(::llvm::StringRef name);
  ::mlir::DictionaryAttr materializeDerivedAttributes();
  void mlir_moduleAttr(::mlir::StringAttr attr);
  void metadataAttr(::mlir::StringAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type compilation_status, ::mlir::TypeRange program, ::mlir::ValueRange dynamic_shapes, ::mlir::StringAttr mlir_module, ::mlir::StringAttr metadata);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange dynamic_shapes, ::mlir::StringAttr mlir_module, ::mlir::StringAttr metadata);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type compilation_status, ::mlir::TypeRange program, ::mlir::ValueRange dynamic_shapes, ::llvm::StringRef mlir_module, ::llvm::StringRef metadata);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange dynamic_shapes, ::llvm::StringRef mlir_module, ::llvm::StringRef metadata);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 4 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace TF
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::TF::_TPUCompileMlirOp)

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_TPUCompileMlirPlaceholderProgramKeyOp declarations
//===----------------------------------------------------------------------===//

class _TPUCompileMlirPlaceholderProgramKeyOpAdaptor {
public:
  _TPUCompileMlirPlaceholderProgramKeyOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  _TPUCompileMlirPlaceholderProgramKeyOpAdaptor(_TPUCompileMlirPlaceholderProgramKeyOp&op);
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
class _TPUCompileMlirPlaceholderProgramKeyOp : public ::mlir::Op<_TPUCompileMlirPlaceholderProgramKeyOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = _TPUCompileMlirPlaceholderProgramKeyOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf._TPUCompileMlirPlaceholderProgramKey");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value program();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type program);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
};
} // namespace TF
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::TF::_TPUCompileMlirPlaceholderProgramKeyOp)

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_TPUDeviceOrdinalPlaceholderOp declarations
//===----------------------------------------------------------------------===//

class _TPUDeviceOrdinalPlaceholderOpAdaptor {
public:
  _TPUDeviceOrdinalPlaceholderOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  _TPUDeviceOrdinalPlaceholderOpAdaptor(_TPUDeviceOrdinalPlaceholderOp&op);
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
class _TPUDeviceOrdinalPlaceholderOp : public ::mlir::Op<_TPUDeviceOrdinalPlaceholderOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = _TPUDeviceOrdinalPlaceholderOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf._TPUDeviceOrdinalPlaceholder");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value device_ordinal();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type device_ordinal);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
};
} // namespace TF
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::TF::_TPUDeviceOrdinalPlaceholderOp)

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_UnaryOpsCompositionOp declarations
//===----------------------------------------------------------------------===//

class _UnaryOpsCompositionOpAdaptor {
public:
  _UnaryOpsCompositionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  _UnaryOpsCompositionOpAdaptor(_UnaryOpsCompositionOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value x();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr op_names();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class _UnaryOpsCompositionOp : public ::mlir::Op<_UnaryOpsCompositionOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::OneOperand, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::TF::SameOperandsAndResultTypeResolveRef, ::mlir::DerivedAttributeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = _UnaryOpsCompositionOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("op_names"), ::llvm::StringRef("T")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier op_namesAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier op_namesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier TAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier TAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf._UnaryOpsComposition");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value x();
  ::mlir::MutableOperandRange xMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value y();
  ::mlir::ArrayAttr op_namesAttr();
  ::mlir::ArrayAttr op_names();
  ::mlir::Type T();
  static bool isDerivedAttribute(::llvm::StringRef name);
  ::mlir::DictionaryAttr materializeDerivedAttributes();
  void op_namesAttr(::mlir::ArrayAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type y, ::mlir::Value x, ::mlir::ArrayAttr op_names);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value x, ::mlir::ArrayAttr op_names);
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
} // namespace TF
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::TF::_UnaryOpsCompositionOp)

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_XlaHostComputeMlirOp declarations
//===----------------------------------------------------------------------===//

class _XlaHostComputeMlirOpAdaptor {
public:
  _XlaHostComputeMlirOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  _XlaHostComputeMlirOpAdaptor(_XlaHostComputeMlirOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr send_key();
  ::mlir::StringAttr recv_key();
  ::mlir::IntegerAttr tpu_core();
  ::mlir::StringAttr host_mlir_module();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class _XlaHostComputeMlirOp : public ::mlir::Op<_XlaHostComputeMlirOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::DerivedAttributeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = _XlaHostComputeMlirOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("send_key"), ::llvm::StringRef("recv_key"), ::llvm::StringRef("tpu_core"), ::llvm::StringRef("host_mlir_module"), ::llvm::StringRef("Tinputs"), ::llvm::StringRef("Toutputs")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier send_keyAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier send_keyAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier recv_keyAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier recv_keyAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier tpu_coreAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier tpu_coreAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  ::mlir::Identifier host_mlir_moduleAttrName() {
    return getAttributeNameForIndex(3);
  }
  static ::mlir::Identifier host_mlir_moduleAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }
  ::mlir::Identifier TinputsAttrName() {
    return getAttributeNameForIndex(4);
  }
  static ::mlir::Identifier TinputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }
  ::mlir::Identifier ToutputsAttrName() {
    return getAttributeNameForIndex(5);
  }
  static ::mlir::Identifier ToutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf._XlaHostComputeMlir");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::MutableOperandRange inputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range outputs();
  ::mlir::StringAttr send_keyAttr();
  ::llvm::StringRef send_key();
  ::mlir::StringAttr recv_keyAttr();
  ::llvm::StringRef recv_key();
  ::mlir::IntegerAttr tpu_coreAttr();
  uint64_t tpu_core();
  ::mlir::StringAttr host_mlir_moduleAttr();
  ::llvm::StringRef host_mlir_module();
  mlir::OperandElementTypeRange Tinputs();
  mlir::ResultElementTypeRange Toutputs();
  static bool isDerivedAttribute(::llvm::StringRef name);
  ::mlir::DictionaryAttr materializeDerivedAttributes();
  void send_keyAttr(::mlir::StringAttr attr);
  void recv_keyAttr(::mlir::StringAttr attr);
  void tpu_coreAttr(::mlir::IntegerAttr attr);
  void host_mlir_moduleAttr(::mlir::StringAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outputs, ::mlir::ValueRange inputs, ::mlir::StringAttr send_key, ::mlir::StringAttr recv_key, ::mlir::IntegerAttr tpu_core, ::mlir::StringAttr host_mlir_module);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outputs, ::mlir::ValueRange inputs, ::llvm::StringRef send_key, ::llvm::StringRef recv_key, uint64_t tpu_core, ::llvm::StringRef host_mlir_module);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();

    FuncOp GetHostFunc(mlir::OwningModuleRef* mlir_module);
  

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 6 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace TF
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::TF::_XlaHostComputeMlirOp)

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_XlaRecvAtHostOp declarations
//===----------------------------------------------------------------------===//

class _XlaRecvAtHostOpAdaptor {
public:
  _XlaRecvAtHostOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  _XlaRecvAtHostOpAdaptor(_XlaRecvAtHostOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value dynamic_key();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr key();
  ::mlir::IntegerAttr device_ordinal();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class _XlaRecvAtHostOp : public ::mlir::Op<_XlaRecvAtHostOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::OneOperand, ::mlir::DerivedAttributeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = _XlaRecvAtHostOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("key"), ::llvm::StringRef("device_ordinal"), ::llvm::StringRef("Toutputs")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier keyAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier keyAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier device_ordinalAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier device_ordinalAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier ToutputsAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier ToutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf._XlaRecvAtHost");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value dynamic_key();
  ::mlir::MutableOperandRange dynamic_keyMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range outputs();
  ::mlir::StringAttr keyAttr();
  ::llvm::StringRef key();
  ::mlir::IntegerAttr device_ordinalAttr();
  uint64_t device_ordinal();
  mlir::ResultElementTypeRange Toutputs();
  static bool isDerivedAttribute(::llvm::StringRef name);
  ::mlir::DictionaryAttr materializeDerivedAttributes();
  void keyAttr(::mlir::StringAttr attr);
  void device_ordinalAttr(::mlir::IntegerAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outputs, ::mlir::Value dynamic_key, ::mlir::StringAttr key, ::mlir::IntegerAttr device_ordinal);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outputs, ::mlir::Value dynamic_key, ::llvm::StringRef key, uint64_t device_ordinal);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 3 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace TF
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::TF::_XlaRecvAtHostOp)

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_XlaRecvAtHostV2Op declarations
//===----------------------------------------------------------------------===//

class _XlaRecvAtHostV2OpAdaptor {
public:
  _XlaRecvAtHostV2OpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  _XlaRecvAtHostV2OpAdaptor(_XlaRecvAtHostV2Op&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value dynamic_key();
  ::mlir::Value device_ordinal();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr key();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class _XlaRecvAtHostV2Op : public ::mlir::Op<_XlaRecvAtHostV2Op, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::DerivedAttributeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = _XlaRecvAtHostV2OpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("key"), ::llvm::StringRef("Toutputs")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier keyAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier keyAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier ToutputsAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier ToutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf._XlaRecvAtHostV2");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value dynamic_key();
  ::mlir::Value device_ordinal();
  ::mlir::MutableOperandRange dynamic_keyMutable();
  ::mlir::MutableOperandRange device_ordinalMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range outputs();
  ::mlir::StringAttr keyAttr();
  ::llvm::StringRef key();
  mlir::ResultElementTypeRange Toutputs();
  static bool isDerivedAttribute(::llvm::StringRef name);
  ::mlir::DictionaryAttr materializeDerivedAttributes();
  void keyAttr(::mlir::StringAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outputs, ::mlir::Value dynamic_key, ::mlir::Value device_ordinal, ::mlir::StringAttr key);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outputs, ::mlir::Value dynamic_key, ::mlir::Value device_ordinal, ::llvm::StringRef key);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 2 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace TF
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::TF::_XlaRecvAtHostV2Op)

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_XlaSendFromHostOp declarations
//===----------------------------------------------------------------------===//

class _XlaSendFromHostOpAdaptor {
public:
  _XlaSendFromHostOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  _XlaSendFromHostOpAdaptor(_XlaSendFromHostOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::Value dynamic_key();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr key();
  ::mlir::IntegerAttr device_ordinal();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class _XlaSendFromHostOp : public ::mlir::Op<_XlaSendFromHostOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::DerivedAttributeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = _XlaSendFromHostOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("key"), ::llvm::StringRef("device_ordinal"), ::llvm::StringRef("Tinputs")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier keyAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier keyAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier device_ordinalAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier device_ordinalAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier TinputsAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier TinputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf._XlaSendFromHost");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Value dynamic_key();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange dynamic_keyMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::StringAttr keyAttr();
  ::llvm::StringRef key();
  ::mlir::IntegerAttr device_ordinalAttr();
  uint64_t device_ordinal();
  mlir::OperandElementTypeRange Tinputs();
  static bool isDerivedAttribute(::llvm::StringRef name);
  ::mlir::DictionaryAttr materializeDerivedAttributes();
  void keyAttr(::mlir::StringAttr attr);
  void device_ordinalAttr(::mlir::IntegerAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange inputs, ::mlir::Value dynamic_key, ::mlir::StringAttr key, ::mlir::IntegerAttr device_ordinal);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange inputs, ::mlir::Value dynamic_key, ::mlir::StringAttr key, ::mlir::IntegerAttr device_ordinal);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange inputs, ::mlir::Value dynamic_key, ::llvm::StringRef key, uint64_t device_ordinal);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange inputs, ::mlir::Value dynamic_key, ::llvm::StringRef key, uint64_t device_ordinal);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 3 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace TF
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::TF::_XlaSendFromHostOp)

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_XlaSendFromHostV2Op declarations
//===----------------------------------------------------------------------===//

class _XlaSendFromHostV2OpAdaptor {
public:
  _XlaSendFromHostV2OpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  _XlaSendFromHostV2OpAdaptor(_XlaSendFromHostV2Op&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::Value dynamic_key();
  ::mlir::Value device_ordinal();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr key();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class _XlaSendFromHostV2Op : public ::mlir::Op<_XlaSendFromHostV2Op, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::DerivedAttributeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = _XlaSendFromHostV2OpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("key"), ::llvm::StringRef("Tinputs")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier keyAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier keyAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier TinputsAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier TinputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf._XlaSendFromHostV2");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Value dynamic_key();
  ::mlir::Value device_ordinal();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange dynamic_keyMutable();
  ::mlir::MutableOperandRange device_ordinalMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::StringAttr keyAttr();
  ::llvm::StringRef key();
  mlir::OperandElementTypeRange Tinputs();
  static bool isDerivedAttribute(::llvm::StringRef name);
  ::mlir::DictionaryAttr materializeDerivedAttributes();
  void keyAttr(::mlir::StringAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange inputs, ::mlir::Value dynamic_key, ::mlir::Value device_ordinal, ::mlir::StringAttr key);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange inputs, ::mlir::Value dynamic_key, ::mlir::Value device_ordinal, ::mlir::StringAttr key);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange inputs, ::mlir::Value dynamic_key, ::mlir::Value device_ordinal, ::llvm::StringRef key);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange inputs, ::mlir::Value dynamic_key, ::mlir::Value device_ordinal, ::llvm::StringRef key);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 2 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace TF
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::TF::_XlaSendFromHostV2Op)


#endif  // GET_OP_CLASSES

