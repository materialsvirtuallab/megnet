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
namespace disc_ral {
class DispatchOp;
} // namespace disc_ral
} // namespace mlir
namespace mlir {
namespace disc_ral {
class RecvInputOp;
} // namespace disc_ral
} // namespace mlir
namespace mlir {
namespace disc_ral {
class SendOutputOp;
} // namespace disc_ral
} // namespace mlir
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace disc_ral {

//===----------------------------------------------------------------------===//
// ::mlir::disc_ral::DispatchOp declarations
//===----------------------------------------------------------------------===//

class DispatchOpAdaptor {
public:
  DispatchOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  DispatchOpAdaptor(DispatchOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value ctx();
  ::mlir::ValueRange args();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr call_target_name();
  ::mlir::BoolAttr has_side_effect();
  ::mlir::StringAttr backend_config();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class DispatchOp : public ::mlir::Op<DispatchOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<1>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = DispatchOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("call_target_name"), ::llvm::StringRef("has_side_effect"), ::llvm::StringRef("backend_config")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier call_target_nameAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier call_target_nameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier has_side_effectAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier has_side_effectAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier backend_configAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier backend_configAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("disc_ral.dispatch");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value ctx();
  ::mlir::Operation::operand_range args();
  ::mlir::MutableOperandRange ctxMutable();
  ::mlir::MutableOperandRange argsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::StringAttr call_target_nameAttr();
  ::llvm::StringRef call_target_name();
  ::mlir::BoolAttr has_side_effectAttr();
  bool has_side_effect();
  ::mlir::StringAttr backend_configAttr();
  ::llvm::StringRef backend_config();
  void call_target_nameAttr(::mlir::StringAttr attr);
  void has_side_effectAttr(::mlir::BoolAttr attr);
  void backend_configAttr(::mlir::StringAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultType0, ::mlir::Value ctx, ::mlir::ValueRange args, ::mlir::StringAttr call_target_name, ::mlir::BoolAttr has_side_effect, ::mlir::StringAttr backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultType0, ::mlir::Value ctx, ::mlir::ValueRange args, ::llvm::StringRef call_target_name, bool has_side_effect, ::llvm::StringRef backend_config);
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
} // namespace disc_ral
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::disc_ral::DispatchOp)

namespace mlir {
namespace disc_ral {

//===----------------------------------------------------------------------===//
// ::mlir::disc_ral::RecvInputOp declarations
//===----------------------------------------------------------------------===//

class RecvInputOpAdaptor {
public:
  RecvInputOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  RecvInputOpAdaptor(RecvInputOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value ctx();
  ::mlir::Value input_idx();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class RecvInputOp : public ::mlir::Op<RecvInputOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::MemRefType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<2>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = RecvInputOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("disc_ral.recv_input");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value ctx();
  ::mlir::Value input_idx();
  ::mlir::MutableOperandRange ctxMutable();
  ::mlir::MutableOperandRange input_idxMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value ctx, ::mlir::Value input_idx);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ctx, ::mlir::Value input_idx);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
};
} // namespace disc_ral
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::disc_ral::RecvInputOp)

namespace mlir {
namespace disc_ral {

//===----------------------------------------------------------------------===//
// ::mlir::disc_ral::SendOutputOp declarations
//===----------------------------------------------------------------------===//

class SendOutputOpAdaptor {
public:
  SendOutputOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  SendOutputOpAdaptor(SendOutputOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value ctx();
  ::mlir::Value output_idx();
  ::mlir::Value result();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class SendOutputOp : public ::mlir::Op<SendOutputOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<3>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SendOutputOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("disc_ral.send_output");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value ctx();
  ::mlir::Value output_idx();
  ::mlir::Value result();
  ::mlir::MutableOperandRange ctxMutable();
  ::mlir::MutableOperandRange output_idxMutable();
  ::mlir::MutableOperandRange resultMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value ctx, ::mlir::Value output_idx, ::mlir::Value result);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ctx, ::mlir::Value output_idx, ::mlir::Value result);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
};
} // namespace disc_ral
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::disc_ral::SendOutputOp)


#endif  // GET_OP_CLASSES

