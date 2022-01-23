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
namespace tf_device {
class ClusterFuncOp;
} // namespace tf_device
} // namespace mlir
namespace mlir {
namespace tf_device {
class ClusterOp;
} // namespace tf_device
} // namespace mlir
namespace mlir {
namespace tf_device {
class LaunchFuncOp;
} // namespace tf_device
} // namespace mlir
namespace mlir {
namespace tf_device {
class LaunchOp;
} // namespace tf_device
} // namespace mlir
namespace mlir {
namespace tf_device {
class ParallelExecuteOp;
} // namespace tf_device
} // namespace mlir
namespace mlir {
namespace tf_device {
class ReceiveOp;
} // namespace tf_device
} // namespace mlir
namespace mlir {
namespace tf_device {
class RemoteRunOp;
} // namespace tf_device
} // namespace mlir
namespace mlir {
namespace tf_device {
class ReplicateOp;
} // namespace tf_device
} // namespace mlir
namespace mlir {
namespace tf_device {
class ReturnOp;
} // namespace tf_device
} // namespace mlir
namespace mlir {
namespace tf_device {
class SendOp;
} // namespace tf_device
} // namespace mlir
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace tf_device {

//===----------------------------------------------------------------------===//
// ::mlir::tf_device::ClusterFuncOp declarations
//===----------------------------------------------------------------------===//

class ClusterFuncOpAdaptor {
public:
  ClusterFuncOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ClusterFuncOpAdaptor(ClusterFuncOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange operands();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::FlatSymbolRefAttr func();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ClusterFuncOp : public ::mlir::Op<ClusterFuncOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ClusterFuncOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("func")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier funcAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier funcAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf_device.cluster_func");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range operands();
  ::mlir::MutableOperandRange operandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range results();
  ::mlir::FlatSymbolRefAttr funcAttr();
  ::llvm::StringRef func();
  void funcAttr(::mlir::FlatSymbolRefAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::FlatSymbolRefAttr func, ::mlir::ValueRange operands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::llvm::StringRef func, ::mlir::ValueRange operands);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();

    // returns the function that this operation will launch.
    FuncOp getFunc() {
      return SymbolTable::lookupNearestSymbolFrom<FuncOp>(*this, funcAttr());
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
} // namespace tf_device
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::tf_device::ClusterFuncOp)

namespace mlir {
namespace tf_device {

//===----------------------------------------------------------------------===//
// ::mlir::tf_device::ClusterOp declarations
//===----------------------------------------------------------------------===//

class ClusterOpAdaptor {
public:
  ClusterOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ClusterOpAdaptor(ClusterOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr policy();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &body();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ClusterOp : public ::mlir::Op<ClusterOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<ReturnOp>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ClusterOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("policy")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier policyAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier policyAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf_device.cluster");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range results();
  ::mlir::Region &body();
  ::mlir::StringAttr policyAttr();
  ::llvm::Optional< ::llvm::StringRef > policy();
  void policyAttr(::mlir::StringAttr attr);
  ::mlir::Attribute removePolicyAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTypes);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, /*optional*/::mlir::StringAttr policy);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);

    Block &GetBody() { return getOperation()->getRegion(0).front(); }
  

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 1 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace tf_device
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::tf_device::ClusterOp)

namespace mlir {
namespace tf_device {

//===----------------------------------------------------------------------===//
// ::mlir::tf_device::LaunchFuncOp declarations
//===----------------------------------------------------------------------===//

class LaunchFuncOpAdaptor {
public:
  LaunchFuncOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  LaunchFuncOpAdaptor(LaunchFuncOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange operands();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr device();
  ::mlir::FlatSymbolRefAttr func();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class LaunchFuncOp : public ::mlir::Op<LaunchFuncOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LaunchFuncOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("device"), ::llvm::StringRef("func")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier deviceAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier deviceAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier funcAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier funcAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf_device.launch_func");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range operands();
  ::mlir::MutableOperandRange operandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range results();
  ::mlir::StringAttr deviceAttr();
  ::llvm::StringRef device();
  ::mlir::FlatSymbolRefAttr funcAttr();
  ::llvm::StringRef func();
  void deviceAttr(::mlir::StringAttr attr);
  void funcAttr(::mlir::FlatSymbolRefAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::StringAttr device, ::mlir::FlatSymbolRefAttr func, ::mlir::ValueRange operands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::llvm::StringRef device, ::llvm::StringRef func, ::mlir::ValueRange operands);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();

    StringRef getFunc() { return func(); }
    StringRef getDevice() { return device(); }
  

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 2 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace tf_device
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::tf_device::LaunchFuncOp)

namespace mlir {
namespace tf_device {

//===----------------------------------------------------------------------===//
// ::mlir::tf_device::LaunchOp declarations
//===----------------------------------------------------------------------===//

class LaunchOpAdaptor {
public:
  LaunchOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  LaunchOpAdaptor(LaunchOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr device();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &body();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class LaunchOp : public ::mlir::Op<LaunchOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<ReturnOp>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LaunchOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("device")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier deviceAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier deviceAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf_device.launch");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range results();
  ::mlir::Region &body();
  ::mlir::StringAttr deviceAttr();
  ::llvm::StringRef device();
  void deviceAttr(::mlir::StringAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, StringAttr device, TypeRange result_types);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::StringAttr device);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::llvm::StringRef device);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);

    Block &GetBody() { return getOperation()->getRegion(0).front(); }
    StringRef getDevice() { return device(); }
    bool WrapsSingleOp();
  

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 1 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace tf_device
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::tf_device::LaunchOp)

namespace mlir {
namespace tf_device {

//===----------------------------------------------------------------------===//
// ::mlir::tf_device::ParallelExecuteOp declarations
//===----------------------------------------------------------------------===//

class ParallelExecuteOpAdaptor {
public:
  ParallelExecuteOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ParallelExecuteOpAdaptor(ParallelExecuteOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::RegionRange regions();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ParallelExecuteOp : public ::mlir::Op<ParallelExecuteOp, ::mlir::OpTrait::VariadicRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<ReturnOp>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ParallelExecuteOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf_device.parallel_execute");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range execute_outputs();
  ::mlir::MutableArrayRef<::mlir::Region> regions();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, int num_regions, TypeRange output_types);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange execute_outputs, unsigned regionsCount);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes, unsigned numRegions);
  ::mlir::LogicalResult verify();

    Block& GetRegionBlockWithIndex(unsigned index);
    Operation::result_range GetRegionOutputs(unsigned region_index);

    // Checks if a tf_device.parallel_execute index'th region block wraps a
    // single operation and the single operation results are perfectly forwarded
    // to the region block's return.
    bool RegionWrapsSingleOp(unsigned index);
  
};
} // namespace tf_device
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::tf_device::ParallelExecuteOp)

namespace mlir {
namespace tf_device {

//===----------------------------------------------------------------------===//
// ::mlir::tf_device::ReceiveOp declarations
//===----------------------------------------------------------------------===//

class ReceiveOpAdaptor {
public:
  ReceiveOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ReceiveOpAdaptor(ReceiveOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr key();
  ::mlir::StringAttr src_host();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ReceiveOp : public ::mlir::Op<ReceiveOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ReceiveOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("key"), ::llvm::StringRef("src_host")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier keyAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier keyAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier src_hostAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier src_hostAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf_device.receive");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value result();
  ::mlir::StringAttr keyAttr();
  ::llvm::StringRef key();
  ::mlir::StringAttr src_hostAttr();
  ::llvm::StringRef src_host();
  void keyAttr(::mlir::StringAttr attr);
  void src_hostAttr(::mlir::StringAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::StringAttr key, ::mlir::StringAttr src_host);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr key, ::mlir::StringAttr src_host);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::llvm::StringRef key, ::llvm::StringRef src_host);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef key, ::llvm::StringRef src_host);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 2 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace tf_device
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::tf_device::ReceiveOp)

namespace mlir {
namespace tf_device {

//===----------------------------------------------------------------------===//
// ::mlir::tf_device::RemoteRunOp declarations
//===----------------------------------------------------------------------===//

class RemoteRunOpAdaptor {
public:
  RemoteRunOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  RemoteRunOpAdaptor(RemoteRunOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange callee_args();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr host();
  ::mlir::FlatSymbolRefAttr callee();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class RemoteRunOp : public ::mlir::Op<RemoteRunOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<ReturnOp>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = RemoteRunOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("host"), ::llvm::StringRef("callee")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier hostAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier hostAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier calleeAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier calleeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf_device.remote_run");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range callee_args();
  ::mlir::MutableOperandRange callee_argsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range results();
  ::mlir::StringAttr hostAttr();
  ::llvm::StringRef host();
  ::mlir::FlatSymbolRefAttr calleeAttr();
  ::llvm::StringRef callee();
  void hostAttr(::mlir::StringAttr attr);
  void calleeAttr(::mlir::FlatSymbolRefAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::StringAttr host, ::mlir::FlatSymbolRefAttr callee, ::mlir::ValueRange callee_args);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::llvm::StringRef host, ::llvm::StringRef callee, ::mlir::ValueRange callee_args);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 2 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace tf_device
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::tf_device::RemoteRunOp)

namespace mlir {
namespace tf_device {

//===----------------------------------------------------------------------===//
// ::mlir::tf_device::ReplicateOp declarations
//===----------------------------------------------------------------------===//

class ReplicateOpAdaptor {
public:
  ReplicateOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  ReplicateOpAdaptor(ReplicateOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange replicated_inputs();
  ::mlir::ValueRange packed_inputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr operand_segment_sizes();
  ::mlir::IntegerAttr n();
  ::mlir::DictionaryAttr devices();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &body();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ReplicateOp : public ::mlir::Op<ReplicateOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<ReturnOp>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ReplicateOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("n"), ::llvm::StringRef("devices")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier nAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier nAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier devicesAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier devicesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf_device.replicate");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range replicated_inputs();
  ::mlir::Operation::operand_range packed_inputs();
  ::mlir::MutableOperandRange replicated_inputsMutable();
  ::mlir::MutableOperandRange packed_inputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range replicated_outputs();
  ::mlir::Region &body();
  ::mlir::DenseIntElementsAttr operand_segment_sizesAttr();
  ::mlir::DenseIntElementsAttr operand_segment_sizes();
  ::mlir::IntegerAttr nAttr();
  uint32_t n();
  ::mlir::DictionaryAttr devicesAttr();
  ::llvm::Optional< ::mlir::DictionaryAttr > devices();
  void operand_segment_sizesAttr(::mlir::DenseIntElementsAttr attr);
  void nAttr(::mlir::IntegerAttr attr);
  void devicesAttr(::mlir::DictionaryAttr attr);
  ::mlir::Attribute removeDevicesAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, int n, const llvm::SmallDenseMap<StringRef, llvm::SmallVector<StringRef, 4>>& devices, llvm::ArrayRef<std::pair<ValueRange, Type>> replicated_inputs, ValueRange packed_inputs, TypeRange replica_output_types);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, int n, llvm::Optional<DictionaryAttr> devices, llvm::ArrayRef<std::pair<ValueRange, Type>> replicated_inputs, ValueRange packed_inputs, TypeRange replica_output_types);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange replicated_outputs, ::mlir::ValueRange replicated_inputs, ::mlir::ValueRange packed_inputs, ::mlir::DenseIntElementsAttr operand_segment_sizes, ::mlir::IntegerAttr n, /*optional*/::mlir::DictionaryAttr devices);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange replicated_outputs, ::mlir::ValueRange replicated_inputs, ::mlir::ValueRange packed_inputs, ::mlir::DenseIntElementsAttr operand_segment_sizes, uint32_t n, /*optional*/::mlir::DictionaryAttr devices);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();

    Block &GetBody() { return getOperation()->getRegion(0).front(); }
    unsigned GetNumReplicatedBlockArguments();
    unsigned GetNumPackedBlockArguments();
    llvm::ArrayRef<BlockArgument> GetPackedBlockArguments();
    llvm::ArrayRef<BlockArgument> GetReplicatedBlockArguments();
    bool IsReplicatedBlockArgument(BlockArgument block_arg);
    bool IsPackedBlockArgument(BlockArgument block_arg);
    unsigned GetReplicaOperandIndexForBlockArgument(BlockArgument block_arg, unsigned replica);
    Value GetReplicaOperandForBlockArgument(BlockArgument block_arg, unsigned replica);
    MutableArrayRef<OpOperand> GetOperandsForBlockArgument(BlockArgument block_arg);
    bool WrapsSingleOp();
  

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 3 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace tf_device
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::tf_device::ReplicateOp)

namespace mlir {
namespace tf_device {

//===----------------------------------------------------------------------===//
// ::mlir::tf_device::ReturnOp declarations
//===----------------------------------------------------------------------===//

class ReturnOpAdaptor {
public:
  ReturnOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ReturnOpAdaptor(ReturnOp&op);
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
class ReturnOp : public ::mlir::Op<ReturnOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::ReturnLike, ::mlir::OpTrait::IsTerminator> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ReturnOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf_device.return");
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
} // namespace tf_device
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::tf_device::ReturnOp)

namespace mlir {
namespace tf_device {

//===----------------------------------------------------------------------===//
// ::mlir::tf_device::SendOp declarations
//===----------------------------------------------------------------------===//

class SendOpAdaptor {
public:
  SendOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  SendOpAdaptor(SendOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value value();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr key();
  ::mlir::StringAttr dst_host();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class SendOp : public ::mlir::Op<SendOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::OneOperand> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SendOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("key"), ::llvm::StringRef("dst_host")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier keyAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier keyAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier dst_hostAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier dst_hostAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf_device.send");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value value();
  ::mlir::MutableOperandRange valueMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::StringAttr keyAttr();
  ::llvm::StringRef key();
  ::mlir::StringAttr dst_hostAttr();
  ::llvm::StringRef dst_host();
  void keyAttr(::mlir::StringAttr attr);
  void dst_hostAttr(::mlir::StringAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::StringAttr key, ::mlir::StringAttr dst_host);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::StringAttr key, ::mlir::StringAttr dst_host);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::llvm::StringRef key, ::llvm::StringRef dst_host);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::llvm::StringRef key, ::llvm::StringRef dst_host);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 2 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace tf_device
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::tf_device::SendOp)


#endif  // GET_OP_CLASSES

