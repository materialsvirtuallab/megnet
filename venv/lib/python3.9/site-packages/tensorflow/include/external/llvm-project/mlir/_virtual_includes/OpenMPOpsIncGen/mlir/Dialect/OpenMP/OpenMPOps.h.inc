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
namespace omp {
class BarrierOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class CriticalDeclareOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class CriticalOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class FlushOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class MasterOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class ParallelOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class ReductionDeclareOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class ReductionOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class TargetOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class TaskwaitOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class TaskyieldOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class TerminatorOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class WsLoopOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class YieldOp;
} // namespace omp
} // namespace mlir
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::BarrierOp declarations
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
    return ::llvm::StringLiteral("omp.barrier");
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
} // namespace omp
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::BarrierOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::CriticalDeclareOp declarations
//===----------------------------------------------------------------------===//

class CriticalDeclareOpAdaptor {
public:
  CriticalDeclareOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  CriticalDeclareOpAdaptor(CriticalDeclareOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr sym_name();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class CriticalDeclareOp : public ::mlir::Op<CriticalDeclareOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::SymbolOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CriticalDeclareOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("sym_name")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier sym_nameAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier sym_nameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.critical.declare");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::StringAttr sym_nameAttr();
  ::llvm::StringRef sym_name();
  void sym_nameAttr(::mlir::StringAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr sym_name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr sym_name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef sym_name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef sym_name);
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
} // namespace omp
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::CriticalDeclareOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::CriticalOp declarations
//===----------------------------------------------------------------------===//

class CriticalOpAdaptor {
public:
  CriticalOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  CriticalOpAdaptor(CriticalOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::FlatSymbolRefAttr name();
  ::mlir::omp::SyncHintKindAttr hint();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class CriticalOp : public ::mlir::Op<CriticalOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CriticalOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("name"), ::llvm::StringRef("hint")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier nameAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier nameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier hintAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier hintAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.critical");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &region();
  ::mlir::FlatSymbolRefAttr nameAttr();
  ::llvm::Optional< ::llvm::StringRef > name();
  ::mlir::omp::SyncHintKindAttr hintAttr();
  ::llvm::Optional<::mlir::omp::SyncHintKind> hint();
  void nameAttr(::mlir::FlatSymbolRefAttr attr);
  void hintAttr(::mlir::omp::SyncHintKindAttr attr);
  ::mlir::Attribute removeNameAttr();
  ::mlir::Attribute removeHintAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::FlatSymbolRefAttr name, /*optional*/::mlir::omp::SyncHintKindAttr hint);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::FlatSymbolRefAttr name, /*optional*/::mlir::omp::SyncHintKindAttr hint);
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
} // namespace omp
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::CriticalOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::FlushOp declarations
//===----------------------------------------------------------------------===//

class FlushOpAdaptor {
public:
  FlushOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  FlushOpAdaptor(FlushOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange varList();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class FlushOp : public ::mlir::Op<FlushOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = FlushOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.flush");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range varList();
  ::mlir::MutableOperandRange varListMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange varList);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
};
} // namespace omp
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::FlushOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::MasterOp declarations
//===----------------------------------------------------------------------===//

class MasterOpAdaptor {
public:
  MasterOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  MasterOpAdaptor(MasterOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class MasterOp : public ::mlir::Op<MasterOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = MasterOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.master");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
};
} // namespace omp
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::MasterOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::ParallelOp declarations
//===----------------------------------------------------------------------===//

class ParallelOpAdaptor {
public:
  ParallelOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  ParallelOpAdaptor(ParallelOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value if_expr_var();
  ::mlir::Value num_threads_var();
  ::mlir::ValueRange private_vars();
  ::mlir::ValueRange firstprivate_vars();
  ::mlir::ValueRange shared_vars();
  ::mlir::ValueRange copyin_vars();
  ::mlir::ValueRange allocate_vars();
  ::mlir::ValueRange allocators_vars();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr default_val();
  ::mlir::StringAttr proc_bind_val();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ParallelOp : public ::mlir::Op<ParallelOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ParallelOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("default_val"), ::llvm::StringRef("proc_bind_val"), ::llvm::StringRef("operand_segment_sizes")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier default_valAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier default_valAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier proc_bind_valAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier proc_bind_valAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.parallel");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value if_expr_var();
  ::mlir::Value num_threads_var();
  ::mlir::Operation::operand_range private_vars();
  ::mlir::Operation::operand_range firstprivate_vars();
  ::mlir::Operation::operand_range shared_vars();
  ::mlir::Operation::operand_range copyin_vars();
  ::mlir::Operation::operand_range allocate_vars();
  ::mlir::Operation::operand_range allocators_vars();
  ::mlir::MutableOperandRange if_expr_varMutable();
  ::mlir::MutableOperandRange num_threads_varMutable();
  ::mlir::MutableOperandRange private_varsMutable();
  ::mlir::MutableOperandRange firstprivate_varsMutable();
  ::mlir::MutableOperandRange shared_varsMutable();
  ::mlir::MutableOperandRange copyin_varsMutable();
  ::mlir::MutableOperandRange allocate_varsMutable();
  ::mlir::MutableOperandRange allocators_varsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &region();
  ::mlir::StringAttr default_valAttr();
  ::llvm::Optional< ::llvm::StringRef > default_val();
  ::mlir::StringAttr proc_bind_valAttr();
  ::llvm::Optional< ::llvm::StringRef > proc_bind_val();
  void default_valAttr(::mlir::StringAttr attr);
  void proc_bind_valAttr(::mlir::StringAttr attr);
  ::mlir::Attribute removeDefault_valAttr();
  ::mlir::Attribute removeProc_bind_valAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value if_expr_var, /*optional*/::mlir::Value num_threads_var, /*optional*/::mlir::StringAttr default_val, ::mlir::ValueRange private_vars, ::mlir::ValueRange firstprivate_vars, ::mlir::ValueRange shared_vars, ::mlir::ValueRange copyin_vars, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/::mlir::StringAttr proc_bind_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value if_expr_var, /*optional*/::mlir::Value num_threads_var, /*optional*/::mlir::StringAttr default_val, ::mlir::ValueRange private_vars, ::mlir::ValueRange firstprivate_vars, ::mlir::ValueRange shared_vars, ::mlir::ValueRange copyin_vars, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/::mlir::StringAttr proc_bind_val);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
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
} // namespace omp
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::ParallelOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::ReductionDeclareOp declarations
//===----------------------------------------------------------------------===//

class ReductionDeclareOpAdaptor {
public:
  ReductionDeclareOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ReductionDeclareOpAdaptor(ReductionDeclareOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr sym_name();
  ::mlir::TypeAttr type();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &initializerRegion();
  ::mlir::Region &reductionRegion();
  ::mlir::Region &atomicReductionRegion();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ReductionDeclareOp : public ::mlir::Op<ReductionDeclareOp, ::mlir::OpTrait::NRegions<3>::Impl, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::SymbolOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ReductionDeclareOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("sym_name"), ::llvm::StringRef("type")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier sym_nameAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier sym_nameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier typeAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier typeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.reduction.declare");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &initializerRegion();
  ::mlir::Region &reductionRegion();
  ::mlir::Region &atomicReductionRegion();
  ::mlir::StringAttr sym_nameAttr();
  ::llvm::StringRef sym_name();
  ::mlir::TypeAttr typeAttr();
  ::mlir::Type type();
  void sym_nameAttr(::mlir::StringAttr attr);
  void typeAttr(::mlir::TypeAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr sym_name, ::mlir::TypeAttr type);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr sym_name, ::mlir::TypeAttr type);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef sym_name, ::mlir::Type type);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef sym_name, ::mlir::Type type);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);

    PointerLikeType getAccumulatorType() {
      if (atomicReductionRegion().empty())
        return {};

      return atomicReductionRegion().front().getArgument(0).getType();
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
} // namespace omp
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::ReductionDeclareOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::ReductionOp declarations
//===----------------------------------------------------------------------===//

class ReductionOpAdaptor {
public:
  ReductionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ReductionOpAdaptor(ReductionOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value operand();
  ::mlir::Value accumulator();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ReductionOp : public ::mlir::Op<ReductionOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<2>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ReductionOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.reduction");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value operand();
  ::mlir::Value accumulator();
  ::mlir::MutableOperandRange operandMutable();
  ::mlir::MutableOperandRange accumulatorMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand, ::mlir::Value accumulator);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand, ::mlir::Value accumulator);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
};
} // namespace omp
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::ReductionOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::TargetOp declarations
//===----------------------------------------------------------------------===//

class TargetOpAdaptor {
public:
  TargetOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  TargetOpAdaptor(TargetOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value if_expr();
  ::mlir::Value device();
  ::mlir::Value thread_limit();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr nowait();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class TargetOp : public ::mlir::Op<TargetOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TargetOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("nowait"), ::llvm::StringRef("operand_segment_sizes")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier nowaitAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier nowaitAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.target");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value if_expr();
  ::mlir::Value device();
  ::mlir::Value thread_limit();
  ::mlir::MutableOperandRange if_exprMutable();
  ::mlir::MutableOperandRange deviceMutable();
  ::mlir::MutableOperandRange thread_limitMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &region();
  ::mlir::UnitAttr nowaitAttr();
  bool nowait();
  void nowaitAttr(::mlir::UnitAttr attr);
  ::mlir::Attribute removeNowaitAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, /*optional*/::mlir::Value thread_limit, /*optional*/::mlir::UnitAttr nowait);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, /*optional*/::mlir::Value thread_limit, /*optional*/::mlir::UnitAttr nowait);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, /*optional*/::mlir::Value thread_limit, /*optional*/bool nowait);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, /*optional*/::mlir::Value thread_limit, /*optional*/bool nowait);
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
} // namespace omp
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::TargetOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::TaskwaitOp declarations
//===----------------------------------------------------------------------===//

class TaskwaitOpAdaptor {
public:
  TaskwaitOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  TaskwaitOpAdaptor(TaskwaitOp&op);
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
class TaskwaitOp : public ::mlir::Op<TaskwaitOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TaskwaitOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.taskwait");
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
} // namespace omp
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::TaskwaitOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::TaskyieldOp declarations
//===----------------------------------------------------------------------===//

class TaskyieldOpAdaptor {
public:
  TaskyieldOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  TaskyieldOpAdaptor(TaskyieldOp&op);
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
class TaskyieldOp : public ::mlir::Op<TaskyieldOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TaskyieldOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.taskyield");
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
} // namespace omp
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::TaskyieldOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::TerminatorOp declarations
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
class TerminatorOp : public ::mlir::Op<TerminatorOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::IsTerminator> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TerminatorOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.terminator");
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
} // namespace omp
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::TerminatorOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::WsLoopOp declarations
//===----------------------------------------------------------------------===//

class WsLoopOpAdaptor {
public:
  WsLoopOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  WsLoopOpAdaptor(WsLoopOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange lowerBound();
  ::mlir::ValueRange upperBound();
  ::mlir::ValueRange step();
  ::mlir::ValueRange private_vars();
  ::mlir::ValueRange firstprivate_vars();
  ::mlir::ValueRange lastprivate_vars();
  ::mlir::ValueRange linear_vars();
  ::mlir::ValueRange linear_step_vars();
  ::mlir::ValueRange reduction_vars();
  ::mlir::Value schedule_chunk_var();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr reductions();
  ::mlir::StringAttr schedule_val();
  ::mlir::IntegerAttr collapse_val();
  ::mlir::UnitAttr nowait();
  ::mlir::IntegerAttr ordered_val();
  ::mlir::StringAttr order_val();
  ::mlir::UnitAttr inclusive();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class WsLoopOp : public ::mlir::Op<WsLoopOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = WsLoopOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("reductions"), ::llvm::StringRef("schedule_val"), ::llvm::StringRef("collapse_val"), ::llvm::StringRef("nowait"), ::llvm::StringRef("ordered_val"), ::llvm::StringRef("order_val"), ::llvm::StringRef("inclusive"), ::llvm::StringRef("operand_segment_sizes")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier reductionsAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier reductionsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  ::mlir::Identifier schedule_valAttrName() {
    return getAttributeNameForIndex(1);
  }
  static ::mlir::Identifier schedule_valAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }
  ::mlir::Identifier collapse_valAttrName() {
    return getAttributeNameForIndex(2);
  }
  static ::mlir::Identifier collapse_valAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }
  ::mlir::Identifier nowaitAttrName() {
    return getAttributeNameForIndex(3);
  }
  static ::mlir::Identifier nowaitAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }
  ::mlir::Identifier ordered_valAttrName() {
    return getAttributeNameForIndex(4);
  }
  static ::mlir::Identifier ordered_valAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }
  ::mlir::Identifier order_valAttrName() {
    return getAttributeNameForIndex(5);
  }
  static ::mlir::Identifier order_valAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }
  ::mlir::Identifier inclusiveAttrName() {
    return getAttributeNameForIndex(6);
  }
  static ::mlir::Identifier inclusiveAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }
  ::mlir::Identifier operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(7);
  }
  static ::mlir::Identifier operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 7);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.wsloop");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range lowerBound();
  ::mlir::Operation::operand_range upperBound();
  ::mlir::Operation::operand_range step();
  ::mlir::Operation::operand_range private_vars();
  ::mlir::Operation::operand_range firstprivate_vars();
  ::mlir::Operation::operand_range lastprivate_vars();
  ::mlir::Operation::operand_range linear_vars();
  ::mlir::Operation::operand_range linear_step_vars();
  ::mlir::Operation::operand_range reduction_vars();
  ::mlir::Value schedule_chunk_var();
  ::mlir::MutableOperandRange lowerBoundMutable();
  ::mlir::MutableOperandRange upperBoundMutable();
  ::mlir::MutableOperandRange stepMutable();
  ::mlir::MutableOperandRange private_varsMutable();
  ::mlir::MutableOperandRange firstprivate_varsMutable();
  ::mlir::MutableOperandRange lastprivate_varsMutable();
  ::mlir::MutableOperandRange linear_varsMutable();
  ::mlir::MutableOperandRange linear_step_varsMutable();
  ::mlir::MutableOperandRange reduction_varsMutable();
  ::mlir::MutableOperandRange schedule_chunk_varMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &region();
  ::mlir::ArrayAttr reductionsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > reductions();
  ::mlir::StringAttr schedule_valAttr();
  ::llvm::Optional< ::llvm::StringRef > schedule_val();
  ::mlir::IntegerAttr collapse_valAttr();
  ::llvm::Optional<uint64_t> collapse_val();
  ::mlir::UnitAttr nowaitAttr();
  bool nowait();
  ::mlir::IntegerAttr ordered_valAttr();
  ::llvm::Optional<uint64_t> ordered_val();
  ::mlir::StringAttr order_valAttr();
  ::llvm::Optional< ::llvm::StringRef > order_val();
  ::mlir::UnitAttr inclusiveAttr();
  bool inclusive();
  void reductionsAttr(::mlir::ArrayAttr attr);
  void schedule_valAttr(::mlir::StringAttr attr);
  void collapse_valAttr(::mlir::IntegerAttr attr);
  void nowaitAttr(::mlir::UnitAttr attr);
  void ordered_valAttr(::mlir::IntegerAttr attr);
  void order_valAttr(::mlir::StringAttr attr);
  void inclusiveAttr(::mlir::UnitAttr attr);
  ::mlir::Attribute removeReductionsAttr();
  ::mlir::Attribute removeSchedule_valAttr();
  ::mlir::Attribute removeCollapse_valAttr();
  ::mlir::Attribute removeNowaitAttr();
  ::mlir::Attribute removeOrdered_valAttr();
  ::mlir::Attribute removeOrder_valAttr();
  ::mlir::Attribute removeInclusiveAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange lowerBound, ValueRange upperBound, ValueRange step, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTypes, ValueRange lowerBound, ValueRange upperBound, ValueRange step, ValueRange privateVars, ValueRange firstprivateVars, ValueRange lastprivate_vars, ValueRange linear_vars, ValueRange linear_step_vars, ValueRange reduction_vars, StringAttr schedule_val, Value schedule_chunk_var, IntegerAttr collapse_val, UnitAttr nowait, IntegerAttr ordered_val, StringAttr order_val, UnitAttr inclusive, bool buildBody = true);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();

    /// Returns the number of loops in the workshape loop nest.
    unsigned getNumLoops() { return lowerBound().size(); }

    /// Returns the number of reduction variables.
    unsigned getNumReductionVars() { return reduction_vars().size(); }
  

private:
  ::mlir::Identifier getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }
  static ::mlir::Identifier getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
  assert(index < 8 && "invalid attribute index");
  return name.getAbstractOperation()->getAttributeNames()[index];
  }
};
} // namespace omp
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::WsLoopOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::YieldOp declarations
//===----------------------------------------------------------------------===//

class YieldOpAdaptor {
public:
  YieldOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  YieldOpAdaptor(YieldOp&op);
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
class YieldOp : public ::mlir::Op<YieldOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::ReturnLike, ::mlir::OpTrait::IsTerminator, ::mlir::OpTrait::HasParent<WsLoopOp, ReductionDeclareOp>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = YieldOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.yield");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range results();
  ::mlir::MutableOperandRange resultsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange results);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
};
} // namespace omp
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::YieldOp)


#endif  // GET_OP_CLASSES

