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
namespace arm_sve {
class ScalableAddFOp;
} // namespace arm_sve
} // namespace mlir
namespace mlir {
namespace arm_sve {
class ScalableAddIOp;
} // namespace arm_sve
} // namespace mlir
namespace mlir {
namespace arm_sve {
class ScalableCmpFOp;
} // namespace arm_sve
} // namespace mlir
namespace mlir {
namespace arm_sve {
class ScalableCmpIOp;
} // namespace arm_sve
} // namespace mlir
namespace mlir {
namespace arm_sve {
class ScalableDivFOp;
} // namespace arm_sve
} // namespace mlir
namespace mlir {
namespace arm_sve {
class ScalableLoadOp;
} // namespace arm_sve
} // namespace mlir
namespace mlir {
namespace arm_sve {
class ScalableMaskedAddFIntrOp;
} // namespace arm_sve
} // namespace mlir
namespace mlir {
namespace arm_sve {
class ScalableMaskedAddFOp;
} // namespace arm_sve
} // namespace mlir
namespace mlir {
namespace arm_sve {
class ScalableMaskedAddIIntrOp;
} // namespace arm_sve
} // namespace mlir
namespace mlir {
namespace arm_sve {
class ScalableMaskedAddIOp;
} // namespace arm_sve
} // namespace mlir
namespace mlir {
namespace arm_sve {
class ScalableMaskedDivFIntrOp;
} // namespace arm_sve
} // namespace mlir
namespace mlir {
namespace arm_sve {
class ScalableMaskedDivFOp;
} // namespace arm_sve
} // namespace mlir
namespace mlir {
namespace arm_sve {
class ScalableMaskedMulFIntrOp;
} // namespace arm_sve
} // namespace mlir
namespace mlir {
namespace arm_sve {
class ScalableMaskedMulFOp;
} // namespace arm_sve
} // namespace mlir
namespace mlir {
namespace arm_sve {
class ScalableMaskedMulIIntrOp;
} // namespace arm_sve
} // namespace mlir
namespace mlir {
namespace arm_sve {
class ScalableMaskedMulIOp;
} // namespace arm_sve
} // namespace mlir
namespace mlir {
namespace arm_sve {
class ScalableMaskedSDivIIntrOp;
} // namespace arm_sve
} // namespace mlir
namespace mlir {
namespace arm_sve {
class ScalableMaskedSDivIOp;
} // namespace arm_sve
} // namespace mlir
namespace mlir {
namespace arm_sve {
class ScalableMaskedSubFIntrOp;
} // namespace arm_sve
} // namespace mlir
namespace mlir {
namespace arm_sve {
class ScalableMaskedSubFOp;
} // namespace arm_sve
} // namespace mlir
namespace mlir {
namespace arm_sve {
class ScalableMaskedSubIIntrOp;
} // namespace arm_sve
} // namespace mlir
namespace mlir {
namespace arm_sve {
class ScalableMaskedSubIOp;
} // namespace arm_sve
} // namespace mlir
namespace mlir {
namespace arm_sve {
class ScalableMaskedUDivIIntrOp;
} // namespace arm_sve
} // namespace mlir
namespace mlir {
namespace arm_sve {
class ScalableMaskedUDivIOp;
} // namespace arm_sve
} // namespace mlir
namespace mlir {
namespace arm_sve {
class ScalableMulFOp;
} // namespace arm_sve
} // namespace mlir
namespace mlir {
namespace arm_sve {
class ScalableMulIOp;
} // namespace arm_sve
} // namespace mlir
namespace mlir {
namespace arm_sve {
class ScalableSDivIOp;
} // namespace arm_sve
} // namespace mlir
namespace mlir {
namespace arm_sve {
class ScalableStoreOp;
} // namespace arm_sve
} // namespace mlir
namespace mlir {
namespace arm_sve {
class ScalableSubFOp;
} // namespace arm_sve
} // namespace mlir
namespace mlir {
namespace arm_sve {
class ScalableSubIOp;
} // namespace arm_sve
} // namespace mlir
namespace mlir {
namespace arm_sve {
class ScalableUDivIOp;
} // namespace arm_sve
} // namespace mlir
namespace mlir {
namespace arm_sve {
class SdotIntrOp;
} // namespace arm_sve
} // namespace mlir
namespace mlir {
namespace arm_sve {
class SdotOp;
} // namespace arm_sve
} // namespace mlir
namespace mlir {
namespace arm_sve {
class SmmlaIntrOp;
} // namespace arm_sve
} // namespace mlir
namespace mlir {
namespace arm_sve {
class SmmlaOp;
} // namespace arm_sve
} // namespace mlir
namespace mlir {
namespace arm_sve {
class UdotIntrOp;
} // namespace arm_sve
} // namespace mlir
namespace mlir {
namespace arm_sve {
class UdotOp;
} // namespace arm_sve
} // namespace mlir
namespace mlir {
namespace arm_sve {
class UmmlaIntrOp;
} // namespace arm_sve
} // namespace mlir
namespace mlir {
namespace arm_sve {
class UmmlaOp;
} // namespace arm_sve
} // namespace mlir
namespace mlir {
namespace arm_sve {
class VectorScaleIntrOp;
} // namespace arm_sve
} // namespace mlir
namespace mlir {
namespace arm_sve {
class VectorScaleOp;
} // namespace arm_sve
} // namespace mlir
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableAddFOp declarations
//===----------------------------------------------------------------------===//

class ScalableAddFOpAdaptor {
public:
  ScalableAddFOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ScalableAddFOpAdaptor(ScalableAddFOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value src1();
  ::mlir::Value src2();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ScalableAddFOp : public ::mlir::Op<ScalableAddFOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::IsCommutative> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ScalableAddFOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("arm_sve.addf");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value src1();
  ::mlir::Value src2();
  ::mlir::MutableOperandRange src1Mutable();
  ::mlir::MutableOperandRange src2Mutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value dst();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dst, ::mlir::Value src1, ::mlir::Value src2);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src1, ::mlir::Value src2);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
};
} // namespace arm_sve
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableAddFOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableAddIOp declarations
//===----------------------------------------------------------------------===//

class ScalableAddIOpAdaptor {
public:
  ScalableAddIOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ScalableAddIOpAdaptor(ScalableAddIOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value src1();
  ::mlir::Value src2();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ScalableAddIOp : public ::mlir::Op<ScalableAddIOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::IsCommutative> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ScalableAddIOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("arm_sve.addi");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value src1();
  ::mlir::Value src2();
  ::mlir::MutableOperandRange src1Mutable();
  ::mlir::MutableOperandRange src2Mutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value dst();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dst, ::mlir::Value src1, ::mlir::Value src2);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src1, ::mlir::Value src2);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
};
} // namespace arm_sve
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableAddIOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableCmpFOp declarations
//===----------------------------------------------------------------------===//

class ScalableCmpFOpAdaptor {
public:
  ScalableCmpFOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ScalableCmpFOpAdaptor(ScalableCmpFOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value lhs();
  ::mlir::Value rhs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::CmpFPredicateAttr predicate();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ScalableCmpFOp : public ::mlir::Op<ScalableCmpFOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SameTypeOperands> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ScalableCmpFOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("predicate")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier predicateAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier predicateAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("arm_sve.cmpf");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value lhs();
  ::mlir::Value rhs();
  ::mlir::MutableOperandRange lhsMutable();
  ::mlir::MutableOperandRange rhsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value result();
  ::mlir::CmpFPredicateAttr predicateAttr();
  ::mlir::CmpFPredicate predicate();
  void predicateAttr(::mlir::CmpFPredicateAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, CmpFPredicate predicate, Value lhs, Value rhs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::CmpFPredicateAttr predicate, ::mlir::Value lhs, ::mlir::Value rhs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::CmpFPredicateAttr predicate, ::mlir::Value lhs, ::mlir::Value rhs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::CmpFPredicate predicate, ::mlir::Value lhs, ::mlir::Value rhs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::CmpFPredicate predicate, ::mlir::Value lhs, ::mlir::Value rhs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    static StringRef getPredicateAttrName() { return "predicate"; }
    static CmpFPredicate getPredicateByName(StringRef name);

    CmpFPredicate getPredicate() {
      return (CmpFPredicate)(*this)->getAttrOfType<IntegerAttr>(
          getPredicateAttrName()).getInt();
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
} // namespace arm_sve
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableCmpFOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableCmpIOp declarations
//===----------------------------------------------------------------------===//

class ScalableCmpIOpAdaptor {
public:
  ScalableCmpIOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ScalableCmpIOpAdaptor(ScalableCmpIOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value lhs();
  ::mlir::Value rhs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::CmpIPredicateAttr predicate();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ScalableCmpIOp : public ::mlir::Op<ScalableCmpIOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SameTypeOperands> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ScalableCmpIOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("predicate")};
  return ::llvm::makeArrayRef(attrNames);
  }
  ::mlir::Identifier predicateAttrName() {
    return getAttributeNameForIndex(0);
  }
  static ::mlir::Identifier predicateAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("arm_sve.cmpi");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value lhs();
  ::mlir::Value rhs();
  ::mlir::MutableOperandRange lhsMutable();
  ::mlir::MutableOperandRange rhsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value result();
  ::mlir::CmpIPredicateAttr predicateAttr();
  ::mlir::CmpIPredicate predicate();
  void predicateAttr(::mlir::CmpIPredicateAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, CmpIPredicate predicate, Value lhs, Value rhs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::CmpIPredicateAttr predicate, ::mlir::Value lhs, ::mlir::Value rhs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::CmpIPredicateAttr predicate, ::mlir::Value lhs, ::mlir::Value rhs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::CmpIPredicate predicate, ::mlir::Value lhs, ::mlir::Value rhs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::CmpIPredicate predicate, ::mlir::Value lhs, ::mlir::Value rhs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    static StringRef getPredicateAttrName() { return "predicate"; }
    static CmpIPredicate getPredicateByName(StringRef name);

    CmpIPredicate getPredicate() {
      return (CmpIPredicate)(*this)->getAttrOfType<IntegerAttr>(
          getPredicateAttrName()).getInt();
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
} // namespace arm_sve
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableCmpIOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableDivFOp declarations
//===----------------------------------------------------------------------===//

class ScalableDivFOpAdaptor {
public:
  ScalableDivFOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ScalableDivFOpAdaptor(ScalableDivFOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value src1();
  ::mlir::Value src2();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ScalableDivFOp : public ::mlir::Op<ScalableDivFOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<2>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ScalableDivFOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("arm_sve.divf");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value src1();
  ::mlir::Value src2();
  ::mlir::MutableOperandRange src1Mutable();
  ::mlir::MutableOperandRange src2Mutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value dst();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dst, ::mlir::Value src1, ::mlir::Value src2);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src1, ::mlir::Value src2);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
};
} // namespace arm_sve
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableDivFOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableLoadOp declarations
//===----------------------------------------------------------------------===//

class ScalableLoadOpAdaptor {
public:
  ScalableLoadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ScalableLoadOpAdaptor(ScalableLoadOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value base();
  ::mlir::Value index();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ScalableLoadOp : public ::mlir::Op<ScalableLoadOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ScalableLoadOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("arm_sve.load");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value base();
  ::mlir::Value index();
  ::mlir::MutableOperandRange baseMutable();
  ::mlir::MutableOperandRange indexMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value result();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value base, ::mlir::Value index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value base, ::mlir::Value index);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    MemRefType getMemRefType() {
      return base().getType().cast<MemRefType>();
    }
  
};
} // namespace arm_sve
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableLoadOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableMaskedAddFIntrOp declarations
//===----------------------------------------------------------------------===//

class ScalableMaskedAddFIntrOpAdaptor {
public:
  ScalableMaskedAddFIntrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ScalableMaskedAddFIntrOpAdaptor(ScalableMaskedAddFIntrOp&op);
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
class ScalableMaskedAddFIntrOp : public ::mlir::Op<ScalableMaskedAddFIntrOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<3>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ScalableMaskedAddFIntrOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("arm_sve.intr.fadd");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
};
} // namespace arm_sve
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableMaskedAddFIntrOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableMaskedAddFOp declarations
//===----------------------------------------------------------------------===//

class ScalableMaskedAddFOpAdaptor {
public:
  ScalableMaskedAddFOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ScalableMaskedAddFOpAdaptor(ScalableMaskedAddFOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value mask();
  ::mlir::Value src1();
  ::mlir::Value src2();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ScalableMaskedAddFOp : public ::mlir::Op<ScalableMaskedAddFOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::OpTrait::IsCommutative> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ScalableMaskedAddFOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("arm_sve.masked.addf");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value mask();
  ::mlir::Value src1();
  ::mlir::Value src2();
  ::mlir::MutableOperandRange maskMutable();
  ::mlir::MutableOperandRange src1Mutable();
  ::mlir::MutableOperandRange src2Mutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value mask, ::mlir::Value src1, ::mlir::Value src2);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value mask, ::mlir::Value src1, ::mlir::Value src2);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
};
} // namespace arm_sve
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableMaskedAddFOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableMaskedAddIIntrOp declarations
//===----------------------------------------------------------------------===//

class ScalableMaskedAddIIntrOpAdaptor {
public:
  ScalableMaskedAddIIntrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ScalableMaskedAddIIntrOpAdaptor(ScalableMaskedAddIIntrOp&op);
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
class ScalableMaskedAddIIntrOp : public ::mlir::Op<ScalableMaskedAddIIntrOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<3>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ScalableMaskedAddIIntrOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("arm_sve.intr.add");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
};
} // namespace arm_sve
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableMaskedAddIIntrOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableMaskedAddIOp declarations
//===----------------------------------------------------------------------===//

class ScalableMaskedAddIOpAdaptor {
public:
  ScalableMaskedAddIOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ScalableMaskedAddIOpAdaptor(ScalableMaskedAddIOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value mask();
  ::mlir::Value src1();
  ::mlir::Value src2();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ScalableMaskedAddIOp : public ::mlir::Op<ScalableMaskedAddIOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::OpTrait::IsCommutative> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ScalableMaskedAddIOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("arm_sve.masked.addi");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value mask();
  ::mlir::Value src1();
  ::mlir::Value src2();
  ::mlir::MutableOperandRange maskMutable();
  ::mlir::MutableOperandRange src1Mutable();
  ::mlir::MutableOperandRange src2Mutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value mask, ::mlir::Value src1, ::mlir::Value src2);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value mask, ::mlir::Value src1, ::mlir::Value src2);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
};
} // namespace arm_sve
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableMaskedAddIOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableMaskedDivFIntrOp declarations
//===----------------------------------------------------------------------===//

class ScalableMaskedDivFIntrOpAdaptor {
public:
  ScalableMaskedDivFIntrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ScalableMaskedDivFIntrOpAdaptor(ScalableMaskedDivFIntrOp&op);
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
class ScalableMaskedDivFIntrOp : public ::mlir::Op<ScalableMaskedDivFIntrOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<3>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ScalableMaskedDivFIntrOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("arm_sve.intr.fdiv");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
};
} // namespace arm_sve
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableMaskedDivFIntrOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableMaskedDivFOp declarations
//===----------------------------------------------------------------------===//

class ScalableMaskedDivFOpAdaptor {
public:
  ScalableMaskedDivFOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ScalableMaskedDivFOpAdaptor(ScalableMaskedDivFOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value mask();
  ::mlir::Value src1();
  ::mlir::Value src2();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ScalableMaskedDivFOp : public ::mlir::Op<ScalableMaskedDivFOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<3>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ScalableMaskedDivFOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("arm_sve.masked.divf");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value mask();
  ::mlir::Value src1();
  ::mlir::Value src2();
  ::mlir::MutableOperandRange maskMutable();
  ::mlir::MutableOperandRange src1Mutable();
  ::mlir::MutableOperandRange src2Mutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value mask, ::mlir::Value src1, ::mlir::Value src2);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value mask, ::mlir::Value src1, ::mlir::Value src2);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
};
} // namespace arm_sve
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableMaskedDivFOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableMaskedMulFIntrOp declarations
//===----------------------------------------------------------------------===//

class ScalableMaskedMulFIntrOpAdaptor {
public:
  ScalableMaskedMulFIntrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ScalableMaskedMulFIntrOpAdaptor(ScalableMaskedMulFIntrOp&op);
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
class ScalableMaskedMulFIntrOp : public ::mlir::Op<ScalableMaskedMulFIntrOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<3>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ScalableMaskedMulFIntrOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("arm_sve.intr.fmul");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
};
} // namespace arm_sve
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableMaskedMulFIntrOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableMaskedMulFOp declarations
//===----------------------------------------------------------------------===//

class ScalableMaskedMulFOpAdaptor {
public:
  ScalableMaskedMulFOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ScalableMaskedMulFOpAdaptor(ScalableMaskedMulFOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value mask();
  ::mlir::Value src1();
  ::mlir::Value src2();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ScalableMaskedMulFOp : public ::mlir::Op<ScalableMaskedMulFOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::OpTrait::IsCommutative> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ScalableMaskedMulFOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("arm_sve.masked.mulf");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value mask();
  ::mlir::Value src1();
  ::mlir::Value src2();
  ::mlir::MutableOperandRange maskMutable();
  ::mlir::MutableOperandRange src1Mutable();
  ::mlir::MutableOperandRange src2Mutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value mask, ::mlir::Value src1, ::mlir::Value src2);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value mask, ::mlir::Value src1, ::mlir::Value src2);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
};
} // namespace arm_sve
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableMaskedMulFOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableMaskedMulIIntrOp declarations
//===----------------------------------------------------------------------===//

class ScalableMaskedMulIIntrOpAdaptor {
public:
  ScalableMaskedMulIIntrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ScalableMaskedMulIIntrOpAdaptor(ScalableMaskedMulIIntrOp&op);
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
class ScalableMaskedMulIIntrOp : public ::mlir::Op<ScalableMaskedMulIIntrOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<3>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ScalableMaskedMulIIntrOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("arm_sve.intr.mul");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
};
} // namespace arm_sve
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableMaskedMulIIntrOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableMaskedMulIOp declarations
//===----------------------------------------------------------------------===//

class ScalableMaskedMulIOpAdaptor {
public:
  ScalableMaskedMulIOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ScalableMaskedMulIOpAdaptor(ScalableMaskedMulIOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value mask();
  ::mlir::Value src1();
  ::mlir::Value src2();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ScalableMaskedMulIOp : public ::mlir::Op<ScalableMaskedMulIOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::OpTrait::IsCommutative> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ScalableMaskedMulIOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("arm_sve.masked.muli");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value mask();
  ::mlir::Value src1();
  ::mlir::Value src2();
  ::mlir::MutableOperandRange maskMutable();
  ::mlir::MutableOperandRange src1Mutable();
  ::mlir::MutableOperandRange src2Mutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value mask, ::mlir::Value src1, ::mlir::Value src2);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value mask, ::mlir::Value src1, ::mlir::Value src2);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
};
} // namespace arm_sve
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableMaskedMulIOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableMaskedSDivIIntrOp declarations
//===----------------------------------------------------------------------===//

class ScalableMaskedSDivIIntrOpAdaptor {
public:
  ScalableMaskedSDivIIntrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ScalableMaskedSDivIIntrOpAdaptor(ScalableMaskedSDivIIntrOp&op);
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
class ScalableMaskedSDivIIntrOp : public ::mlir::Op<ScalableMaskedSDivIIntrOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<3>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ScalableMaskedSDivIIntrOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("arm_sve.intr.sdiv");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
};
} // namespace arm_sve
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableMaskedSDivIIntrOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableMaskedSDivIOp declarations
//===----------------------------------------------------------------------===//

class ScalableMaskedSDivIOpAdaptor {
public:
  ScalableMaskedSDivIOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ScalableMaskedSDivIOpAdaptor(ScalableMaskedSDivIOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value mask();
  ::mlir::Value src1();
  ::mlir::Value src2();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ScalableMaskedSDivIOp : public ::mlir::Op<ScalableMaskedSDivIOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<3>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ScalableMaskedSDivIOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("arm_sve.masked.divi_signed");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value mask();
  ::mlir::Value src1();
  ::mlir::Value src2();
  ::mlir::MutableOperandRange maskMutable();
  ::mlir::MutableOperandRange src1Mutable();
  ::mlir::MutableOperandRange src2Mutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value mask, ::mlir::Value src1, ::mlir::Value src2);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value mask, ::mlir::Value src1, ::mlir::Value src2);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
};
} // namespace arm_sve
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableMaskedSDivIOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableMaskedSubFIntrOp declarations
//===----------------------------------------------------------------------===//

class ScalableMaskedSubFIntrOpAdaptor {
public:
  ScalableMaskedSubFIntrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ScalableMaskedSubFIntrOpAdaptor(ScalableMaskedSubFIntrOp&op);
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
class ScalableMaskedSubFIntrOp : public ::mlir::Op<ScalableMaskedSubFIntrOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<3>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ScalableMaskedSubFIntrOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("arm_sve.intr.fsub");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
};
} // namespace arm_sve
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableMaskedSubFIntrOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableMaskedSubFOp declarations
//===----------------------------------------------------------------------===//

class ScalableMaskedSubFOpAdaptor {
public:
  ScalableMaskedSubFOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ScalableMaskedSubFOpAdaptor(ScalableMaskedSubFOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value mask();
  ::mlir::Value src1();
  ::mlir::Value src2();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ScalableMaskedSubFOp : public ::mlir::Op<ScalableMaskedSubFOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<3>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ScalableMaskedSubFOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("arm_sve.masked.subf");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value mask();
  ::mlir::Value src1();
  ::mlir::Value src2();
  ::mlir::MutableOperandRange maskMutable();
  ::mlir::MutableOperandRange src1Mutable();
  ::mlir::MutableOperandRange src2Mutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value mask, ::mlir::Value src1, ::mlir::Value src2);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value mask, ::mlir::Value src1, ::mlir::Value src2);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
};
} // namespace arm_sve
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableMaskedSubFOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableMaskedSubIIntrOp declarations
//===----------------------------------------------------------------------===//

class ScalableMaskedSubIIntrOpAdaptor {
public:
  ScalableMaskedSubIIntrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ScalableMaskedSubIIntrOpAdaptor(ScalableMaskedSubIIntrOp&op);
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
class ScalableMaskedSubIIntrOp : public ::mlir::Op<ScalableMaskedSubIIntrOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<3>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ScalableMaskedSubIIntrOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("arm_sve.intr.sub");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
};
} // namespace arm_sve
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableMaskedSubIIntrOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableMaskedSubIOp declarations
//===----------------------------------------------------------------------===//

class ScalableMaskedSubIOpAdaptor {
public:
  ScalableMaskedSubIOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ScalableMaskedSubIOpAdaptor(ScalableMaskedSubIOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value mask();
  ::mlir::Value src1();
  ::mlir::Value src2();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ScalableMaskedSubIOp : public ::mlir::Op<ScalableMaskedSubIOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<3>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ScalableMaskedSubIOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("arm_sve.masked.subi");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value mask();
  ::mlir::Value src1();
  ::mlir::Value src2();
  ::mlir::MutableOperandRange maskMutable();
  ::mlir::MutableOperandRange src1Mutable();
  ::mlir::MutableOperandRange src2Mutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value mask, ::mlir::Value src1, ::mlir::Value src2);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value mask, ::mlir::Value src1, ::mlir::Value src2);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
};
} // namespace arm_sve
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableMaskedSubIOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableMaskedUDivIIntrOp declarations
//===----------------------------------------------------------------------===//

class ScalableMaskedUDivIIntrOpAdaptor {
public:
  ScalableMaskedUDivIIntrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ScalableMaskedUDivIIntrOpAdaptor(ScalableMaskedUDivIIntrOp&op);
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
class ScalableMaskedUDivIIntrOp : public ::mlir::Op<ScalableMaskedUDivIIntrOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<3>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ScalableMaskedUDivIIntrOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("arm_sve.intr.udiv");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
};
} // namespace arm_sve
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableMaskedUDivIIntrOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableMaskedUDivIOp declarations
//===----------------------------------------------------------------------===//

class ScalableMaskedUDivIOpAdaptor {
public:
  ScalableMaskedUDivIOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ScalableMaskedUDivIOpAdaptor(ScalableMaskedUDivIOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value mask();
  ::mlir::Value src1();
  ::mlir::Value src2();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ScalableMaskedUDivIOp : public ::mlir::Op<ScalableMaskedUDivIOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<3>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ScalableMaskedUDivIOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("arm_sve.masked.divi_unsigned");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value mask();
  ::mlir::Value src1();
  ::mlir::Value src2();
  ::mlir::MutableOperandRange maskMutable();
  ::mlir::MutableOperandRange src1Mutable();
  ::mlir::MutableOperandRange src2Mutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value mask, ::mlir::Value src1, ::mlir::Value src2);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value mask, ::mlir::Value src1, ::mlir::Value src2);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
};
} // namespace arm_sve
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableMaskedUDivIOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableMulFOp declarations
//===----------------------------------------------------------------------===//

class ScalableMulFOpAdaptor {
public:
  ScalableMulFOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ScalableMulFOpAdaptor(ScalableMulFOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value src1();
  ::mlir::Value src2();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ScalableMulFOp : public ::mlir::Op<ScalableMulFOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::IsCommutative> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ScalableMulFOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("arm_sve.mulf");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value src1();
  ::mlir::Value src2();
  ::mlir::MutableOperandRange src1Mutable();
  ::mlir::MutableOperandRange src2Mutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value dst();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dst, ::mlir::Value src1, ::mlir::Value src2);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src1, ::mlir::Value src2);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
};
} // namespace arm_sve
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableMulFOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableMulIOp declarations
//===----------------------------------------------------------------------===//

class ScalableMulIOpAdaptor {
public:
  ScalableMulIOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ScalableMulIOpAdaptor(ScalableMulIOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value src1();
  ::mlir::Value src2();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ScalableMulIOp : public ::mlir::Op<ScalableMulIOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::IsCommutative> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ScalableMulIOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("arm_sve.muli");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value src1();
  ::mlir::Value src2();
  ::mlir::MutableOperandRange src1Mutable();
  ::mlir::MutableOperandRange src2Mutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value dst();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dst, ::mlir::Value src1, ::mlir::Value src2);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src1, ::mlir::Value src2);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
};
} // namespace arm_sve
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableMulIOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableSDivIOp declarations
//===----------------------------------------------------------------------===//

class ScalableSDivIOpAdaptor {
public:
  ScalableSDivIOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ScalableSDivIOpAdaptor(ScalableSDivIOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value src1();
  ::mlir::Value src2();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ScalableSDivIOp : public ::mlir::Op<ScalableSDivIOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<2>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ScalableSDivIOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("arm_sve.divi_signed");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value src1();
  ::mlir::Value src2();
  ::mlir::MutableOperandRange src1Mutable();
  ::mlir::MutableOperandRange src2Mutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value dst();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dst, ::mlir::Value src1, ::mlir::Value src2);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src1, ::mlir::Value src2);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
};
} // namespace arm_sve
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableSDivIOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableStoreOp declarations
//===----------------------------------------------------------------------===//

class ScalableStoreOpAdaptor {
public:
  ScalableStoreOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ScalableStoreOpAdaptor(ScalableStoreOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value base();
  ::mlir::Value index();
  ::mlir::Value value();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ScalableStoreOp : public ::mlir::Op<ScalableStoreOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ScalableStoreOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("arm_sve.store");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value base();
  ::mlir::Value index();
  ::mlir::Value value();
  ::mlir::MutableOperandRange baseMutable();
  ::mlir::MutableOperandRange indexMutable();
  ::mlir::MutableOperandRange valueMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value base, ::mlir::Value index, ::mlir::Value value);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value base, ::mlir::Value index, ::mlir::Value value);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);

    MemRefType getMemRefType() {
      return base().getType().cast<MemRefType>();
    }
  
};
} // namespace arm_sve
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableStoreOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableSubFOp declarations
//===----------------------------------------------------------------------===//

class ScalableSubFOpAdaptor {
public:
  ScalableSubFOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ScalableSubFOpAdaptor(ScalableSubFOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value src1();
  ::mlir::Value src2();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ScalableSubFOp : public ::mlir::Op<ScalableSubFOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<2>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ScalableSubFOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("arm_sve.subf");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value src1();
  ::mlir::Value src2();
  ::mlir::MutableOperandRange src1Mutable();
  ::mlir::MutableOperandRange src2Mutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value dst();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dst, ::mlir::Value src1, ::mlir::Value src2);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src1, ::mlir::Value src2);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
};
} // namespace arm_sve
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableSubFOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableSubIOp declarations
//===----------------------------------------------------------------------===//

class ScalableSubIOpAdaptor {
public:
  ScalableSubIOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ScalableSubIOpAdaptor(ScalableSubIOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value src1();
  ::mlir::Value src2();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ScalableSubIOp : public ::mlir::Op<ScalableSubIOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<2>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ScalableSubIOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("arm_sve.subi");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value src1();
  ::mlir::Value src2();
  ::mlir::MutableOperandRange src1Mutable();
  ::mlir::MutableOperandRange src2Mutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value dst();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dst, ::mlir::Value src1, ::mlir::Value src2);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src1, ::mlir::Value src2);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
};
} // namespace arm_sve
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableSubIOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::ScalableUDivIOp declarations
//===----------------------------------------------------------------------===//

class ScalableUDivIOpAdaptor {
public:
  ScalableUDivIOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  ScalableUDivIOpAdaptor(ScalableUDivIOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value src1();
  ::mlir::Value src2();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ScalableUDivIOp : public ::mlir::Op<ScalableUDivIOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<2>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ScalableUDivIOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("arm_sve.divi_unsigned");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value src1();
  ::mlir::Value src2();
  ::mlir::MutableOperandRange src1Mutable();
  ::mlir::MutableOperandRange src2Mutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value dst();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dst, ::mlir::Value src1, ::mlir::Value src2);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src1, ::mlir::Value src2);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
};
} // namespace arm_sve
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ScalableUDivIOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::SdotIntrOp declarations
//===----------------------------------------------------------------------===//

class SdotIntrOpAdaptor {
public:
  SdotIntrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  SdotIntrOpAdaptor(SdotIntrOp&op);
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
class SdotIntrOp : public ::mlir::Op<SdotIntrOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<3>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SdotIntrOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("arm_sve.intr.sdot");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
};
} // namespace arm_sve
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::arm_sve::SdotIntrOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::SdotOp declarations
//===----------------------------------------------------------------------===//

class SdotOpAdaptor {
public:
  SdotOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  SdotOpAdaptor(SdotOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value acc();
  ::mlir::Value src1();
  ::mlir::Value src2();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class SdotOp : public ::mlir::Op<SdotOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::arm_sve::ScalableVectorType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SdotOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("arm_sve.sdot");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value acc();
  ::mlir::Value src1();
  ::mlir::Value src2();
  ::mlir::MutableOperandRange accMutable();
  ::mlir::MutableOperandRange src1Mutable();
  ::mlir::MutableOperandRange src2Mutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value dst();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dst, ::mlir::Value acc, ::mlir::Value src1, ::mlir::Value src2);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value acc, ::mlir::Value src1, ::mlir::Value src2);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
};
} // namespace arm_sve
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::arm_sve::SdotOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::SmmlaIntrOp declarations
//===----------------------------------------------------------------------===//

class SmmlaIntrOpAdaptor {
public:
  SmmlaIntrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  SmmlaIntrOpAdaptor(SmmlaIntrOp&op);
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
class SmmlaIntrOp : public ::mlir::Op<SmmlaIntrOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<3>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SmmlaIntrOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("arm_sve.intr.smmla");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
};
} // namespace arm_sve
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::arm_sve::SmmlaIntrOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::SmmlaOp declarations
//===----------------------------------------------------------------------===//

class SmmlaOpAdaptor {
public:
  SmmlaOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  SmmlaOpAdaptor(SmmlaOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value acc();
  ::mlir::Value src1();
  ::mlir::Value src2();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class SmmlaOp : public ::mlir::Op<SmmlaOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::arm_sve::ScalableVectorType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SmmlaOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("arm_sve.smmla");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value acc();
  ::mlir::Value src1();
  ::mlir::Value src2();
  ::mlir::MutableOperandRange accMutable();
  ::mlir::MutableOperandRange src1Mutable();
  ::mlir::MutableOperandRange src2Mutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value dst();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dst, ::mlir::Value acc, ::mlir::Value src1, ::mlir::Value src2);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value acc, ::mlir::Value src1, ::mlir::Value src2);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
};
} // namespace arm_sve
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::arm_sve::SmmlaOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::UdotIntrOp declarations
//===----------------------------------------------------------------------===//

class UdotIntrOpAdaptor {
public:
  UdotIntrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  UdotIntrOpAdaptor(UdotIntrOp&op);
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
class UdotIntrOp : public ::mlir::Op<UdotIntrOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<3>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = UdotIntrOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("arm_sve.intr.udot");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
};
} // namespace arm_sve
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::arm_sve::UdotIntrOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::UdotOp declarations
//===----------------------------------------------------------------------===//

class UdotOpAdaptor {
public:
  UdotOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  UdotOpAdaptor(UdotOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value acc();
  ::mlir::Value src1();
  ::mlir::Value src2();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class UdotOp : public ::mlir::Op<UdotOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::arm_sve::ScalableVectorType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = UdotOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("arm_sve.udot");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value acc();
  ::mlir::Value src1();
  ::mlir::Value src2();
  ::mlir::MutableOperandRange accMutable();
  ::mlir::MutableOperandRange src1Mutable();
  ::mlir::MutableOperandRange src2Mutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value dst();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dst, ::mlir::Value acc, ::mlir::Value src1, ::mlir::Value src2);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value acc, ::mlir::Value src1, ::mlir::Value src2);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
};
} // namespace arm_sve
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::arm_sve::UdotOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::UmmlaIntrOp declarations
//===----------------------------------------------------------------------===//

class UmmlaIntrOpAdaptor {
public:
  UmmlaIntrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  UmmlaIntrOpAdaptor(UmmlaIntrOp&op);
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
class UmmlaIntrOp : public ::mlir::Op<UmmlaIntrOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<3>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = UmmlaIntrOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("arm_sve.intr.ummla");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
};
} // namespace arm_sve
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::arm_sve::UmmlaIntrOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::UmmlaOp declarations
//===----------------------------------------------------------------------===//

class UmmlaOpAdaptor {
public:
  UmmlaOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  UmmlaOpAdaptor(UmmlaOp&op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value acc();
  ::mlir::Value src1();
  ::mlir::Value src2();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class UmmlaOp : public ::mlir::Op<UmmlaOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::arm_sve::ScalableVectorType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = UmmlaOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("arm_sve.ummla");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value acc();
  ::mlir::Value src1();
  ::mlir::Value src2();
  ::mlir::MutableOperandRange accMutable();
  ::mlir::MutableOperandRange src1Mutable();
  ::mlir::MutableOperandRange src2Mutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value dst();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dst, ::mlir::Value acc, ::mlir::Value src1, ::mlir::Value src2);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value acc, ::mlir::Value src1, ::mlir::Value src2);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
};
} // namespace arm_sve
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::arm_sve::UmmlaOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::VectorScaleIntrOp declarations
//===----------------------------------------------------------------------===//

class VectorScaleIntrOpAdaptor {
public:
  VectorScaleIntrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  VectorScaleIntrOpAdaptor(VectorScaleIntrOp&op);
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
class VectorScaleIntrOp : public ::mlir::Op<VectorScaleIntrOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = VectorScaleIntrOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("arm_sve.vscale");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
};
} // namespace arm_sve
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::arm_sve::VectorScaleIntrOp)

namespace mlir {
namespace arm_sve {

//===----------------------------------------------------------------------===//
// ::mlir::arm_sve::VectorScaleOp declarations
//===----------------------------------------------------------------------===//

class VectorScaleOpAdaptor {
public:
  VectorScaleOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});
  VectorScaleOpAdaptor(VectorScaleOp&op);
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
class VectorScaleOp : public ::mlir::Op<VectorScaleOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::IndexType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = VectorScaleOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("arm_sve.vector_scale");
  }
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
};
} // namespace arm_sve
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::arm_sve::VectorScaleOp)


#endif  // GET_OP_CLASSES

