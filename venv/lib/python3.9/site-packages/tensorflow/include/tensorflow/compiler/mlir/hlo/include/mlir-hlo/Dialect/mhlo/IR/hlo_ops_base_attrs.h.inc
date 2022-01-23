/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES


namespace mlir {
class DialectAsmParser;
class DialectAsmPrinter;
} // namespace mlir
namespace mlir {
namespace mhlo {
  class GatherDimensionNumbersAttr;
  class ScatterDimensionNumbersAttr;

  namespace detail {
    struct GatherDimensionNumbersAttrStorage;
  } // end namespace detail
  class GatherDimensionNumbersAttr : public ::mlir::Attribute::AttrBase<GatherDimensionNumbersAttr, ::mlir::Attribute,
                                         detail::GatherDimensionNumbersAttrStorage> {
  public:
    /// Inherit some necessary constructors from 'AttrBase'.
    using Base::Base;

    static GatherDimensionNumbersAttr get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> offsetDims, ::llvm::ArrayRef<int64_t> collapsedSliceDims, ::llvm::ArrayRef<int64_t> startIndexMap, int64_t indexVectorDim);
    static constexpr ::llvm::StringLiteral getMnemonic() {
      return ::llvm::StringLiteral("gather");
    }

    static ::mlir::Attribute parse(::mlir::MLIRContext *context,
                             ::mlir::DialectAsmParser &parser, ::mlir::Type type);
    void print(::mlir::DialectAsmPrinter &printer) const;
    ::llvm::ArrayRef<int64_t> getOffsetDims() const;
    ::llvm::ArrayRef<int64_t> getCollapsedSliceDims() const;
    ::llvm::ArrayRef<int64_t> getStartIndexMap() const;
    int64_t getIndexVectorDim() const;
  };

  namespace detail {
    struct ScatterDimensionNumbersAttrStorage;
  } // end namespace detail
  class ScatterDimensionNumbersAttr : public ::mlir::Attribute::AttrBase<ScatterDimensionNumbersAttr, ::mlir::Attribute,
                                         detail::ScatterDimensionNumbersAttrStorage> {
  public:
    /// Inherit some necessary constructors from 'AttrBase'.
    using Base::Base;

    static ScatterDimensionNumbersAttr get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> updateWindowDims, ::llvm::ArrayRef<int64_t> insertedWindowDims, ::llvm::ArrayRef<int64_t> scatterDimsToOperandDims, int64_t indexVectorDim);
    static constexpr ::llvm::StringLiteral getMnemonic() {
      return ::llvm::StringLiteral("scatter");
    }

    static ::mlir::Attribute parse(::mlir::MLIRContext *context,
                             ::mlir::DialectAsmParser &parser, ::mlir::Type type);
    void print(::mlir::DialectAsmPrinter &printer) const;
    ::llvm::ArrayRef<int64_t> getUpdateWindowDims() const;
    ::llvm::ArrayRef<int64_t> getInsertedWindowDims() const;
    ::llvm::ArrayRef<int64_t> getScatterDimsToOperandDims() const;
    int64_t getIndexVectorDim() const;
  };
} // namespace mhlo
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::mhlo::GatherDimensionNumbersAttr)
DECLARE_EXPLICIT_TYPE_ID(::mlir::mhlo::ScatterDimensionNumbersAttr)

#endif  // GET_ATTRDEF_CLASSES

