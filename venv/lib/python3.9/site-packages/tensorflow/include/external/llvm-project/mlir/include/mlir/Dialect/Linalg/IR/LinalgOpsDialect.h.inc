/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace linalg {

class LinalgDialect : public ::mlir::Dialect {
  explicit LinalgDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context,
      ::mlir::TypeID::get<LinalgDialect>()) {
    
    getContext()->getOrLoadDialect<AffineDialect>();

    getContext()->getOrLoadDialect<math::MathDialect>();

    getContext()->getOrLoadDialect<memref::MemRefDialect>();

    getContext()->getOrLoadDialect<StandardOpsDialect>();

    getContext()->getOrLoadDialect<tensor::TensorDialect>();

    initialize();
  }

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~LinalgDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("linalg");
  }

  /// Parse a type registered to this dialect.
  ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;

  /// Print a type registered to this dialect.
  void printType(::mlir::Type type,
                 ::mlir::DialectAsmPrinter &os) const override;

  /// Register canonicalization patterns.
  void getCanonicalizationPatterns(
      ::mlir::RewritePatternSet &results) const override;

    /// Provides a hook for verifying dialect attributes attached to the given
    /// op.
    ::mlir::LogicalResult verifyOperationAttribute(
        ::mlir::Operation *op, ::mlir::NamedAttribute attribute) override;

    /// Attribute name used to to memoize indexing maps for named ops.
    constexpr const static ::llvm::StringLiteral
        kMemoizedIndexingMapsAttrName = "linalg.memoized_indexing_maps";

    /// Attribute name used to mark region arguments that can be bufferized
    /// in-place during linalg comprehensive bufferization.
    constexpr const static ::llvm::StringLiteral
      kInplaceableAttrName = "linalg.inplaceable";

    /// Attribute name used to mark the bufferization layout for region
    /// arguments during linalg comprehensive bufferization.
    constexpr const static ::llvm::StringLiteral
      kBufferLayoutAttrName = "linalg.buffer_layout";

    using RegionBuilderFunType =
      llvm::function_ref<void(ImplicitLocOpBuilder &b, Block &)>;
    RegionBuilderFunType getRegionBuilder(StringRef name) {
      return namedStructuredOpRegionBuilders.lookup(name);
    }
    private:
      llvm::StringMap<RegionBuilderFunType> namedStructuredOpRegionBuilders;
  };
} // namespace linalg
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::LinalgDialect)
