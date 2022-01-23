/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class SubElementAttrInterface;
namespace detail {
struct SubElementAttrInterfaceInterfaceTraits {
  struct Concept {
    void (*walkImmediateSubElements)(const Concept *impl, ::mlir::Attribute , llvm::function_ref<void(mlir::Attribute)>, llvm::function_ref<void(mlir::Type)>);
  };
  template<typename ConcreteAttr>
  class Model : public Concept {
  public:
    using Interface = ::mlir::SubElementAttrInterface;
    Model() : Concept{walkImmediateSubElements} {}

    static inline void walkImmediateSubElements(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, llvm::function_ref<void(mlir::Attribute)> walkAttrsFn, llvm::function_ref<void(mlir::Type)> walkTypesFn);
  };
  template<typename ConcreteAttr>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::SubElementAttrInterface;
    FallbackModel() : Concept{walkImmediateSubElements} {}

    static inline void walkImmediateSubElements(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, llvm::function_ref<void(mlir::Attribute)> walkAttrsFn, llvm::function_ref<void(mlir::Type)> walkTypesFn);
  };
  template<typename ConcreteModel, typename ConcreteAttr>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
  };
};template <typename ConcreteAttr>
struct SubElementAttrInterfaceTrait;

} // end namespace detail
class SubElementAttrInterface : public ::mlir::AttributeInterface<SubElementAttrInterface, detail::SubElementAttrInterfaceInterfaceTraits> {
public:
  using ::mlir::AttributeInterface<SubElementAttrInterface, detail::SubElementAttrInterfaceInterfaceTraits>::AttributeInterface;
  template <typename ConcreteAttr>
  struct Trait : public detail::SubElementAttrInterfaceTrait<ConcreteAttr> {};
  void walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn, llvm::function_ref<void(mlir::Type)> walkTypesFn) const;

    /// Walk all of the held sub-attributes.
    void walkSubAttrs(llvm::function_ref<void(mlir::Attribute)> walkFn) {
      walkSubElements(walkFn, /*walkTypesFn=*/[](mlir::Type) {});
    }

    /// Walk all of the held sub-types.
    void walkSubTypes(llvm::function_ref<void(mlir::Type)> walkFn) {
      walkSubElements(/*walkAttrsFn=*/[](mlir::Attribute) {}, walkFn);
    }

    /// Walk all of the held sub-attributes and sub-types.
    void walkSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                         llvm::function_ref<void(mlir::Type)> walkTypesFn);
  
};
namespace detail {
  template <typename ConcreteAttr>
  struct SubElementAttrInterfaceTrait : public ::mlir::AttributeInterface<SubElementAttrInterface, detail::SubElementAttrInterfaceInterfaceTraits>::Trait<ConcreteAttr> {

    /// Walk all of the held sub-attributes.
    void walkSubAttrs(llvm::function_ref<void(mlir::Attribute)> walkFn) {
      walkSubElements(walkFn, /*walkTypesFn=*/[](mlir::Type) {});
    }

    /// Walk all of the held sub-types.
    void walkSubTypes(llvm::function_ref<void(mlir::Type)> walkFn) {
      walkSubElements(/*walkAttrsFn=*/[](mlir::Attribute) {}, walkFn);
    }

    /// Walk all of the held sub-attributes and sub-types.
    void walkSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                         llvm::function_ref<void(mlir::Type)> walkTypesFn) {
      SubElementAttrInterface interface((*static_cast<const ConcreteAttr *>(this)));
      interface.walkSubElements(walkAttrsFn, walkTypesFn);
    }
  
  };
}// namespace detail
template<typename ConcreteAttr>
void detail::SubElementAttrInterfaceInterfaceTraits::Model<ConcreteAttr>::walkImmediateSubElements(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, llvm::function_ref<void(mlir::Attribute)> walkAttrsFn, llvm::function_ref<void(mlir::Type)> walkTypesFn) {
  return (tablegen_opaque_val.cast<ConcreteAttr>()).walkImmediateSubElements(walkAttrsFn, walkTypesFn);
}
template<typename ConcreteAttr>
void detail::SubElementAttrInterfaceInterfaceTraits::FallbackModel<ConcreteAttr>::walkImmediateSubElements(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, llvm::function_ref<void(mlir::Attribute)> walkAttrsFn, llvm::function_ref<void(mlir::Type)> walkTypesFn) {
  return static_cast<const ConcreteAttr *>(impl)->walkImmediateSubElements(tablegen_opaque_val, walkAttrsFn, walkTypesFn);
}
} // namespace mlir
