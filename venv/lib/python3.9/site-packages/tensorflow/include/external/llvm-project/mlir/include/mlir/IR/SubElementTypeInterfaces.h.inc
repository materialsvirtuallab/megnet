/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class SubElementTypeInterface;
namespace detail {
struct SubElementTypeInterfaceInterfaceTraits {
  struct Concept {
    void (*walkImmediateSubElements)(const Concept *impl, ::mlir::Type , llvm::function_ref<void(mlir::Attribute)>, llvm::function_ref<void(mlir::Type)>);
  };
  template<typename ConcreteType>
  class Model : public Concept {
  public:
    using Interface = ::mlir::SubElementTypeInterface;
    Model() : Concept{walkImmediateSubElements} {}

    static inline void walkImmediateSubElements(const Concept *impl, ::mlir::Type tablegen_opaque_val, llvm::function_ref<void(mlir::Attribute)> walkAttrsFn, llvm::function_ref<void(mlir::Type)> walkTypesFn);
  };
  template<typename ConcreteType>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::SubElementTypeInterface;
    FallbackModel() : Concept{walkImmediateSubElements} {}

    static inline void walkImmediateSubElements(const Concept *impl, ::mlir::Type tablegen_opaque_val, llvm::function_ref<void(mlir::Attribute)> walkAttrsFn, llvm::function_ref<void(mlir::Type)> walkTypesFn);
  };
  template<typename ConcreteModel, typename ConcreteType>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
  };
};template <typename ConcreteType>
struct SubElementTypeInterfaceTrait;

} // end namespace detail
class SubElementTypeInterface : public ::mlir::TypeInterface<SubElementTypeInterface, detail::SubElementTypeInterfaceInterfaceTraits> {
public:
  using ::mlir::TypeInterface<SubElementTypeInterface, detail::SubElementTypeInterfaceInterfaceTraits>::TypeInterface;
  template <typename ConcreteType>
  struct Trait : public detail::SubElementTypeInterfaceTrait<ConcreteType> {};
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
  template <typename ConcreteType>
  struct SubElementTypeInterfaceTrait : public ::mlir::TypeInterface<SubElementTypeInterface, detail::SubElementTypeInterfaceInterfaceTraits>::Trait<ConcreteType> {

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
      SubElementTypeInterface interface((*static_cast<const ConcreteType *>(this)));
      interface.walkSubElements(walkAttrsFn, walkTypesFn);
    }
  
  };
}// namespace detail
template<typename ConcreteType>
void detail::SubElementTypeInterfaceInterfaceTraits::Model<ConcreteType>::walkImmediateSubElements(const Concept *impl, ::mlir::Type tablegen_opaque_val, llvm::function_ref<void(mlir::Attribute)> walkAttrsFn, llvm::function_ref<void(mlir::Type)> walkTypesFn) {
  return (tablegen_opaque_val.cast<ConcreteType>()).walkImmediateSubElements(walkAttrsFn, walkTypesFn);
}
template<typename ConcreteType>
void detail::SubElementTypeInterfaceInterfaceTraits::FallbackModel<ConcreteType>::walkImmediateSubElements(const Concept *impl, ::mlir::Type tablegen_opaque_val, llvm::function_ref<void(mlir::Attribute)> walkAttrsFn, llvm::function_ref<void(mlir::Type)> walkTypesFn) {
  return static_cast<const ConcreteType *>(impl)->walkImmediateSubElements(tablegen_opaque_val, walkAttrsFn, walkTypesFn);
}
} // namespace mlir
