/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class MemRefElementTypeInterface;
namespace detail {
struct MemRefElementTypeInterfaceInterfaceTraits {
  struct Concept {
  };
  template<typename ConcreteType>
  class Model : public Concept {
  public:
    using Interface = ::mlir::MemRefElementTypeInterface;
    Model() : Concept{} {}

  };
  template<typename ConcreteType>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::MemRefElementTypeInterface;
    FallbackModel() : Concept{} {}

  };
  template<typename ConcreteModel, typename ConcreteType>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
  };
};template <typename ConcreteType>
struct MemRefElementTypeInterfaceTrait;

} // end namespace detail
class MemRefElementTypeInterface : public ::mlir::TypeInterface<MemRefElementTypeInterface, detail::MemRefElementTypeInterfaceInterfaceTraits> {
public:
  using ::mlir::TypeInterface<MemRefElementTypeInterface, detail::MemRefElementTypeInterfaceInterfaceTraits>::TypeInterface;
  template <typename ConcreteType>
  struct Trait : public detail::MemRefElementTypeInterfaceTrait<ConcreteType> {};
};
namespace detail {
  template <typename ConcreteType>
  struct MemRefElementTypeInterfaceTrait : public ::mlir::TypeInterface<MemRefElementTypeInterface, detail::MemRefElementTypeInterfaceInterfaceTraits>::Trait<ConcreteType> {
  };
}// namespace detail
} // namespace mlir
