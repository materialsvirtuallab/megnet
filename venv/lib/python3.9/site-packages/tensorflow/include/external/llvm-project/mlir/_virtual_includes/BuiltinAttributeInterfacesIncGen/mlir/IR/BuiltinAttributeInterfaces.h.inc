/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class ElementsAttr;
namespace detail {
struct ElementsAttrInterfaceTraits {
  struct Concept {
    ::mlir::FailureOr<::mlir::detail::ElementsAttrIndexer> (*getValuesImpl)(const Concept *impl, ::mlir::Attribute , ::mlir::TypeID);
    bool (*isSplat)(const Concept *impl, ::mlir::Attribute );
  };
  template<typename ConcreteAttr>
  class Model : public Concept {
  public:
    using Interface = ::mlir::ElementsAttr;
    Model() : Concept{getValuesImpl, isSplat} {}

    static inline ::mlir::FailureOr<::mlir::detail::ElementsAttrIndexer> getValuesImpl(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::TypeID elementID);
    static inline bool isSplat(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
  };
  template<typename ConcreteAttr>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::ElementsAttr;
    FallbackModel() : Concept{getValuesImpl, isSplat} {}

    static inline ::mlir::FailureOr<::mlir::detail::ElementsAttrIndexer> getValuesImpl(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::TypeID elementID);
    static inline bool isSplat(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteAttr>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    ::mlir::FailureOr<::mlir::detail::ElementsAttrIndexer> getValuesImpl(::mlir::Attribute tablegen_opaque_val, ::mlir::TypeID elementID) const;
    bool isSplat(::mlir::Attribute tablegen_opaque_val) const;
  };
};template <typename ConcreteAttr>
struct ElementsAttrTrait;

} // end namespace detail
class ElementsAttr : public ::mlir::AttributeInterface<ElementsAttr, detail::ElementsAttrInterfaceTraits> {
public:
  using ::mlir::AttributeInterface<ElementsAttr, detail::ElementsAttrInterfaceTraits>::AttributeInterface;
  template <typename ConcreteAttr>
  struct Trait : public detail::ElementsAttrTrait<ConcreteAttr> {};
  ::mlir::FailureOr<::mlir::detail::ElementsAttrIndexer> getValuesImpl(::mlir::TypeID elementID) const;
  bool isSplat() const;

    template <typename T>
    using iterator = detail::ElementsAttrIterator<T>;
    template <typename T>
    using iterator_range = llvm::iterator_range<iterator<T>>;

    //===------------------------------------------------------------------===//
    // Accessors
    //===------------------------------------------------------------------===//

    /// Return the type of this attribute.
    ShapedType getType() const;

    /// Return the element type of this ElementsAttr.
    Type getElementType() const { return getElementType(*this); }
    static Type getElementType(Attribute elementsAttr);

    /// Return if the given 'index' refers to a valid element in this attribute.
    bool isValidIndex(ArrayRef<uint64_t> index) const {
      return isValidIndex(*this, index);
    }
    static bool isValidIndex(ShapedType type, ArrayRef<uint64_t> index);
    static bool isValidIndex(Attribute elementsAttr, ArrayRef<uint64_t> index);

    /// Return the 1 dimensional flattened row-major index from the given
    /// multi-dimensional index.
    uint64_t getFlattenedIndex(ArrayRef<uint64_t> index) const {
      return getFlattenedIndex(*this, index);
    }
    static uint64_t getFlattenedIndex(Attribute elementsAttr,
                                      ArrayRef<uint64_t> index);

    /// Returns the number of elements held by this attribute.
    int64_t getNumElements() const { return getNumElements(*this); }
    static int64_t getNumElements(Attribute elementsAttr);

    //===------------------------------------------------------------------===//
    // Value Iteration
    //===------------------------------------------------------------------===//

    template <typename T>
    using DerivedAttrValueCheckT =
        typename std::enable_if_t<std::is_base_of<Attribute, T>::value &&
                                  !std::is_same<Attribute, T>::value>;
    template <typename T, typename ResultT>
    using DefaultValueCheckT =
        typename std::enable_if_t<std::is_same<Attribute, T>::value ||
                                  !std::is_base_of<Attribute, T>::value,
                                  ResultT>;

    /// Return the element of this attribute at the given index as a value of
    /// type 'T'.
    template <typename T>
    T getFlatValue(uint64_t index) const {
      return *std::next(value_begin<T>(), index);
    }

    /// Return the splat value for this attribute. This asserts that the
    /// attribute corresponds to a splat.
    template <typename T>
    T getSplatValue() const {
      assert(isSplat() && "expected splat attribute");
      return *value_begin<T>();
    }

    /// Return the elements of this attribute as a value of type 'T'.
    template <typename T>
    DefaultValueCheckT<T, iterator_range<T>> getValues() const {
      return iterator_range<T>(value_begin<T>(), value_end<T>());
    }
    template <typename T>
    DefaultValueCheckT<T, iterator<T>> value_begin() const;
    template <typename T>
    DefaultValueCheckT<T, iterator<T>> value_end() const {
      return iterator<T>({}, size());
    }

    /// Return the held element values a range of T, where T is a derived
    /// attribute type.
    template <typename T>
    using DerivedAttrValueIterator =
      llvm::mapped_iterator<iterator<Attribute>, T (*)(Attribute)>;
    template <typename T>
    using DerivedAttrValueIteratorRange =
      llvm::iterator_range<DerivedAttrValueIterator<T>>;
    template <typename T, typename = DerivedAttrValueCheckT<T>>
    DerivedAttrValueIteratorRange<T> getValues() const {
      auto castFn = [](Attribute attr) { return attr.template cast<T>(); };
      return llvm::map_range(getValues<Attribute>(),
                             static_cast<T (*)(Attribute)>(castFn));
    }
    template <typename T, typename = DerivedAttrValueCheckT<T>>
    DerivedAttrValueIterator<T> value_begin() const {
      return getValues<T>().begin();
    }
    template <typename T, typename = DerivedAttrValueCheckT<T>>
    DerivedAttrValueIterator<T> value_end() const {
      return {value_end<Attribute>(), nullptr};
    }

    //===------------------------------------------------------------------===//
    // Failable Value Iteration

    /// If this attribute supports iterating over element values of type `T`,
    /// return the iterable range. Otherwise, return llvm::None.
    template <typename T>
    DefaultValueCheckT<T, Optional<iterator_range<T>>> tryGetValues() const {
      if (Optional<iterator<T>> beginIt = try_value_begin<T>())
        return iterator_range<T>(*beginIt, value_end<T>());
      return llvm::None;
    }
    template <typename T>
    DefaultValueCheckT<T, Optional<iterator<T>>> try_value_begin() const;

    /// If this attribute supports iterating over element values of type `T`,
    /// return the iterable range. Otherwise, return llvm::None.
    template <typename T, typename = DerivedAttrValueCheckT<T>>
    Optional<DerivedAttrValueIteratorRange<T>> tryGetValues() const {
      auto castFn = [](Attribute attr) { return attr.template cast<T>(); };
      if (auto values = tryGetValues<Attribute>())
        return llvm::map_range(*values, static_cast<T (*)(Attribute)>(castFn));
      return llvm::None;
    }
    template <typename T, typename = DerivedAttrValueCheckT<T>>
    Optional<DerivedAttrValueIterator<T>> try_value_begin() const {
      if (auto values = tryGetValues<T>())
        return values->begin();
      return llvm::None;
    }
  
    /// Return the attribute value at the given index. The index is expected to
    /// refer to a valid element.
    Attribute getValue(ArrayRef<uint64_t> index) const {
      return getValue<Attribute>(index);
    }

    /// Return the value of type 'T' at the given index, where 'T' corresponds
    /// to an Attribute type.
    template <typename T>
    std::enable_if_t<!std::is_same<T, ::mlir::Attribute>::value &&
                     std::is_base_of<T, ::mlir::Attribute>::value>
    getValue(ArrayRef<uint64_t> index) const {
      return getValue(index).template dyn_cast_or_null<T>();
    }

    /// Return the value of type 'T' at the given index.
    template <typename T>
    T getValue(ArrayRef<uint64_t> index) const {
      return getFlatValue<T>(getFlattenedIndex(index));
    }

    /// Return the number of elements held by this attribute.
    int64_t size() const { return getNumElements(); }

    /// Return if the attribute holds no elements.
    bool empty() const { return size() == 0; }
  
};
namespace detail {
  template <typename ConcreteAttr>
  struct ElementsAttrTrait : public ::mlir::AttributeInterface<ElementsAttr, detail::ElementsAttrInterfaceTraits>::Trait<ConcreteAttr> {
    ::mlir::FailureOr<::mlir::detail::ElementsAttrIndexer> getValuesImpl(::mlir::TypeID elementID) const {
      auto result = getValueImpl(
        (typename ConcreteAttr::ContiguousIterableTypesT *)nullptr, elementID,
        /*isContiguous=*/std::true_type());
      if (succeeded(result))
        return std::move(result);

      return getValueImpl(
        (typename ConcreteAttr::NonContiguousIterableTypesT *)nullptr,
        elementID, /*isContiguous=*/std::false_type());
    }
    bool isSplat() const {
      // By default, only check for a single element splat.
        return (*static_cast<const ConcreteAttr *>(this)).getNumElements() == 1;
    }

    // By default, no types are iterable.
    using ContiguousIterableTypesT = std::tuple<>;
    using NonContiguousIterableTypesT = std::tuple<>;

    //===------------------------------------------------------------------===//
    // Accessors
    //===------------------------------------------------------------------===//

    /// Return the element type of this ElementsAttr.
    Type getElementType() const {
      return ::mlir::ElementsAttr::getElementType((*static_cast<const ConcreteAttr *>(this)));
    }

    /// Returns the number of elements held by this attribute.
    int64_t getNumElements() const {
      return ::mlir::ElementsAttr::getNumElements((*static_cast<const ConcreteAttr *>(this)));
    }

    /// Return if the given 'index' refers to a valid element in this attribute.
    bool isValidIndex(ArrayRef<uint64_t> index) const {
      return ::mlir::ElementsAttr::isValidIndex((*static_cast<const ConcreteAttr *>(this)), index);
    }

  protected:
    /// Returns the 1-dimensional flattened row-major index from the given
    /// multi-dimensional index.
    uint64_t getFlattenedIndex(ArrayRef<uint64_t> index) const {
      return ::mlir::ElementsAttr::getFlattenedIndex((*static_cast<const ConcreteAttr *>(this)), index);
    }

    //===------------------------------------------------------------------===//
    // Value Iteration Internals
    //===------------------------------------------------------------------===//
  protected:
    /// This class is used to allow specifying function overloads for different
    /// types, without actually taking the types as parameters. This avoids the
    /// need to build complicated SFINAE to select specific overloads.
    template <typename T>
    struct OverloadToken {};

  private:
    /// This function unpacks the types within a given tuple and then forwards
    /// on to the unwrapped variant.
    template <typename... Ts, typename IsContiguousT>
    auto getValueImpl(std::tuple<Ts...> *, ::mlir::TypeID elementID,
                      IsContiguousT isContiguous) const {
      return getValueImpl<Ts...>(elementID, isContiguous);
    }
    /// Check to see if the given `elementID` matches the current type `T`. If
    /// it does, build a value result using the current type. If it doesn't,
    /// keep looking for the desired type.
    template <typename T, typename... Ts, typename IsContiguousT>
    auto getValueImpl(::mlir::TypeID elementID,
                      IsContiguousT isContiguous) const {
      if (::mlir::TypeID::get<T>() == elementID)
        return buildValueResult<T>(isContiguous);
      return getValueImpl<Ts...>(elementID, isContiguous);
    }
    /// Bottom out case for no matching type.
    template <typename IsContiguousT>
    ::mlir::FailureOr<::mlir::detail::ElementsAttrIndexer>
    getValueImpl(::mlir::TypeID, IsContiguousT) const {
      return failure();
    }

    /// Build an indexer for the given type `T`, which is represented via a
    /// contiguous range.
    template <typename T>
    ::mlir::FailureOr<::mlir::detail::ElementsAttrIndexer> buildValueResult(
        /*isContiguous*/std::true_type) const {
      auto valueIt = (*static_cast<const ConcreteAttr *>(this)).value_begin_impl(OverloadToken<T>());
      return ::mlir::detail::ElementsAttrIndexer::contiguous(
        (*static_cast<const ConcreteAttr *>(this)).isSplat(), &*valueIt);
    }
    /// Build an indexer for the given type `T`, which is represented via a
    /// non-contiguous range.
    template <typename T>
    ::mlir::FailureOr<::mlir::detail::ElementsAttrIndexer> buildValueResult(
        /*isContiguous*/std::false_type) const {
      auto valueIt = (*static_cast<const ConcreteAttr *>(this)).value_begin_impl(OverloadToken<T>());
      return ::mlir::detail::ElementsAttrIndexer::nonContiguous(
        (*static_cast<const ConcreteAttr *>(this)).isSplat(), valueIt);
    }

  public:
    //===------------------------------------------------------------------===//
    // Value Iteration
    //===------------------------------------------------------------------===//

    /// Return an iterator to the first element of this attribute as a value of
    /// type `T`.
    template <typename T>
    auto value_begin() const {
      return (*static_cast<const ConcreteAttr *>(this)).value_begin_impl(OverloadToken<T>());
    }

    /// Return the elements of this attribute as a value of type 'T'.
    template <typename T>
    auto getValues() const {
      auto beginIt = (*static_cast<const ConcreteAttr *>(this)).template value_begin<T>();
      return llvm::make_range(beginIt, std::next(beginIt, size()));
    }
    /// Return the value at the given flattened index.
    template <typename T> T getFlatValue(uint64_t index) const {
      return *std::next((*static_cast<const ConcreteAttr *>(this)).template value_begin<T>(), index);
    }
  
    /// Return the attribute value at the given index. The index is expected to
    /// refer to a valid element.
    Attribute getValue(ArrayRef<uint64_t> index) const {
      return getValue<Attribute>(index);
    }

    /// Return the value of type 'T' at the given index, where 'T' corresponds
    /// to an Attribute type.
    template <typename T>
    std::enable_if_t<!std::is_same<T, ::mlir::Attribute>::value &&
                     std::is_base_of<T, ::mlir::Attribute>::value>
    getValue(ArrayRef<uint64_t> index) const {
      return getValue(index).template dyn_cast_or_null<T>();
    }

    /// Return the value of type 'T' at the given index.
    template <typename T>
    T getValue(ArrayRef<uint64_t> index) const {
      return getFlatValue<T>(getFlattenedIndex(index));
    }

    /// Return the number of elements held by this attribute.
    int64_t size() const { return getNumElements(); }

    /// Return if the attribute holds no elements.
    bool empty() const { return size() == 0; }
  
  };
}// namespace detail
template<typename ConcreteAttr>
::mlir::FailureOr<::mlir::detail::ElementsAttrIndexer> detail::ElementsAttrInterfaceTraits::Model<ConcreteAttr>::getValuesImpl(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::TypeID elementID) {
  return (tablegen_opaque_val.cast<ConcreteAttr>()).getValuesImpl(elementID);
}
template<typename ConcreteAttr>
bool detail::ElementsAttrInterfaceTraits::Model<ConcreteAttr>::isSplat(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (tablegen_opaque_val.cast<ConcreteAttr>()).isSplat();
}
template<typename ConcreteAttr>
::mlir::FailureOr<::mlir::detail::ElementsAttrIndexer> detail::ElementsAttrInterfaceTraits::FallbackModel<ConcreteAttr>::getValuesImpl(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::TypeID elementID) {
  return static_cast<const ConcreteAttr *>(impl)->getValuesImpl(tablegen_opaque_val, elementID);
}
template<typename ConcreteAttr>
bool detail::ElementsAttrInterfaceTraits::FallbackModel<ConcreteAttr>::isSplat(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->isSplat(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteAttr>
::mlir::FailureOr<::mlir::detail::ElementsAttrIndexer> detail::ElementsAttrInterfaceTraits::ExternalModel<ConcreteModel, ConcreteAttr>::getValuesImpl(::mlir::Attribute tablegen_opaque_val, ::mlir::TypeID elementID) const {
auto result = getValueImpl(
        (typename ConcreteAttr::ContiguousIterableTypesT *)nullptr, elementID,
        /*isContiguous=*/std::true_type());
      if (succeeded(result))
        return std::move(result);

      return getValueImpl(
        (typename ConcreteAttr::NonContiguousIterableTypesT *)nullptr,
        elementID, /*isContiguous=*/std::false_type());
}
template<typename ConcreteModel, typename ConcreteAttr>
bool detail::ElementsAttrInterfaceTraits::ExternalModel<ConcreteModel, ConcreteAttr>::isSplat(::mlir::Attribute tablegen_opaque_val) const {
// By default, only check for a single element splat.
        return (tablegen_opaque_val.cast<ConcreteAttr>()).getNumElements() == 1;
}
} // namespace mlir
