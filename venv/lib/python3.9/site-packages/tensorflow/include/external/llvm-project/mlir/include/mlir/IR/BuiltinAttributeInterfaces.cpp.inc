/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

::mlir::FailureOr<::mlir::detail::ElementsAttrIndexer> mlir::ElementsAttr::getValuesImpl(::mlir::TypeID elementID) const {
      return getImpl()->getValuesImpl(getImpl(), *this, elementID);
  }
bool mlir::ElementsAttr::isSplat() const {
      return getImpl()->isSplat(getImpl(), *this);
  }
