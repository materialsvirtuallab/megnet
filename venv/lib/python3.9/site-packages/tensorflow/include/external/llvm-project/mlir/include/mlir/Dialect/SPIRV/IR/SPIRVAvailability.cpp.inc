/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Availability Interface Definitions                                         *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

::mlir::spirv::Version QueryMinVersionInterface::getMinVersion() {
  return getImpl()->getMinVersion(getImpl(), getOperation());
}
::mlir::spirv::Version QueryMaxVersionInterface::getMaxVersion() {
  return getImpl()->getMaxVersion(getImpl(), getOperation());
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> QueryExtensionInterface::getExtensions() {
  return getImpl()->getExtensions(getImpl(), getOperation());
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> QueryCapabilityInterface::getCapabilities() {
  return getImpl()->getCapabilities(getImpl(), getOperation());
}
