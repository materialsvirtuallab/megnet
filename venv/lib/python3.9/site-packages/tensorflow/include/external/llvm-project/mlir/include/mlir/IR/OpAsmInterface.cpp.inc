/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

void mlir::OpAsmOpInterface::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
      return getImpl()->getAsmResultNames(getImpl(), getOperation(), setNameFn);
  }
StringRef mlir::OpAsmOpInterface::getDefaultDialect() {
      return getImpl()->getDefaultDialect();
  }
