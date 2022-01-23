/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace omp {
::llvm::StringRef stringifyClauseDefault(ClauseDefault val) {
  switch (val) {
    case ClauseDefault::defprivate: return "defprivate";
    case ClauseDefault::deffirstprivate: return "deffirstprivate";
    case ClauseDefault::defshared: return "defshared";
    case ClauseDefault::defnone: return "defnone";
  }
  return "";
}

::llvm::Optional<ClauseDefault> symbolizeClauseDefault(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<ClauseDefault>>(str)
      .Case("defprivate", ClauseDefault::defprivate)
      .Case("deffirstprivate", ClauseDefault::deffirstprivate)
      .Case("defshared", ClauseDefault::defshared)
      .Case("defnone", ClauseDefault::defnone)
      .Default(::llvm::None);
}
} // namespace omp
} // namespace mlir

namespace mlir {
namespace omp {
::llvm::StringRef stringifyClauseOrderKind(ClauseOrderKind val) {
  switch (val) {
  }
  return "";
}

::llvm::Optional<ClauseOrderKind> symbolizeClauseOrderKind(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<ClauseOrderKind>>(str)
      .Default(::llvm::None);
}
::llvm::Optional<ClauseOrderKind> symbolizeClauseOrderKind(unsigned value) {
  switch (value) {
  default: return ::llvm::None;
  }
}

} // namespace omp
} // namespace mlir

namespace mlir {
namespace omp {
::llvm::StringRef stringifyClauseProcBindKind(ClauseProcBindKind val) {
  switch (val) {
    case ClauseProcBindKind::primary: return "primary";
    case ClauseProcBindKind::master: return "master";
    case ClauseProcBindKind::close: return "close";
    case ClauseProcBindKind::spread: return "spread";
  }
  return "";
}

::llvm::Optional<ClauseProcBindKind> symbolizeClauseProcBindKind(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<ClauseProcBindKind>>(str)
      .Case("primary", ClauseProcBindKind::primary)
      .Case("master", ClauseProcBindKind::master)
      .Case("close", ClauseProcBindKind::close)
      .Case("spread", ClauseProcBindKind::spread)
      .Default(::llvm::None);
}
} // namespace omp
} // namespace mlir

namespace mlir {
namespace omp {
::llvm::StringRef stringifyClauseScheduleKind(ClauseScheduleKind val) {
  switch (val) {
    case ClauseScheduleKind::Static: return "Static";
    case ClauseScheduleKind::Dynamic: return "Dynamic";
    case ClauseScheduleKind::Guided: return "Guided";
    case ClauseScheduleKind::Auto: return "Auto";
    case ClauseScheduleKind::Runtime: return "Runtime";
  }
  return "";
}

::llvm::Optional<ClauseScheduleKind> symbolizeClauseScheduleKind(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<ClauseScheduleKind>>(str)
      .Case("Static", ClauseScheduleKind::Static)
      .Case("Dynamic", ClauseScheduleKind::Dynamic)
      .Case("Guided", ClauseScheduleKind::Guided)
      .Case("Auto", ClauseScheduleKind::Auto)
      .Case("Runtime", ClauseScheduleKind::Runtime)
      .Default(::llvm::None);
}
} // namespace omp
} // namespace mlir

namespace mlir {
namespace omp {
::llvm::StringRef ConvertToString(SyncHintKind val) {
  switch (val) {
    case SyncHintKind::none: return "none";
    case SyncHintKind::uncontended: return "uncontended";
    case SyncHintKind::contended: return "contended";
    case SyncHintKind::nonspeculative: return "nonspeculative";
    case SyncHintKind::speculative: return "speculative";
  }
  return "";
}

::llvm::Optional<SyncHintKind> ConvertToEnum(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<SyncHintKind>>(str)
      .Case("none", SyncHintKind::none)
      .Case("uncontended", SyncHintKind::uncontended)
      .Case("contended", SyncHintKind::contended)
      .Case("nonspeculative", SyncHintKind::nonspeculative)
      .Case("speculative", SyncHintKind::speculative)
      .Default(::llvm::None);
}
::llvm::Optional<SyncHintKind> symbolizeSyncHintKind(uint32_t value) {
  switch (value) {
  case 0: return SyncHintKind::none;
  case 1: return SyncHintKind::uncontended;
  case 2: return SyncHintKind::contended;
  case 3: return SyncHintKind::nonspeculative;
  case 4: return SyncHintKind::speculative;
  default: return ::llvm::None;
  }
}

bool SyncHintKindAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 1)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 2)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 3)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 4)));
}
SyncHintKindAttr SyncHintKindAttr::get(::mlir::MLIRContext *context, SyncHintKind val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return baseAttr.cast<SyncHintKindAttr>();
}
SyncHintKind SyncHintKindAttr::getValue() const {
  return static_cast<SyncHintKind>(::mlir::IntegerAttr::getInt());
}
} // namespace omp
} // namespace mlir

