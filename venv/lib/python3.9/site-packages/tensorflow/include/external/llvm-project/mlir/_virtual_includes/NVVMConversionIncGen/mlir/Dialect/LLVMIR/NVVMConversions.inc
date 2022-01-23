if (auto op = dyn_cast<::mlir::NVVM::Barrier0Op>(opInst)) {

      createIntrinsicCall(builder, llvm::Intrinsic::nvvm_barrier0);
  
  return success();
}
if (auto op = dyn_cast<::mlir::NVVM::BlockDimXOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_x,
        { 
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::NVVM::BlockDimYOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_y,
        { 
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::NVVM::BlockDimZOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_z,
        { 
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::NVVM::BlockIdXOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_x,
        { 
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::NVVM::BlockIdYOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_y,
        { 
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::NVVM::BlockIdZOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_z,
        { 
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::NVVM::GridDimXOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::nvvm_read_ptx_sreg_nctaid_x,
        { 
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::NVVM::GridDimYOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::nvvm_read_ptx_sreg_nctaid_y,
        { 
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::NVVM::GridDimZOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::nvvm_read_ptx_sreg_nctaid_z,
        { 
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::NVVM::LaneIdOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::nvvm_read_ptx_sreg_laneid,
        { 
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::NVVM::MmaOp>(opInst)) {

    moduleTranslation.mapValue(op.res()) = createIntrinsicCall(
        builder, llvm::Intrinsic::nvvm_mma_m8n8k4_row_col_f32_f32, moduleTranslation.lookupValues(op.args()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::NVVM::ShflBflyOp>(opInst)) {

      auto intId = getShflBflyIntrinsicId(
          moduleTranslation.convertType(op.getResult().getType()), static_cast<bool>(op.return_value_and_is_valid()));
      moduleTranslation.mapValue(op.res()) = createIntrinsicCall(builder,
          intId, {moduleTranslation.lookupValue(op.dst()), moduleTranslation.lookupValue(op.val()), moduleTranslation.lookupValue(op.offset()), moduleTranslation.lookupValue(op.mask_and_clamp())});
  
  return success();
}
if (auto op = dyn_cast<::mlir::NVVM::ThreadIdXOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x,
        { 
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::NVVM::ThreadIdYOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::nvvm_read_ptx_sreg_tid_y,
        { 
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::NVVM::ThreadIdZOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::nvvm_read_ptx_sreg_tid_z,
        { 
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::NVVM::VoteBallotOp>(opInst)) {

      moduleTranslation.mapValue(op.res()) = createIntrinsicCall(builder,
            llvm::Intrinsic::nvvm_vote_ballot_sync, {moduleTranslation.lookupValue(op.mask()), moduleTranslation.lookupValue(op.pred())});
  
  return success();
}
if (auto op = dyn_cast<::mlir::NVVM::WMMALoadAM16N16K16Op>(opInst)) {

    moduleTranslation.mapValue(op.res()) = createNvvmIntrinsicCall(
      builder, llvm::Intrinsic::nvvm_wmma_m16n16k16_load_a_f16_row_stride, moduleTranslation.lookupValues(op.args()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::NVVM::WMMALoadBM16N16K16Op>(opInst)) {

    moduleTranslation.mapValue(op.res()) = createNvvmIntrinsicCall(
      builder, llvm::Intrinsic::nvvm_wmma_m16n16k16_load_b_f16_row_stride, moduleTranslation.lookupValues(op.args()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::NVVM::WMMALoadCF16M16N16K16Op>(opInst)) {

    moduleTranslation.mapValue(op.res()) = createNvvmIntrinsicCall(
      builder, llvm::Intrinsic::nvvm_wmma_m16n16k16_load_c_f16_row_stride, moduleTranslation.lookupValues(op.args()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::NVVM::WMMALoadCF32M16N16K16Op>(opInst)) {

    moduleTranslation.mapValue(op.res()) = createNvvmIntrinsicCall(
      builder, llvm::Intrinsic::nvvm_wmma_m16n16k16_load_c_f32_row_stride, moduleTranslation.lookupValues(op.args()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::NVVM::WMMAMmaF16F16M16N16K16Op>(opInst)) {

    moduleTranslation.mapValue(op.res()) = createNvvmIntrinsicCall(
        builder, llvm::Intrinsic::nvvm_wmma_m16n16k16_mma_row_row_f16_f16, moduleTranslation.lookupValues(op.args()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::NVVM::WMMAMmaF32F32M16N16K16Op>(opInst)) {

    moduleTranslation.mapValue(op.res()) = createNvvmIntrinsicCall(
        builder, llvm::Intrinsic::nvvm_wmma_m16n16k16_mma_row_row_f32_f32, moduleTranslation.lookupValues(op.args()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::NVVM::WMMAStoreF16M16N16K16Op>(opInst)) {

        createNvvmIntrinsicCall(
        builder, llvm::Intrinsic::nvvm_wmma_m16n16k16_store_d_f16_row_stride, moduleTranslation.lookupValues(op.args()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::NVVM::WMMAStoreF32M16N16K16Op>(opInst)) {

        createNvvmIntrinsicCall(
        builder, llvm::Intrinsic::nvvm_wmma_m16n16k16_store_d_f32_row_stride, moduleTranslation.lookupValues(op.args()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::NVVM::WarpSizeOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::nvvm_read_ptx_sreg_warpsize,
        { 
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;(void) inst;moduleTranslation.mapValue(op.res()) = inst;
  return success();
}
