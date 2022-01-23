
//===----------------------------------------------------------------------===//
// Op definition for MatmulOp
//===----------------------------------------------------------------------===//

def MatmulOp : LinalgStructuredBase_Op<"matmul", !listconcat([
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  SingleBlockImplicitTerminator<"YieldOp">],
  /*extraInterfaces=*/[LinalgContractionOpInterface])> {
    
  let summary = [{ Performs a matrix multiplication of two 2D inputs. }];
  let description = [{
    Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<MatmulOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<MatmulOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let printer = [{ return ::printNamedStructuredOp(p, *this); }];
    let parser = [{
      return ::parseNamedStructuredOp<MatmulOp>(parser, result);
    }];
    let hasFolder = 1;

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for QuantizedMatmulOp
//===----------------------------------------------------------------------===//

def QuantizedMatmulOp : LinalgStructuredBase_Op<"quantized_matmul", !listconcat([
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  SingleBlockImplicitTerminator<"YieldOp">],
  /*extraInterfaces=*/[])> {
    
  let summary = [{ Performs a matrix multiplication of two 2D inputs. }];
  let description = [{
    Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output. The quantized variant
includes zero-point adjustments for the left and right operands of the
matmul.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<QuantizedMatmulOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<QuantizedMatmulOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let printer = [{ return ::printNamedStructuredOp(p, *this); }];
    let parser = [{
      return ::parseNamedStructuredOp<QuantizedMatmulOp>(parser, result);
    }];
    let hasFolder = 1;

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for Mmt4DOp
//===----------------------------------------------------------------------===//

def Mmt4DOp : LinalgStructuredBase_Op<"mmt4d", !listconcat([
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  SingleBlockImplicitTerminator<"YieldOp">],
  /*extraInterfaces=*/[LinalgContractionOpInterface])> {
    
  let summary = [{ Performs a matrix-matrix-transpose multiplication of two 4D inputs. }];
  let description = [{
    Differences from linalg.matmul:
* The right hand side is transposed, whence the 't' in 'mmt'.
* The input and output tensors have a 4D shape instead of a 2D shape. They
  are interpreted as 2D matrices with one level of 2D tile subdivision,
  whence the 2+2=4 dimensions. The inner tile dimensions are identified with
  '0' suffixes below, for instance the LHS matrix shape (M, K, M0, K0) reads
  as: MxK tiles, each of shape M0xK0.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<Mmt4DOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<Mmt4DOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let printer = [{ return ::printNamedStructuredOp(p, *this); }];
    let parser = [{
      return ::parseNamedStructuredOp<Mmt4DOp>(parser, result);
    }];
    let hasFolder = 1;

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for BatchMatmulOp
//===----------------------------------------------------------------------===//

def BatchMatmulOp : LinalgStructuredBase_Op<"batch_matmul", !listconcat([
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  SingleBlockImplicitTerminator<"YieldOp">],
  /*extraInterfaces=*/[LinalgContractionOpInterface])> {
    
  let summary = [{ Performs a batched matrix multiplication of two 3D inputs. }];
  let description = [{
    Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<BatchMatmulOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<BatchMatmulOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let printer = [{ return ::printNamedStructuredOp(p, *this); }];
    let parser = [{
      return ::parseNamedStructuredOp<BatchMatmulOp>(parser, result);
    }];
    let hasFolder = 1;

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for QuantizedBatchMatmulOp
//===----------------------------------------------------------------------===//

def QuantizedBatchMatmulOp : LinalgStructuredBase_Op<"quantized_batch_matmul", !listconcat([
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  SingleBlockImplicitTerminator<"YieldOp">],
  /*extraInterfaces=*/[])> {
    
  let summary = [{ Performs a batched matrix multiplication of two 3D inputs. }];
  let description = [{
    Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output. The quantized variant
includes zero-point adjustments for the left and right operands of the
matmul.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<QuantizedBatchMatmulOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<QuantizedBatchMatmulOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let printer = [{ return ::printNamedStructuredOp(p, *this); }];
    let parser = [{
      return ::parseNamedStructuredOp<QuantizedBatchMatmulOp>(parser, result);
    }];
    let hasFolder = 1;

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for MatvecOp
//===----------------------------------------------------------------------===//

def MatvecOp : LinalgStructuredBase_Op<"matvec", !listconcat([
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  SingleBlockImplicitTerminator<"YieldOp">],
  /*extraInterfaces=*/[LinalgContractionOpInterface])> {
    
  let summary = [{ Performs a matrix-vector multiplication. }];
  let description = [{
    Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<MatvecOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<MatvecOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let printer = [{ return ::printNamedStructuredOp(p, *this); }];
    let parser = [{
      return ::parseNamedStructuredOp<MatvecOp>(parser, result);
    }];
    let hasFolder = 1;

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for VecmatOp
//===----------------------------------------------------------------------===//

def VecmatOp : LinalgStructuredBase_Op<"vecmat", !listconcat([
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  SingleBlockImplicitTerminator<"YieldOp">],
  /*extraInterfaces=*/[LinalgContractionOpInterface])> {
    
  let summary = [{ Performs a vector-matrix multiplication. }];
  let description = [{
    Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<VecmatOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<VecmatOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let printer = [{ return ::printNamedStructuredOp(p, *this); }];
    let parser = [{
      return ::parseNamedStructuredOp<VecmatOp>(parser, result);
    }];
    let hasFolder = 1;

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for BatchMatvecOp
//===----------------------------------------------------------------------===//

def BatchMatvecOp : LinalgStructuredBase_Op<"batch_matvec", !listconcat([
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  SingleBlockImplicitTerminator<"YieldOp">],
  /*extraInterfaces=*/[LinalgContractionOpInterface])> {
    
  let summary = [{ Performs a batched matrix-vector multiplication. }];
  let description = [{
    Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<BatchMatvecOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<BatchMatvecOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let printer = [{ return ::printNamedStructuredOp(p, *this); }];
    let parser = [{
      return ::parseNamedStructuredOp<BatchMatvecOp>(parser, result);
    }];
    let hasFolder = 1;

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for DotOp
//===----------------------------------------------------------------------===//

def DotOp : LinalgStructuredBase_Op<"dot", !listconcat([
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  SingleBlockImplicitTerminator<"YieldOp">],
  /*extraInterfaces=*/[LinalgContractionOpInterface])> {
    
  let summary = [{ Performs a dot product of two vectors to a scalar result. }];
  let description = [{
    Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<DotOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<DotOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let printer = [{ return ::printNamedStructuredOp(p, *this); }];
    let parser = [{
      return ::parseNamedStructuredOp<DotOp>(parser, result);
    }];
    let hasFolder = 1;

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for Conv1DOp
//===----------------------------------------------------------------------===//

def Conv1DOp : LinalgStructuredBase_Op<"conv_1d", !listconcat([
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  SingleBlockImplicitTerminator<"YieldOp">],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{ Performs 1-D convolution with no channels. }];
  let description = [{
    Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<Conv1DOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<Conv1DOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let printer = [{ return ::printNamedStructuredOp(p, *this); }];
    let parser = [{
      return ::parseNamedStructuredOp<Conv1DOp>(parser, result);
    }];
    let hasFolder = 1;

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for Conv2DOp
//===----------------------------------------------------------------------===//

def Conv2DOp : LinalgStructuredBase_Op<"conv_2d", !listconcat([
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  SingleBlockImplicitTerminator<"YieldOp">],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{ Performs 2-D convolution with no channels. }];
  let description = [{
    Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<Conv2DOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<Conv2DOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let printer = [{ return ::printNamedStructuredOp(p, *this); }];
    let parser = [{
      return ::parseNamedStructuredOp<Conv2DOp>(parser, result);
    }];
    let hasFolder = 1;

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for Conv3DOp
//===----------------------------------------------------------------------===//

def Conv3DOp : LinalgStructuredBase_Op<"conv_3d", !listconcat([
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  SingleBlockImplicitTerminator<"YieldOp">],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{ Performs 3-D convolution with no channels. }];
  let description = [{
    Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<Conv3DOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<Conv3DOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let printer = [{ return ::printNamedStructuredOp(p, *this); }];
    let parser = [{
      return ::parseNamedStructuredOp<Conv3DOp>(parser, result);
    }];
    let hasFolder = 1;

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for Conv1DNwcWcfOp
//===----------------------------------------------------------------------===//

def Conv1DNwcWcfOp : LinalgStructuredBase_Op<"conv_1d_nwc_wcf", !listconcat([
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  SingleBlockImplicitTerminator<"YieldOp">],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{ Performs 1-D convolution. }];
  let description = [{
    Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
RankedI64ElementsAttr<[1]>:$strides,
RankedI64ElementsAttr<[1]>:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<Conv1DNwcWcfOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<Conv1DNwcWcfOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(inputs);
    $_state.addOperands(outputs);
    $_state.addTypes(resultTensorTypes);
    $_state.addAttribute(
      "operand_segment_sizes",
      $_builder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<Conv1DNwcWcfOp>(
      $_builder,
      $_state,
      TypeRange(inputs),
      TypeRange(outputs));
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    $_state.addAttributes(attributes);
  }]>

    ];
    let printer = [{ return ::printNamedStructuredOp(p, *this); }];
    let parser = [{
      return ::parseNamedStructuredOp<Conv1DNwcWcfOp>(parser, result);
    }];
    let hasFolder = 1;

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();
    
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for Conv2DNhwcHwcfOp
//===----------------------------------------------------------------------===//

def Conv2DNhwcHwcfOp : LinalgStructuredBase_Op<"conv_2d_nhwc_hwcf", !listconcat([
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  SingleBlockImplicitTerminator<"YieldOp">],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{ Performs 2-D convolution. }];
  let description = [{
    Layout:
  * Input: NHWC.
  * Kernel: HWCF.

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
RankedI64ElementsAttr<[2]>:$strides,
RankedI64ElementsAttr<[2]>:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<Conv2DNhwcHwcfOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<Conv2DNhwcHwcfOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(inputs);
    $_state.addOperands(outputs);
    $_state.addTypes(resultTensorTypes);
    $_state.addAttribute(
      "operand_segment_sizes",
      $_builder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<Conv2DNhwcHwcfOp>(
      $_builder,
      $_state,
      TypeRange(inputs),
      TypeRange(outputs));
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    $_state.addAttributes(attributes);
  }]>

    ];
    let printer = [{ return ::printNamedStructuredOp(p, *this); }];
    let parser = [{
      return ::parseNamedStructuredOp<Conv2DNhwcHwcfOp>(parser, result);
    }];
    let hasFolder = 1;

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();
    
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for Conv2DNhwcHwcfQOp
//===----------------------------------------------------------------------===//

def Conv2DNhwcHwcfQOp : LinalgStructuredBase_Op<"conv_2d_nhwc_hwcf_q", !listconcat([
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  SingleBlockImplicitTerminator<"YieldOp">],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{ Performs 2-D convolution with zero point offsets. }];
  let description = [{
    Layout:
  * Input: NHWC.
  * Kernel: HWCF.

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output. This includes the zero
point offsets common to quantized operations.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
RankedI64ElementsAttr<[2]>:$strides,
RankedI64ElementsAttr<[2]>:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<Conv2DNhwcHwcfQOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<Conv2DNhwcHwcfQOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(inputs);
    $_state.addOperands(outputs);
    $_state.addTypes(resultTensorTypes);
    $_state.addAttribute(
      "operand_segment_sizes",
      $_builder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<Conv2DNhwcHwcfQOp>(
      $_builder,
      $_state,
      TypeRange(inputs),
      TypeRange(outputs));
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    $_state.addAttributes(attributes);
  }]>

    ];
    let printer = [{ return ::printNamedStructuredOp(p, *this); }];
    let parser = [{
      return ::parseNamedStructuredOp<Conv2DNhwcHwcfQOp>(parser, result);
    }];
    let hasFolder = 1;

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();
    
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for Conv2DNchwFchwOp
//===----------------------------------------------------------------------===//

def Conv2DNchwFchwOp : LinalgStructuredBase_Op<"conv_2d_nchw_fchw", !listconcat([
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  SingleBlockImplicitTerminator<"YieldOp">],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{ Performs 2-D convolution. }];
  let description = [{
    Layout:
  * Input: NCHW.
  * Kernel: FCHW.

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
RankedI64ElementsAttr<[2]>:$strides,
RankedI64ElementsAttr<[2]>:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<Conv2DNchwFchwOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<Conv2DNchwFchwOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(inputs);
    $_state.addOperands(outputs);
    $_state.addTypes(resultTensorTypes);
    $_state.addAttribute(
      "operand_segment_sizes",
      $_builder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<Conv2DNchwFchwOp>(
      $_builder,
      $_state,
      TypeRange(inputs),
      TypeRange(outputs));
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    $_state.addAttributes(attributes);
  }]>

    ];
    let printer = [{ return ::printNamedStructuredOp(p, *this); }];
    let parser = [{
      return ::parseNamedStructuredOp<Conv2DNchwFchwOp>(parser, result);
    }];
    let hasFolder = 1;

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();
    
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for Conv3DNdhwcDhwcfOp
//===----------------------------------------------------------------------===//

def Conv3DNdhwcDhwcfOp : LinalgStructuredBase_Op<"conv_3d_ndhwc_dhwcf", !listconcat([
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  SingleBlockImplicitTerminator<"YieldOp">],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{ Performs 3-D convolution. }];
  let description = [{
    Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
RankedI64ElementsAttr<[3]>:$strides,
RankedI64ElementsAttr<[3]>:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<Conv3DNdhwcDhwcfOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<Conv3DNdhwcDhwcfOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(inputs);
    $_state.addOperands(outputs);
    $_state.addTypes(resultTensorTypes);
    $_state.addAttribute(
      "operand_segment_sizes",
      $_builder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<Conv3DNdhwcDhwcfOp>(
      $_builder,
      $_state,
      TypeRange(inputs),
      TypeRange(outputs));
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    $_state.addAttributes(attributes);
  }]>

    ];
    let printer = [{ return ::printNamedStructuredOp(p, *this); }];
    let parser = [{
      return ::parseNamedStructuredOp<Conv3DNdhwcDhwcfOp>(parser, result);
    }];
    let hasFolder = 1;

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();
    
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for DepthwiseConv2DNhwOp
//===----------------------------------------------------------------------===//

def DepthwiseConv2DNhwOp : LinalgStructuredBase_Op<"depthwise_conv2D_nhw", !listconcat([
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  SingleBlockImplicitTerminator<"YieldOp">],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{ Performs depth-wise 2-D convolution. }];
  let description = [{
    Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output. Multiplier is set to 1
which is a special case for most dpethwise convolutions.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
RankedI64ElementsAttr<[2]>:$strides,
RankedI64ElementsAttr<[2]>:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<DepthwiseConv2DNhwOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<DepthwiseConv2DNhwOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(inputs);
    $_state.addOperands(outputs);
    $_state.addTypes(resultTensorTypes);
    $_state.addAttribute(
      "operand_segment_sizes",
      $_builder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<DepthwiseConv2DNhwOp>(
      $_builder,
      $_state,
      TypeRange(inputs),
      TypeRange(outputs));
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    $_state.addAttributes(attributes);
  }]>

    ];
    let printer = [{ return ::printNamedStructuredOp(p, *this); }];
    let parser = [{
      return ::parseNamedStructuredOp<DepthwiseConv2DNhwOp>(parser, result);
    }];
    let hasFolder = 1;

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();
    
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for DepthwiseConv2DNhwQOp
//===----------------------------------------------------------------------===//

def DepthwiseConv2DNhwQOp : LinalgStructuredBase_Op<"depthwise_conv2D_nhw_q", !listconcat([
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  SingleBlockImplicitTerminator<"YieldOp">],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{ Performs depth-wise 2-D convolution. }];
  let description = [{
    Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
RankedI64ElementsAttr<[2]>:$strides,
RankedI64ElementsAttr<[2]>:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<DepthwiseConv2DNhwQOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<DepthwiseConv2DNhwQOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(inputs);
    $_state.addOperands(outputs);
    $_state.addTypes(resultTensorTypes);
    $_state.addAttribute(
      "operand_segment_sizes",
      $_builder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<DepthwiseConv2DNhwQOp>(
      $_builder,
      $_state,
      TypeRange(inputs),
      TypeRange(outputs));
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    $_state.addAttributes(attributes);
  }]>

    ];
    let printer = [{ return ::printNamedStructuredOp(p, *this); }];
    let parser = [{
      return ::parseNamedStructuredOp<DepthwiseConv2DNhwQOp>(parser, result);
    }];
    let hasFolder = 1;

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();
    
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for DepthwiseConv2DNhwcOp
//===----------------------------------------------------------------------===//

def DepthwiseConv2DNhwcOp : LinalgStructuredBase_Op<"depthwise_conv2D_nhwc", !listconcat([
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  SingleBlockImplicitTerminator<"YieldOp">],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{ Performs depth-wise 2-D convolution. }];
  let description = [{
    Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
RankedI64ElementsAttr<[2]>:$strides,
RankedI64ElementsAttr<[2]>:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<DepthwiseConv2DNhwcOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<DepthwiseConv2DNhwcOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(inputs);
    $_state.addOperands(outputs);
    $_state.addTypes(resultTensorTypes);
    $_state.addAttribute(
      "operand_segment_sizes",
      $_builder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<DepthwiseConv2DNhwcOp>(
      $_builder,
      $_state,
      TypeRange(inputs),
      TypeRange(outputs));
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    $_state.addAttributes(attributes);
  }]>

    ];
    let printer = [{ return ::printNamedStructuredOp(p, *this); }];
    let parser = [{
      return ::parseNamedStructuredOp<DepthwiseConv2DNhwcOp>(parser, result);
    }];
    let hasFolder = 1;

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();
    
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for DepthwiseConv2DNhwcQOp
//===----------------------------------------------------------------------===//

def DepthwiseConv2DNhwcQOp : LinalgStructuredBase_Op<"depthwise_conv2D_nhwc_q", !listconcat([
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  SingleBlockImplicitTerminator<"YieldOp">],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{ Performs depth-wise 2-D convolution. }];
  let description = [{
    Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
RankedI64ElementsAttr<[2]>:$strides,
RankedI64ElementsAttr<[2]>:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<DepthwiseConv2DNhwcQOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<DepthwiseConv2DNhwcQOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(inputs);
    $_state.addOperands(outputs);
    $_state.addTypes(resultTensorTypes);
    $_state.addAttribute(
      "operand_segment_sizes",
      $_builder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<DepthwiseConv2DNhwcQOp>(
      $_builder,
      $_state,
      TypeRange(inputs),
      TypeRange(outputs));
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    $_state.addAttributes(attributes);
  }]>

    ];
    let printer = [{ return ::printNamedStructuredOp(p, *this); }];
    let parser = [{
      return ::parseNamedStructuredOp<DepthwiseConv2DNhwcQOp>(parser, result);
    }];
    let hasFolder = 1;

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();
    
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for PoolingNhwcSumOp
//===----------------------------------------------------------------------===//

def PoolingNhwcSumOp : LinalgStructuredBase_Op<"pooling_nhwc_sum", !listconcat([
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  SingleBlockImplicitTerminator<"YieldOp">],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{ Performs sum pooling. }];
  let description = [{
    Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
RankedI64ElementsAttr<[2]>:$strides,
RankedI64ElementsAttr<[2]>:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<PoolingNhwcSumOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<PoolingNhwcSumOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(inputs);
    $_state.addOperands(outputs);
    $_state.addTypes(resultTensorTypes);
    $_state.addAttribute(
      "operand_segment_sizes",
      $_builder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<PoolingNhwcSumOp>(
      $_builder,
      $_state,
      TypeRange(inputs),
      TypeRange(outputs));
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    $_state.addAttributes(attributes);
  }]>

    ];
    let printer = [{ return ::printNamedStructuredOp(p, *this); }];
    let parser = [{
      return ::parseNamedStructuredOp<PoolingNhwcSumOp>(parser, result);
    }];
    let hasFolder = 1;

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();
    
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for PoolingNhwcMaxOp
//===----------------------------------------------------------------------===//

def PoolingNhwcMaxOp : LinalgStructuredBase_Op<"pooling_nhwc_max", !listconcat([
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  SingleBlockImplicitTerminator<"YieldOp">],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{ Performs max pooling. }];
  let description = [{
    Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
RankedI64ElementsAttr<[2]>:$strides,
RankedI64ElementsAttr<[2]>:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<PoolingNhwcMaxOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<PoolingNhwcMaxOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(inputs);
    $_state.addOperands(outputs);
    $_state.addTypes(resultTensorTypes);
    $_state.addAttribute(
      "operand_segment_sizes",
      $_builder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<PoolingNhwcMaxOp>(
      $_builder,
      $_state,
      TypeRange(inputs),
      TypeRange(outputs));
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    $_state.addAttributes(attributes);
  }]>

    ];
    let printer = [{ return ::printNamedStructuredOp(p, *this); }];
    let parser = [{
      return ::parseNamedStructuredOp<PoolingNhwcMaxOp>(parser, result);
    }];
    let hasFolder = 1;

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();
    
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for PoolingNchwMaxOp
//===----------------------------------------------------------------------===//

def PoolingNchwMaxOp : LinalgStructuredBase_Op<"pooling_nchw_max", !listconcat([
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  SingleBlockImplicitTerminator<"YieldOp">],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{ Performs max pooling. }];
  let description = [{
    Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
RankedI64ElementsAttr<[2]>:$strides,
RankedI64ElementsAttr<[2]>:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<PoolingNchwMaxOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<PoolingNchwMaxOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(inputs);
    $_state.addOperands(outputs);
    $_state.addTypes(resultTensorTypes);
    $_state.addAttribute(
      "operand_segment_sizes",
      $_builder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<PoolingNchwMaxOp>(
      $_builder,
      $_state,
      TypeRange(inputs),
      TypeRange(outputs));
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    $_state.addAttributes(attributes);
  }]>

    ];
    let printer = [{ return ::printNamedStructuredOp(p, *this); }];
    let parser = [{
      return ::parseNamedStructuredOp<PoolingNchwMaxOp>(parser, result);
    }];
    let hasFolder = 1;

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();
    
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for PoolingNhwcMinOp
//===----------------------------------------------------------------------===//

def PoolingNhwcMinOp : LinalgStructuredBase_Op<"pooling_nhwc_min", !listconcat([
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  SingleBlockImplicitTerminator<"YieldOp">],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{ Performs min pooling. }];
  let description = [{
    Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
RankedI64ElementsAttr<[2]>:$strides,
RankedI64ElementsAttr<[2]>:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<PoolingNhwcMinOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<PoolingNhwcMinOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(inputs);
    $_state.addOperands(outputs);
    $_state.addTypes(resultTensorTypes);
    $_state.addAttribute(
      "operand_segment_sizes",
      $_builder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<PoolingNhwcMinOp>(
      $_builder,
      $_state,
      TypeRange(inputs),
      TypeRange(outputs));
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    $_state.addAttributes(attributes);
  }]>

    ];
    let printer = [{ return ::printNamedStructuredOp(p, *this); }];
    let parser = [{
      return ::parseNamedStructuredOp<PoolingNhwcMinOp>(parser, result);
    }];
    let hasFolder = 1;

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();
    
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for PoolingNdhwcSumOp
//===----------------------------------------------------------------------===//

def PoolingNdhwcSumOp : LinalgStructuredBase_Op<"pooling_ndhwc_sum", !listconcat([
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  SingleBlockImplicitTerminator<"YieldOp">],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{ Performs 3D sum pooling. }];
  let description = [{
    Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
RankedI64ElementsAttr<[3]>:$strides,
RankedI64ElementsAttr<[3]>:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<PoolingNdhwcSumOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<PoolingNdhwcSumOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(inputs);
    $_state.addOperands(outputs);
    $_state.addTypes(resultTensorTypes);
    $_state.addAttribute(
      "operand_segment_sizes",
      $_builder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<PoolingNdhwcSumOp>(
      $_builder,
      $_state,
      TypeRange(inputs),
      TypeRange(outputs));
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    $_state.addAttributes(attributes);
  }]>

    ];
    let printer = [{ return ::printNamedStructuredOp(p, *this); }];
    let parser = [{
      return ::parseNamedStructuredOp<PoolingNdhwcSumOp>(parser, result);
    }];
    let hasFolder = 1;

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();
    
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for PoolingNdhwcMaxOp
//===----------------------------------------------------------------------===//

def PoolingNdhwcMaxOp : LinalgStructuredBase_Op<"pooling_ndhwc_max", !listconcat([
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  SingleBlockImplicitTerminator<"YieldOp">],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{ Performs 3D max pooling. }];
  let description = [{
    Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
RankedI64ElementsAttr<[3]>:$strides,
RankedI64ElementsAttr<[3]>:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<PoolingNdhwcMaxOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<PoolingNdhwcMaxOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(inputs);
    $_state.addOperands(outputs);
    $_state.addTypes(resultTensorTypes);
    $_state.addAttribute(
      "operand_segment_sizes",
      $_builder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<PoolingNdhwcMaxOp>(
      $_builder,
      $_state,
      TypeRange(inputs),
      TypeRange(outputs));
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    $_state.addAttributes(attributes);
  }]>

    ];
    let printer = [{ return ::printNamedStructuredOp(p, *this); }];
    let parser = [{
      return ::parseNamedStructuredOp<PoolingNdhwcMaxOp>(parser, result);
    }];
    let hasFolder = 1;

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();
    
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for PoolingNdhwcMinOp
//===----------------------------------------------------------------------===//

def PoolingNdhwcMinOp : LinalgStructuredBase_Op<"pooling_ndhwc_min", !listconcat([
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  SingleBlockImplicitTerminator<"YieldOp">],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{ Performs 3D min pooling. }];
  let description = [{
    Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
RankedI64ElementsAttr<[3]>:$strides,
RankedI64ElementsAttr<[3]>:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<PoolingNdhwcMinOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<PoolingNdhwcMinOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(inputs);
    $_state.addOperands(outputs);
    $_state.addTypes(resultTensorTypes);
    $_state.addAttribute(
      "operand_segment_sizes",
      $_builder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<PoolingNdhwcMinOp>(
      $_builder,
      $_state,
      TypeRange(inputs),
      TypeRange(outputs));
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    $_state.addAttributes(attributes);
  }]>

    ];
    let printer = [{ return ::printNamedStructuredOp(p, *this); }];
    let parser = [{
      return ::parseNamedStructuredOp<PoolingNdhwcMinOp>(parser, result);
    }];
    let hasFolder = 1;

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();
    
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for FillRng2DOp
//===----------------------------------------------------------------------===//

def FillRng2DOp : LinalgStructuredBase_Op<"fill_rng_2d", !listconcat([
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  SingleBlockImplicitTerminator<"YieldOp">],
  /*extraInterfaces=*/[])> {
    
  let summary = [{ Fills the output tensor with pseudo random numbers. }];
  let description = [{
    The operation generations pseudo random numbers using a linear congruential
generator. It provides no guarantees regarding the distribution of the
generated random numbers. Instead of generating the random numbers
sequentially, it instantiates one random number generator per data element
and runs them in parallel. The seed operand and the indices of the data
element seed the random number generation. The min and max operands limit
the range of the generated random numbers.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<FillRng2DOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<FillRng2DOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let printer = [{ return ::printNamedStructuredOp(p, *this); }];
    let parser = [{
      return ::parseNamedStructuredOp<FillRng2DOp>(parser, result);
    }];
    let hasFolder = 1;

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for SoftPlus2DOp
//===----------------------------------------------------------------------===//

def SoftPlus2DOp : LinalgStructuredBase_Op<"soft_plus_2d", !listconcat([
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  SingleBlockImplicitTerminator<"YieldOp">],
  /*extraInterfaces=*/[])> {
    
  let summary = [{ Implements the soft plus operator. }];
  let description = [{
    Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<SoftPlus2DOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<SoftPlus2DOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let printer = [{ return ::printNamedStructuredOp(p, *this); }];
    let parser = [{
      return ::parseNamedStructuredOp<SoftPlus2DOp>(parser, result);
    }];
    let hasFolder = 1;

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}
