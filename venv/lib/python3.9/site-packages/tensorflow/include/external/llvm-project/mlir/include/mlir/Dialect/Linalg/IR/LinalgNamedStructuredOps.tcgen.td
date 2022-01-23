  def MatmulI8I8I32Op : LinalgStructuredBase_Op<"matmul_i8_i8_i32", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SingleBlockImplicitTerminator<"YieldOp">
    /*extraInterfaces=*/, LinalgContractionOpInterface]> {
      
      let arguments = (ins
        Variadic<AnyShaped>:$inputs,
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
          $_state.addAttribute(
            "operand_segment_sizes",
            $_builder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          $_state.addAttributes(attributes);
          createAndFillStructuredOpRegion<MatmulI8I8I32Op>(
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
          $_state.addAttribute(
            "operand_segment_sizes",
            $_builder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          $_state.addAttributes(attributes);
          createAndFillStructuredOpRegion<MatmulI8I8I32Op>(
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
        return ::parseNamedStructuredOp<MatmulI8I8I32Op>(parser, result);
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
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }

        
      }];
  }
