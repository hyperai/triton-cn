### `tt.call` (triton::CallOp)


*调用操作*


语法：

```plain
operation ::= `tt.call` $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
```


`tt.call` 操作表示对位于调用位置相同符号作用域中的函数的直接调用。调用的操作和结果类型必须与指定的函数类型匹配。被调用者被编码为名为 「callee」的符号引用属性。


示例：

```plain
%2 = tt.call @my_add(%0, %1) : (f32, f32) -> f32
```


特征：`TensorSizeTrait`, `VerifyTensorLayoutsTrait`


接口：`CallOpInterface`, `SymbolUserOpInterface`


#### 属性：

|属性|MLIR 类型|描述|
|:----|:----|:----|
|callee|::mlir::FlatSymbolRefAttr|扁平符号引用属性|

#### 操作：

|操作|描述|
|:----|:----|
|operands|任何类型的变量|

#### 结果：

|结果|描述|
|:----|:----|
|«unnamed»|任何类型的变量|

### `tt.func` (triton::FuncOp)


*名称包含单个*`SSACFG`*区域的操作。*


函数内部的操作不能隐式捕获定义在函数外部的值，即函数是 `IsolatedFromAbove`  的。所有外部引用必须使用函数参数或属性来建立符号连接（例如通过类似 `SymbolRefAttr` 这样的字符串属性按名称引用的符号）。外部函数声明（用于引用在其他模块中声明的函数）没有函数体。虽然 MLIR 文本形式提供了对函数参数的内联语法，但它们在内部表示为区域中第一个块的「块参数」。


只能在函数参数、结果或函数本身的属性字典中指定 dialect 属性名称。


示例：

```plain
// External function definitions.
// 外部函数定义
tt.func @abort()
tt.func @scribble(i32, i64, memref<? x 128 x f32, #layout_map0>) -> f64


// A function that returns its argument twice:
// 返回其参数两次的函数：
tt.func @count(%x: i64) -> (i64, i64)
  attributes {fruit: "banana"} {
  return %x, %x: i64, i64
}


// A function with an argument attribute
// 带有参数属性的函数
tt.func @example_fn_arg(%x: i32 {swift.self = unit})


// A function with a result attribute
// 带有结果属性的函数
tt.func @example_fn_result() -> (f64 {dialectName.attrName = 0 : i64})


// A function with an attribute
// 带有属性的函数
tt.func @example_fn_attr() attributes {dialectName.attrName = false}
```


特征：`AffineScope`, `AutomaticAllocationScope`, `IsolatedFromAbove`, `TensorSizeTrait`, `VerifyTensorLayoutsTrait`


接口：`CallableOpInterface`, `FunctionOpInterface`, `OpAsmOpInterface`, `Symbol`

#### 

#### 属性：

|属性|MLIR 类型|描述|
|:----|:----|:----|
|sym_name|::mlir::StringAttr|字符串属性|
|function_type|::mlir::TypeAttr|函数类型的类型属性|
|sym_visibility|::mlir::StringAttr|字符串属性|
|arg_attrs|::mlir::ArrayAttr|字典属性的数组|
|res_attrs|::mlir::ArrayAttr|字典属性的数组|

### `tt.return` (triton::ReturnOp)


*函数返回操作*


语法：

```plain
operation ::= `tt.return` attr-dict ($srcs^ `:` type($srcs))?
```


`tt.return` 操作表示函数内部的返回操作。操作接受可变数量的操作，不产生结果。操作的数量和类型必须与包含操作的函数的签名匹配。


示例：

```plain
tt.func @foo() : (i32, f8) {
  ...
  tt.return %0, %1 : i32, f8
}
```


特征：`AlwaysSpeculatableImplTrait`, `HasParent<FuncOp>`, `ReturnLike`, `TensorSizeTrait`, `Terminator`, `VerifyTensorLayoutsTrait`


接口：`ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `RegionBranchTerminatorOpInterface`


效果：`MemoryEffects::Effect{}`

#### 

#### 操作：

|操作|描述|
|:----|:----|
|srcs|任何类型的变量|

### `tt.addptr` (triton::AddPtrOp)


语法：

```plain
operation ::= `tt.addptr` $ptr `,` $offset attr-dict `:` type($result) `,` type($offset)
```


特征：`AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultEncoding`, `SameOperandsAndResultShape`, `TensorSizeTrait`, `VerifyTensorLayoutsTrait`


接口：`ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`


效果：`MemoryEffects::Effect{}`

#### 

#### 操作：

|操作|描述|
|:----|:----|
|ptr|指针/指针的有序张量|
|offset|整数/整数值的有序张量|

#### 结果：

|结果|描述|
|:----|:----|
|result|指针/指针值的有序张量|

### `tt.advance` (triton::AdvanceOp)


*通过偏移量推进张量指针*


语法：

```plain
operation ::= `tt.advance` $ptr `,` `[` $offsets `]` attr-dict `:` type($result)
```


特征：`AlwaysSpeculatableImplTrait`, `TensorSizeTrait`, `VerifyTensorLayoutsTrait`


接口：`ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`


效果：`MemoryEffects::Effect{}`

#### 

#### 操作：

|操作|描述|
|:----|:----|
|ptr|指针|
|offsets|32-bit 无符号整数的可变参数|

#### 结果：

|结果|描述|
|:----|:----|
|result|指针|

### `tt.assert` (triton::AssertOp)


*设备端断言，类似于 CUDA 用于正确性检查*


语法：

```plain
operation ::= `tt.assert` $condition `,` $message `,` $file `,` $func `,` $line attr-dict `:` type($condition)
```


`tt.assert` 接受条件张量、消息字符串、文件字符串、函数字符串和行号。如果条件为 false，则打印消息并终止程序。


特征：`TensorSizeTrait`, `VerifyTensorLayoutsTrait`、


接口：`MemoryEffectOpInterface (MemoryEffectOpInterface)`


效果：`MemoryEffects::Effect{MemoryEffects::Write on ::mlir::triton::GlobalMemory}`

#### 

#### 属性：

|属性|MLIR 类型|描述|
|:----|:----|:----|
|message|::mlir::StringAttr|字符串属性|
|file|::mlir::StringAttr|字符串属性|
|func|::mlir::StringAttr|字符串属性|
|line|::mlir::IntegerAttr|32-bit 无符号整数属性|

#### 操作：

|操作数|描述|
|:----|:----|
|condition|浮点/整数/指针的有序张量|

### `tt.atomic_cas` (triton::AtomicCASOp)


*原子比较并交换 (cas)*


语法：

```plain
operation ::= `tt.atomic_cas` $sem `,` $scope `,` $ptr `,` $cmp `,` $val attr-dict `:`
              functional-type(operands, $result)
```


将 $cmp 与位置 $ptr 处的数据 $old 进行比较，


如果 $old == $cmp，则将 $val 存储到 $ptr，


否则将 $old 存储到 $ptr，


返回 $old


特征：`SameOperandsAndResultEncoding`, `SameOperandsAndResultShape`, `TensorSizeTrait`, `VerifyTensorLayoutsTrait`


接口：`MemoryEffectOpInterface (MemoryEffectOpInterface)`


效果：`MemoryEffects::Effect{MemoryEffects::Read on ::mlir::triton::GlobalMemory}`, `MemoryEffects::Effect{MemoryEffects::Write on ::mlir::triton::GlobalMemory}`

#### 

#### 属性：

|属性|MLIR 类型|描述|
|:----|:----|:----|
|sem|::mlir::triton::MemSemanticAttr|允许的 32-bit 无符号整数值：1, 2, 3, 4|
|scope|::mlir::triton::MemSyncScopeAttr|允许的 32-bit 无符号整数值：1, 2, 3|

#### 操作：

|操作数|描述|
|:----|:----|
|ptr|指针/指针值的有序张量|
|cmp|浮点/浮点值的有序张量，整数/整数值的有序张量，指针或指针值的有序张量，指针|
|val|浮点/浮点值的有序张量，整数/整数值的有序张量，指针或指针值的有序张量，指针|

#### 结果：

|结果|描述|
|:----|:----|
|result|浮点/浮点值的有序张量，整数/整数值的有序张量，指针或指针值的有序张量，指针|

### `tt.atomic_rmw` (triton::AtomicRMWOp)


*原子读改写操作*


语法：

```plain
operation ::= `tt.atomic_rmw` $atomic_rmw_op `,` $sem `,` $scope `,` $ptr `,` $val (`,` $mask^)?  attr-dict `:`
              functional-type(operands, $result)
```


加载 $ptr 处的数据，使用 $rmw_op 对 $val 进行操作，并将结果存储到 $ptr。


返回 `$`ptr 的旧值。


特征：`SameOperandsAndResultEncoding`, `SameOperandsAndResultShape`, `TensorSizeTrait`, `VerifyTensorLayoutsTrait`


接口：`MemoryEffectOpInterface (MemoryEffectOpInterface)`


效果：`MemoryEffects::Effect{MemoryEffects::Read on ::mlir::triton::GlobalMemory}`, `MemoryEffects::Effect{MemoryEffects::Write on ::mlir::triton::GlobalMemory}`

#### 

#### 属性：

|属性|MLIR 类型|描述|
|:----|:----|:----|
|atomic_rmw_op|::mlir::triton::RMWOpAttr|允许的 32-bit 无符号整数值：1, 2, 3, 4, 5, 6, 7, 8, 9, 10|
|sem|::mlir::triton::MemSemanticAttr|允许的 32-bit 无符号整数值：1, 2, 3, 4|
|scope|::mlir::triton::MemSyncScopeAttr|允许的 32-bit 无符号整数值：1, 2, 3|

#### 操作：

|操作数|描述|
|:----|:----|
|ptr|指针/指针值的有序张量|
|val|浮点/浮点值的有序张量，整数/整数值的有序张量，指针或指针值的有序张量，指针|
|mask|1-bit 无符号整数/1-bit 无符号整数值的有序张量|

#### 结果：

|结果|描述|
|:----|:----|
|result|浮点/浮点值的有序张量，整数/整数值的有序张量，指针或指针值的有序张量，指针|

### `tt.bitcast` (triton::BitcastOp)


*在相同位宽的类型之间进行转换*


语法：

```plain
operation ::= `tt.bitcast` $src attr-dict `:` type($src) `->` type($result)
```


特征：`AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultEncoding`, `SameOperandsAndResultShape`, `TensorSizeTrait`, `VerifyTensorLayoutsTrait`


接口：`ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`


效果：`MemoryEffects::Effect{}`

#### 

#### 操作：

|操作数|描述|
|:----|:----|
|src|浮点/浮点值的有序张量，整数/整数值的有序张量，指针或指针值的有序张量，指针|

#### 结果：

|结果|描述|
|:----|:----|
|result|浮点/浮点值的有序张量，整数/整数值的有序张量，指针/指针值的有序张量，指针|

### `tt.broadcast` (triton::BroadcastOp)

*广播张量*


语法：

```plain
operation ::= `tt.broadcast` $src attr-dict `:` type($src) `->` type($result)
```


对于给定的张量，广播操作将尺寸为 1 的一个或多个维度扩展为新的尺寸，例如，tensor<1x32x1xf32> -> tensor<2x32x4xf32>。不能改变非 1 维度的尺寸。


特征：`AlwaysSpeculatableImplTrait`, `SameOperandsAndResultElementType`, `SameOperandsAndResultEncoding`, `TensorSizeTrait`, `VerifyTensorLayoutsTrait`


接口：`ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`


效果：`MemoryEffects::Effect{}`

#### 

#### 操作：

|操作|描述|
|:----|:----|
|src|浮点/整数/指针值的有序张量|

#### 结果：

|结果|描述|
|:----|:----|
|result|浮点/整数/指针值的有序张量|

### `tt.cat` (triton::CatOp)


*连接 2 个张量*


语法：

```plain
operation ::= `tt.cat` $lhs `,` $rhs attr-dict `:` type($lhs) `->` type($result)
```


特征：`SameOperandsAndResultElementType`, `SameTypeOperands`, `TensorSizeTrait`, `VerifyTensorLayoutsTrait`


接口：`NoMemoryEffect (MemoryEffectOpInterface)`


效果：`MemoryEffects::Effect{}`

#### 

#### 操作：

|操作|描述|
|:----|:----|
|lhs|浮点/整数/指针值的有序张量|
|rhs|浮点/整数/指针值的有序张量|

#### 结果：

|结果|描述|
|:----|:----|
|result|浮点数/整数/指针值的有序张量|

### `tt.clampf` (triton::ClampFOp)


*浮点数类型的**钳位**操作*


语法：

```plain
operation ::= `tt.clampf` $x `,` $min `,` $max `,` `propagateNan` `=` $propagateNan attr-dict `:` type($result)
```


浮点数类型的钳位操作。


该操作接受三个参数：x、min 和 max。它返回一个与 x 相同形状的张量，其中的值被钳制在 [min, max] 范围内。


特征：`AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`, `TensorSizeTrait`, `VerifyTensorLayoutsTrait`


接口：`ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`


效果：`MemoryEffects::Effect{}`

#### 

#### 属性：

|属性|MLIR 类型|描述|
|:----|:----|:----|
|propagateNan|::mlir::triton::PropagateNanAttr|允许的32-bit 无符号整数 case：0, 65535|

#### 操作：

|操作|描述|
|:----|:----|
|x|浮点/浮点值的有序张量|
|min|浮点/浮点值的有序张量|
|max|浮点/浮点值的有序张量|

#### 结果：

|结果|描述|
|:----|:----|
|result|浮点/浮点值的有序张量|

### `tt.dot` (triton::DotOp)


*点乘*


语法：

```plain
operation ::= `tt.dot` $a`,` $b`,` $c (`,` `inputPrecision` `=` $inputPrecision^)? attr-dict `:`
              type($a) `*` type($b) `->` type($d)
```


$d = matrix_multiply($a, $b) + $c。$inputPrecision 描述了在输入为 f32 时如何使用 TC。它可以是以下之一：tf32、tf32x3、ieee。

*  tf32：使用 TC 的 tf32 操作。
* tf32x3：实现 3xTF32 技巧,更多信息请参见 F32DotTC.cpp 中的代码段。
* ieee：不使用 TC，在软件中实现点积操作。

 如果 GPU 没有 Tensor 核心或输入不是 f32，则忽略此标志。


特征：`AlwaysSpeculatableImplTrait`, `DotLike`, `TensorSizeTrait`, `VerifyTensorLayoutsTrait`

接口：`ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

效果：`MemoryEffects::Effect{}`

#### 

#### 属性：

|属性|MLIR 类型|描述|
|:----|:----|:----|
|inputPrecision|::mlir::triton::InputPrecisionAttr|允许的 32-bit 无符号整数 case : 0, 1, 2|
|maxNumImpreciseAcc|::mlir::IntegerAttr|32-bit 无符号整数属性|

#### 操作：

|操作|描述|
|:----|:----|
|a|浮点/整数值的有序张量|
|b|浮点数/整数值的有序张量|
|c|浮点数/整数值的有序张量|

#### 结果：

|结果|描述|
|:----|:----|
|d|浮点数/整数值的有序张量|

### `tt.elementwise_inline_asm` (triton::ElementwiseInlineAsmOp)


*应用内联汇编一个逐元素操作到一组打包元素。*


语法：

```plain
operation ::= `tt.elementwise_inline_asm` $asm_string attr-dict ($args^ `:` type($args))? `->` type($result)
```


运行一个内联 asm 块以生成一个或多个张量。


asm 块一次接收 `packed_element` 元素。具体接收哪些元素是未指定的。


特征：`Elementwise`, `SameOperandsAndResultEncoding`, `TensorSizeTrait`, `VerifyTensorLayoutsTrait`


接口：`MemoryEffectOpInterface`

#### 

#### 属性：

|属性|MLIR 类型|描述|
|:----|:----|:----|
|asm_string|::mlir::StringAttr|字符串属性|
|constraints|::mlir::StringAttr|字符串属性|
|pure|::mlir::BoolAttr|布尔属性|
|packed_element|::mlir::IntegerAttr|32-bit 无符号整数属性|

#### 操作：

|操作|描述|
|:----|:----|
|args|浮点/浮点值的有序张量，整数/整数值的有序张量，指针/指针值的有序张量，指针的可变参数。|

#### 结果：

|结果|描述|
|:----|:----|
|result|浮点/浮点值的有序张量，整数/整数值的有序张量，指针/指针值的有序张量，指针的可变参数。|

### `tt.expand_dims` (triton::ExpandDimsOp)


*扩展维度*


语法：

```plain
operation ::= `tt.expand_dims` $src attr-dict `:` type($src) `->` type($result)
```


特征: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultElementType`, `TensorSizeTrait`, `VerifyTensorLayoutsTrait`


接口: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`


效果: `MemoryEffects::Effect{}`

#### 

#### 属性:

|属性|MLIR 类型|描述|
|:----|:----|:----|
|axis|::mlir::IntegerAttr|32-bit 无符号整数属性|

#### 操作:

|操作|描述|
|:----|:----|
|src|浮点/整数/指针值的有序张量|

#### 结果:

|结果|描述|
|:----|:----|
|result|浮点/整数或指针值的有序张量|

### `tt.experimental_descriptor_load` (triton::ExperimentalDescriptorLoadOp)


*从描述符中加载*


语法:

```plain
operation ::= `tt.experimental_descriptor_load` $desc_ptr `[` $indices `]`
              oilist(
              `cacheModifier` `=` $cache |
              `evictionPolicy` `=` $evict
              )
              attr-dict `:` qualified(type($desc_ptr)) `->` type($result)
```


此操作在支持它的目标上会被降低为 Nvidia TMA 加载操作。`desc_ptr` 是指向在全局内存中分配的 TMA 描述符的指针。目标张量的类型和形状必须与描述符匹配，否则结果是未定义的。


这是一个后门，仅用于测试/实验。该操作将在将来被移除。


特征: `TensorSizeTrait`, `VerifyTensorLayoutsTrait`


接口: `MemoryEffectOpInterface (MemoryEffectOpInterface)`


效果: `MemoryEffects::Effect{MemoryEffects::Read on ::mlir::triton::GlobalMemory}`

#### 

#### 属性：

|属性|MLIR 类型|描述|
|:----|:----|:----|
|cache|::mlir::triton::CacheModifierAttr|允许的 32-bit 无符号整数 case: 1, 2, 3, 4, 5, 6|
|evict|::mlir::triton::EvictionPolicyAttr|允许的 32-bit 无符号整数 case: 1, 2, 3|

#### 操作：

|操作|描述|
|:----|:----|
|desc_ptr|Triton IR 类型系统中的指针类型(::mlir::triton::PointerType)|
|indices|32-bit 无符号整数的可变参数|

#### 结果：

|结果|描述|
|:----|:----|
|result|浮点/整数/指针值的有序张量|

### `tt.experimental_descriptor_store` (triton::ExperimentalDescriptorStoreOp)


*基于描述符保存值*


语法：

```plain
operation ::= `tt.experimental_descriptor_store` $desc_ptr `[` $indices `]` `,` $src
              attr-dict `:` qualified(type($desc_ptr)) `,` type($src)
```


此操作将在支持的目标上降级为 Nvidia TMA 存储操作。`desc_ptr` 是指向分配在全局内存中的 TMA 描述符的指针。`src` 的形状和类型必须与描述符匹配，否则结果是未定义的。


这是一个后门，仅用于测试/实验。该操作将在将来被移除。


特征：`TensorSizeTrait`, `VerifyTensorLayoutsTrait`


接口：`MemoryEffectOpInterface (MemoryEffectOpInterface)`


效果：`MemoryEffects::Effect{MemoryEffects::Write on ::mlir::triton::GlobalMemory}`

#### 

#### 操作：

|操作|描述|
|:----|:----|
|desc_ptr|Triton IR 类型系统中的指针类型(::mlir::triton::PointerType)|
|src|浮点数/整数/指针值的有序张量|
|indices|32-bit 无符号整数的可变参数|

### `tt.extern_elementwise` (triton::ExternElementwiseOp)


语法：

```plain
operation ::= `tt.extern_elementwise` operands attr-dict `:` functional-type(operands, $result)
```


调用外部函数 `$symbol`，该函数由 `$libpath/$libname` 实现，参数为 `$args`，返回结果为 `$libpath/$libname:$symbol($args...)`。


特征： `Elementwise`, `SameOperandsAndResultEncoding`, `SameVariadicOperandSize`, `TensorSizeTrait`, `VerifyTensorLayoutsTrait`


接口：`MemoryEffectOpInterface`

#### 

#### 属性：

|属性|MLIR 类型|描述|
|:----|:----|:----|
|libname|::mlir::StringAttr|字符串属性|
|libpath|::mlir::StringAttr|字符串属性|
|symbol|::mlir::StringAttr|字符串属性|
|pure|::mlir::BoolAttr|布尔属性|

#### 操作：

|操作|描述|
|:----|:----|
|srcs|浮点/浮点值的有序张量，整数/整数值的有序张量，指针/指针值的有序张量，指针的可变参数。|

#### 结果：

|结果|描述|
|:----|:----|
|result|浮点/浮点值的有序张量，整数/整数值的有序张量，指针/指针值的有序张量，指针。|

### `tt.fp_to_fp` (triton::FpToFpOp)


*自定义类型的浮点数转换*


语法：

```plain
operation ::= `tt.fp_to_fp` $src attr-dict  (`,` `rounding` `=` $rounding^)? `:` type($src) `->` type($result)
```


自定义类型 (F8) 的浮点数转换，以及非默认的舍入模式。


F8 <-> FP16, BF16, FP32, FP64


特征：`AlwaysSpeculatableImplTrait`, `SameOperandsAndResultEncoding`, `SameOperandsAndResultShape`, `TensorSizeTrait`, `VerifyTensorLayoutsTrait`


接口：`ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`


效果：`MemoryEffects::Effect{}`

#### 

#### 属性：

|属性|MLIR 类型|描述|
|:----|:----|:----|
|rounding|::mlir::triton::RoundingModeAttr|允许的32-bit 无符号整数 case: 0, 1|

#### 操作：

|操作|描述|
|:----|:----|
|src|浮点值有序张量|

#### 结果：

|结果|描述|
|:----|:----|
|result|浮点值有序张量|

### `tt.get_num_programs` (triton::GetNumProgramsOp)


语法：

```plain
operation ::= `tt.get_num_programs` $axis attr-dict `:` type($result)
```


特征：`AlwaysSpeculatableImplTrait`, `TensorSizeTrait`, `VerifyTensorLayoutsTrait`


接口：`ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`


效果：`MemoryEffects::Effect{}`

#### 

#### 属性：

|Attribute|MLIR Type|Description|
|:----|:----|:----|
|axis|::mlir::triton::ProgramIDDimAttr|allowed 32-bit signless integer cases: 0, 1, 2|

|属性|MLIR 类型|描述|
|:----|:----|:----|
|axis|::mlir::triton::ProgramIDDimAttr|允许的32-bit 无符号整数 case: 0, 1, 2|

#### 操作：

|结果|描述|
|:----|:----|
|result|32-bit 无符号整数|

### `tt.get_program_id` (triton::GetProgramIdOp)


语法：

```plain
operation ::= `tt.get_program_id` $axis attr-dict `:` type($result)
```


特征：`AlwaysSpeculatableImplTrait`, `TensorSizeTrait`, `VerifyTensorLayoutsTrait`


接口：`ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`


效果：`MemoryEffects::Effect{}`

#### 

#### 属性：

|Attribute|MLIR Type|Description|
|:----|:----|:----|
|axis|::mlir::triton::ProgramIDDimAttr|allowed 32-bit signless integer cases: 0, 1, 2|

|Attribute|MLIR Type|Description|
|:----|:----|:----|
|axis|::mlir::triton::ProgramIDDimAttr|允许的 32-bit 无符号整数 case: 0, 1, 2|

#### 结果:

|结果|描述|
|:----|:----|
|result|32-bit 无符号整数|

### `tt.histogram` (triton::HistogramOp)


*返回输入的直方图*

 

语法：

```plain
operation ::= `tt.histogram` $src attr-dict `:` type($src) `->` type($result)
```


返回输入张量的直方图。输出张量的维度等于 bins 的数量，每个 bin 的宽度为 1，bins 从 0 开始。


特征：`AlwaysSpeculatableImplTrait`, `TensorSizeTrait`, `VerifyTensorLayoutsTrait`


接口：`ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`


效果：`MemoryEffects::Effect{}`

#### 

#### 操作:

|操作|描述|
|:----|:----|
|src|整数值有序张量|

#### 结果:

|结果|描述|
|:----|:----|
|result|整数值有序张量|

### `tt.int_to_ptr` (triton::IntToPtrOp)

 

*转换 int64 为 指针*


语法：

```plain
operation ::= `tt.int_to_ptr` $src attr-dict `:` type($src) `->` type($result)
```


特征：`AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultEncoding`, `SameOperandsAndResultShape`, `TensorSizeTrait`, `VerifyTensorLayoutsTrait`


接口：`ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`


效果：`MemoryEffects::Effect{}`

#### 

#### 操作:

|操作|描述|
|:----|:----|
|src|64-bit 无符号整数/64-bit 无符号整数张量|

#### 结果：

|结果|描述|
|:----|:----|
|result|指针/指针值的有序张量|

### `tt.join` (triton::JoinOp)


*沿着新的次要维度连接两个张量*


语法：

```plain
operation ::= `tt.join` $lhs `,` $rhs attr-dict `:` type($lhs) `->` type($result)
```


例如，如果两个输入张量的形状为 4x8xf32，则返回形状为 4x8x2xf32 的张量。


由于 Triton 张量始终具有 2 的幂次方个元素，因此两个输入张量必须具有相同的形状。


特征：`SameTypeOperands`, `TensorSizeTrait`, `VerifyTensorLayoutsTrait`


接口：`InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`


效果：`MemoryEffects::Effect{}`

#### 

#### 操作:

|操作|描述|
|:----|:----|
|lhs|浮点数/整数/指针值的有序张量|
|rhs|浮点数/整数/指针值的有序张量|

#### 结果:

|结果|描述|
|:----|:----|
|result|浮点数/整数/指针值的有序张量|

### `tt.load` (triton::LoadOp)


*从指针张量或张量指针中加载*


语法：

```plain
operation ::= `tt.load` $ptr (`,` $mask^)? (`,` $other^)?
              oilist(
              `cacheModifier` `=` $cache |
              `evictionPolicy` `=` $evict
              )
              attr-dict `:` type($ptr)
```


特征：`AttrSizedOperandSegments`, `SameLoadStoreOperandsAndResultEncoding`, `SameLoadStoreOperandsAndResultShape`, `TensorSizeTrait`, `VerifyTensorLayoutsTrait`


接口：`InferTypeOpInterface`, `MemoryEffectOpInterface`

#### 

#### 属性：

|属性|MLIR 类型|描述|
|:----|:----|:----|
|boundaryCheck|::mlir::DenseI32ArrayAttr|i32 稠密数组属性|
|padding|::mlir::triton::PaddingOptionAttr|允许的 32-bit 无符号整数选项：1, 2|
|cache|::mlir::triton::CacheModifierAttr|允许的 32-bit 无符号整数选项：1, 2, 3, 4, 5, 6|
|evict|::mlir::triton::EvictionPolicyAttr|允许的 32-bit 无符号整数选项：1, 2, 3|
|isVolatile|::mlir::BoolAttr|布尔属性|

#### 操作：

|操作|描述|
|:----|:----|
|ptr|ptr or ranked tensor of ptr values or ptr 指针，指针/指针的有序张量|
|mask|1-bit 无符号整数/1-bit 无符号整数值的有序张量|
|other|浮点/浮点值的有序张量，整数/整数值的有序张量，指针/指针值的有序张量，指针。|

#### 结果：

|结果|描述|
|:----|:----|
|result|浮点/浮点值的有序张量，整数/整数值的有序张量，指针/指针值的有序张量，指针。|

### `tt.make_range` (triton::MakeRangeOp)


*生成 range*


语法：

```plain
operation ::= `tt.make_range` attr-dict `:` type($result)
```


返回一个一维的 int32 张量。

 

数值范围从 $start 到 $end（不包括 $end），步长为 1。


特征：`AlwaysSpeculatableImplTrait`, `TensorSizeTrait`, `VerifyTensorLayoutsTrait`


接口：`ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`


效果：`MemoryEffects::Effect{}`

#### 

#### 属性：

|属性|MLIR 类型|描述|
|:----|:----|:----|
|start|::mlir::IntegerAttr|32-bit 无符号整数属性|
|end|::mlir::IntegerAttr|32-bit 无符号整数属性|

#### 结果：

|结果|描述|
|:----|:----|
|result|整数值有序张量|

### `tt.make_tensor_ptr` (triton::MakeTensorPtrOp)

 

*创建一个张量指针类型，其中包含父张量和指定块的元信息。*


语法：

```plain
operation ::= `tt.make_tensor_ptr` $base `,` `[` $shape `]` `,` `[` $strides `]` `,` `[` $offsets `]` attr-dict `:` type($result)
```


`tt.make_tensor_ptr` 接受父张量和块张量的元信息，然后返回指向块张量的指针，例如返回类型为 `tt.ptr<tensor<8x8xf16>>`。


特征：`AlwaysSpeculatableImplTrait`, `SameVariadicOperandSize`, `TensorSizeTrait`, `VerifyTensorLayoutsTrait`


接口：`ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`


效果：`MemoryEffects::Effect{}`

#### 

#### 属性：

|属性|MLIR 类型|描述|
|:----|:----|:----|
|order|::mlir::DenseI32ArrayAttr|i32 稠密数组属性|

#### 操作:

|操作|描述|
|:----|:----|
|base|指针|
|shape|64-bit 无符号整数可变参数|
|strides|64-bit 无符号整数可变参数|
|offsets|64-bit 无符号整数可变参数|

#### 结果：

|结果|描述|
|:----|:----|
|result|指针|

### `tt.mulhiui` (triton::MulhiUIOp)


*最大的 N 位是两个整数的 2N 位乘积的前 N 位*


语法：

```plain
operation ::= `tt.mulhiui` $x `,` $y attr-dict `:` type($x)
```


两个整数的 2N 位乘积的最高 N 位


特征：`AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`, `TensorSizeTrait`, `VerifyTensorLayoutsTrait`


接口：`ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`


效果：`MemoryEffects::Effect{}`

#### 

#### 操作:


|操作|描述|
|:----|:----|
|x|整数/整数值有序张量|
|y|整数/整数值有序张量|

#### 结果：

|结果|描述|
|:----|:----|
|result|整数/整数值有序张量|

### `tt.precise_divf` (triton::PreciseDivFOp)


*浮点类型的精确除法*


语法：

```plain
operation ::= `tt.precise_divf` $x `,` $y attr-dict `:` type($x)
```


浮点数类型的精确除法


特征：`AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`, `TensorSizeTrait`, `VerifyTensorLayoutsTrait`


接口：`ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`


效果：`MemoryEffects::Effect{}`

#### 

#### 操作:

|操作|描述|
|:----|:----|
|x|浮点/浮点值的有序张量|
|y|浮点/浮点值的有序张量|

#### 结果：

|结果|描述|
|:----|:----|
|result|浮点/浮点值的有序张量|

### `tt.precise_sqrt` (triton::PreciseSqrtOp)


*浮点数类型的精确**平**方**根*


语法：

```plain
operation ::= `tt.precise_sqrt` $x attr-dict `:` type($x)
```


浮点数类型的精确平方根


特征：`AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`, `TensorSizeTrait`, `VerifyTensorLayoutsTrait`


接口：`ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`


效果：`MemoryEffects::Effect{}`

#### 

#### 操作:

|操作|描述|
|:----|:----|
|x|浮点/浮点值的有序张量|

#### 结果：

|结果|描述|
|:----|:----|
|result|浮点/浮点值的有序张量|

### `tt.print` (triton::PrintOp)


*设备端打印，如* *CUDA* *用于调试*


语法：

```plain
operation ::= `tt.print` $prefix attr-dict (`:` $args^ `:` type($args))?
```


`tt.print` 接受一个字面字符串前缀和任意数量的标量或张量参数，这些参数将被打印，格式将根据参数自动生成。



特征：`TensorSizeTrait`, `VerifyTensorLayoutsTrait`


接口：`MemoryEffectOpInterface (MemoryEffectOpInterface)`


效果：`MemoryEffects::Effect{MemoryEffects::Write on ::mlir::triton::GlobalMemory}`

#### 

#### 属性:

|属性|MLIR 类型|描述|
|:----|:----|:----|
|prefix|::mlir::StringAttr|字符串属性|
|hex|::mlir::BoolAttr|布尔属性|

#### 操作：

|操作|描述|
|:----|:----|
|args|浮点/浮点值的有序张量，整数/整数值的有序张量，指针/指针值的有序张量，指针的可变参数。|

### `tt.ptr_to_int` (triton::PtrToIntOp)


*转换指针为 int64*


语法：

```plain
operation ::= `tt.ptr_to_int` $src attr-dict `:` type($src) `->` type($result)
```


特征：`AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultEncoding`, `SameOperandsAndResultShape`, `TensorSizeTrait`, `VerifyTensorLayoutsTrait`


接口：`ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`


效果：`MemoryEffects::Effect{}`

#### 

#### 操作:

|操作|描述|
|:----|:----|
|src|指针/指针有序张量|

#### 结果：

|结果|描述|
|:----|:----|
|result|64-bit 无符号整数/64-bit 无符号整数张量|

### `tt.reduce` (triton::ReduceOp)


使用通用组合算法进行归约


特征：`AlwaysSpeculatableImplTrait`, `SameOperandsEncoding`, `SingleBlock`, `TensorSizeTrait`, `VerifyTensorLayoutsTrait`


接口：`ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`


效果：`MemoryEffects::Effect{}`

#### 

#### 属性：

|属性|MLIR 类型|描述|
|:----|:----|:----|
|axis|::mlir::IntegerAttr|32-bit 无符号整数属性|

#### 操作：

|操作|描述|
|:----|:----|
|srcs|浮点/整数/指针的有序张量的可变参数|

#### 结果：

|结果|描述|
|:----|:----|
|result|浮点/浮点值的有序张量，整数/整数值的有序张量，指针/指针值的有序张量，指针的可变参数。|

### `tt.reduce.return` (triton::ReduceReturnOp)


*归**约运算符**的终止符*


语法：

```plain
operation ::= `tt.reduce.return` $result attr-dict `:` type($result)
```


特征：`AlwaysSpeculatableImplTrait`, `HasParent<ReduceOp>`, `ReturnLike`, `TensorSizeTrait`, `Terminator`, `VerifyTensorLayoutsTrait`


接口：`ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `RegionBranchTerminatorOpInterface`


效果：`MemoryEffects::Effect{}`

#### 

#### 操作:

|操作|描述|
|:----|:----|
|result|任意类型的可变参数|

### `tt.reshape` (triton::ReshapeOp)


*将张量重新解释为不同的形状**。如果设置了属性，可能会改变元素的顺序。*


语法：

```plain
operation ::= `tt.reshape` $src attr-dict `:` type($src) `->` type($result)
```


将张量重新解释为不同的形状。


如果设置了 allow_reorder，编译器可以自由地改变元素的顺序以生成更高效的代码。


如果设置了 efficient_layout，这是一个提示，表明出于性能原因应保持目标布局。但编译器仍可为了更好的性能而进行改变。


特征：`AlwaysSpeculatableImplTrait`, `SameOperandsAndResultElementType`, `TensorSizeTrait`, `VerifyTensorLayoutsTrait`


接口：`ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`


效果：`MemoryEffects::Effect{}`

#### 

#### 属性:

|属性|MLIR 类型|描述|
|:----|:----|:----|
|allow_reorder|::mlir::BoolAttr|布尔属性|
|efficient_layout|::mlir::UnitAttr|单元属性|

#### 操作：

|操作|描述|
|:----|:----|
|src|浮点/整数/指针值的有序张量|

#### 结果：

|结果|描述|
|:----|:----|
|result|浮点/整数/指针值的有序张量|

### `tt.scan` (triton::ScanOp)

_Associative scan using generic combination algorithm_


使用通用组合算法进行关联扫描。


特征：`AlwaysSpeculatableImplTrait`, `SameOperandsAndResultEncoding`, `SameOperandsAndResultShape`, `SingleBlock`, `TensorSizeTrait`, `VerifyTensorLayoutsTrait`


接口：`ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`


效果：`MemoryEffects::Effect{}`

#### 

#### 属性：

|Attribute|MLIR Type|Description|
|:----|:----|:----|
|axis|::mlir::IntegerAttr|32-bit signless integer attribute|
|reverse|::mlir::BoolAttr|bool attribute|

|属性|MLIR 类型|描述|
|:----|:----|:----|
|axis|::mlir::IntegerAttr|32-bit 无符号整数属性|
|reverse|::mlir::BoolAttr|布尔属性|

#### 操作：

|操作|描述|
|:----|:----|
|srcs|浮点数/整数/指针值的有序张量的可变参数|

#### 结果：

|结果|描述|
|:----|:----|
|result|浮点数/整数/指针的有序张量的可变参数|

### `tt.scan.return` (triton::ScanReturnOp)


*扫描操作的终止符*


语法：

```plain
operation ::= `tt.scan.return` $result attr-dict `:` type($result)
```


特征：`AlwaysSpeculatableImplTrait`, `HasParent<ScanOp>`, `ReturnLike`, `TensorSizeTrait`, `Terminator`, `VerifyTensorLayoutsTrait`


接口：`ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `RegionBranchTerminatorOpInterface`


效果：`MemoryEffects::Effect{}`

#### 

#### 操作:


|操作|描述|
|:----|:----|
|result|任意类型的可变参数|

### `tt.splat` (triton::SplatOp)


*广播*


语法:

```plain
operation ::= `tt.splat` $src attr-dict `:` type($src) `->` type($result)
```


特征：`AlwaysSpeculatableImplTrait`, `SameOperandsAndResultElementType`, `SameOperandsAndResultEncoding`, `TensorSizeTrait`, `VerifyTensorLayoutsTrait`


接口：`ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`


效果：`MemoryEffects::Effect{}`

#### 

#### 操作:


|操作|描述|
|:----|:----|
|src|浮点/浮点值的有序张量，整数/整数值的有序张量，指针/指针值的有序张量，指针的可变参数。|

#### 结果：

|结果|描述|
|:----|:----|
|result|浮点/整数/指针值的有序张量|

### `tt.split` (triton::SplitOp)


*将一个张量沿着最后**1**个维度分割成**2**个*


语法：

```plain
operation ::= `tt.split` $src attr-dict `:` type($src) `->` type($outLHS)
```


输入必须是最后 1 个维度大小为 2 的张量。返回 2 个张量，分别是 src[…, 0] 和 src[…, 1]。


例如，如果输入形状为 4x8x2xf32，则返回 2 个形状为 4x8xf32 的张量。


特征：`TensorSizeTrait`, `VerifyTensorLayoutsTrait`


接口：`InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`


效果：`MemoryEffects::Effect{}`

#### 

#### 操作：

|操作|描述|
|:----|:----|
|src|浮点/整数/指针值的有序张量|

#### 结果：

|结果|描述|
|:----|:----|
|outLHS|浮点/整数/指针值的有序张量|
|outRHS|浮点/整数/指针值的有序张量|

### `tt.store` (triton::StoreOp)


*使用指针张量存储或使用张量指针存储*


语法：

```plain
operation ::= `tt.store` $ptr `,` $value (`,` $mask^)?
              oilist(`cacheModifier` `=` $cache | `evictionPolicy` `=` $evict)
              attr-dict `:` type($ptr)
```


特征：`SameLoadStoreOperandsEncoding`, `SameLoadStoreOperandsShape`, `TensorSizeTrait`, `VerifyTensorLayoutsTrait`


接口：`MemoryEffectOpInterface (MemoryEffectOpInterface)`


效果：`MemoryEffects::Effect{MemoryEffects::Write on ::mlir::triton::GlobalMemory}`

#### 

#### 属性：

|属性|MLIR 类型|描述|
|:----|:----|:----|
|boundaryCheck|::mlir::DenseI32ArrayAttr|i32 密集数组属性|
|cache|::mlir::triton::CacheModifierAttr|允许的 32-bit 无符号整数 case：1, 2, 3, 4, 5, 6|
|evict|::mlir::triton::EvictionPolicyAttr|允许的 32-bit 无符号整数 case：1, 2, 3|

#### 操作：

|操作|描述|
|:----|:----|
|ptr|ptr or ranked tensor of ptr values or ptr 指针，指针/指针值的有序张量|
|value|浮点/浮点值的有序张量，整数/整数值的有序张量，指针/指针值的有序张量，指针。|
|mask|1-bit 无符号整数 1-bit 无符号整数值的有序张量|

### `tt.trans` (triton::TransOp)


*重新排列张量的维度*


语法：

```plain
operation ::= `tt.trans` $src attr-dict `:` type($src) `->` type($result)
```


例如，给定一个形状为 `[1,2,4]` 的张量 `x`，`transpose(x)` 使用 `order=[2,0,1]` 将张量重新排列为形状 `[4,1,2]`。


尽管这个操作称为「trans」，但它实现了 tl.trans() 和 tl.permute() 两个功能。(「permute」可能是一个更好的名称，但它被称为「trans」是因为最初它只支持 2D 张量。)

#### 

#### 编码实现说明：


在 TritonGPU dialect（及可能的其他 dialect）中，为了使这个操作在代码生成的角度上成为「无操作」（nop），会为该操作的输出选择一种编码方式。


例如，假设张量 x 有一个编码，使得 GPU 线程 [i,j,k] 包含张量的元素 [i,j,k]。现在我们按顺序 [2,1,0] 对 x 进行转置，即颠倒其维度的顺序。在 TritonGPU 中，我们将为转置操作的输出选择一种布局，以使 GPU 线程 [i,j,k] 包含转置后 x 的元素 [k,j,i]。但实际上，这与之前的元素是相同的，我们所做的只是「重命名」线程 [i,j,k] 所含有的元素。


真正的转置操作 -- 即在 GPU 线程之间移动数据 -- 发生在出现在操作之前和/或之后的 convertLayout 操作中。


我们这样做是为了您可以在每个操作后无需进入共享内存即可链式多次数据移动操作（例如转置+重塑+连接）。


特征：`AlwaysSpeculatableImplTrait`, `SameOperandsAndResultElementType`, `TensorSizeTrait`, `VerifyTensorLayoutsTrait`


接口：`ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`


效果：`MemoryEffects::Effect{}`

#### 

#### 属性：

|属性|MLIR 类型|描述|
|:----|:----|:----|
|order|::mlir::DenseI32ArrayAttr|i32 密集数组属性|

#### 操作:


|操作|描述|
|:----|:----|
|src|TensorOrMemDesc 实例|

#### 结果：

|结果|描述|
|:----|:----|
|result|TensorOrMemDesc 实例|

# 

