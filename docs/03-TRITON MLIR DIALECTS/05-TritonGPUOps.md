## `triton_gpu.async_commit_group`

## (triton::gpu::AsyncCommitGroupOp)


*异步提交组*


语法 (Syntax)：

```plain
operation ::= `triton_gpu.async_commit_group` $inputTokens attr-dict
```


特征 (Traits)：`VerifyTensorLayoutsTrait`


接口 (Interfaces)：`InferTypeOpInterface`


### 操作 (Operands)：

|**操作**|**描述**|
|:----|:----|
|inputTokens|异步 token 类型的可变参数|


### 结果 (Results)：

|**结果**|**描述**|
|:----|:----|
|asyncToken|异步 token 类型|


## `triton_gpu.async_copy_global_to_local`

## (triton::gpu::AsyncCopyGlobalToLocalOp)


*从全局内存中异步拷贝数据到**本地**内存*


语法:

```plain
operation ::= `triton_gpu.async_copy_global_to_local` $src `,` $result (`mask` $mask^)? (`other` $other^)?
              oilist(`cacheModifier` `=` $cache | `evictionPolicy` `=` $evict)
              attr-dict `:` type($src) `->` type($result)
```


此操作将数据从全局内存异步复制到本地内存。这类似于 tt.load，但数据被复制到由内存描述符指向的本地内存，而不是分布式张量。其余 operands 与 tt.load 相同。


特征：`AttrSizedOperandSegments`, `VerifyTensorLayoutsTrait`


接口：`MemoryEffectOpInterface`, `MemoryEffectOpInterface`


### 属性 （Attributes):

|属性|MLIR 类型|描述|
|:----|:----|:----|
|cache|::mlir::triton::CacheModifierAttr|允许的 32-bit 无符号整数情况：1，2，3，4，5，6|
|evict|::mlir::triton::EvictionPolicyAttr|允许的 32-bit 无符号整数值：1, 2, 3|
|isVolatile|::mlir::BoolAttr|布尔属性|

### 操作 (Operands):

|**操作**|**描述**|
|:----|:----|
|src|指针值的有序张量|
|result|在 Triton IR 类型系统中的内存描述符类型 (::mlir::triton::MemDescType)|
|mask|1-bits 无符号整数的张量|
|other|浮点/浮点值的有序张量，整数/整数值的有序张量，指针/指针值的有序张量，指针。|


### 结果 (Results):

|**结果**|**描述**|
|:----|:----|
|token|异步 token 类型|


## `triton_gpu.async_wait`(triton::gpu::AsyncWaitOp)


*异步等待*


语法：

```plain
operation ::= `triton_gpu.async_wait` $asyncToken attr-dict
```


特征：`AttrSizedOperandSegments`


接口：`InferTypeOpInterface`


### 属性：

|属性|MLIR 类型|描述|
|:----|:----|:----|
|num|::mlir::IntegerAttr|32-bit 无符号整数属性|


### 操作：

|**操作**|**描述**|
|:----|:----|
|asyncToken| 异步 token 类型的可变参数|


### 结果：

|**结果**|**描述**|
|:----|:----|
|retToken|异步 token 类型|


## `triton_gpu.convert_layout`(triton::gpu::ConvertLayoutOp)


*转换布局*


语法:

```plain
operation ::= `triton_gpu.convert_layout` $src attr-dict `:` type($src) `->` type($result)
```


特征：`AlwaysSpeculatableImplTrait`, `SameOperandsAndResultElementType`, `SameOperandsAndResultShape`, `VerifyTensorLayoutsTrait`


接口：`ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`


效果：`MemoryEffects::Effect{}`


### 操作：

|**操作**|**描述**|
|:----|:----|
|src| 浮点/整数/指针值的有序张量|


### 结果：

|**结果**|**描述**|
|:----|:----|
|result|浮点/整数/指针值的有序张量|


## `triton_gpu.local_alloc`(triton::gpu::LocalAllocOp)


*分配张量*


语法:

```plain
operation ::= `triton_gpu.local_alloc` $src attr-dict `:` functional-type(operands, results)
```


该操作在共享内存中分配 buffer，返回含有 buffer 的地址和视图的描述符


显示释放 buffer 是可选的，见 local_dealloc


特征：`VerifyTensorLayoutsTrait`


接口：`MemoryEffectOpInterface`


### 操作：

|**操作**|**描述**|
|:----|:----|
|src|浮点/整数/指针值的有序张量|


### 结果：

|**结果**|**描述**|
|:----|:----|
|result|在 Triton IR 类型系统中的内存描述符类型 (::mlir::triton::MemDescType)|


## `triton_gpu.local_dealloc`(triton::gpu::LocalDeallocOp)


*释放 buffer*


语法:

```plain
operation ::= `triton_gpu.local_dealloc` $src attr-dict `:` qualified(type($src))
```


此操作显式释放了一个 buffer，此操作后使用该 buffer 是未定义的。


这项操作是可选的。如果不明确地释放一个 buffer，编译器会假定在所有对该 buffer 的使用之后的第一个点进行释放。


因为我们假定 memdesc 在第一个后支配其使用的点就失效，所以等待 memdesc 上异步操作完成的操作（例如 triton_nvidia_gpu.warp_group_dot_wait）也应将 memdesc 作为操作数。


特征：`VerifyTensorLayoutsTrait`


接口：`MemoryEffectOpInterface (MemoryEffectOpInterface)`


效果：`MemoryEffects::Effect{MemoryEffects::Free on ::mlir::triton::gpu::SharedMemory}`


### 操作：

|**操作**|**描述**|
|:----|:----|
|src| 在 Triton IR 类型系统中的内存描述符类型 (::mlir::triton::MemDescType)|


## `triton_gpu.local_load`(triton::gpu::LocalLoadOp)


*从本地内存中读取 buffer 到分布式张量中*


语法:

```plain
operation ::= `triton_gpu.local_dealloc` $src attr-dict `:` qualified(type($src))
```


从本地内存描述符中读取张量到分布式张量中。


特征：`VerifyTensorLayoutsTrait`


接口：`MemoryEffectOpInterface`


### 操作:

|**操作**|**描述**|
|:----|:----|
|src|在 Triton IR 类型系统中的内存描述符类型 (::mlir::triton::MemDescType)|
|token|异步 token 类型|


### 结果


|**结果**|**描述**|
|:----|:----|
|result|浮点/整数/指针值的有序张量|


## `triton_gpu.local_store`(triton::gpu::LocalStoreOp)


*保存分布式张量到本地内存的 buffer 中*


语法：

```plain
operation ::= `triton_gpu.local_dealloc` $src attr-dict `:` qualified(type($src))
```


*将一个分布式张量存储到本地内存中的 buffer 中*


特征：`VerifyTensorLayoutsTrait`


接口：`MemoryEffectOpInterface`


### 操作:

|**操作**|**描述**|
|:----|:----|
|src|浮点/整数/指针值的有序张量|
|dst|在 Triton IR 类型系统中的内存描述符类型 (::mlir::triton::MemDescType)|


## `triton_gpu.memdesc_subview`(triton::gpu::MemDescSubviewOp)


*获取描述符的子视图*


语法:

```plain
operation ::= `triton_gpu.memdesc_subview` $src `[` $offsets `]` attr-dict `:` qualified(type($src)) `->` qualified(type($result))
```


这个操作返回一个表示 buffer 子视图的新描述符，它不会影响底层内存，子视图可以具有更低的维度。


例如，假设


* input 的形状是 2x4x16xf16
* output 的形状是 4x4xf16，并且
* offsets = [1, 0, 4].

在 Python 语法中，子视图涵盖 input[1] [0:4][4:8]


特征：`AlwaysSpeculatableImplTrait`,   `VerifyTensorLayoutsTrait`


接口：`ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`


效果：`MemoryEffects::Effect{}`


### 操作:

|**操作**|**描述**|
|:----|:----|
|src|在 Triton IR 类型系统中的内存描述符类型 (::mlir::triton::MemDescType)|
|offsets|32-bit 无符号整数可变参数|


### 结果：

|**结果**|**描述**|
|:----|:----|
|result|在 Triton IR 类型系统中的内存描述符类型 (::mlir::triton::MemDescType)|


