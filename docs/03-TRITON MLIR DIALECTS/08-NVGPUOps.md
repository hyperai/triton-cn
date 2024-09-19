### `nvgpu.cluster_arrive` (triton::nvgpu::ClusterArriveOp)


语法：


```plain
operation ::= `nvgpu.cluster_arrive` attr-dict
```


#### 属性：

|属性|MLIR 类型|描述|
|:----|:----|:----|
|relaxed|::mlir::IntegerAttr|1-bit 无符号整数属性|


### `nvgpu.cluster_id` (triton::nvgpu::ClusterCTAIdOp)


语法：


```plain
operation ::= `nvgpu.cluster_id` attr-dict
```



特征：`AlwaysSpeculatableImplTrait`


接口：`ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`


效果：`MemoryEffects::Effect{}`


#### 结果:

|**结果**|**描述**|
|:----|:----|
| result | 32-bit 无符号整数|


### `nvgpu.cluster_wait` (triton::nvgpu::ClusterWaitOp)


语法：


```plain
operation ::= `nvgpu.cluster_wait` attr-dict
```


### `nvgpu.fence_async_shared` (triton::nvgpu::FenceAsyncSharedOp)


语法：


```plain
operation ::= `nvgpu.fence_async_shared` attr-dict
```


#### 属性：

|属性|MLIR 类型|描述|
|:----|:----|:----|
|bCluster|::mlir::BoolAttr|布尔属性|


### `nvgpu.stmatrix` (triton::nvgpu::StoreMatrixOp)


语法：


```plain
operation ::= `nvgpu.stmatrix` operands attr-dict `:` type(operands)
```



接口：`MemoryEffectOpInterface (MemoryEffectOpInterface)`


效果：`MemoryEffects::Effect{MemoryEffects::Write on ::mlir::SideEffects::DefaultResource}`


#### 操作：

|**操作**|**描述**|
|:----|:----|
| addr |地址空间 3 中的 LLVM 指针|
| datas |32-bit 无符号整数的可变参数|


### `nvgpu.wgmma_commit_group` (triton::nvgpu::WGMMACommitGroupOp)


语法：


```plain
operation ::= `nvgpu.wgmma_commit_group` attr-dict
```


### `nvgpu.wgmma_fence` (triton::nvgpu::WGMMAFenceOp)


语法：


```plain
operation ::= `nvgpu.wgmma_fence` attr-dict
```


### `nvgpu.wgmma` (triton::nvgpu::WGMMAOp)


语法：


```plain
operation ::= `nvgpu.wgmma` $opA `,` $opB (`,` $opC^)? attr-dict `:` functional-type(operands, $res)
```


#### Attributes: 属性:

|属性|MLIR 类型|描述|
|:----|:----|:----|
|m|::mlir::IntegerAttr|32-bit 无符号整数属性|
|n|::mlir::IntegerAttr|32-bit 无符号整数属性|
|k|::mlir::IntegerAttr|32-bit 无符号整数属性|
|eltTypeC|::mlir::triton::nvgpu::WGMMAEltTypeAttr|wgmma 操作数类型为 's8', 's32', 'e4m3', 'e5m2', 'f16', 'bf16', 'tf32', 或者 'f32'|
|eltTypeA|::mlir::triton::nvgpu::WGMMAEltTypeAttr|wgmma 操作数类型为 's8', 's32', 'e4m3', 'e5m2', 'f16', 'bf16', 'tf32', 或者 'f32'|
|eltTypeB|::mlir::triton::nvgpu::WGMMAEltTypeAttr|wgmma 操作数类型为 's8', 's32', 'e4m3', 'e5m2', 'f16', 'bf16', 'tf32', 或者 'f32'|
|layoutA|::mlir::triton::nvgpu::WGMMALayoutAttr|wgmma 布局，可以为 'row'（行）或 'col'（列）|
|layoutB|::mlir::triton::nvgpu::WGMMALayoutAttr|wgmma 布局，可以为 'row'（行）或 'col'（列）|

#### 操作：


|**操作**|**描述**|
|:----|:----|
| opA | wgmma operand A/B type wgmma 操作 A/B 类型|
| opB | wgmma operand A/B type wgmma 操作 A/B 类型|
| opC | LLVM structure type LLVM 结构体类型|

#### 结果：

|**结果**|**描述**|
|:----|:----|
| res |LLVM 结构体类型|


### `nvgpu.wgmma_wait_group` (triton::nvgpu::WGMMAWaitGroupOp)


语法：


```plain
operation ::= `nvgpu.wgmma_wait_group` $input attr-dict `:` type($input)
```


接口：`InferTypeOpInterface`


#### 属性：

|属性|MLIR 类型|描述|
|:----|:----|:----|
|pendings|::mlir::IntegerAttr|32-bit 无符号整数属性|


#### 操作：

|**操作**|**描述**|
|:----|:----|
| input | LLVM structure type LLVM 结构体类型|


#### 结果：


|**结果**|**描述**|
|:----|:----|
| output | LLVM structure type LLVM 结构体类型|


