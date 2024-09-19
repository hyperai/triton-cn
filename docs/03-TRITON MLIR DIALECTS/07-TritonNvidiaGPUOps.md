### `triton_nvidia_gpu.async_tma_copy_global_to_local` (triton::nvidia_gpu::AsyncTMACopyGlobalToLocalOp)


根据描述符将数据从全局内存异步复制到本地内存


语法：


```plain
operation ::= `triton_nvidia_gpu.async_tma_copy_global_to_local` $desc_ptr `[` $coord `]` $result `,` $barrier `,` $pred
              oilist(`cacheModifier` `=` $cache | `evictionPolicy` `=` $evict)
              attr-dict `:` type($desc_ptr) `,` type($barrier) `->` type($result)
```


该操作是数据从全局内存异步复制到本地内存。类似于 tt.load，不同之处在于数据被复制到由内存描述符指向的本地内存，而不是分布式张量。被复制的数据取决于由 `desc_ptr` 指向的全局内存描述符。


特征：`VerifyTensorLayoutsTrait`


接口：`MemoryEffectOpInterface`


#### 属性:

|Attribute|MLIR Type|Description|
|:----|:----|:----|
|cache|::mlir::triton::CacheModifierAttr|允许 32-bit 无符号整数情况：1, 2, 3, 4, 5, 6|
|evict|::mlir::triton::EvictionPolicyAttr|允许 32-bit 无符号整数情况：1, 2, 3|
|isVolatile|::mlir::BoolAttr|布尔属性|


#### 操作:

|操作|描述|
|:----|:----|
|desc_ptr|Triton IR 类型系统中的指针类型|
|coord|32-bit 无符号整数可变参数|
|barrier|Triton IR 类型系统中的内存描述符 (::mlir::triton::MemDescType)|
|result|Triton IR 类型系统中的内存描述符 (::mlir::triton::MemDescType)|
|pred|1-bit 无符号整数|


### `triton_nvidia_gpu.async_tma_copy_local_to_global` (triton::nvidia_gpu::AsyncTMACopyLocalToGlobalOp)


*基于描述符**将数据从**本地内存异步复制到全局内存*


语法：


```plain
operation ::= `triton_nvidia_gpu.async_tma_copy_local_to_global` $desc_ptr `[` $coord `]` $src
              attr-dict `:` type($desc_ptr) `,` type($src)
```


这个操作是异步将数据从本地内存复制到全局内存。类似于 tt.store，不同之处在于数据是从由内存描述符指向的本地内存复制出来，而不是从分布式张量。被复制的数据取决于由 `desc_ptr` 指向的全局内存描述符。


特征：`VerifyTensorLayoutsTrait`


接口：`MemoryEffectOpInterface`


#### 操作：

|**操作**|**描述**|
|:----|:----|
| desc_ptr |Triton IR 类型系统中的指针类型|
| coord |32-bit 无符号整数可变参数|
| src |Triton IR 类型系统中的内存描述符 (::mlir::triton::MemDescType)|


### `triton_nvidia_gpu.barrier_expect` (triton::nvidia_gpu::BarrierExpectOp)


*发出一个预期要复制的字节数的屏障信号。*


语法:


```plain
operation ::= `triton_nvidia_gpu.barrier_expect` $alloc `,` $size attr-dict `,` $pred `:` type($alloc)
```


这个操作发出一个屏障信号，表示预期将复制 `size` 字节。相关的屏障等待将阻塞，直到预期的字节数被复制完成。


特征：`VerifyTensorLayoutsTrait`


接口：`MemoryEffectOpInterface`


#### 属性:

|属性|MLIR 类型|描述|
|:----|:----|:----|
|size|::mlir::IntegerAttr|32-bit 无符号整数属性|


#### 操作:

|**操作**|**描述**|
|:----|:----|
| alloc |Triton IR 类型系统中的内存描述符 (::mlir::triton::MemDescType)|
| pred |1-bit 无符号整数|


### `triton_nvidia_gpu.cluster_arrive` (triton::nvidia_gpu::ClusterArriveOp)


语法：


```plain
operation ::= `triton_nvidia_gpu.cluster_arrive` attr-dict
```


特征：`VerifyTensorLayoutsTrait`


#### 属性：

|属性|MLIR 类型|描述|
|:----|:----|:----|
|relaxed|::mlir::IntegerAttr|1-bit 无符号整数属性|


### `triton_nvidia_gpu.cluster_wait` (triton::nvidia_gpu::ClusterWaitOp)


语法：


```plain
operation ::= `triton_nvidia_gpu.cluster_wait` attr-dict
```


特征：`VerifyTensorLayoutsTrait`


### `triton_nvidia_gpu.fence_async_shared` (triton::nvidia_gpu::FenceAsyncSharedOp)


*异步代理屏障*


语法：


```plain
operation ::= `triton_nvidia_gpu.fence_async_shared` attr-dict
```


特征：`VerifyTensorLayoutsTrait`


#### 属性：

|属性|MLIR 类型|描述|
|:----|:----|:----|
|bCluster|::mlir::BoolAttr|布尔属性|


### `triton_nvidia_gpu.init_barrier` (triton::nvidia_gpu::InitBarrierOp)


*初始化给定的共享内存分配的屏障*


语法：


```plain
operation ::= `triton_nvidia_gpu.init_barrier` $alloc `,` $count attr-dict `:` type($alloc)
```


初始化带有 mbarrier 信息的共享内存分配。`alloc` 是指向共享内存分配的描述符。`count` 是屏障预期到达的次数。


这将降级为 PTX 指令 `mbarrier.init.shared::cta.b64`。


特征：`VerifyTensorLayoutsTrait`


接口：`MemoryEffectOpInterface`


#### 属性：

|属性|MLIR 类型|描述|
|:----|:----|:----|
|count|::mlir::IntegerAttr|32-bit 无符号整数属性|

#### 操作：

|**操作**|**描述**|
|:----|:----|
| alloc |Triton IR 类型系统中的内存描述符(::mlir::triton::MemDescType)|


### `triton_nvidia_gpu.inval_barrier` (triton::nvidia_gpu::InvalBarrierOp)


无效屏障分配


语法：


```plain
operation ::= `triton_nvidia_gpu.inval_barrier` $alloc attr-dict `:` type($alloc)
```


使屏障分配失效，以便可以重新使用。根据 PTX 规范，这必须在 mbarrier 使用的内存重新使用之前完成。


https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-inval


特征：`VerifyTensorLayoutsTrait`


接口：`MemoryEffectOpInterface`


#### 操作：


|**操作**|**描述**|
|:----|:----|
| alloc |Triton IR 类型系统中的内存描述符 (::mlir::triton::MemDescType)|


### `triton_nvidia_gpu.async_tma_store_wait` (triton::nvidia_gpu::TMAStoreWait)


*等待所有输入被读取*


语法：


```plain
operation ::= `triton_nvidia_gpu.async_tma_store_wait` attr-dict
```


在写入共享内存之前，必须等待所有与相关存储操作关联的读取操作完成。


特征：`VerifyTensorLayoutsTrait`


#### 属性:

|属性|MLIR 类型|描述|
|:----|:----|:----|
|pendings|::mlir::IntegerAttr|32-bit 无符号整数属性|


### `triton_nvidia_gpu.wait_barrier` (triton::nvidia_gpu::WaitBarrierOp)


*等待屏障段完成*


语法：


```plain
operation ::= `triton_nvidia_gpu.wait_barrier` $alloc `,` $phase attr-dict `:` type($alloc)
```


阻塞程序进度，直到 `alloc` 中的 mbarrier 对象完成其当前阶段。


这会使用 PTX 指令 `mbarrier.try_wait.parity.shared.b64` 来降低等待循环。


屏障行为描述见：

https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-asynchronous-copy-completion-mechanisms


特征：`VerifyTensorLayoutsTrait`


接口：`M``emoryEffectOpInterface`


#### 操作:

|**操作**|**描述**|
|:----|:----|
| alloc |Triton IR 类型系统中的内存描述符 (::mlir::triton::MemDescType)|
| phase | 32-bit 无符号整数|


### `triton_nvidia_gpu.warp_group_dot` (triton::nvidia_gpu::WarpGroupDotOp)


*Warp 组点乘*


语法：


```plain
operation ::= `triton_nvidia_gpu.warp_group_dot` $a`,` $b`,` $c attr-dict `:` type($a) `*` type($b) `->` type($d)
```


$d = matrix_multiply($a, $b) + $c。 InputPrecisionAttr 的文档可见 TT_DotOp


特征：`DotLike`, `VerifyTensorLayoutsTrait`


接口：`InferTypeOpInterface`, `MemoryEffectOpInterface`


#### 属性:


|Attribute|MLIR Type|Description|
|:----|:----|:----|
|inputPrecision|::mlir::triton::InputPrecisionAttr|允许 32-bit 无符号整数情况: 0, 1, 2|
|maxNumImpreciseAcc|::mlir::IntegerAttr|32-bit 无符号整数属性|
|isAsync|::mlir::BoolAttr|布尔属性|

#### 

#### 操作：

|**操作**|**描述**|
|:----|:----|
| a |TensorOrMemDesc 实例|
| b |  TensorOrMemDesc 实例|
| c |浮点\整数值有序张量|


#### 结果：

|**Result**|**Description**|
|:----|:----|
| d | 浮点\整数值有序张量|


### `triton_nvidia_gpu.warp_group_dot_wait` (triton::nvidia_gpu::WarpGroupDotWaitOp)


*等待 Warp 组点乘完成*


语法：


```plain
operation ::= `triton_nvidia_gpu.warp_group_dot_wait` $inputs attr-dict `:` type($inputs)
```



等待直到未完成的异步点积操作数量为 $pendings 或更少。


$inputs 必须是对应于我们正在等待的异步点乘操作的张量。例如，如果有 N 个待处理的异步点乘操作，并且我们调用 `warp_group_dot_wait 1`，那么 $inputs 必须是第一个点乘操作的结果。


特征：`VerifyTensorLayoutsTrait`


接口：`InferTypeOpInterface`


#### 属性：

|属性|MLIR 类型|描述|
|:----|:----|:----|
|pendings|::mlir::IntegerAttr|32-bit 无符号整数属性|

#### 操作：

|**操作**|**描述**|
|:----|:----|
| inputs |TensorOrMemDesc 实例的可变参数|

#### 结果：

|**Result**|**Description**|
|:----|:----|
| outputs |TensorOrMemDesc 实例的可变参数|


