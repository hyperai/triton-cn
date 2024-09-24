MLIR Triton Dialect 中的 Triton IR。


Dependent Dialects：


* Arith:
   * addf, addi, andi, cmpf, cmpi, divf, fptosi, …
* Math:
   * exp, sin, cos, log, …
* StructuredControlFlow:
   * for, if, while, yield, condition
* ControlFlow:
   * br, cond_br

[TOC]

# 类型 (Types)

## 内存描述符类型 (MemDescType)


在 Triton IR 类型系统中的内存描述符类型 (`::mlir::triton::MemDescType`) 。


内存描述符 (Memory descriptor) 包含一个基本指针（标量 Scalar）和内存的描述符。如果可变内存为 false，即内存是常量，则只能分配和存储一次。常量内存分配与张量不同，因为它可以有多个视图，并且可以更改描述符而不更改底层内存。


### 参数:


|**参数**|**C++ 类型**|**描述**|
|:----|:----|:----|
|shape|::llvm::ArrayRef<int64_t>| |
|elementType|Type| |
|encoding|Attribute| |
|memorySpace|Attribute| |
|mutable_memory|bool| |


## PointerType


在 Triton IR 类型系统中的指针类型 (`::mlir::triton::PointerType`)。


在 Triton IR 类型系统中的指针类型可以指向标量或张量


### 参数:

|**参数**|**C++ 类型**|**描述**|
|:----|:----|:----|
|pointeeType|Type| |
|addressSpace|int| |


