---
title: Triton 语义
---


Triton 在大多数情况下遵守 NumPy 的语义，但也有一些例外。在本文档中，我们将介绍 Triton 支持的一些数组计算功能，并讨论 Triton 语义与 NumPy 不同的例外情况。


## 类型提升

**类型提升** (Type Promotion) 是在不同数据类型的张量参与运算时发生的。对于与[双下划线方法](https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types)相关的二元运算和三元函数 `tl.where` 的最后两个参数，Triton 会自动将输入张量转换为一个通用的数据类型，这一转换遵循数据类型种类的层级顺序：`{bool} < {integral dypes} < {floating point dtypes}`。


该算法如下：


1. **类型**：如果一个张量的数据类型属于更高级的类型，则另一个张量将被提升至该数据类型，例如：`(int32, bfloat16) -> bfloat16`。

2. **宽度**：如果两个张量的数据类型属于同一类，但其中一个张量具有更大的宽度，则另一个张量将被提升至此数据类型，例如：`(float32, float16) -> float32`。

3. **上限**：如果两个张量的宽度和符号相同，但数据类型不同，则它们都将被提升为下一个更大的数据类型，例如：`(float16, bfloat16) -> float32`。

	3.1 如果两个张量的数据类型都是不同的 `fp8` 类型，它们将都被统一转换为 `float16`。

4. **无符号类型优先**：在其他情况下（相同宽度，不同符号），它们将被提升为相应的无符号数据类型：`(int32, uint32) -> uint32`。


当涉及标量时，规则会有所不同。在此处，标量是指数值字面量、标记为 *tl.constexpr* 的变量或这些的组合。它们由 NumPy 标量表示，并具有`bool`、`int`和`float`等类型。


当一个操作涉及张量和标量时：


1. 如果标量的类型低于或等于张量的类型，则标量不会参与类型提升：`(uint8, int) -> uint8`。

2. 如果标量的类型高于张量，我们将选择最适合标量的数据类型，在 `int32` < `uint32` < `int64` < `uint64` 中为整数选择，在 `float32` < `float64` 中选择浮点数。然后，张量和标量都将提升为这种数据类型：`(int16, 4.0) -> float32`。


## 广播

**广播** (Broadcasting) 允许对不同形状的张量进行操作，它会自动将张量的形状扩展至兼容的大小，且在此过程中无需复制数据。其遵循以下规则：


1. 如果其中一个张量的形状较短，则在左侧填充 1，直到两个张量的维数相同，例如：`((3, 4), (5, 3, 4)) -> ((1, 3, 4), (5, 3, 4))`。

2. 如果两个维度相等，或者其中之一为 1，那么这两个维度是兼容的。维度为 1 的那个维度将会被扩展以匹配另一个张量的维度，例如：`((1, 3, 4), (5, 3, 4)) -> ((5, 3, 4), (5, 3, 4))`。


## 与 NumPy 的不同之处

为了提高效率，Triton 的整数除法运算符遵守 C 语义，而不是 Python 语义。因此，混合符号的整数的 `int // int` 运算实现为 [C ](https://en.wikipedia.org/wiki/Modulo#In_programming_languages)[语言](https://en.wikipedia.org/wiki/Modulo#In_programming_languages)[中的向零舍入](https://en.wikipedia.org/wiki/Modulo#In_programming_languages)，而不是 Python 中的向负无穷舍入。同样，取模运算符 `int % int`（定义为 `a % b = a - b * (a // b)`）也遵守 C 语义，而不是 Python 语义。


在所有输入都是标量的情况下，整数除法和取模运算遵循 Python 语义，这点可能会造成迷惑。

