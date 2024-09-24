---
title: triton_language.dot
---

```python
triton.language.dot(input, other, acc=None, input_precision=None, allow_tf32=None, max_num_imprecise_acc=None, out_dtype=triton.language.float32)
```


返回 2 个块的矩阵乘积。


这 2 个块必须都是二维或三维的并且有兼容的内部维度。对于三维的块，tl.dot 执行批量矩阵乘积，其中每个块的第一维度代表批量维度。


**参数****：**

* **input**（标量类型为 {`int8`,`float8_e5m2`,`float16`,`bf``loat16`,`float32`} 中的 2D 或 3D 张量）- 第 1 个要相乘的张量。
* **other****（**标量类型为 {`int8`,`float8_e5m2`,`float16`, `bf``loat16`,`float32`} 中的 2D 或 3D 张量）- 第 2 个要相乘的张量。
* **acc**（标量类型为 {`int8`,`float8_e5m2`,`float16`,`bf``loat16`,`float32`} 中的 2D 或 3D 张量）- 累加器张量。如果不为 None，则将结果添加到该张量中。
* **input_precision** (*string**。*对于 nvidia 可用选项为：`"tf32"`,`"tf32x3"`,`"ieee"`。默认为 `"tf32"`。对于 amd 可用选项为 `"ieee"`) - 用于确定如何使用 Tensor Cores 进行 f32 x f32 的计算。如果设备没有 Tensor Cores 或输入不是 dtype f32，则此选项将被忽略。对于具有 Tensor Cores 的设备，默认精度为 tf32。
* **allow_tf32** - 已弃用。如果为 true，则 input_precision 设置为「tf32」。只能指定 `input_precision` 和 `allow_tf32` 中的 1 个（即至少 1 个必须为 `None`）。

