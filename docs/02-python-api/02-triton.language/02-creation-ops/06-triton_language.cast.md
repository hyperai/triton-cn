---
title: triton_language.cast
---

```python
triton.language.cast(input, dtype: dtype, fp_downcast_rounding: str | None = None, bitcast: bool = False)
```

将张量转换为指定的 `dtype`。


**参数****：**

* **dtype** (*tl.dtype*) - 目标数据类型。

* **fp_downcast_rounding** (*str*, *optional*) - 向下转换浮点值的舍入模式。仅当 self 是浮点张量且 dtype 是比特宽度较小的浮点类型时使用。支持的值为 `"rtne"`（四舍五入到最接近的偶数）和 `"rtz"`（向零舍入）。

* **bitcast** (*bool*, *optional*) - 如果为 true，则将张量位转换为给定的 `dtype`，而不是进行数值转换。

此函数也可以作为 `tensor` 上的成员函数调用，作为 `x.cast (...)` 而不是  `cast (x，...)`。

