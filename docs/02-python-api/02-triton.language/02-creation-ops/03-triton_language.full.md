---
title: triton_language.full
---

```python
triton.language.full(shape, value, dtype)
```


返回一个张量，该张量填充了指定 `shape` 和 `dtype` 的标量值。


**参数****：**

* **shape** (*tuple of ints*) - 新数组的形状，例如 (8, 16) 或 (8,)。

* **value** (*scalar*) - 用于填充数组的标量值。

* **dtype** (*tl.dtype*) - 新数组的数据类型，例如 `tl.float16`。

