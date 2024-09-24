---
title: triton_language.reduce
---

```python
triton.language.reduce(input, axis, combine_fn, keep_dims=False)
```


将 combine_fn 应用于沿指定 `axis` 轴上 `input` 张量中的所有元素。 


**参数****：**

* **input** (*Tensor*) - 输入张量，或张量的元组。
* **axis** (*int | None*) - 要进行归约操作的维度。如果为 None，则归约所有维度。
* **combine_fn** (*Callable*) - 1 个用于组合 2 组标量张量的函数（必须使用 @triton.jit 标记）。
* **keep_dims** (*bool*) - 如果为 true，保留长度为 1 的归约维度。

这个函数也可作为 `reduce` 的成员函数调用，使用 `x.reduce(...)` 而不是 `reduce(x, ...)`。


