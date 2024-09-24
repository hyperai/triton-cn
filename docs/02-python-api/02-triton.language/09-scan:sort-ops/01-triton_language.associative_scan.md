---
title: triton_language.associative_scan
---

```python
triton.language.associative_scan(input, axis, combine_fn, reverse=False)
```


沿指定 `axis` 将 combine_fn 应用于 `input` 张量的每个元素和携带的值，并更新携带的值。 


**参数****：**

* **input** (*Tensor*) - 输入张量，或张量的元组。
* **axis** (*int*) - 要进行归约操作的维度。
* **combine_fn** (*Callable*) - 1 个用于组合 2 组标量张量的函数（必须使用 @triton.jit 标记）。
* **reverse** (*bool*) - 是否沿着轴进行反向关联扫描。

这个函数也可作为 `tensor` 的成员函数调用，使用 `x.associative_scan(...)` 而不是 `associative_scan(x, ...)`。


