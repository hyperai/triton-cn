---
title: triton_language.cumprod
---

```python
triton.language.cumprod(input, axis=0, reverse=False)
```


返回沿指定 `axis` 的 `input` 张量中所有元素的累积乘积。 


**参数****：**

* **input** (*Tensor*) - 输入值。
* **axis** (*int*) - 应进行扫描的维度。

这个函数也可作为 `tensor` 的成员函数调用，使用 `x.cumprod(...)` 而不是 `cumprod(x, ...)`。


