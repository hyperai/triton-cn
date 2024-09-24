---
title: triton_language.softmax
---

```python
triton.language.softmax(x, ieee_rounding=False)
```


计算 `x` 的逐元素 softmax 函数值。 


**参数****：**

* **x** (*Block*) - 输入值。

这个函数也可作为 `tensor` 的成员函数调用，使用 `x.softmax(...)` 而不是 `softmax(x, ...)`。


