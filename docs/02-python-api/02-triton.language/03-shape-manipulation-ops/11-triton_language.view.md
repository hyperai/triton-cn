---
title: triton_language.view
---

```python
triton.language.view(input, *shape)
```


返回具有与输入相同元素但形状不同的张量，元素的顺序可能无法保持。


**参数****：**

* **input** (*Block*) - 输入张量。
* **shape** - 所需的形状。

`shape` 可以作为 1 个元组或作为单独的参数被传递：

```python
# These are equivalent
# 这些是等价的
view(x, (32, 32))
view(x, 32, 32)
```


这个函数也可作为 `tensor` 的成员函数调用，使用 `x.view(...)` 而不是 `view(x, ...)`。


