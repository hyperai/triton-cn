---
title: triton_language.static_range
---

```python
classtriton.language.static_range(self, arg1, arg2=None, step=None)
```


永远向上计数的迭代器。


```python
@triton.jit
def kernel(...):
    for i in tl.static_range(10):
        ...
```


**注意****：**


这是一个特殊的迭代器，用于在 `triton.jit` 函数的上下文中实现类似于 Python 中 `range` 的语义。此外，它还引导编译器主动展开循环。


**参数****：**

* **arg1** - 起始值。
* **arg2** - 结束值。
* **step** - 步长值。

```python
__init__(self, arg1, arg2=None, step=None)
```


**方法**

|**__init__(self, arg1[, arg2, step])**|
|:----|


