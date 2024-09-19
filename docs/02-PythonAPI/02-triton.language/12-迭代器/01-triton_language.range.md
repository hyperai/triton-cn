```python
classtriton.language.range(self, arg1, arg2=None, step=None, num_stages=None)
```


永远向上计数的迭代器 。


```python
@triton.jit
def kernel(...):
    for i in tl.range(10, num_stages=3):
        ...
```


**注意****：**

这是一个特殊的迭代器，用于在 `triton.jit` 函数的上下文中实现类似于 Python 中 `range` 的语义。此外，它允许用户向编译器传递额外的属性。


**参数****：**

* **arg1** - 起始值。
* **arg2** - 结束值。
* **step** - 步长值。
* **num_stages** - 将循环流水线化为多个阶段（因此一次有 `num_stages` 次循环迭代在同时执行中）。
* 注意，这与将 `num_stages` 作为内核参数传递略有不同。内核参数仅对点积操作的加载进行流水线处理，而这个属性尝试在这个循环中将大多数（但不是全部）加载进行流水线处理。

```python
__init__(self, arg1, arg2=None, step=None, num_stages=None)
```


**方法**

|**__init__(self, arg1[, arg2, step, num_stages])**|
|:----|


