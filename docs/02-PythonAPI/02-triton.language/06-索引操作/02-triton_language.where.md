```python
triton.language.where(condition, x, y)
```


根据 `condition` 返回来自 `x` 或 `y` 元素的张量。


注意：无论 `condition` 的值是什么，`x` 和 `y` 总是会被求值。


如果希望避免意外的内存操作，请使用 *triton.load* 和 *triton.store* 中的 `mask` 参数。


`x` 和 `y` 的形状都会被广播到 `condition` 的形状。`x` 和 `y` 必须具有相同的数据类型。


**参数****：**

* **condition**（*triton.bool 的块*）- 当为 True（非零）时，产生 x，否则产生 y。
* **x** - 在条件为 True 的索引处选择的值。
* **y** - 在条件为 False 的索引处选择的值。


