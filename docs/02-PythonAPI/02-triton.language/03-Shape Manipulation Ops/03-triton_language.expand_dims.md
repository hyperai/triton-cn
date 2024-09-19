```python
triton.language.expand_dims(input, axis)
```


通过插入新的长度为 1 的维度来扩展张量的形状。


轴索引是相对于生成的张量而言的，因此对于每个轴，`result.shape[axis]` 将为 1。


**参数****：**

* **input** (*tl.tensor*) - 输入张量。
* **axis** (*int* | *Sequence[int]*) - 要添加新轴的索引。

该函数也可作为 `tensor` 的成员函数调用，使用 `x.expand_dims(...)` 而不是 `expand_dims(x, ...)`。


