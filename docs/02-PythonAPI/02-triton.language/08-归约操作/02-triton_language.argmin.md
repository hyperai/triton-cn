```python
triton.language.argmin(input, axis, tie_break_left=True, keep_dims=False)
```


返回沿指定 `axis` 的 `input` 张量中所有元素的最小索引。 


**参数****：**

* **input** (*Tensor*) - 输入值。
* **axis** (*int*) - 要进行归约操作的维度。
* **keep_dims** (*bool*) - 如果为 true，则保留长度为 1 的归约维度。
* **tie_break_left** (*bool*) - 如果为 true，在出现平局的情况下（即多个元素具有相同的最小索引值），对于非 NaN 的值返回最左边的索引。

这个函数也可作为 `tensor` 的成员函数调用，使用 `x.argmin(...)` 而不是 `argmin(x, ...)`。


