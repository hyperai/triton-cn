```python
triton.language.sum(input, axis=None, keep_dims=False)
```


返回 `input` 张量中，沿指定 `axis` 的所有元素的总和。 


**参数****：**

* **input** (*Tensor*) - 输入值。
* **axis** (*int*) - 要进行归约操作的维度。
* **keep_dims** (*bool*) - 如果为 true，保留长度为 1 的归约维度。

这个函数也可作为 `tensor` 的成员函数调用，使用 `x.sum(...)` 而不是 `sum(x, ...)`。


