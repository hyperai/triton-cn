```python
triton.language.xor_sum(input, axis=None, keep_dims=False)
```


沿指定 `axis` 的 `input` 张量中所有元素的异或和。 


**参数****：**

* **input** (*Tensor*) - 输入值。
* **axis** (*int*) - 要进行归约操作的维度。
* **keep_dims** (*bool*) - 如果为 true，保留长度为 1 的归约维度。

这个函数也可作为 `tensor` 的成员函数调用，使用 `x.xor_sum(...)` 而不是 `xor_sum(x, ...)`。


