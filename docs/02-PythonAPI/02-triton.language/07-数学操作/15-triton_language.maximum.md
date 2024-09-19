```python
triton.language.maximum(x, y, propagate_nan: ~triton.language.core.constexpr = <PROPAGATE_NAN.NONE: 0>
```


计算 `x` 和 `y` 的逐元素最大值。 


**参数****：**

* **x** (*Block*) - 第 1 个输入的张量。
* **y** (*Block*) - 第 2 个输入的张量。
* **propagate_nan** (*tl.PropagateNan*) - 是否传播 NaN 值。

> 另请参阅  
> tl.PropagateNan

