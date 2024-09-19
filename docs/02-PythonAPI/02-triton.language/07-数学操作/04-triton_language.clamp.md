```python
triton.language.clamp(x, min, max, propagate_nan: ~triton.language.core.constexpr = <PROPAGATE_NAN.NONE: 0>)
```
 将输入张量 `x` 的值限制在 [min, max] 范围内。 

**参数：**

* **x** (*Block*) - 输入值。
* **min** (*Block*) – 限制操作的下界值。
* **max** (*Block*) – 限制操作的上界值。
* **propagate_nan** (*tl.PropagateNan*) – 是否将 NaN 值传播出去。此设置仅对张量`x`有效。如果`min`或`max`中任一值为NaN，则最终结果将无法确定。

> 另请参阅  
> tl.PropagateNan

