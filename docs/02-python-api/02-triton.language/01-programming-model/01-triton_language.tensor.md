---
title: triton_language.tensor
---

```python
class triton.language.tensor(self, handle, type: dtype)
```
表示一个值或指针的 N 维数组。

在 Triton 程序中，`tensor` 是最基本的数据结构。`triton.language` 中的大多数函数对 tensors 进行操作并返回。


这里大多数命名的成员函数都是 `triton.language` 中自由函数的重复。例如，`triton.language.sqrt(x)` 等同于 `x.sqrt()`。


`tensor` 还定义了大部分的魔法/双下划线方法，因此可以像写 `x+y`、`x << 2` 等等。


**构造函数**

```python
__init__(self, handle, type: dtype)
```
不被用户代码调用。

**方法**

| __init__(self, handle, type) |不被用户代码调用|
|:----|:----|
| abs(self) |转发到 abs() 自由函数|
| advance(self, offsets) |转发到 advance() 自由函数|
| argmax(self, *kwargs) |转发到 argmax() 自由函数|
| argmin(self, *kwargs) |转发到 argmin() 自由函数|
| associative_scan(self, axis, combine_fn[, ...]) |转发到 associative_scan() 自由函数|
| atomic_add(self, val[, mask, sem, scope]) |转发到 atomic_add() 自由函数|
| atomic_and(self, val[, mask, sem, scope]) |转发到 atomic_and() 自由函数|
| atomic_cas(self, cmp, val[, sem, scope]) |转发到 atomic_cas() 自由函数|
| atomic_max(self, val[, mask, sem, scope]) |转发到 atomic_max() 自由函数|
| atomic_min(self, val[, mask, sem, scope]) |转发到 atomic_min() 自由函数|
| atomic_or(self, val[, mask, sem, scope]) |转发到 atomic_or() 自由函数|
| atomic_xchg(self, val[, mask, sem, scope]) |转发到 atomic_xchg() 自由函数|
| atomic_xor(self, val[, mask, sem, scope]) |转发到 atomic_xor() 自由函数|
| broadcast_to(self, *shape) |转发到 broadcast_to() 自由函数|
| cast(self, dtype[, fp_downcast_rounding, ...]) |转发到 cast() 自由函数|
| cdiv(*self,**kwargs) |转发到 cdiv() 自由函数|
| ceil(self) |转发到 ceil() 自由函数|
| cos(self) |转发到 cos() 自由函数|
| cumprod(*self,**kwargs) |转发到 cumprod() 自由函数|
| cumsum(*self,**kwargs) |转发到 cumsum() 自由函数|
| erf(self) |转发到 erf() 自由函数|
| exp(self) |转发到 exp() 自由函数|
| exp2(self) |转发到 exp2() 自由函数|
| expand_dims(self, axis) |转发到 expand_dims() 自由函数|
| flip(*self,**kwargs) |转发到 flip() 自由函数|
| floor(self) |转发到 floor() 自由函数|
| histogram(self, num_bins) |转发到 histogram() 自由函数|
| log(self) |转发到 log() 自由函数|
| log2(self) |转发到 log2() 自由函数|
| logical_and(self, other) | |
| logical_or(self, other) ||
| max(*self,**kwargs) |转发到 max() 自由函数|
| min(*self,**kwargs) |转发到 min() 自由函数|
| permute(self, *dims) |转发到 permute() 自由函数|
| ravel(*self,**kwargs) |转发到 ravel() 自由函数|
| reduce(self, axis, combine_fn[, keep_dims]) |转发到 reduce() 自由函数|
| reshape(self, *shape[, can_reorder]) |转发到 reshape() 自由函数|
| rsqrt(self) |转发到 rsqrt() 自由函数|
| sigmoid(*self,**kwargs) |转发到 sigmoid() 自由函数|
| sin(self) |转发到 sin() 自由函数|
| softmax(*self,**kwargs) |转发到 softmax() 自由函数|
| sort(*self,**kwargs) |转发到 sort() 自由函数|
| split(self) |转发到 split() 自由函数|
| sqrt(self) |转发到 sqrt() 自由函数|
| sqrt_rn(self) |转发到 sqrt_rn() 自由函数|
| store(self, value[, mask, boundary_check, ...]) |转发到 store() 自由函数|
| sum(*self,**kwargs) |转发到 sum() 自由函数|
| to(self, dtype[, fp_downcast_rounding, bitcast]) |tensor.cast() 的别名|
| trans(self, *dims) |转发到 trans() 自由函数|
| view(self, *shape) |转发到 view() 自由函数|
| xor_sum(self[, axis, keep_dims]) |转发到 xor_sum() 自由函数|


**属性**

| T |转置  1 个 2D 张量|
|:----|:----|


