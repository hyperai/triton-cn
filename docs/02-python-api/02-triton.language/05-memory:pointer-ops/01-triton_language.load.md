---
title: triton_language.load
---

```python
triton.language.load(pointer, mask=None, other=None, boundary_check=(), padding_option='', cache_modifier='', eviction_policy='', volatile=False)
```


返回 1 个数据张量，其值从由指针所定义的内存位置处加载：


1.如果 `pointer` 是单元素指针，则加载 1 个标量。在这种情况下：

* `mask` 和 `other` 必须也是标量，

    

* `other` 会隐式地转换为 `pointer.dtype.element_ty` 类型，

    

* `boundary_check` 和 `padding_option` 必须为空。

2.如果 `pointer` 是 1 个 N 维指针张量，则加载 1 个 N 维张量。在这种情况下：

* `mask` 和 `other` 会被隐式地广播到 `pointer.shape`，

    

* `other` 会隐式地转换为 `pointer.dtype.element_ty` 类型，

    

* `boundary_check` 和 `padding_option` 必须为空。

   

3.如果 `pointer` 是由 `make_block_ptr` 定义的块指针，则加载 1 个张量。在这种情况下：

* `mask` 和 `other` 必须为 `None`，

    

* 可以指定 `boundary_check` 和 `padding_option` 来控制超出越界访问的行为。

**参数****：**

* **pointer**（*triton.PointerType**，**或 dtype=triton.PointerType 的块*）- 指向要加载的数据的指针。
* **mask**（*triton.int1 的块**，**可选*）- 如果 mask[idx] 为 false，则不加载 pointer[idx] 处的数据（对于块指针必须为 None）。
* **other** (*块*, *可选*) - 如果 mask[idx] 为 false，则返回 other[idx]。
* **boundary_check**（*整数元组**，**可选*）- 表示应进行边界检查维度的元组。
* **padding_option** - 应为 {“”, “zero”, “nan”} 中的一个，越界时进行填充。
* **cache_modifier****（***str**，可选，**应为 {“”, “ca”, “cg”} 中的一个*）- 其中「ca」表示在所有层级进行缓存，「cg」表示在全局层级缓存（在 L2 及以下缓存，不是 L1），详细信息请参见[缓存操作符](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators)。）在 NVIDIA PTX 中更改缓存选项。
* **eviction_policy**（*str*, *可选）*- 更改 NVIDIA PTX 中的驱逐策略。
* **volatile**（*bool*, *可选）*- 更改 NVIDIA PTX 中的易失性选项。

