```python
triton.language.store(pointer, value, mask=None, boundary_check=(), cache_modifier='', eviction_policy='')
```

将数据张量存储到由指针定义的内存位置。

1.如果 `pointer` 是单元素指针，则加载 1 个标量。在这种情况下：

- `mask`必须是标量，

- `boundary_check` 和 `padding_option` 必须为空。

  2.如果 `pointer` 是 1 个 N 维指针张量，则会存储 1 个 N 维张量。在这种情况下：

- `mask` 会被隐式地广播到 `pointer.shape`，

- `boundary_check` 必须为空。

  3.如果 `pointer` 是由 `make_block_ptr` 定义的块指针，则会存储 1 个张量。在这种情况下：

- `mask` 必须为 `None`，

- 可以指定 `boundary_check` 以控制越界访问的行为。

_value_ 会被隐式地广播为 _pointer.shaoe_ ，并转换为 _pointer.element_ty_ 类型。

**参数\*\***：\*\*

- **pointer** (_triton.PointerType, 或 dtype=triton.PointerType 的块_) - 存储 value 元素的内存位置。
- **value** (_Block_) - 要存储的元素张量。
- **mask**（_triton.int1 的块_, _可选_）- 如果 mask[idx] 为 false，则不将 value[idx] 存储在 pointer[idx] 处。
- **boundary_check**（_整数元组_, _可选_）- 整数元组，表示应该进行边界检查的维度。
- **cache_modifier** (str，可选，应为 {“”（空字符串）、“.wb”（表示缓存回写所有一致性层级）、“.cg”（表示全局缓存）、“.cs”（表示缓存流）、“.wt”（表示缓存直写）} 中的一个）更多详情请参见[缓存操作符](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators)）—— 在 NVIDIA PTX 中更改缓存选项。
- **eviction_policy**（_str_，_可选_，_应为 {"", "evict_first", "evict_last"} 中的一个_）- 更改 NVIDIA PTX 中的驱逐策略。

这个函数也可作为 `tensor` 的成员函数调用，使用 `x.store(...)` 方式而不是 `store(x, ...)`。
