```python
triton.language.make_block_ptr(base: tensor, shape, strides, offsets, block_shape, order)
```


返回指向父张量中 1 个块的指针。


**参数****：**

* **base** - 父张量的基础指针。
* **shape** - 父张量的形状。
* **strides** - 父张量的步幅。
* **offsets** - 块的偏移量。
* **block_shape** - 块的形状。
* **order** - 原始数据格式的顺序。

