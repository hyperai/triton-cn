---
title: triton_language.cat
---

```python
triton.language.cat(input, other, can_reorder=False)
```


连接给定的块。


**参数****：**

* **input** (*Tensor*) - 第一个输入张量。

* **other** (*Tensor*) - 第二个输入张量。

* **reorder** - 编译器提示。如果为 true，则允许编译器在连接输入时重新排序元素。仅在顺序不重要时使用（例如，结果仅用于缩减操作）。当前的 cat 实现仅支持 can_reorder=True。

