```python
triton.language.interleave(a, b)
```


沿着最后 1 个维度交错 2 个张量的值。这 2 个张量必须有相同形状。等同于 *tl.join(a, b).reshape(a.shape[-1:] + [2*****a.shape[-1]])**。*


**参数****：**

* **a** (*Tensor*) – 第 1 个输入张量。
* **b** (*Tensor*) – 第 2 个输入张量。

