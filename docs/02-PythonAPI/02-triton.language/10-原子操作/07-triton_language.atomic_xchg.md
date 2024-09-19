```python
triton.language.atomic_xchg(pointer, val, mask=None, sem=None, scope=None)
```


在由 `pointer` 指定的内存位置执行 1 个原子交换操作。 


在原子操作前返回保存在 `pointer` 处的数据。


**参数****：**

* **pointer** (*Block of dtype=triton.PointerDType*) - 要操作的内存位置。
* **val** (*Block of dtype=pointer.dtype.element_ty*) - 执行原子操作所需的值。
* **sem** (*str, optional*) - 指定操作的内存语义。可接受的值为「acquire」（获取）、「release」（释放）、「acq_rel」（获取释放）和「relaxed」（放松）。如果未提供，则函数默认使用「acq_rel」语义。
* **scope** (*str, optional*) - 定义观察原子操作同步效果的线程范围。可接受的值为「gpu」（默认）、「cta」（协作线程数组，线程块）或「sys」（代表「SYSTEM」）。默认值为「gpu」。

这个函数也可作为 `tensor` 的成员函数调用，使用 `x.atomic_xchg(...)` 而不是 `atomic_xchg(x, ...)`。


