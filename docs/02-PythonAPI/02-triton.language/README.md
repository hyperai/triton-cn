## 编程模式

| tensor |表示一个值或指针的 N 维数组|
|:----|:----|
| program_id |沿指定轴返回当前程序实例的 id|
| num_programs |沿指定轴返回当前程序实例的数量|

## 创建操作

| arange |返回半开区间 [start, end) 内的连续值|
|:----|:----|
| cat |连接给定的块 |
| full |返回一个张量，该张量填充了指定 shape 和 dtype 的标量值|
| zeros |返回一个张量，该张量用指定 shape 和 dtype 填充了标量值 0|
| zeros_like |返回一个 shape 和 dtype 与给定张量相同的全零张量|
| cast |将张量转换为指定的 dtype|

## Shape Manipulation Ops

| broadcast |尝试将两个给定的块广播到一个共同兼容的 shape|
|:----|:----|
| broadcast_to |尝试将给定的张量广播到新的 shape|
| expand_dims |通过插入新的长度为 1 的维度来扩展张量的形状|
| interleave |沿着最后一个维度交错两个张量的值|
| join |在一个新的次要维度中连接给定的张量|
| permute |排列张量的维度|
| ravel |返回 x 的连续扁平视图|
| reshape |返回一个具有与输入相同元素数但具有提供的形状的张量|
| split |将张量沿其最后一个维度分成两部分，该维度大小必须为 2 |
| trans |排列张量的维度。 |
| view |返回具有与输入相同元素但形状不同的张量|

## Linear Algebra Ops

| dot |返回两个块的矩阵乘积|
|:----|:----|

## 内存/指针操作

| load |返回一个张量，其值从由*指针*定义的内存位置加载 |
|:----|:----|
| store |将数据张量存储到由指针定义的内存位置 |
| make_block_ptr |返回指向父张量中某个块的指针 |
| advance |推进一个块指针|

## 索引操作

| flip |沿着维度 *dim* 翻转张量 *x*|
|:----|:----|
| where |根据 condition 返回来自 x 或 y 的元素组成的张量|
| swizzle2d |将行主序排列为 size_i ****size_j* 的矩阵的索引，转换为每组 *size_g* 行的列主序矩阵的索引|

## 数学操作

| abs |计算 x 的逐元素绝对值|
|:----|:----|
| cdiv |计算 x 除以 div 的向上取整除法|
| ceil |计算 x 的逐元素向上取整值|
| clamp |将输入张量 x 的值限制在 [min, max] 范围内|
| cos |计算 x 的逐元素余弦值|
| div_rn |计算 x 和 y 的逐元素精确除法（根据 IEEE 标准四舍五入到最近的值）|
| erf |计算 x 的逐元素误差函数|
| exp |计算 x 的逐元素指数|
| exp2 |计算 x 的逐元素指数（以 2 为底）|
| fdiv |计算 x 和 y 的逐元素快速除法|
| floor |计算 x 的逐元素向下取整|
| fma |计算 x、y 和 z 的逐元素融合乘加运算|
| log |计算 x 的逐元素自然对数|
| log2 |计算 x 的逐元素对数（以 2 为底）|
| maximum |计算 x 和 y 的逐元素最大值|
| minimum |计算 x 和 y 的逐元素最小值|
| rsqrt |计算 x 的逐元素的平方根倒数|
| sigmoid |计算 x 的逐元素 sigmoid 函数值|
| sin | Computes the element-wise sine of x. 计算 x 的逐元素正弦值|
| softmax |计算 x 的逐元素 softmax 值|
| sqrt |计算 x 的逐元素快速平方根|
| sqrt_rn |计算 x 的逐元素精确平方根（根据 IEEE 标准四舍五入到最近的值）|
| umulhi |计算 x 和 y 的 2N 位乘积的逐元素最高有效 N 位|

## 归约操作

| argmax |返回沿指定 axis 轴上 input 张量中所有元素的最大索引|
|:----|:----|
| argmin |返回沿指定 axis 轴上 input 张量中所有元素的最小索引|
| max |返回沿指定 axis 轴上 input 张量中所有元素的最大值|
| min |返回沿指定 axis 轴上 input 张量中所有元素的最小值|
| reduce |将 combine_fn 应用于沿指定 axis 的 input 张量中的所有元素|
| sum |返回 input 张量中，沿指定 axis 的所有元素的总和|
| xor_sum |返回 input 张量中，沿指定 axis 的所有元素的异或和|

## 扫描/排序操作

| associative_scan|沿指定 axis 将 combine_fn 应用于 input 张量的每个元素和携带的值，并更新携带的值|
|:----|:----|
| cumprod |返回沿指定 axis 的 input 张量中所有元素的累积乘积|
| cumsum |返回沿指定 axis 的 input 张量中所有元素的累积和|
| histogram |基于 input 张量计算 1 个具有 num_bins 个 bin 的直方图，每个 bin 宽度为 1，起始于 0|
| sort |沿着指定维度对张量进行排序|

## 原子操作

| atomic_add |在由 pointer 指定的内存位置执行原子加法|
|:----|:----|
| atomic_and |在由 pointer 指定的内存位置执行原子逻辑和操作|
| atomic_cas |在由 pointer 指定的内存位置执行 1 个原子比较并交换操作|
| atomic_max |在由 pointer 指定的内存位置执行 1 个原子最大值操作|
| atomic_min |在由 pointer 指定的内存位置执行 1 个原子最小值操作|
| atomic_or |在由 pointer 指定的内存位置执行 1 个原子逻辑或操作|
| atomic_xchg |在由 pointer 指定的内存位置执行 1 个原子交换操作|
| atomic_xor |在由 pointer 指定的内存位置执行原子逻辑异或操作|


## 随机数生成

| randint4x |给定  1 个seed 标量和 1 个offset 块，返回 4 个 int32 类型的随机块|
|:----|:----|
| randint |给定 1 个 seed 标量和 1 个 offset 块，返回 1 个 int32 类型的随机块|
| rand |给定 1 个 seed 标量和 1 个 offset 块，返回 1 个在 $$U(0,1)$$ 中的 float32 类型的随机块|
| randn |给定 1 个 seed 标量和 1 个 offset 块，返回 1 个在 N(0,1) 中的 float32 类型的随机块|


## 迭代器

| range |永远向上计数的迭代器 |
|:----|:----|
| static_range |永远向上计数的迭代器 |


## 内联汇编

| inline_asm_elementwise |在张量上执行内联汇编 |
|:----|:----|


## 编译器提示操作

| debug_barrier |插入 1 个屏障以同步 1 个块中的所有线程|
|:----|:----|
| max_constancy |告知编译器 input 中的第 1 个值是常量|
| max_contiguous |告知编译器 input 中的第 1 个值是连续 |
| multiple_of |告知编译器 input 中的所有值都是 value 的倍数|


## 调试操作

| static_print |在编译时打印数值 |
|:----|:----|
| static_assert |在编译时断言条件|
| device_print |在运行时从设备打印数值|
| device_assert |在运行时从设备上断言条件|


