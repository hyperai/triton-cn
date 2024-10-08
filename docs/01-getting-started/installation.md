---
title: 安装
---

已支持的平台/操作系统及硬件，请查看 [Github Compatibility 板块](https://github.com/triton-lang/triton?tab=readme-ov-file#compatibility)。


## Binary 版本


通过 pip 安装最新 Triton 稳定版：

```plain
pip install triton
```


针对 CPython 3.8-3.12 及 PyPy 3.8-3.9 的 Binary wheels 现已可用。


最新 Nightly 版本如下：

```plain
pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly
```


## 源码安装


### Python Package


运行以下命令从源代码安装 Python 软件包：

```plain
git clone https://github.com/triton-lang/triton.git;
cd triton/python;
pip install ninja cmake wheel; # build-time dependencies
pip install -e .
```


注意：如果系统上没有安装 llvm，可以通过 setup.py 脚本下载官方 LLVM 静态库并自动链接。


如需使用自定义 LLVM 进行构建，请查看 Github [Building with a custom LLVM](https://github.com/triton-lang/triton?tab=readme-ov-file#building-with-a-custom-llvm) 板块。


然后可通过运行单元测试对安装情况进行测试：

```plain
pip install -e '.[tests]'
pytest -vs test/unit/
```


Benchmarks 如下：

```plain
cd bench
python -m run --with-plots --result-dir /tmp/triton-bench
```
