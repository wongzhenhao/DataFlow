# DataFlow-Preview-开发文档

先安装：
```shell
pip install -e .
```

选择性安装:
```shell
pip install -e .[all]
```

安装text组件
```shell
pip install -e .[text]
```

## 基于命令行的调用方式
从pypi查看是否是最新版本
```
dataflow -v 
```

在本地某一路径生成算子运行所需的shell脚本和yaml脚本 (目前todo，可以讨论潜在的指令)
```shell
dataflow init all
dataflow init reasoning
```
