# 安装命令清单

当前机器 Python 版本是 `3.12.2`，下面这套命令按 Python 3.12 写。

## 推荐安装方式

### 1. 进入项目目录

```bash
cd /home/xwl/药物禁忌
```

### 2. 创建虚拟环境

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. 升级安装工具

```bash
python -m pip install -U pip setuptools wheel
```

### 4. 安装完整依赖

```bash
python -m pip install -r /home/xwl/药物禁忌/requirements.txt
```

### 5. 检查脚本依赖是否齐全

```bash
python /home/xwl/药物禁忌/scripts/train_fingerprint_models.py --check-env
```

## 不想一次装完时的分步安装命令

### 基础数值与绘图

```bash
python -m pip install "numpy>=1.26,<3" "pandas>=2.2,<3" "scikit-learn>=1.5,<2" "matplotlib>=3.9,<4"
```

### 特征选择

```bash
python -m pip install "mrmr-selection>=0.2.8"
```

安装后在代码里导入方式仍然是：

```python
import mrmr
```

### 独立模型

```bash
python -m pip install "xgboost==2.1.0" "catboost>=1.2.10,<2"
```

### AutoGluon 平台

```bash
python -m pip install "autogluon.tabular[catboost,xgboost]==1.5.0"
```

## 当前这次报错的原因

你遇到的冲突是因为：

- 原来清单里写了 `xgboost>=3.2,<4`
- 但 `autogluon.tabular 1.5.0` 自己要求 `xgboost<3.2,>=2.0`

所以 `pip` 无法同时满足两边约束。

我已经把依赖改成了：

```text
xgboost==2.1.0
autogluon.tabular[catboost,xgboost]==1.5.0
```

这样主要是为了两点：

1. 和 AutoGluon 1.5.0 的依赖范围保持一致。
2. `xgboost 2.1.0` 在 PyPI 上提供 `manylinux2014_x86_64` 轮子，通常比更新版本在 Linux 上更稳。

## 建议你现在直接执行

如果你已经在虚拟环境里，直接重新运行：

```bash
cd /home/xwl/药物禁忌
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
```

如果还想更稳一点，可以分两步装：

```bash
python -m pip install "numpy>=1.26,<3" "pandas>=2.2,<3" "scikit-learn>=1.5,<2" "matplotlib>=3.9,<4" "mrmr-selection>=0.2.8"
python -m pip install "xgboost==2.1.0" "catboost>=1.2.10,<2" "autogluon.tabular[catboost,xgboost]==1.5.0"
```

装完后检查：

```bash
python /home/xwl/药物禁忌/scripts/train_fingerprint_models.py --check-env
```

## 安装完成后的运行命令

### 环境检查

```bash
python /home/xwl/药物禁忌/scripts/train_fingerprint_models.py --check-env
```

### 用默认配置启动训练

```bash
bash /home/xwl/药物禁忌/scripts/run_training.sh
```

### 指定参数启动训练

```bash
bash /home/xwl/药物禁忌/scripts/run_training.sh /home/xwl/药物禁忌/configs/default_experiment.json \
  --seed 2025 \
  --test-size 0.2 \
  --validation-size 0.2 \
  --ifs-step 100
```

## 版本说明

这份依赖范围是按当前官方安装信息和你机器上的 `Python 3.12.2` 写的，重点参考了这些来源：

- AutoGluon 官方安装文档写明 `autogluon.tabular[all]` 是完整安装方式，同时也支持按需安装可选依赖。  
- PyPI 当前信息显示 `autogluon.tabular 1.5.0` 支持 Python 3.10 到 3.13。  
- PyPI 当前信息显示 `autogluon.tabular 1.5.0` 对 `xgboost` 的要求是 `<3.2, >=2.0`。  
- PyPI 当前信息显示 `xgboost 2.1.0` 提供 `manylinux2014_x86_64` 轮子。  
- PyPI 当前信息显示 `catboost 1.2.10` 已提供 CPython 3.12 的包。  
- `mrmr-selection` 的安装包名是 `mrmr-selection`，导入名是 `mrmr`。

如果你后面想把环境再拆得更干净，我也可以继续帮你补一份 `requirements-minimal.txt` 和 `requirements-full.txt` 双版本。
