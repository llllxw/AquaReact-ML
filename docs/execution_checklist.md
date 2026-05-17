# 指纹建模实验执行清单

## 一、具体执行清单

1. 固定数据输入  
   使用 `/home/xwl/药物禁忌/元数据/训练集` 和 `/home/xwl/药物禁忌/元数据/测试集` 里的现成特征矩阵，按同名文件重新拼接成完整样本池，再重新划分新的内部训练集、验证集、测试集。

2. 固定随机划分  
   通过 `random_seed`、`test_size`、`validation_size` 固定本轮实验划分，避免 notebook 手工划分造成结果不可复现。

3. 单指纹与组合指纹逐一建模  
   当前脚本会自动识别并遍历：
   `ECFP4`
   `FCFP4`
   `MACCS`
   `RDKit`
   `E+F`
   `E+M`
   `E+R`
   `F+M`
   `F+R`
   `R+M`
   `E+F+M`
   `E+F+R`
   `E+R+M`
   `F+R+M`
   `E+F+R+M`

4. 特征排序与 IFS  
   默认先用 `mRMR` 做特征排序。若本机没有安装 `mrmr`，脚本会回退到 `mutual_info` 排序。  
   IFS 会对每个最终模型分别打分，并为每个模型单独选择自己的最优特征维度，而不是所有模型共用同一个特征数。

5. 模型训练与评估  
   训练模型包括：
   `DT`
   `RF`
   `ExtraTrees`
   `KNN`
   `GB`
   `XGBoost`
   `CatBoost`
   `AutoGluon`

6. 结果汇总  
   输出每个指纹方案在内部独立测试集上的 `Accuracy`、`AUC`、`Precision`、`Recall`、`F1Score`、`Sensitivity`、`Specificity`、`MCC`。

7. 最优组合图件  
   自动从“分子指纹组合”里筛选测试集 AUC 最高的组合，并生成：
   模型比较柱状图  
   ROC 曲线图

8. AutoGluon 平台内部模型图件  
   若最优组合的最优模型来自 `AutoGluon`，自动补充内部子模型性能指标表和热图。

## 二、输出文件清单

### 1. 根目录输出

每次运行会在 `outputs/run_时间戳/` 下生成一套完整结果。

核心文件包括：

`resolved_config.json`  
本次实验最终实际使用的配置。

`split_manifest.csv`  
每条样本的新划分结果，标记为 `train / validation / test`。

### 2. 总表目录

目录：`tables/`

`all_metrics_long.csv`  
所有指纹方案、所有模型、所有指标的长表。

`summary_accuracy_wide.csv`  
适合直接整理成论文表格的 Accuracy 宽表。

`summary_auc_wide.csv`  
适合直接整理成论文表格的 AUC 宽表。

`selected_feature_counts.csv`  
每个指纹方案在不同模型下最终选中的特征维度。

`paper_ready_summary.csv`  
合并特征维度、Accuracy、AUC 的汇总表。

### 3. 每个指纹方案目录

目录：`feature_sets/<feature_set>/`

`ranked_features.csv`  
特征排序结果。

`ifs_results.csv`  
IFS 每个特征数对应的验证集结果。

`ifs_curve.png`  
IFS 曲线图。

`selected_feature_counts_by_model.csv`  
当前指纹方案下，不同模型各自的最优特征维度。

`selected_features_by_model.csv`  
当前指纹方案下，不同模型各自最终使用的特征列表。

`test_metrics.csv`  
该指纹方案下各模型在内部独立测试集上的结果。

`roc_curve_data.csv`  
该指纹方案下各模型的 ROC 源数据。

`autogluon_model/`  
如果训练了 AutoGluon，会保存对应模型目录。

### 4. 最优组合目录

目录：`best_feature_combination/`

`best_selection.json`  
记录最优分子指纹组合和对应最优模型。

`model_metrics.csv`  
最优组合下不同模型的指标表。

`model_metrics_bar.png`  
最优组合下各模型多指标柱状图。

`roc_curve_data.csv`  
最优组合下各模型 ROC 源数据。

`roc_curves.png`  
最优组合下各模型 ROC 曲线图。

`autogluon_internal_metrics.csv`  
若最优模型来自 AutoGluon，则保存平台内部模型指标。

`autogluon_internal_heatmap.png`  
若最优模型来自 AutoGluon，则生成内部模型性能热图。

## 三、命令行运行方式

### 环境检查

```bash
python /home/xwl/药物禁忌/scripts/train_fingerprint_models.py --check-env
```

### 使用默认配置直接训练

```bash
bash /home/xwl/药物禁忌/scripts/run_training.sh
```

### 指定配置文件训练

```bash
bash /home/xwl/药物禁忌/scripts/run_training.sh /home/xwl/药物禁忌/configs/default_experiment.json
```

### 运行时覆盖部分参数

```bash
bash /home/xwl/药物禁忌/scripts/run_training.sh /home/xwl/药物禁忌/configs/default_experiment.json \
  --seed 2025 \
  --test-size 0.2 \
  --validation-size 0.2 \
  --ifs-step 100
```

## 四、当前环境提醒

我在当前机器上检查到以下包缺失：

`matplotlib`
`mrmr`
`xgboost`
`catboost`
`autogluon.tabular`

这意味着脚本结构已经可以直接使用，但若要完整跑出论文所需全部图表和模型结果，需要先补齐这些依赖。
