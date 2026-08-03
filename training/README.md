# 本地训练与 ONNX 导出

本目录保留 AnZhuoFace 的模型训练配置、训练脚本和 ONNX 导出脚本。FairFace 原始图片和标注不随仓库发布，使用前请自行准备数据并遵守其许可。

## 目录

- `configs/`：不同主干网络、输入尺寸和训练策略的 YAML 配置。
- `scripts/train_multitask_fairface.py`：年龄段 + 性别多任务训练。
- `scripts/export_onnx.py`：从 PyTorch checkpoint 导出 ONNX。
- `artifacts/`：本地训练输出目录，默认被 Git 忽略。

## 训练前检查

训练属于 GPU 密集型任务。开始前请确认当前 Python 环境能看到 CUDA，并确认显卡资源可用：

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
nvidia-smi
```

只有在输出确认 CUDA 可用后再开始训练。数据集目录、批大小和设备编号按实际环境调整。

## 训练示例

```powershell
python .\training\scripts\train_multitask_fairface.py `
  --config .\training\configs\fairface_multitask_improved.yaml
```

导出 ONNX：

```powershell
python .\training\scripts\export_onnx.py `
  --checkpoint .\training\artifacts\fairface_multitask_improved\best.pt `
  --output .\training\artifacts\fairface_multitask_improved\best.onnx
```

导出的模型通过独立验证后，复制到 `app/src/main/assets/`，然后执行 Android 构建。

## 标签与指标

年龄标签为 `0-2`、`3-9`、`10-19`、`20-29`、`30-39`、`40-49`、`50-59`、`60-69` 和 `70+`。模型报告中的 MAE/CS(5) 使用年龄段中点计算，目的是比较模型版本，不等同于精确年龄回归误差。
