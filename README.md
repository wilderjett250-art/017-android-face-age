# AnZhuoFace · Android 端侧人脸年龄段与性别估计

AnZhuoFace 是一个可以直接安装和运行的 Android 端侧视觉工具：从相册或相机取得图片，先用 ML Kit 检测人脸，再用本地 ONNX 模型完成年龄段估计与性别辅助识别，并在手机上展示人脸框、预测标签、置信度、耗时和历史记录。

项目的重点是完整的“模型训练 → ONNX 导出 → Android 部署 → 本地推理”闭环。图片不会上传到服务器，默认推理过程在设备端完成。

> Age is reported as an age range. The number shown beside it is the midpoint of that range and is intended for visual reference, not exact age measurement.

## 先运行：直接安装 APK

打开 [Releases](https://github.com/wilderjett250-art/017-anzhuoface/releases/latest)，下载 `AnZhuoFace-v1.0-debug.apk`，在 Android 设备上安装即可。

首次使用时：

1. 允许相机权限（只有实时拍摄功能需要）。
2. 可以先点击“演示样例”确认模型加载正常。
3. 选择“相册图片”导入照片，或进入相机页面进行连续采样。
4. 在“模型选择”中切换标准版、轻量版和论文复现版进行对比。

APK 校验值保存在 [`release/SHA256SUMS.txt`](release/SHA256SUMS.txt)。

## 主要功能

- 相册导入、相机输入和内置样例三种输入方式。
- ML Kit 端侧人脸检测与人脸区域裁剪。
- OpenCV DNN 加载 ONNX 模型，离线完成年龄段推理。
- 独立的 ONNX 性别分类器作为辅助输出。
- 标准版、轻量版、论文复现版三套模型可切换。
- 在原图绘制人脸框、年龄段、折算年龄和置信度。
- 相机页面对连续采样帧进行质量评分和稳定结果汇总。
- 本地保存最近识别记录，展示输入尺寸、耗时、人脸数量和结果摘要。

## 模型版本

| 版本 | 主干网络 | 输入 | 年龄段 Top-1 | 相邻年龄段容错 | ONNX 体积 |
| --- | --- | ---: | ---: | ---: | ---: |
| 论文复现版 | MobileNetV2 + DEX 风格期望输出 | 160 × 160 | 54.57% | 93.94% | 12.03 MB |
| 标准版 | MobileNetV3 Large | 160 × 160 | 53.21% | 93.92% | 14.39 MB |
| 轻量版 | MobileNetV3 Small | 128 × 128 | 49.72% | 90.61% | 5.44 MB |

这些指标来自项目内部验证集。FairFace 的标签是年龄段，因此界面中的“约 XX 岁”是年龄段中点换算值，不能替代精确年龄测量。更完整的模型说明见 [`docs/model-report.md`](docs/model-report.md)。

## 从源码构建

### 环境

- Android Studio 最新稳定版
- JDK 17
- Android SDK Platform 35
- Android 设备或 Android Emulator，最低 Android 8.0（API 26）

### 构建 Debug APK

在项目根目录执行：

```powershell
.\gradlew.bat :app:assembleDebug --no-daemon
```

构建结果位于：

```text
app/build/outputs/apk/debug/app-debug.apk
```

也可以直接用 Android Studio 打开项目根目录，选择 `app` 模块运行。

### 训练与导出模型

训练脚本和配置位于 [`training/`](training/)。原始 FairFace 数据集不包含在仓库中，使用前请自行准备数据并遵守数据集许可。

```powershell
python .\training\scripts\train_multitask_fairface.py `
  --config .\training\configs\fairface_multitask_improved.yaml

python .\training\scripts\export_onnx.py `
  --checkpoint .\training\artifacts\fairface_multitask_improved\best.pt `
  --output .\training\artifacts\fairface_multitask_improved\best.onnx
```

训练产物默认写入 `training/artifacts/`，该目录属于本地实验输出，不纳入公开源码提交。导出的 ONNX 文件复制到 `app/src/main/assets/` 后即可由 Android 端加载。

## 目录结构

```text
app/                         Android 应用源码、界面、推理逻辑和内置模型
training/configs/            FairFace 训练配置
training/scripts/            多任务训练和 ONNX 导出脚本
docs/system-design.md        系统架构与处理流程
docs/model-report.md         模型版本和指标说明
release/                     APK 校验信息及下载说明（APK 本体作为 Release 附件）
```

核心代码：

- `MainActivity.kt`：图片输入、模型选择、结果展示和历史记录。
- `CameraActivity.kt`：相机预览、连续采样和稳定结果汇总。
- `FaceAgeAnalyzer.kt`：人脸检测、裁剪、归一化和 ONNX 推理。
- `AccurateGenderClassifier.kt`：端侧 ONNX 性别辅助分类。
- `ModelProfile.kt`：三套模型的输入尺寸、指标和资源映射。

## 隐私与使用边界

- 推理默认在本机完成，项目没有内置图片上传接口。
- 识别历史保存在 App 私有数据中，可在界面中清空。
- 年龄输出是年龄段估计，性别输出也是模型预测结果，适合学习、研究和演示场景。
- 请在获得照片本人授权的前提下使用，不要将结果用于身份核验、招聘、信贷或其他高风险决策。

## 开源许可

源码采用 MIT License，见 [`LICENSE`](LICENSE)。Android、OpenCV、ONNX Runtime、ML Kit 以及 FairFace 的使用仍需遵守各自的上游许可和条款；本仓库不包含 FairFace 原始图片数据集。

## English

AnZhuoFace is an offline Android computer-vision tool for face age-range estimation and auxiliary gender classification. It detects faces with ML Kit, preprocesses each crop locally, runs ONNX models through OpenCV DNN/ONNX Runtime, and displays annotated results, confidence scores, latency, and local history.

The project keeps the complete engineering path from FairFace-based training and ONNX export to Android deployment. No image upload service is required for the default workflow.

Download the ready-to-use APK from the [latest GitHub Release](https://github.com/wilderjett250-art/017-anzhuoface/releases/latest). See the Chinese sections above for build commands, model details, privacy boundaries, and repository structure.
