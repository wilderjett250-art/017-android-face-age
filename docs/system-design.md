# AnZhuoFace 系统设计

## 1. 定位

AnZhuoFace 将人脸检测、年龄段估计、性别辅助识别和移动端展示组合成一个离线 Android 工具。它既能作为普通用户直接安装的演示应用，也保留了训练脚本、模型导出和端侧部署代码，方便继续研究。

## 2. 总体架构

```text
输入层：相册图片 / 相机预览帧 / 内置样例
              ↓
检测层：Google ML Kit Face Detection
              ↓
预处理层：人脸框扩展、裁剪、缩放、颜色转换、归一化
              ↓
推理层：OpenCV DNN 年龄模型 + ONNX Runtime 性别模型
              ↓
展示层：人脸框、年龄段、折算年龄、性别、置信度、耗时
              ↓
本地记录：SharedPreferences 保存最近识别摘要
```

## 3. 输入与交互

主页面由 `MainActivity` 管理，支持：

- 从系统相册选择图片。
- 使用相机页面连续采样并选取质量更好的帧。
- 加载内置样例，方便首次运行和现场演示。
- 在三套模型之间切换。
- 查看识别历史并清空本地记录。

相机页面由 `CameraActivity` 管理。它每隔一段时间读取预览帧，结合人脸面积、图像清晰度和模型置信度计算帧质量，再对最近样本做加权汇总，减少单帧抖动带来的结果跳变。

## 4. 检测与预处理

`FaceAgeAnalyzer` 首先通过 ML Kit 获取人脸边界框。对每张脸做适度扩展后裁剪，随后统一到模型要求的输入尺寸，并完成：

1. RGBA 到 BGR/RGB 的颜色转换。
2. 像素值归一化。
3. OpenCV `blobFromImage` 组织模型输入。
4. 端侧 ONNX 推理和输出后处理。

每张人脸都会独立生成年龄预测和性别辅助预测，结果再回写到原图框选区域。

## 5. 模型与资源

模型资源位于 `app/src/main/assets/`：

- `fairface_mobilenetv2_dex_distill.onnx`
- `fairface_multitask.onnx`
- `fairface_multitask_lite.onnx`
- `gender_classifier.onnx`

`ModelProfile` 将模型文件、输入尺寸、展示名称和验证指标绑定在一起，界面只需要切换配置即可完成对照测试。

## 6. 本地数据与隐私

默认流程不依赖远程接口，图片在设备内完成检测与推理。历史记录仅保存时间、输入来源、模型名称、人脸数量、摘要和耗时等信息，不上传到服务器。用户可在主页面清空历史。

## 7. 训练与部署闭环

`training/scripts/train_multitask_fairface.py` 负责训练多任务模型，`training/scripts/export_onnx.py` 负责导出 ONNX。典型流程：

1. 准备遵守许可的数据集和训练环境。
2. 使用 `training/configs/` 中的配置训练模型。
3. 导出 ONNX 并进行独立验证。
4. 将选定模型复制到 Android `assets` 目录。
5. 使用 Gradle 构建 APK，在设备上验证图片、相机和历史记录流程。

## 8. 工程边界

当前产品输出是年龄段估计和性别辅助分类。年龄段中点仅用于界面展示；模型对光照、姿态、遮挡、人脸大小和数据分布变化敏感。后续可以继续扩展更完整的实时相机流、量化部署、消融实验和更广泛的设备测试。
