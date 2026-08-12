# 017 Android 人脸年龄分析 / Android Face Age Analysis

> 在 Android 端导入图片或打开摄像头，展示人脸框、年龄段和辅助识别结果。
>
> **English:** An Android app that imports images or opens the camera to display face boxes, age ranges, and auxiliary recognition results.

## 解决什么问题 / Problem

移动端人脸分析依赖服务器时，离线环境无法使用，图片也不便留在设备内处理。

**English:** Server-based face analysis fails offline and requires images to leave the device.

## 项目展示 / Demo

![示例人脸输入 / Sample face](app/src/main/assets/sample_face.jpg)

导入示例图片或打开摄像头，查看设备端推理结果。

模型和示例资源随工程交付，便于在 Android Studio 中复现。

**English:** The model and sample assets are delivered with the project for Android Studio reproduction.

## 高光亮点 / Highlights

- Kotlin Android 应用。
  **English:** Kotlin Android application.
- ONNX 模型本地推理。
  **English:** Local inference with an ONNX model.
- 图片、拍照、实时摄像头和本地视频。
  **English:** Images, photos, live camera, and local video.
- 模型与示例资源随工程提供。
  **English:** Model and sample assets are included.

## 技术名词 / Tech

`Kotlin · Android · CameraX · ONNX Runtime · OpenCV/ML Kit`

## 从 ZIP 开始复现 / Reproduce from ZIP

1. 解压 ZIP，用 Android Studio 打开工程根目录。
2. 连接设备或启动模拟器。
3. 执行 `gradlew.bat assembleDebug` 并安装 APK。
4. 从首页选择图片、拍照或摄像头进行测试。

**Expected result:** 完成上述步骤后，应能看到项目的页面、窗口、设备输出或测试结果。

**Expected result:** After these steps, you should see the project's page, window, device output, or test result.

## 范围与安全 / Scope and Safety

年龄段和辅助识别结果不是身份、医疗或安全结论；使用前应取得人脸图像授权。

**English:** Age ranges and auxiliary recognition are not identity, medical, or safety conclusions; obtain permission before using face images.

## 交流 / Contact

欢迎交流技术。

Open to technical exchange.

[English full version](README.en.md)
