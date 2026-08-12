# 017 Android 人脸年龄分析 | Android Face Age

> 在手机本地运行 ONNX 模型，完成图片、摄像头和视频中的人脸与年龄段分析。
>
> **English:** A practical, runnable project with a documented workflow for the problem described above.

## 项目展示 / Demo

![示例人脸输入](app/src/main/assets/sample_face.jpg)

## 解决什么问题 / Problem

解决移动端人脸分析依赖服务器、离线环境无法运行和数据不便出端的问题。

**English:** This project addresses the problem above with a reproducible local workflow.

## 有什么用 / Use

导入图片或打开摄像头，在 Android 端显示人脸框、年龄段和辅助识别结果。

**English:** Run the workflow locally, inspect the output, and extend the project from the provided source.

## 高光亮点 / Highlights

- Kotlin Android 应用
- ONNX 模型本地推理
- 图片、拍照、实时摄像头、本地视频
- 模型和示例资源随工程交付

## 技术名词 / Tech

`Kotlin · Android · CameraX · ONNX Runtime · OpenCV/ML Kit`

## 从 ZIP 开始复现 / Reproduce from ZIP

1. 下载 ZIP 并解压，用 Android Studio 打开根目录。
2. 连接设备或启动模拟器。
3. 执行 gradlew.bat assembleDebug 并安装 APK。
4. 从首页选择图片、拍照或摄像头进行测试。
5. 模型和样例位于 app/src/main/assets。

**Expected result:** 应用启动后可以查看人脸检测框和年龄分析结果；真机效果应在目标设备上复验。

## 目录提示 / Notes

- 先阅读本 README，再按项目内更详细的中文/英文文档补充配置。
- 不要把真实密码、Token、数据库业务数据和本机运行结果提交回仓库。
- 下载 ZIP 后的第一次运行应使用测试数据或示例图片，确认链路正常后再接入自己的环境。

[English documentation](README.en.md)
