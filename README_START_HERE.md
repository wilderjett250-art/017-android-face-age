# AnZhuoFace 快速开始

## 只想体验

1. 从 GitHub Releases 下载 `AnZhuoFace-v1.0-debug.apk`。
2. 安装到 Android 设备。
3. 点击“演示样例”，确认本地模型可以运行。
4. 再使用“选择图片”或“相机识别”。

## 想看源码

用 Android Studio 打开本文件所在的项目根目录，不要只打开 `app` 子目录。项目需要 JDK 17 和 Android SDK 35。

```powershell
.\gradlew.bat :app:assembleDebug --no-daemon
```

## 先看哪些文件

- `app/src/main/java/com/anzhuoface/app/MainActivity.kt`：主页面和图片分析流程。
- `app/src/main/java/com/anzhuoface/app/FaceAgeAnalyzer.kt`：人脸检测、预处理和 ONNX 推理。
- `app/src/main/java/com/anzhuoface/app/CameraActivity.kt`：相机采样和稳定结果。
- `training/`：训练配置、训练脚本和 ONNX 导出脚本。
- `docs/`：架构与模型说明。

完整说明请阅读根目录 [`README.md`](README.md)。

## English

Install the APK from the latest GitHub Release for a quick demo. To inspect or build the source, open the repository root in Android Studio with JDK 17 and Android SDK 35. The main processing path is implemented in `MainActivity.kt` and `FaceAgeAnalyzer.kt`.
