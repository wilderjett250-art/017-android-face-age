# 公开版验证记录

## Android 构建

```text
命令：gradlew.bat :app:assembleDebug --no-daemon --stacktrace
结果：成功
```

构建过程中出现的 SDK XML 版本提示来自本机 Android SDK 工具版本组合，不影响本次构建结果。

## APK 内容检查

发布 APK：`release/AnZhuoFace-v1.0-debug.apk`

- 文件大小：293,345,647 bytes
- SHA-256：`3434BB367D71D61587D74701EEB36D307609A4F7BFB4E50F2A571DA997EEC7A7`
- 已包含 `AndroidManifest.xml`、`classes.dex` 和四个端侧 ONNX 模型资源。
- APK 通过 ZIP 条目检查，模型资源可在包内找到。

## 公开源码检查

- Git 跟踪文件：53 个。
- 未跟踪 APK、Gradle 构建目录、训练权重、`__pycache__` 或本地运行输出。
- 源码仓库不包含论文 Word/PDF、交付截图、训练曲线和本地路径信息。
