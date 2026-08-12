# 017 Android Face Age Analysis

> An Android app that imports images or opens the camera to display face boxes, age ranges, and auxiliary recognition results.

## Problem

Server-based face analysis fails offline and requires images to leave the device.

## Demo

![Sample face input](app/src/main/assets/sample_face.jpg)

Import the sample image or open the camera to inspect the on-device result.

The model and sample assets are delivered with the project for Android Studio reproduction.

## Highlights

- Kotlin Android application.
- Local inference with an ONNX model.
- Images, photos, live camera, and local video.
- Model and sample assets are included.

## Tech

`Kotlin · Android · CameraX · ONNX Runtime · OpenCV/ML Kit`

## Reproduce from ZIP

1. Extract the ZIP and open the project root in Android Studio.
2. Connect a device or start an emulator.
3. Run `gradlew.bat assembleDebug` and install the APK.
4. Choose an image, photo, or camera input from the home page.

**Expected result:** After these steps, you should see the project's page, window, device output, or test result.

## Scope and Safety

Age ranges and auxiliary recognition are not identity, medical, or safety conclusions; obtain permission before using face images.

## Contact

Open to technical exchange.
