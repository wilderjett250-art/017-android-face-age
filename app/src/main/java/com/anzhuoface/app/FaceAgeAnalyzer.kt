package com.anzhuoface.app

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import kotlinx.coroutines.tasks.await
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect as CvRect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.dnn.Dnn
import org.opencv.dnn.Net
import org.opencv.imgproc.Imgproc
import kotlin.math.max
import kotlin.math.min

class FaceAgeAnalyzer(private val context: Context) {

    private val faceDetector by lazy {
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setContourMode(FaceDetectorOptions.CONTOUR_MODE_NONE)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
            .build()
        FaceDetection.getClient(options)
    }

    private val accurateGenderClassifier by lazy {
        AccurateGenderClassifier(context)
    }

    private val ageBuckets = listOf(
        "0-2" to 1,
        "3-9" to 6,
        "10-19" to 15,
        "20-29" to 25,
        "30-39" to 35,
        "40-49" to 45,
        "50-59" to 55,
        "60-69" to 65,
        "70+" to 75
    )

    private val netCache = mutableMapOf<String, Net>()

    suspend fun analyze(bitmap: Bitmap, profile: ModelProfile): List<Pair<Face, AgePrediction>> {
        val faces = faceDetector.process(InputImage.fromBitmap(bitmap, 0)).await()
        if (faces.isEmpty()) return emptyList()

        return faces.mapNotNull { face ->
            predictAgeAndGender(bitmap, face, profile)?.let { prediction -> face to prediction }
        }
    }

    private fun predictAgeAndGender(sourceBitmap: Bitmap, face: Face, profile: ModelProfile): AgePrediction? {
        OpenCvRuntime.ensureLoaded()
        val expandedRect = face.boundingBox.expand(sourceBitmap.width, sourceBitmap.height, 0.18f) ?: return null
        val inputSize = Size(profile.inputSize.toDouble(), profile.inputSize.toDouble())
        val ageNet = netCache.getOrPut(profile.key) {
            val model = AssetFileUtil.copyAssetToFile(context, profile.assetName)
            Dnn.readNetFromONNX(model.absolutePath)
        }

        val srcMat = Mat()
        val rgbaMat = Mat()
        val bgrMat = Mat()
        var faceMat: Mat? = null
        var resizedMat: Mat? = null
        var rgbMat: Mat? = null
        var floatMat: Mat? = null
        var normalizedMat: Mat? = null
        var blob: Mat? = null
        var ageOutput: Mat? = null

        return try {
            val croppedBitmap = Bitmap.createBitmap(
                sourceBitmap,
                expandedRect.left,
                expandedRect.top,
                expandedRect.width(),
                expandedRect.height()
            )

            Utils.bitmapToMat(sourceBitmap, srcMat)
            if (srcMat.type() != CvType.CV_8UC4) {
                srcMat.convertTo(rgbaMat, CvType.CV_8UC4)
            } else {
                srcMat.copyTo(rgbaMat)
            }

            Imgproc.cvtColor(rgbaMat, bgrMat, Imgproc.COLOR_RGBA2BGR)
            val safeRect = expandedRect.toSafeCvRect(bgrMat.width(), bgrMat.height()) ?: return null
            faceMat = Mat(bgrMat, safeRect)
            resizedMat = Mat()
            rgbMat = Mat()
            floatMat = Mat()
            normalizedMat = Mat()

            Imgproc.resize(faceMat, resizedMat, inputSize)
            Imgproc.cvtColor(resizedMat, rgbMat, Imgproc.COLOR_BGR2RGB)
            rgbMat.convertTo(floatMat, CvType.CV_32FC3, 1.0 / 255.0)
            Core.subtract(floatMat, Scalar(0.485, 0.456, 0.406), normalizedMat)
            Core.divide(normalizedMat, Scalar(0.229, 0.224, 0.225), normalizedMat)

            blob = Dnn.blobFromImage(
                normalizedMat,
                1.0,
                inputSize,
                Scalar(0.0, 0.0, 0.0),
                false,
                false
            )
            ageNet.setInput(blob)

            val outputNames = ageNet.getUnconnectedOutLayersNames()
            val outputs = mutableListOf<Mat>()
            ageNet.forward(outputs, outputNames)
            ageOutput = outputs[0]

            val ageScores = FloatArray(ageOutput.cols())
            ageOutput.get(0, 0, ageScores)
            val ageIndex = ageScores.indices.maxByOrNull { ageScores[it] } ?: return null
            val (ageLabel, estimatedAge) = ageBuckets[ageIndex]

            val genderResult = accurateGenderClassifier.classify(croppedBitmap)
            croppedBitmap.recycle()

            AgePrediction(
                ageLabel = ageLabel,
                estimatedAge = estimatedAge,
                ageConfidence = softmax(ageScores, ageIndex),
                genderLabel = genderResult.label,
                genderConfidence = genderResult.confidence
            )
        } finally {
            faceMat?.release()
            resizedMat?.release()
            rgbMat?.release()
            floatMat?.release()
            normalizedMat?.release()
            blob?.release()
            ageOutput?.release()
            srcMat.release()
            rgbaMat.release()
            bgrMat.release()
        }
    }

    private fun Rect.expand(imageWidth: Int, imageHeight: Int, ratio: Float): Rect? {
        val dx = (width() * ratio).toInt()
        val dy = (height() * ratio).toInt()
        val expanded = Rect(
            (left - dx).coerceAtLeast(0),
            (top - dy).coerceAtLeast(0),
            (right + dx).coerceAtMost(imageWidth),
            (bottom + dy).coerceAtMost(imageHeight)
        )
        return if (expanded.width() > 0 && expanded.height() > 0) expanded else null
    }

    private fun Rect.toSafeCvRect(maxWidth: Int, maxHeight: Int): CvRect? {
        val safeLeft = max(0, left)
        val safeTop = max(0, top)
        val safeRight = min(maxWidth, right)
        val safeBottom = min(maxHeight, bottom)
        val width = safeRight - safeLeft
        val height = safeBottom - safeTop
        if (width <= 0 || height <= 0) return null
        return CvRect(safeLeft, safeTop, width, height)
    }

    private fun softmax(values: FloatArray, targetIndex: Int): Float {
        val maxValue = values.maxOrNull() ?: return 0f
        val exponentials = values.map { kotlin.math.exp((it - maxValue).toDouble()) }
        val sum = exponentials.sum().takeIf { it > 0.0 } ?: return 0f
        return (exponentials[targetIndex] / sum).toFloat()
    }
}

private object OpenCvRuntime {
    @Volatile
    private var loaded = false

    fun ensureLoaded() {
        if (loaded) return

        synchronized(this) {
            if (loaded) return

            val loadedByJvm = runCatching {
                System.loadLibrary("opencv_java4")
                true
            }.getOrDefault(false)

            val loadedBySdk = if (loadedByJvm) true else OpenCVLoader.initLocal()
            check(loadedBySdk) { "OpenCV native runtime failed to load." }
            loaded = true
        }
    }
}
