package com.anzhuoface.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import java.nio.FloatBuffer

class AccurateGenderClassifier(context: Context) {
    private val environment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        context.assets.open("gender_classifier.onnx").use { stream ->
            session = environment.createSession(stream.readBytes())
        }
    }

    fun classify(bitmap: Bitmap): GenderResult {
        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        val inputBuffer = FloatBuffer.allocate(1 * 3 * INPUT_SIZE * INPUT_SIZE)
        val red = FloatArray(INPUT_SIZE * INPUT_SIZE)
        val green = FloatArray(INPUT_SIZE * INPUT_SIZE)
        val blue = FloatArray(INPUT_SIZE * INPUT_SIZE)

        for (y in 0 until INPUT_SIZE) {
            for (x in 0 until INPUT_SIZE) {
                val pixel = resized.getPixel(x, y)
                val index = y * INPUT_SIZE + x
                red[index] = ((((pixel shr 16) and 0xFF) / 255f) - MEAN[0]) / STD[0]
                green[index] = ((((pixel shr 8) and 0xFF) / 255f) - MEAN[1]) / STD[1]
                blue[index] = (((pixel and 0xFF) / 255f) - MEAN[2]) / STD[2]
            }
        }

        inputBuffer.put(red)
        inputBuffer.put(green)
        inputBuffer.put(blue)
        inputBuffer.rewind()

        val tensor = OnnxTensor.createTensor(
            environment,
            inputBuffer,
            longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong()),
        )

        tensor.use {
            session.run(mapOf("input" to tensor)).use { output ->
                val logits = (output[0].value as Array<FloatArray>)[0]
                val femaleLogit = logits[0]
                val maleLogit = logits[1]
                val maleProbability = sigmoid(maleLogit - femaleLogit)
                val label = if (maleProbability >= 0.5f) "男" else "女"
                val confidence = if (maleProbability >= 0.5f) maleProbability else 1f - maleProbability
                return GenderResult(label, confidence)
            }
        }
    }

    data class GenderResult(
        val label: String,
        val confidence: Float,
    )

    private fun sigmoid(value: Float): Float {
        return (1.0 / (1.0 + kotlin.math.exp((-value).toDouble()))).toFloat()
    }

    companion object {
        private const val INPUT_SIZE = 224
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }
}
