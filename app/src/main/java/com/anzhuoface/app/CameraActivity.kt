package com.anzhuoface.app

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Rect
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.anzhuoface.app.databinding.ActivityCameraBinding
import com.google.mlkit.vision.face.Face
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.util.ArrayDeque
import kotlin.math.abs
import kotlin.math.max

class CameraActivity : AppCompatActivity() {

    companion object {
        const val EXTRA_MODEL_KEY = "model_key"
        const val EXTRA_IMAGE_PATH = "image_path"
    }

    private lateinit var binding: ActivityCameraBinding
    private lateinit var analyzer: FaceAgeAnalyzer
    private lateinit var profile: ModelProfile

    private var samplingJob: Job? = null
    private var analysisRunning = false
    private val recentSamples = ArrayDeque<SampledFrame>()
    private var bestSample: SampledFrame? = null
    private var latestLiveSample: SampledFrame? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityCameraBinding.inflate(layoutInflater)
        setContentView(binding.root)

        analyzer = FaceAgeAnalyzer(applicationContext)
        profile = resolveProfile(intent.getStringExtra(EXTRA_MODEL_KEY))

        binding.modelBadgeText.text = getString(R.string.realtime_model_badge, profile.displayName)
        binding.currentFrameText.text = getString(R.string.realtime_sampling_running)
        binding.bestFrameText.text = getString(R.string.realtime_no_best_frame)

        binding.closeCameraButton.setOnClickListener { finish() }
        binding.useBestFrameButton.setOnClickListener { exportBestFrame() }

        startCameraPreview()
    }

    override fun onStart() {
        super.onStart()
        startSamplingLoop()
    }

    override fun onStop() {
        samplingJob?.cancel()
        super.onStop()
    }

    override fun onDestroy() {
        samplingJob?.cancel()
        clearSamples()
        super.onDestroy()
    }

    private fun startCameraPreview() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.previewView.surfaceProvider)
            }
            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_FRONT_CAMERA, preview)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun startSamplingLoop() {
        samplingJob?.cancel()
        samplingJob = lifecycleScope.launch {
            while (isActive) {
                if (!analysisRunning) {
                    val previewBitmap = binding.previewView.bitmap?.copy(Bitmap.Config.ARGB_8888, false)
                    if (previewBitmap != null) {
                        analyzePreviewFrame(previewBitmap)
                    } else {
                        binding.samplingStatusText.text = getString(R.string.realtime_sampling_waiting)
                    }
                }
                delay(500)
            }
        }
    }

    private fun analyzePreviewFrame(bitmap: Bitmap) {
        analysisRunning = true
        binding.samplingStatusText.text = getString(R.string.realtime_sampling_running)

        lifecycleScope.launch {
            val results = withContext(Dispatchers.Default) {
                analyzer.analyze(bitmap, profile)
            }
            handleFrameResult(bitmap, results)
            analysisRunning = false
        }
    }

    private fun handleFrameResult(bitmap: Bitmap, results: List<Pair<Face, AgePrediction>>) {
        if (results.isEmpty()) {
            bitmap.recycle()
            latestLiveSample = null
            binding.overlayView.clear()
            binding.currentFrameText.text = getString(R.string.realtime_sampling_no_face)
            renderStableResult()
            return
        }

        val filteredResults = results
            .sortedByDescending { it.first.boundingBox.width() * it.first.boundingBox.height() }
            .filter { isFaceLargeEnough(bitmap, it.first.boundingBox) }

        if (filteredResults.isEmpty()) {
            bitmap.recycle()
            latestLiveSample = null
            binding.overlayView.clear()
            binding.currentFrameText.text = getString(R.string.realtime_small_face)
            renderStableResult()
            return
        }

        val primary = filteredResults.first()
        val score = computeFrameScore(bitmap, primary.first.boundingBox, primary.second)
        val sample = SampledFrame(
            bitmap = bitmap,
            frameWidth = bitmap.width,
            frameHeight = bitmap.height,
            results = filteredResults.map { FramePrediction(it.first.boundingBox, it.second) },
            primaryPrediction = primary.second,
            score = score
        )

        latestLiveSample = sample
        recentSamples.addLast(sample)
        while (recentSamples.size > 6) {
            val removed = recentSamples.removeFirst()
            if (removed !== bestSample && removed !== latestLiveSample && !removed.bitmap.isRecycled) {
                removed.bitmap.recycle()
            }
        }

        bestSample = recentSamples.maxByOrNull { it.score }

        binding.currentFrameText.text = when {
            filteredResults.size > 1 -> getString(R.string.realtime_multiple_faces)
            else -> getString(
                R.string.realtime_current_face_result,
                "${primary.second.genderLabel}/${primary.second.ageLabel}",
                primary.second.estimatedAge,
                (primary.second.ageConfidence * 100).toInt(),
                (primary.second.genderConfidence * 100).toInt()
            )
        }

        renderLiveOverlay()
        renderStableResult()
    }

    private fun renderLiveOverlay() {
        val liveSample = latestLiveSample
        if (liveSample == null) {
            binding.overlayView.clear()
            return
        }

        binding.overlayView.update(
            liveSample.frameWidth,
            liveSample.frameHeight,
            liveSample.results.map {
                OverlayItem(
                    rect = it.rect,
                    label = "${it.prediction.genderLabel} ${it.prediction.ageLabel}"
                )
            }
        )
    }

    private fun renderStableResult() {
        val currentBest = bestSample
        if (currentBest == null || recentSamples.isEmpty()) {
            binding.bestFrameText.text = getString(R.string.realtime_no_best_frame)
            binding.useBestFrameButton.isEnabled = false
            return
        }

        val stable = buildStablePrediction(recentSamples.toList())
        binding.bestFrameText.text = getString(
            R.string.realtime_stable_result,
            "${stable.genderLabel}/${stable.ageLabel}",
            stable.estimatedAge,
            (stable.ageConfidence * 100).toInt(),
            recentSamples.size
        )
        binding.useBestFrameButton.isEnabled = true
    }

    private fun buildStablePrediction(samples: List<SampledFrame>): AgePrediction {
        val genderWeights = linkedMapOf("男" to 0f, "女" to 0f)
        val ageWeights = linkedMapOf<String, Float>()
        var weightedAge = 0f
        var ageConfidenceSum = 0f
        var genderConfidenceSum = 0f
        var totalWeight = 0f

        samples.forEach { sample ->
            val weight = sample.score.coerceAtLeast(0.01f)
            val prediction = sample.primaryPrediction
            genderWeights[prediction.genderLabel] = (genderWeights[prediction.genderLabel] ?: 0f) + weight
            ageWeights[prediction.ageLabel] = (ageWeights[prediction.ageLabel] ?: 0f) + weight
            weightedAge += prediction.estimatedAge * weight
            ageConfidenceSum += prediction.ageConfidence * weight
            genderConfidenceSum += prediction.genderConfidence * weight
            totalWeight += weight
        }

        val stableGender = genderWeights.maxByOrNull { it.value }?.key ?: samples.last().primaryPrediction.genderLabel
        val stableAgeLabel = ageWeights.maxByOrNull { it.value }?.key ?: samples.last().primaryPrediction.ageLabel
        val stableAge = if (totalWeight > 0f) (weightedAge / totalWeight).toInt() else samples.last().primaryPrediction.estimatedAge

        return AgePrediction(
            ageLabel = stableAgeLabel,
            estimatedAge = stableAge,
            ageConfidence = if (totalWeight > 0f) ageConfidenceSum / totalWeight else samples.last().primaryPrediction.ageConfidence,
            genderLabel = stableGender,
            genderConfidence = if (totalWeight > 0f) genderConfidenceSum / totalWeight else samples.last().primaryPrediction.genderConfidence
        )
    }

    private fun isFaceLargeEnough(bitmap: Bitmap, rect: Rect): Boolean {
        val frameArea = max(1, bitmap.width * bitmap.height).toFloat()
        val faceRatio = (rect.width() * rect.height()) / frameArea
        return faceRatio >= 0.06f
    }

    private fun computeFrameScore(bitmap: Bitmap, rect: Rect, prediction: AgePrediction): Float {
        val frameArea = max(1, bitmap.width * bitmap.height).toFloat()
        val areaScore = ((rect.width() * rect.height()) / frameArea / 0.35f).coerceIn(0f, 1f)
        val sharpnessScore = estimateSharpness(bitmap, rect).coerceIn(0f, 1f)
        return (
            prediction.ageConfidence * 0.45f +
                prediction.genderConfidence * 0.10f +
                areaScore * 0.20f +
                sharpnessScore * 0.25f
            ).coerceIn(0f, 1f)
    }

    private fun estimateSharpness(bitmap: Bitmap, rect: Rect): Float {
        val safeLeft = rect.left.coerceIn(0, bitmap.width - 1)
        val safeTop = rect.top.coerceIn(0, bitmap.height - 1)
        val safeRight = rect.right.coerceIn(safeLeft + 1, bitmap.width)
        val safeBottom = rect.bottom.coerceIn(safeTop + 1, bitmap.height)
        val step = 4
        var gradientSum = 0.0
        var count = 0

        for (y in safeTop + step until safeBottom step step) {
            for (x in safeLeft + step until safeRight step step) {
                val center = bitmap.getPixel(x, y).luma()
                val left = bitmap.getPixel(x - step, y).luma()
                val up = bitmap.getPixel(x, y - step).luma()
                gradientSum += abs(center - left) + abs(center - up)
                count += 2
            }
        }

        if (count == 0) return 0f
        return (gradientSum / count / 48.0).toFloat()
    }

    private fun exportBestFrame() {
        val sample = bestSample ?: run {
            binding.samplingStatusText.text = getString(R.string.realtime_no_best_frame)
            return
        }

        val outFile = File(cacheDir, "realtime_capture.jpg")
        FileOutputStream(outFile).use { stream ->
            sample.bitmap.compress(Bitmap.CompressFormat.JPEG, 95, stream)
        }

        setResult(
            RESULT_OK,
            Intent().putExtra(EXTRA_IMAGE_PATH, outFile.absolutePath)
        )
        finish()
    }

    private fun clearSamples() {
        val best = bestSample
        val live = latestLiveSample
        recentSamples.forEach { sample ->
            if (sample !== best && sample !== live && !sample.bitmap.isRecycled) {
                sample.bitmap.recycle()
            }
        }
        if (best != null && !best.bitmap.isRecycled) {
            best.bitmap.recycle()
        }
        if (live != null && live !== best && !live.bitmap.isRecycled) {
            live.bitmap.recycle()
        }
        recentSamples.clear()
        bestSample = null
        latestLiveSample = null
    }

    private fun resolveProfile(key: String?): ModelProfile {
        return ModelProfiles.fromKey(key)
    }

    private fun Int.luma(): Double {
        val r = (this shr 16) and 0xFF
        val g = (this shr 8) and 0xFF
        val b = this and 0xFF
        return 0.299 * r + 0.587 * g + 0.114 * b
    }
}

private data class SampledFrame(
    val bitmap: Bitmap,
    val frameWidth: Int,
    val frameHeight: Int,
    val results: List<FramePrediction>,
    val primaryPrediction: AgePrediction,
    val score: Float
)

private data class FramePrediction(
    val rect: Rect,
    val prediction: AgePrediction
)
