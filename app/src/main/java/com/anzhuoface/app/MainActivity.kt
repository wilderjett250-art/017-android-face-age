package com.anzhuoface.app

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts.StartActivityForResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.setPadding
import androidx.lifecycle.lifecycleScope
import com.anzhuoface.app.databinding.ActivityMainBinding
import com.google.android.material.card.MaterialCardView
import com.google.mlkit.vision.face.Face
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "MainActivity"
    }

    private lateinit var binding: ActivityMainBinding
    private lateinit var analyzer: FaceAgeAnalyzer
    private lateinit var historyRepository: HistoryRepository
    private var activeProfile: ModelProfile = ModelProfiles.thesis

    private val pickImageLauncher =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
            if (uri != null) {
                loadAndAnalyze(uri, getString(R.string.source_gallery))
            }
        }

    private val cameraPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                openRealtimeCamera()
            } else {
                showMessage(getString(R.string.camera_permission_denied))
            }
        }

    private val realtimeCameraLauncher =
        registerForActivityResult(StartActivityForResult()) { result ->
            if (result.resultCode != RESULT_OK) return@registerForActivityResult

            val path = result.data?.getStringExtra(CameraActivity.EXTRA_IMAGE_PATH) ?: return@registerForActivityResult
            lifecycleScope.launch {
                val bitmap = withContext(Dispatchers.IO) {
                    BitmapFactory.decodeFile(path)
                }
                if (bitmap != null) {
                    analyzeBitmap(bitmap, getString(R.string.source_camera_realtime))
                } else {
                    showMessage(getString(R.string.load_failed))
                }
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        analyzer = FaceAgeAnalyzer(applicationContext)
        historyRepository = HistoryRepository(applicationContext)

        binding.standardModelChip.setOnClickListener { selectModel(ModelProfiles.standard) }
        binding.liteModelChip.setOnClickListener { selectModel(ModelProfiles.lite) }
        binding.thesisModelChip.setOnClickListener { selectModel(ModelProfiles.thesis) }

        binding.pickImageButton.setOnClickListener { pickImageLauncher.launch("image/*") }
        binding.cameraButton.text = getString(R.string.open_realtime_camera)
        binding.cameraButton.setOnClickListener { openRealtimeCameraWithPermission() }
        binding.sampleButton.setOnClickListener {
            lifecycleScope.launch {
                val bitmap = withContext(Dispatchers.IO) {
                    BitmapFactory.decodeStream(assets.open("sample_face.jpg"))
                }
                if (bitmap != null) {
                    analyzeBitmap(bitmap, getString(R.string.source_sample))
                } else {
                    showMessage(getString(R.string.load_failed))
                }
            }
        }
        binding.clearHistoryButton.setOnClickListener {
            historyRepository.clear()
            renderHistory()
            binding.tipText.text = getString(R.string.history_cleared)
        }

        binding.metricsText.text = getString(R.string.metrics_placeholder)
        binding.pipelineText.text = getString(R.string.pipeline_placeholder)
        selectModel(ModelProfiles.thesis)
        renderHistory()
    }

    private fun openRealtimeCameraWithPermission() {
        val granted = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        if (granted) {
            openRealtimeCamera()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun openRealtimeCamera() {
        val intent = Intent(this, CameraActivity::class.java)
            .putExtra(CameraActivity.EXTRA_MODEL_KEY, activeProfile.key)
        realtimeCameraLauncher.launch(intent)
    }

    private fun selectModel(profile: ModelProfile) {
        activeProfile = profile
        binding.standardModelChip.isChecked = profile.key == ModelProfiles.standard.key
        binding.liteModelChip.isChecked = profile.key == ModelProfiles.lite.key
        binding.thesisModelChip.isChecked = profile.key == ModelProfiles.thesis.key
        binding.modelSummaryText.text = buildModelSummary(profile)
        binding.metricsText.text = getString(R.string.metrics_placeholder_model, profile.displayName)
        binding.pipelineText.text = getString(R.string.pipeline_placeholder_model, profile.displayName, profile.inputSize)
    }

    private fun buildModelSummary(profile: ModelProfile): String {
        return getString(
            R.string.model_summary_template,
            profile.displayName,
            profile.backbone,
            profile.trainingDataset,
            profile.inputSize,
            profile.trainingStrategy,
            profile.ageTop1,
            profile.ageOneOff,
            profile.ageMaeProxy,
            profile.cs5Proxy,
            profile.modelSize,
            profile.lightweightStrategy
        )
    }

    private fun loadAndAnalyze(uri: Uri, source: String) {
        lifecycleScope.launch {
            val bitmap = withContext(Dispatchers.IO) {
                contentResolver.openInputStream(uri)?.use(BitmapFactory::decodeStream)
            }
            if (bitmap != null) {
                analyzeBitmap(bitmap, source)
            } else {
                showMessage(getString(R.string.load_failed))
            }
        }
    }

    private suspend fun analyzeBitmap(bitmap: Bitmap, source: String) {
        val profile = activeProfile
        setLoading(true)
        binding.previewImage.setImageBitmap(bitmap)
        binding.resultText.text = getString(R.string.analyzing)
        binding.tipText.text = getString(R.string.tip_detecting, source, profile.displayName)
        binding.metricsText.text = getString(R.string.metrics_running, profile.displayName)
        binding.pipelineText.text = buildPipelineText(profile, source, bitmap.width, bitmap.height, false)

        val startTime = System.currentTimeMillis()
        runCatching {
            withContext(Dispatchers.Default) {
                analyzer.analyze(bitmap, profile)
            }
        }.onSuccess { results ->
            val duration = System.currentTimeMillis() - startTime
            val success = results.isNotEmpty()
            Log.d(TAG, "Analysis finished: model=${profile.key}, source=$source, faces=${results.size}, durationMs=$duration")
            binding.metricsText.text = getString(
                R.string.metrics_template,
                profile.displayName,
                source,
                bitmap.width,
                bitmap.height,
                results.size,
                duration
            )
            binding.pipelineText.text = buildPipelineText(profile, source, bitmap.width, bitmap.height, true)

            val summary = if (success) buildHistorySummary(results) else getString(R.string.no_face_found)
            historyRepository.save(
                HistoryEntry(
                    timestamp = System.currentTimeMillis(),
                    source = source,
                    modelName = profile.displayName,
                    faceCount = results.size,
                    summary = summary,
                    durationMs = duration,
                    imageWidth = bitmap.width,
                    imageHeight = bitmap.height,
                    success = success
                )
            )

            if (success) {
                binding.previewImage.setImageBitmap(drawResults(bitmap, results))
                binding.resultText.text = buildResultText(profile, results)
                binding.tipText.text = getString(R.string.tip_success, profile.displayName)
            } else {
                binding.resultText.text = getString(R.string.no_face_found)
                binding.tipText.text = getString(R.string.tip_retry)
                binding.previewImage.setImageBitmap(bitmap)
            }

            renderHistory()
        }.onFailure { error ->
            Log.e(TAG, "Analysis failed for model=${profile.key}, source=$source", error)
            binding.resultText.text = getString(R.string.analyze_failed, error.message ?: "unknown")
            binding.tipText.text = getString(R.string.tip_retry)
            binding.metricsText.text = getString(R.string.metrics_failed, profile.displayName)
            binding.pipelineText.text = buildPipelineText(profile, source, bitmap.width, bitmap.height, false)
            historyRepository.save(
                HistoryEntry(
                    timestamp = System.currentTimeMillis(),
                    source = source,
                    modelName = profile.displayName,
                    faceCount = 0,
                    summary = getString(R.string.history_error_summary),
                    durationMs = System.currentTimeMillis() - startTime,
                    imageWidth = bitmap.width,
                    imageHeight = bitmap.height,
                    success = false
                )
            )
            renderHistory()
        }

        setLoading(false)
    }

    private fun buildPipelineText(profile: ModelProfile, source: String, width: Int, height: Int, finished: Boolean): String {
        val state = if (finished) {
            getString(R.string.pipeline_state_finished)
        } else {
            getString(R.string.pipeline_state_running)
        }
        return getString(R.string.pipeline_template, state, profile.displayName, source, width, height, profile.inputSize)
    }

    private fun drawResults(bitmap: Bitmap, results: List<Pair<Face, AgePrediction>>): Bitmap {
        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)
        val strokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#00C853")
            style = Paint.Style.STROKE
            strokeWidth = 6f
        }
        val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textSize = 42f
        }
        val textBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#AA1B5E20")
            style = Paint.Style.FILL
        }

        results.forEach { (face, prediction) ->
            val box = face.boundingBox
            canvas.drawRect(box, strokePaint)
            val label = "${prediction.genderLabel} ${prediction.ageLabel}"
            val textWidth = textPaint.measureText(label)
            val textLeft = box.left.toFloat()
            val textTop = if (box.top > 56) box.top - 12f else box.bottom + 52f
            canvas.drawRect(
                textLeft - 12f,
                textTop - 44f,
                textLeft + textWidth + 12f,
                textTop + 10f,
                textBgPaint
            )
            canvas.drawText(label, textLeft, textTop, textPaint)
        }
        return mutableBitmap
    }

    private fun buildResultText(profile: ModelProfile, results: List<Pair<Face, AgePrediction>>): String {
        return buildString {
            append("${profile.displayName} 检测到 ${results.size} 张人脸\n\n")
            results.forEachIndexed { index, (_, prediction) ->
                append(
                    "人脸 ${index + 1}：${prediction.genderLabel}，年龄段 ${prediction.ageLabel}，约 ${prediction.estimatedAge} 岁，" +
                        "年龄置信度 ${(prediction.ageConfidence * 100).toInt()}%，性别置信度 ${(prediction.genderConfidence * 100).toInt()}%\n"
                )
            }
        }
    }

    private fun buildHistorySummary(results: List<Pair<Face, AgePrediction>>): String {
        return results.joinToString("；") { (_, prediction) ->
            "${prediction.genderLabel}/${prediction.ageLabel}"
        }
    }

    private fun renderHistory() {
        val history = historyRepository.load()
        binding.historyContainer.removeAllViews()
        binding.emptyHistoryText.visibility = if (history.isEmpty()) View.VISIBLE else View.GONE
        binding.testingSummaryText.text = buildTestingSummary(history)

        history.forEach { entry ->
            binding.historyContainer.addView(createHistoryCard(entry))
        }
    }

    private fun buildTestingSummary(history: List<HistoryEntry>): String {
        if (history.isEmpty()) return getString(R.string.testing_summary_empty)

        val successRuns = history.count { it.success }
        val avgDuration = history.map { it.durationMs }.average().toLong()
        val avgFaces = history.map { it.faceCount }.average()
        return getString(
            R.string.testing_summary_template,
            history.size,
            successRuns,
            avgDuration,
            String.format(Locale.getDefault(), "%.1f", avgFaces)
        )
    }

    private fun createHistoryCard(entry: HistoryEntry): View {
        val timeText = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault()).format(Date(entry.timestamp))

        val card = MaterialCardView(this).apply {
            radius = 8f
            setCardBackgroundColor(Color.WHITE)
            strokeColor = Color.parseColor("#D0D7DE")
            strokeWidth = 1
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                bottomMargin = 12.dp()
            }
        }

        val content = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(16.dp())
        }

        val title = TextView(this).apply {
            text = "$timeText  |  ${entry.source}  |  ${entry.modelName}"
            textSize = 14f
            setTextColor(Color.parseColor("#1F2328"))
        }
        val subtitle = TextView(this).apply {
            text = "人脸数：${entry.faceCount}    耗时：${entry.durationMs} ms    结果：${entry.summary}"
            textSize = 13f
            setTextColor(Color.parseColor("#57606A"))
            setPadding(0, 8.dp(), 0, 0)
        }
        val detail = TextView(this).apply {
            text = "输入尺寸：${entry.imageWidth}x${entry.imageHeight}    状态：${if (entry.success) "完成" else "异常"}"
            textSize = 12f
            setTextColor(Color.parseColor("#6E7781"))
            setPadding(0, 8.dp(), 0, 0)
        }

        content.addView(title)
        content.addView(subtitle)
        content.addView(detail)
        card.addView(content)
        return card
    }

    private fun Int.dp(): Int = (this * resources.displayMetrics.density).toInt()

    private fun showMessage(message: String) {
        binding.resultText.text = message
        binding.tipText.text = getString(R.string.tip_idle)
    }

    private fun setLoading(loading: Boolean) {
        binding.progressBar.visibility = if (loading) View.VISIBLE else View.GONE
        binding.pickImageButton.isEnabled = !loading
        binding.cameraButton.isEnabled = !loading
        binding.sampleButton.isEnabled = !loading
        binding.clearHistoryButton.isEnabled = !loading
        binding.standardModelChip.isEnabled = !loading
        binding.liteModelChip.isEnabled = !loading
        binding.thesisModelChip.isEnabled = !loading
    }
}
