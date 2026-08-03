package com.anzhuoface.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.util.AttributeSet
import android.view.View
import kotlin.math.max

class FaceOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#22C55E")
        style = Paint.Style.STROKE
        strokeWidth = 6f
    }

    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 38f
    }

    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#AA166534")
        style = Paint.Style.FILL
    }

    private var frameWidth = 0
    private var frameHeight = 0
    private var items: List<OverlayItem> = emptyList()

    fun update(frameWidth: Int, frameHeight: Int, items: List<OverlayItem>) {
        this.frameWidth = frameWidth
        this.frameHeight = frameHeight
        this.items = items
        invalidate()
    }

    fun clear() {
        frameWidth = 0
        frameHeight = 0
        items = emptyList()
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (frameWidth <= 0 || frameHeight <= 0 || items.isEmpty()) return

        val scaleX = width / frameWidth.toFloat()
        val scaleY = height / frameHeight.toFloat()

        items.forEach { item ->
            val left = item.rect.left * scaleX
            val top = item.rect.top * scaleY
            val right = item.rect.right * scaleX
            val bottom = item.rect.bottom * scaleY

            canvas.drawRect(left, top, right, bottom, boxPaint)

            val textWidth = textPaint.measureText(item.label)
            val labelTop = max(48f, top - 14f)
            canvas.drawRect(
                left - 10f,
                labelTop - 42f,
                left + textWidth + 10f,
                labelTop + 8f,
                labelPaint
            )
            canvas.drawText(item.label, left, labelTop, textPaint)
        }
    }
}

data class OverlayItem(
    val rect: Rect,
    val label: String
)
