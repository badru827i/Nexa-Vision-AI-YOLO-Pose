package com.nexa.visionai

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import kotlin.math.max

class PoseOverlay @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {
    private val linePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 5f
        strokeCap = Paint.Cap.ROUND
        color = Color.GREEN
    }
    private val pointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.YELLOW
    }
    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f
        color = Color.CYAN
    }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textSize = 34f
        typeface = android.graphics.Typeface.DEFAULT_BOLD
        color = Color.WHITE
        setShadowLayer(6f, 0f, 2f, Color.BLACK)
    }
    private var detections: List<YoloPoseEngine.PoseDetection> = emptyList()
    private val edges = arrayOf(
        intArrayOf(0,1), intArrayOf(0,2), intArrayOf(1,3), intArrayOf(2,4),
        intArrayOf(5,6), intArrayOf(5,7), intArrayOf(7,9), intArrayOf(6,8),
        intArrayOf(8,10), intArrayOf(5,11), intArrayOf(6,12), intArrayOf(11,12),
        intArrayOf(11,13), intArrayOf(13,15), intArrayOf(12,14), intArrayOf(14,16)
    )

    fun update(value: List<YoloPoseEngine.PoseDetection>) {
        detections = value
        postInvalidateOnAnimation()
    }

    fun clear() {
        detections = emptyList()
        postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        for (detection in detections) {
            val box = detection.box
            boxPaint.alpha = 235
            canvas.drawRect(box, boxPaint)
            textPaint.alpha = 255
            canvas.drawText(
                "PERSON ${(detection.confidence * 100).toInt()}%",
                box.left,
                max(40f, box.top - 12f),
                textPaint
            )

            val p = detection.keypoints
            for (edge in edges) {
                if (edge[0] >= p.size || edge[1] >= p.size) continue
                val a = p[edge[0]]
                val b = p[edge[1]]
                if (a.confidence < 0.25f || b.confidence < 0.25f) continue
                linePaint.alpha = 255
                canvas.drawLine(a.point.x, a.point.y, b.point.x, b.point.y, linePaint)
            }
            for (kp in p) {
                if (kp.confidence < 0.25f) continue
                pointPaint.alpha = 255
                canvas.drawCircle(kp.point.x, kp.point.y, 8f, pointPaint)
            }
        }
    }
}
