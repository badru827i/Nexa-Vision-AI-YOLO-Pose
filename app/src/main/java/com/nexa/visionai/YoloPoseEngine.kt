package com.nexa.visionai

import android.content.Context
import android.graphics.Bitmap
import android.graphics.PointF
import android.graphics.RectF
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

class YoloPoseEngine(context: Context) : AutoCloseable {
    companion object {
        const val MODEL_FILE = "yolo11x-pose.tflite"
        const val INPUT_SIZE = 640
        const val KEYPOINTS = 17
        private const val CONF_THRESHOLD = 0.30f
        private const val IOU_THRESHOLD = 0.45f
    }

    data class PoseKeypoint(val point: PointF, val confidence: Float)
    data class PoseDetection(val box: RectF, val confidence: Float, val keypoints: List<PoseKeypoint>)

    private val interpreter: Interpreter
    private val input = ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * 3 * 4).order(ByteOrder.nativeOrder())
    private val outputShape: IntArray
    private val outputElements: Int
    private val output: Array<Array<FloatArray>>
    private val threads: Int

    init {
        val fd = context.assets.openFd(MODEL_FILE)
        FileInputStream(fd.fileDescriptor).use { stream ->
            val mapped = stream.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            threads = max(2, min(Runtime.getRuntime().availableProcessors(), 6))
            interpreter = Interpreter(mapped, Interpreter.Options().apply { setNumThreads(threads) })
        }
        fd.close()
        outputShape = interpreter.getOutputTensor(0).shape()
        require(outputShape.size == 3 && outputShape[0] == 1) {
            "Expected YOLO pose output [1,C,N], got ${outputShape.contentToString()}"
        }
        val c = outputShape[1]
        val n = outputShape[2]
        outputElements = c * n
        output = Array(1) { Array(c) { FloatArray(n) } }
        require(c >= 56) { "Expected at least 56 output channels, got $c" }
    }

    fun detect(source: Bitmap): List<PoseDetection> {
        val scale = min(INPUT_SIZE.toFloat() / source.width, INPUT_SIZE.toFloat() / source.height)
        val resizedW = (source.width * scale).toInt().coerceAtLeast(1)
        val resizedH = (source.height * scale).toInt().coerceAtLeast(1)
        val padX = (INPUT_SIZE - resizedW) / 2f
        val padY = (INPUT_SIZE - resizedH) / 2f

        val letterbox = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
        val scaled = Bitmap.createScaledBitmap(source, resizedW, resizedH, true)
        val canvas = android.graphics.Canvas(letterbox)
        canvas.drawColor(android.graphics.Color.rgb(114, 114, 114))
        canvas.drawBitmap(scaled, padX, padY, android.graphics.Paint(android.graphics.Paint.FILTER_BITMAP_FLAG))
        scaled.recycle()

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        letterbox.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        letterbox.recycle()

        input.rewind()
        for (p in pixels) {
            input.putFloat(((p ushr 16) and 255) / 255f)
            input.putFloat(((p ushr 8) and 255) / 255f)
            input.putFloat((p and 255) / 255f)
        }
        input.rewind()

        interpreter.run(input, output)
        val channels = outputShape[1]
        val count = outputShape[2]
        val detections = ArrayList<PoseDetection>()

        fun at(candidate: Int, channel: Int): Float = output[0][channel][candidate]

        for (i in 0 until count) {
            val confidence = at(i, 4)
            if (!confidence.isFinite() || confidence < CONF_THRESHOLD) continue

            val cx = at(i, 0)
            val cy = at(i, 1)
            val w = at(i, 2)
            val h = at(i, 3)
            if (!cx.isFinite() || !cy.isFinite() || !w.isFinite() || !h.isFinite()) continue

            val left = ((cx - w / 2f) - padX) / scale
            val top = ((cy - h / 2f) - padY) / scale
            val right = ((cx + w / 2f) - padX) / scale
            val bottom = ((cy + h / 2f) - padY) / scale
            val box = RectF(
                left.coerceIn(0f, source.width.toFloat()),
                top.coerceIn(0f, source.height.toFloat()),
                right.coerceIn(0f, source.width.toFloat()),
                bottom.coerceIn(0f, source.height.toFloat())
            )
            if (box.width() < 4f || box.height() < 4f) continue

            val points = ArrayList<PoseKeypoint>(KEYPOINTS)
            for (k in 0 until KEYPOINTS) {
                val base = 5 + k * 3
                if (base + 2 >= channels) break
                val x = ((at(i, base) - padX) / scale).coerceIn(0f, source.width.toFloat())
                val y = ((at(i, base + 1) - padY) / scale).coerceIn(0f, source.height.toFloat())
                val kp = at(i, base + 2).coerceIn(0f, 1f)
                points.add(PoseKeypoint(PointF(x, y), kp))
            }
            if (points.size == KEYPOINTS) detections.add(PoseDetection(box, confidence.coerceIn(0f, 1f), points))
        }
        return nms(detections).take(12)
    }

    fun workerThreads(): Int = threads

    private fun nms(items: List<PoseDetection>): List<PoseDetection> {
        val sorted = items.sortedByDescending { it.confidence }.toMutableList()
        val kept = ArrayList<PoseDetection>()
        while (sorted.isNotEmpty()) {
            val best = sorted.removeAt(0)
            kept.add(best)
            sorted.removeAll { iou(best.box, it.box) > IOU_THRESHOLD }
        }
        return kept
    }

    private fun iou(a: RectF, b: RectF): Float {
        val l = max(a.left, b.left)
        val t = max(a.top, b.top)
        val r = min(a.right, b.right)
        val bo = min(a.bottom, b.bottom)
        val inter = max(0f, r - l) * max(0f, bo - t)
        val union = a.width() * a.height() + b.width() * b.height() - inter
        return if (union <= 0f) 0f else inter / union
    }

    override fun close() = interpreter.close()
}
