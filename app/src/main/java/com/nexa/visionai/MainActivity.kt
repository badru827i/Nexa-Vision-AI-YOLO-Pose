package com.nexa.visionai

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.os.SystemClock
import android.view.Gravity
import android.view.View
import android.widget.FrameLayout
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.nio.ByteBuffer
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.max

class MainActivity : ComponentActivity() {
    private lateinit var preview: PreviewView
    private lateinit var overlay: PoseOverlay
    private lateinit var status: TextView
    private val analysisExecutor = Executors.newSingleThreadExecutor()
    private val busy = AtomicBoolean(false)
    private var engine: YoloPoseEngine? = null
    private var lastFpsUpdate = SystemClock.elapsedRealtime()
    private var frames = 0
    private var modelReady = false

    private val cameraPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startCamera() else showStatus("CAMERA PERMISSION REQUIRED")
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.decorView.systemUiVisibility = View.SYSTEM_UI_FLAG_FULLSCREEN or View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
        buildUi()
        loadModel()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            cameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun buildUi() {
        val root = FrameLayout(this).apply { setBackgroundColor(android.graphics.Color.BLACK) }
        preview = PreviewView(this).apply {
            scaleType = PreviewView.ScaleType.FILL_CENTER
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }
        overlay = PoseOverlay(this)
        status = TextView(this).apply {
            text = "NEXA VISION AI • INITIALIZING"
            textSize = 15f
            setTextColor(android.graphics.Color.WHITE)
            setShadowLayer(6f, 0f, 2f, android.graphics.Color.BLACK)
            setPadding(20, 20, 20, 20)
        }
        root.addView(preview, FrameLayout.LayoutParams(-1, -1))
        root.addView(overlay, FrameLayout.LayoutParams(-1, -1))
        root.addView(status, FrameLayout.LayoutParams(-1, -2).apply {
            gravity = Gravity.TOP or Gravity.CENTER_HORIZONTAL
            topMargin = 22
        })
        setContentView(root)
    }

    private fun loadModel() {
        analysisExecutor.execute {
            runCatching { YoloPoseEngine(applicationContext) }.onSuccess { loaded ->
                engine = loaded
                modelReady = true
                runOnUiThread { showStatus("YOLO11x-POSE READY • 640×640 • FP32") }
            }.onFailure { error ->
                modelReady = false
                runOnUiThread { showStatus("MODEL ERROR: ${error.message ?: "yolo11x-pose.tflite missing"}") }
            }
        }
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            try {
                val provider = future.get()
                val previewUseCase = Preview.Builder().build().also { it.surfaceProvider = preview.surfaceProvider }
                val analysis = ImageAnalysis.Builder()
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setTargetResolution(android.util.Size(1280, 720))
                    .build()

                analysis.setAnalyzer(analysisExecutor) { image -> analyze(image) }
                provider.unbindAll()
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, previewUseCase, analysis)
            } catch (t: Throwable) {
                showStatus("CAMERA ERROR: ${t.javaClass.simpleName}")
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyze(image: ImageProxy) {
        if (!modelReady || !busy.compareAndSet(false, true)) {
            image.close()
            return
        }
        try {
            val bitmap = imageToBitmap(image)
            val rotation = image.imageInfo.rotationDegrees
            val rotated = rotate(bitmap, rotation)
            if (rotated !== bitmap) bitmap.recycle()
            val source = rotated
            val detections = engine?.detect(source).orEmpty()
            val mapped = detections.map { d -> mapToPreview(d, source.width, source.height) }
            source.recycle()
            frames++
            val now = SystemClock.elapsedRealtime()
            if (now - lastFpsUpdate >= 1000L) {
                val fps = frames * 1000f / max(1L, now - lastFpsUpdate)
                frames = 0
                lastFpsUpdate = now
                runOnUiThread { showStatus("YOLO11x-POSE • ${fps.toInt()} FPS • ${mapped.size} PERSON • 17 KP") }
            }
            runOnUiThread { overlay.update(mapped) }
        } catch (t: Throwable) {
            runOnUiThread {
                showStatus("AI ERROR: ${t.javaClass.simpleName}: ${t.message ?: "unknown"}")
            }
        } finally {
            busy.set(false)
            image.close()
        }
    }

    private fun mapToPreview(detection: YoloPoseEngine.PoseDetection, imageW: Int, imageH: Int): YoloPoseEngine.PoseDetection {
        val vw = preview.width.toFloat().coerceAtLeast(1f)
        val vh = preview.height.toFloat().coerceAtLeast(1f)
        val scale = max(vw / imageW, vh / imageH)
        val dx = (vw - imageW * scale) / 2f
        val dy = (vh - imageH * scale) / 2f
        fun x(v: Float) = v * scale + dx
        fun y(v: Float) = v * scale + dy
        val b = detection.box
        val box = android.graphics.RectF(x(b.left), y(b.top), x(b.right), y(b.bottom))
        val points = detection.keypoints.map { kp ->
            YoloPoseEngine.PoseKeypoint(android.graphics.PointF(x(kp.point.x), y(kp.point.y)), kp.confidence)
        }
        return YoloPoseEngine.PoseDetection(box, detection.confidence, points)
    }

    private fun imageToBitmap(image: ImageProxy): Bitmap {
        val plane = image.planes[0]
        val width = image.width
        val height = image.height
        val pixelStride = plane.pixelStride
        val rowStride = plane.rowStride
        val rowBytes = width * pixelStride
        val output = ByteBuffer.allocateDirect(width * height * 4)
        val source = plane.buffer
        source.rewind()
        val row = ByteArray(rowBytes)
        for (y in 0 until height) {
            source.position(y * rowStride)
            source.get(row, 0, rowBytes)
            output.put(row)
        }
        output.rewind()
        return Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).also { it.copyPixelsFromBuffer(output) }
    }

    private fun rotate(bitmap: Bitmap, degrees: Int): Bitmap {
        if (degrees == 0) return bitmap
        val matrix = Matrix().apply { postRotate(degrees.toFloat()) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun showStatus(value: String) { status.text = value }

    override fun onDestroy() {
        engine?.close()
        analysisExecutor.shutdownNow()
        super.onDestroy()
    }
}
