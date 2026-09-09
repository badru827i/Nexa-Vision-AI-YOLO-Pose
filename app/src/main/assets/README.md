# YOLO11x-Pose model asset

The app expects this exact file at build/runtime:

`yolo11x-pose.tflite`

The GitHub Actions release workflow generates the official Ultralytics YOLO11x-Pose LiteRT FP32 export automatically, then injects it into the APK build. The large model is intentionally not committed to the Git repository.
