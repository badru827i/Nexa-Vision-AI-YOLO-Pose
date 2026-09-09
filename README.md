# Nexa Vision AI — YOLO11x Pose

A clean Android realtime pose-scanning app built around CameraX and an on-device YOLO11x-Pose LiteRT/TFLite model.

## Scan pipeline

CameraX → RGBA frame → letterbox 640×640 → YOLO11x-Pose FP32 → person boxes → 17 COCO keypoints → skeleton overlay.

The analyzer uses `KEEP_ONLY_LATEST` and drops frames while inference is busy so the camera pipeline stays responsive instead of building a backlog.

## Model

The app expects `app/src/main/assets/yolo11x-pose.tflite` at runtime. The repository intentionally does not commit the large binary model. GitHub Actions downloads the official `yolo11x-pose.pt`, exports it with Ultralytics using the `litert` format at 640×640 FP32, injects the `.tflite` into the build, and produces the APK artifact.

Ultralytics documents YOLO11-Pose LiteRT export at 640×640 with FP32 as the default precision. See https://docs.ultralytics.com/integrations/litert/.

## Build

GitHub Actions is the intended first build path because the model export is performed during CI. Run the **Build Nexa Vision AI APK** workflow manually from the Actions tab or push to `main`.

## Performance target

This project intentionally prioritizes model quality and realtime responsiveness over APK size. The FP32 model is kept for the baseline accuracy/performance build. GPU acceleration can be added as a second tested runtime path after CPU inference is verified on the target phone.
