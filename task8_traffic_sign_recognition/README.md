# Task 8: Traffic Sign Recognition

This project detects and classifies German traffic signs inside full-resolution
road scenes using a lightweight YOLOv8 object detector.

## Dataset

The project uses the
[German Traffic Sign Detection (GTSDB) Dataset](https://www.kaggle.com/datasets/icebearogo/german-traffic-sign-detection-gtsdb-dataset),
a JPEG mirror of the official benchmark. It contains:

- 600 training scenes
- 300 held-out test scenes
- 1360×800 road images
- Zero or more traffic signs per scene
- YOLO bounding boxes for 43 traffic-sign classes

Negative scenes are retained because a real driving detector must also learn
when no relevant sign is present.

## Pipeline

The main script:

1. Downloads the Kaggle archive when needed.
2. Validates every YOLO annotation and filters invalid records.
3. Preserves empty label files for negative images.
4. Fine-tunes pretrained YOLOv8n on complete road scenes.
5. Evaluates precision, recall, mAP@50, and mAP@50:95.
6. Benchmarks model-only and end-to-end FPS after warm-up.
7. Compares detection counts at confidence thresholds 0.10, 0.25, and 0.50.
8. Exports the trained detector to ONNX.

One test annotation in the Kaggle mirror uses invalid class ID `100`; the data
validation step removes and records it instead of silently corrupting
evaluation.

## Why YOLOv8n?

YOLOv8n is the smallest YOLOv8 detector. Its compact architecture is appropriate
for the task's real-time constraint, where a slightly more accurate but much
slower model may be unsuitable for autonomous-driving inference.

## Run

From the repository root:

```bash
python -m pip install -r requirements.txt
python task8_traffic_sign_recognition/src/traffic_sign_detection.py
```

Use existing weights without retraining:

```bash
python task8_traffic_sign_recognition/src/traffic_sign_detection.py \
  --weights task8_traffic_sign_recognition/outputs/models/gtsdb_yolov8n_best.pt
```

Adjust the deployment confidence threshold:

```bash
python task8_traffic_sign_recognition/src/traffic_sign_detection.py \
  --confidence 0.50
```

## Webcam bonus

After training, run:

```bash
python task8_traffic_sign_recognition/src/webcam_inference.py \
  --confidence 0.25
```

Press `q` or Escape to exit. Webcam inference draws bounding boxes, class
labels, confidence scores, and live FPS.

## Results

The committed YOLOv8n checkpoint was trained for 10 epochs at 640-pixel input
resolution and evaluated on all 300 held-out full scenes (360 valid boxes).

| Metric | Result |
| --- | ---: |
| Precision | 0.0134 |
| Recall | 0.5024 |
| mAP@50 | 0.0276 |
| mAP@50:95 | 0.0204 |
| End-to-end speed | 48.3 FPS |
| Model-only speed | 97.0 FPS |
| Mean end-to-end latency | 20.7 ms |
| P95 end-to-end latency | 26.1 ms |

Speed was measured with single-image inference on 100 held-out scenes after
warm-up on an Apple M3 GPU (`mps`). The end-to-end benchmark includes image
loading, preprocessing, inference, and postprocessing. Hardware changes will
change the FPS result.

This experiment clears a 30 FPS real-time reference, but its detection accuracy
is **not deployment-ready**. GTSDB is extremely sparse for a 43-class detector:
the training split has only 506 annotated signs, some classes have a single
example, three classes are absent from training, and the median sign is only
2.8% of the image width. Longer or deployment-grade training should use
class-balanced augmentation, additional labeled scenes, and possibly a
safety-relevant superclass taxonomy.

The confidence filter illustrates the precision/coverage tradeoff on the first
100 test scenes:

| Confidence | Detections | Scenes with detections |
| ---: | ---: | ---: |
| 0.10 | 120 | 47 |
| 0.25 | 30 | 19 |
| 0.50 | 5 | 5 |

The sample-prediction figure deliberately shows selected detected scenes for
visual inspection; the metrics above use the complete, unfiltered test split.

## Generated artifacts

- `outputs/models/gtsdb_yolov8n_best.pt`
- Local ONNX export in `outputs/models/gtsdb_yolov8n.onnx`
- `outputs/model_metrics.json`
- `outputs/confidence_thresholds.json`
- `outputs/dataset_summary.json`
- `outputs/training_history.csv`
- Dataset composition, training curves, sample detections, and an accuracy/FPS
  summary in `outputs/figures/`
