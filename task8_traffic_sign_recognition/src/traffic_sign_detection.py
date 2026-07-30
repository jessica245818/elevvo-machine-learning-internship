"""Train, evaluate, benchmark, and export a YOLOv8 detector on GTSDB."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import random
import shutil
import time
import urllib.request
import zipfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from ultralytics import YOLO


RANDOM_STATE = 42
DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "icebearogo/german-traffic-sign-detection-gtsdb-dataset"
)
CLASS_NAMES = [
    "speed_limit_20",
    "speed_limit_30",
    "speed_limit_50",
    "speed_limit_60",
    "speed_limit_70",
    "speed_limit_80",
    "end_speed_limit_80",
    "speed_limit_100",
    "speed_limit_120",
    "no_passing",
    "no_passing_over_3_5t",
    "right_of_way_next_intersection",
    "priority_road",
    "yield",
    "stop",
    "no_vehicles",
    "vehicles_over_3_5t_prohibited",
    "no_entry",
    "general_caution",
    "dangerous_curve_left",
    "dangerous_curve_right",
    "double_curve",
    "bumpy_road",
    "slippery_road",
    "road_narrows_right",
    "road_work",
    "traffic_signals",
    "pedestrians",
    "children_crossing",
    "bicycles_crossing",
    "beware_ice_snow",
    "wild_animals_crossing",
    "end_all_restrictions",
    "turn_right_ahead",
    "turn_left_ahead",
    "ahead_only",
    "straight_or_right",
    "straight_or_left",
    "keep_right",
    "keep_left",
    "roundabout",
    "end_no_passing",
    "end_no_passing_over_3_5t",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and benchmark YOLOv8n on full-scene GTSDB images."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(
            "task8_traffic_sign_recognition/data/raw/GTSDB_Train_and_Test"
        ),
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("task8_traffic_sign_recognition/data/processed/gtsdb_yolo"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("task8_traffic_sign_recognition/outputs"),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--benchmark-images", type=int, default=100)
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Evaluate existing weights instead of training.",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip ONNX export.",
    )
    return parser.parse_args()


def ensure_dataset(raw_dir: Path) -> None:
    required = [
        raw_dir / "Train/images",
        raw_dir / "Train/labels",
        raw_dir / "Test/images",
        raw_dir / "Test/labels",
    ]
    if all(path.exists() for path in required):
        return

    raw_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Dataset not found. Downloading from {DATASET_URL}")
    with urllib.request.urlopen(DATASET_URL) as response:
        content = response.read()
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        safe_names = [
            name
            for name in archive.namelist()
            if name.startswith("GTSDB_Train_and_Test/")
            and ".." not in Path(name).parts
        ]
        archive.extractall(raw_dir.parent, safe_names)


def link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def prepare_dataset(
    raw_dir: Path,
    processed_dir: Path,
) -> tuple[Path, dict[str, object]]:
    """Create a validated YOLO dataset and filter malformed annotations."""
    ensure_dataset(raw_dir)
    invalid_annotations: list[dict[str, object]] = []
    class_counts = np.zeros(len(CLASS_NAMES), dtype=int)
    box_widths: list[float] = []
    box_heights: list[float] = []
    split_summary: dict[str, dict[str, int]] = {}

    for raw_name, output_name in [("Train", "train"), ("Test", "test")]:
        source_images = raw_dir / raw_name / "images"
        source_labels = raw_dir / raw_name / "labels"
        image_output = processed_dir / "images" / output_name
        label_output = processed_dir / "labels" / output_name
        image_output.mkdir(parents=True, exist_ok=True)
        label_output.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(source_images.glob("*.jpg"))
        positive_images = 0
        valid_boxes = 0
        for image_path in image_paths:
            link_or_copy(image_path, image_output / image_path.name)
            source_label = source_labels / f"{image_path.stem}.txt"
            valid_lines: list[str] = []
            if source_label.exists():
                for line_number, line in enumerate(
                    source_label.read_text(encoding="utf-8").splitlines(),
                    start=1,
                ):
                    parts = line.split()
                    reason = None
                    if len(parts) != 5:
                        reason = "expected five YOLO values"
                    else:
                        try:
                            class_id = int(parts[0])
                            coordinates = [float(value) for value in parts[1:]]
                        except ValueError:
                            reason = "non-numeric value"
                        else:
                            if not 0 <= class_id < len(CLASS_NAMES):
                                reason = "class outside official 0-42 taxonomy"
                            elif not all(0 <= value <= 1 for value in coordinates):
                                reason = "normalized coordinate outside [0, 1]"
                            elif coordinates[2] <= 0 or coordinates[3] <= 0:
                                reason = "non-positive box size"
                    if reason:
                        invalid_annotations.append(
                            {
                                "file": str(source_label),
                                "line": line_number,
                                "content": line,
                                "reason": reason,
                            }
                        )
                        continue
                    valid_lines.append(line)
                    class_counts[class_id] += 1
                    box_widths.append(coordinates[2])
                    box_heights.append(coordinates[3])
                    valid_boxes += 1

            if valid_lines:
                positive_images += 1
            (label_output / f"{image_path.stem}.txt").write_text(
                "\n".join(valid_lines) + ("\n" if valid_lines else ""),
                encoding="utf-8",
            )

        split_summary[output_name] = {
            "images": len(image_paths),
            "positive_images": positive_images,
            "negative_images": len(image_paths) - positive_images,
            "boxes": valid_boxes,
        }

    yaml_path = processed_dir / "gtsdb.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(processed_dir.resolve()),
                "train": "images/train",
                "val": "images/test",
                "test": "images/test",
                "names": {
                    index: name for index, name in enumerate(CLASS_NAMES)
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    present_classes = np.flatnonzero(class_counts).tolist()
    summary: dict[str, object] = {
        "classes_defined": len(CLASS_NAMES),
        "classes_present_in_all_splits": len(present_classes),
        "class_counts": {
            CLASS_NAMES[index]: int(count)
            for index, count in enumerate(class_counts)
            if count
        },
        "splits": split_summary,
        "invalid_annotations_filtered": invalid_annotations,
        "median_box_width_fraction": float(np.median(box_widths)),
        "median_box_height_fraction": float(np.median(box_heights)),
        "dataset_yaml": str(yaml_path),
    }
    return yaml_path, summary


def select_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


def train_detector(
    data_yaml: Path,
    output_dir: Path,
    epochs: int,
    image_size: int,
    batch_size: int,
    device: str,
) -> Path:
    model = YOLO("yolov8n.pt")
    run_dir = (output_dir / "runs").resolve()
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        device=device,
        project=str(run_dir),
        name="train",
        exist_ok=True,
        pretrained=True,
        patience=max(5, epochs),
        seed=RANDOM_STATE,
        deterministic=True,
        workers=0,
        cache=False,
        plots=True,
        verbose=True,
    )
    best_source = run_dir / "train" / "weights" / "best.pt"
    if not best_source.exists():
        raise RuntimeError("Training completed without producing best.pt.")
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    best_output = model_dir / "gtsdb_yolov8n_best.pt"
    shutil.copy2(best_source, best_output)
    results_csv = run_dir / "train" / "results.csv"
    if results_csv.exists():
        shutil.copy2(results_csv, output_dir / "training_history.csv")
    results_plot = run_dir / "train" / "results.png"
    if results_plot.exists():
        shutil.copy2(
            results_plot,
            output_dir / "figures" / "02_training_curves.png",
        )
    return best_output


def evaluate_detector(
    weights: Path,
    data_yaml: Path,
    output_dir: Path,
    image_size: int,
    device: str,
) -> dict[str, float]:
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=image_size,
        conf=0.001,
        iou=0.6,
        device=device,
        project=str((output_dir / "runs").resolve()),
        name="evaluation",
        exist_ok=True,
        plots=True,
        workers=0,
        verbose=True,
    )
    speed = metrics.speed
    return {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "preprocess_ms_per_image": float(speed.get("preprocess", 0)),
        "inference_ms_per_image": float(speed.get("inference", 0)),
        "postprocess_ms_per_image": float(speed.get("postprocess", 0)),
    }


def benchmark_detector(
    model: YOLO,
    image_paths: list[Path],
    image_size: int,
    confidence: float,
    device: str,
) -> dict[str, object]:
    selected = image_paths[: min(len(image_paths), 100)]
    for image_path in selected[:5]:
        model.predict(
            str(image_path),
            imgsz=image_size,
            conf=confidence,
            device=device,
            verbose=False,
        )
    synchronize(device)

    latencies: list[float] = []
    inference_times: list[float] = []
    detection_counts: list[int] = []
    for image_path in selected:
        start = time.perf_counter()
        result = model.predict(
            str(image_path),
            imgsz=image_size,
            conf=confidence,
            device=device,
            verbose=False,
        )[0]
        synchronize(device)
        latencies.append(time.perf_counter() - start)
        inference_times.append(float(result.speed["inference"]))
        detection_counts.append(len(result.boxes))

    total_seconds = sum(latencies)
    return {
        "device": device,
        "image_size": image_size,
        "confidence_threshold": confidence,
        "images_measured": len(selected),
        "end_to_end_fps": len(selected) / total_seconds,
        "mean_end_to_end_ms": float(np.mean(latencies) * 1_000),
        "p95_end_to_end_ms": float(np.percentile(latencies, 95) * 1_000),
        "mean_model_inference_ms": float(np.mean(inference_times)),
        "model_only_fps": 1_000 / float(np.mean(inference_times)),
        "total_detections": int(sum(detection_counts)),
        "images_with_detections": int(np.count_nonzero(detection_counts)),
    }


def confidence_analysis(
    model: YOLO,
    image_paths: list[Path],
    image_size: int,
    device: str,
) -> list[dict[str, float | int]]:
    sample = image_paths[: min(100, len(image_paths))]
    rows: list[dict[str, float | int]] = []
    for threshold in [0.10, 0.25, 0.50]:
        detections = 0
        images_detected = 0
        for image_path in sample:
            result = model.predict(
                str(image_path),
                imgsz=image_size,
                conf=threshold,
                device=device,
                verbose=False,
            )[0]
            count = len(result.boxes)
            detections += count
            images_detected += int(count > 0)
        rows.append(
            {
                "confidence_threshold": threshold,
                "images": len(sample),
                "detections": detections,
                "images_with_detections": images_detected,
            }
        )
    return rows


def save_dataset_figure(
    summary: dict[str, object],
    figure_dir: Path,
) -> None:
    counts = summary["class_counts"]
    top = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:15]
    names = [item[0] for item in reversed(top)]
    values = [item[1] for item in reversed(top)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    bars = axes[0].barh(names, values, color="#2563EB")
    axes[0].bar_label(bars, padding=3)
    axes[0].set(
        title="Most Frequent GTSDB Classes",
        xlabel="Bounding boxes",
    )
    split_names = ["Train", "Test"]
    positive = [
        summary["splits"]["train"]["positive_images"],
        summary["splits"]["test"]["positive_images"],
    ]
    negative = [
        summary["splits"]["train"]["negative_images"],
        summary["splits"]["test"]["negative_images"],
    ]
    axes[1].bar(split_names, positive, label="Contains sign", color="#16A34A")
    axes[1].bar(
        split_names,
        negative,
        bottom=positive,
        label="Negative scene",
        color="#94A3B8",
    )
    axes[1].set(
        title="Full-Scene Dataset Composition",
        ylabel="Images",
    )
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "01_dataset_overview.png", dpi=180)
    plt.close(fig)


def save_sample_predictions(
    model: YOLO,
    test_images: list[Path],
    test_label_dir: Path,
    figure_dir: Path,
    image_size: int,
    confidence: float,
    device: str,
) -> None:
    positive_images = [
        image
        for image in test_images
        if (test_label_dir / f"{image.stem}.txt").read_text().strip()
    ]
    random.Random(RANDOM_STATE).shuffle(positive_images)
    detected: list[tuple[Path, object]] = []
    for image_path in positive_images:
        result = model.predict(
            str(image_path),
            imgsz=image_size,
            conf=confidence,
            device=device,
            verbose=False,
        )[0]
        if len(result.boxes):
            detected.append((image_path, result))
        if len(detected) == 6:
            break

    fig, axes = plt.subplots(2, 3, figsize=(16, 7))
    for ax, (image_path, result) in zip(axes.flat, detected):
        annotated = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
        ax.imshow(annotated)
        ax.set_title(f"{image_path.name}: {len(result.boxes)} detections")
        ax.axis("off")
    for ax in axes.flat[len(detected) :]:
        ax.axis("off")
    fig.suptitle(
        f"Selected YOLOv8n Test-Scene Detections at Confidence ≥ {confidence:.2f}"
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "03_sample_predictions.png", dpi=180)
    plt.close(fig)


def save_performance_figure(
    metrics: dict[str, object],
    threshold_results: list[dict[str, float | int]],
    figure_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    metric_names = ["mAP@50", "mAP@50:95"]
    metric_values = [
        100 * float(metrics["map50"]),
        100 * float(metrics["map50_95"]),
    ]
    bars = axes[0].bar(metric_names, metric_values, color=["#2563EB", "#60A5FA"])
    axes[0].bar_label(bars, fmt="%.2f%%", padding=3)
    axes[0].set(title="Held-Out Detection Accuracy", ylabel="Score (%)")

    fps_names = ["End-to-end", "Model only"]
    fps_values = [
        float(metrics["end_to_end_fps"]),
        float(metrics["model_only_fps"]),
    ]
    bars = axes[1].bar(fps_names, fps_values, color=["#16A34A", "#4ADE80"])
    axes[1].bar_label(bars, fmt="%.1f FPS", padding=3)
    axes[1].axhline(30, color="#DC2626", linestyle="--", label="30 FPS reference")
    axes[1].set(title="Single-Image Inference Speed", ylabel="Frames per second")
    axes[1].legend()
    fig.suptitle(
        "Accuracy and Speed Summary "
        f"(confidence {float(metrics['confidence_threshold']):.2f}; "
        f"{int(threshold_results[1]['detections'])} detections/100 scenes)"
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "04_accuracy_speed_summary.png", dpi=180)
    plt.close(fig)


def export_onnx(
    model: YOLO,
    output_dir: Path,
    image_size: int,
) -> Path:
    exported = Path(
        model.export(
            format="onnx",
            imgsz=image_size,
            opset=12,
            simplify=False,
            dynamic=False,
        )
    )
    destination = output_dir / "models" / "gtsdb_yolov8n.onnx"
    shutil.copy2(exported, destination)
    return destination


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    data_yaml, dataset_summary = prepare_dataset(
        args.raw_dir,
        args.processed_dir,
    )
    save_dataset_figure(dataset_summary, figure_dir)
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(dataset_summary, indent=2) + "\n",
        encoding="utf-8",
    )

    device = select_device()
    print(f"Training/evaluation device: {device}")
    if args.weights:
        weights = args.weights
    else:
        weights = train_detector(
            data_yaml,
            output_dir,
            args.epochs,
            args.image_size,
            args.batch_size,
            device,
        )

    detection_metrics = evaluate_detector(
        weights,
        data_yaml,
        output_dir,
        args.image_size,
        device,
    )
    model = YOLO(str(weights))
    test_images = sorted(
        (args.processed_dir / "images" / "test").glob("*.jpg")
    )
    benchmark_images = test_images[: args.benchmark_images]
    speed_metrics = benchmark_detector(
        model,
        benchmark_images,
        args.image_size,
        args.confidence,
        device,
    )
    threshold_results = confidence_analysis(
        model,
        benchmark_images,
        args.image_size,
        device,
    )
    save_sample_predictions(
        model,
        test_images,
        args.processed_dir / "labels" / "test",
        figure_dir,
        args.image_size,
        args.confidence,
        device,
    )

    onnx_path = None
    onnx_error = None
    if not args.skip_export:
        try:
            onnx_path = export_onnx(model, output_dir, args.image_size)
        except Exception as error:  # Export failure should not discard evaluation.
            onnx_error = f"{type(error).__name__}: {error}"

    results = {
        "model": "YOLOv8n",
        "weights": str(weights),
        "epochs": args.epochs,
        "image_size": args.image_size,
        "device": device,
        **detection_metrics,
        **speed_metrics,
        "onnx_export": str(onnx_path) if onnx_path else None,
        "onnx_export_error": onnx_error,
    }
    save_performance_figure(results, threshold_results, figure_dir)
    (output_dir / "model_metrics.json").write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "confidence_thresholds.json").write_text(
        json.dumps(threshold_results, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nDetection and speed metrics")
    print(json.dumps(results, indent=2))
    print("\nConfidence-threshold analysis")
    print(json.dumps(threshold_results, indent=2))
    print(f"\nArtifacts saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
