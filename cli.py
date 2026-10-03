"""
Vehicle Exterior Damage Detection - Command Line Interface (CLI)
Enables batch or single-image inference and automated report generation.
"""

import os
import argparse
import json
from pathlib import Path
from PIL import Image

from detector import DamageDetector


def main():
    parser = argparse.ArgumentParser(
        description="Run vehicle exterior damage detection on an image or directory of images."
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to an input vehicle image file."
    )
    parser.add_argument(
        "--batch-dir",
        type=str,
        help="Path to a directory containing images to process."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="latest_model.pth",
        help="Path to the trained PyTorch .pth checkpoint (default: latest_model.pth)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save annotated images and detection reports (default: output)."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.30,
        help="Confidence detection threshold between 0.0 and 1.0 (default: 0.30)."
    )
    parser.add_argument(
        "--nms",
        type=float,
        default=0.30,
        help="Non-Maximum Suppression IoU threshold (default: 0.30)."
    )
    parser.add_argument(
        "--mask-alpha",
        type=float,
        default=0.45,
        help="Segmentation mask opacity alpha (default: 0.45)."
    )
    parser.add_argument(
        "--save-crops",
        action="store_true",
        help="Save individual cropped defect images."
    )

    args = parser.parse_args()

    if not args.image and not args.batch_dir:
        print("Error: Please provide either --image <path> or --batch-dir <path>.")
        parser.print_help()
        return

    os.makedirs(args.output_dir, exist_ok=True)
    detector = DamageDetector(weights_path=args.weights)

    image_paths = []
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image not found at {args.image}")
            return
        image_paths.append(args.image)

    if args.batch_dir:
        if not os.path.isdir(args.batch_dir):
            print(f"Error: Directory not found at {args.batch_dir}")
            return
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            image_paths.extend(Path(args.batch_dir).glob(ext))

    print(f"\nProcessing {len(image_paths)} image(s)...")

    for idx, img_path in enumerate(image_paths, 1):
        path_str = str(img_path)
        base_name = Path(path_str).stem
        print(f"\n[{idx}/{len(image_paths)}] Analyzing: {path_str}")

        try:
            pil_img = Image.open(path_str)
        except Exception as e:
            print(f"  Failed to load image: {e}")
            continue

        results = detector.predict(
            image=pil_img,
            conf_threshold=args.conf,
            nms_threshold=args.nms,
            mask_alpha=args.mask_alpha
        )

        num_defects = int(results["kpi_total_defects"])
        print(f"  Result: {num_defects} defects | Primary: {results['kpi_primary_damage']} | Severity: {results['kpi_severity_level']} (Latency: {results['latency_sec']}s)")

        # Save annotated image
        annotated_path = os.path.join(args.output_dir, f"{base_name}_annotated.jpg")
        results["annotated_image"].save(annotated_path, quality=95)
        print(f"  Saved visual overlay: {annotated_path}")

        # Save JSON inspection report
        report_path = os.path.join(args.output_dir, f"{base_name}_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results["report_json"], f, indent=2)
        print(f"  Saved inspection report: {report_path}")

        # Optionally save crops
        if args.save_crops and results["defects_gallery"]:
            crops_dir = os.path.join(args.output_dir, f"{base_name}_crops")
            os.makedirs(crops_dir, exist_ok=True)
            for c_idx, (crop_img, cap) in enumerate(results["defects_gallery"], 1):
                clean_cap = cap.replace("#", "").replace(":", "_").replace(" ", "_").replace("%", "pct").replace("(", "").replace(")", "")
                crop_path = os.path.join(crops_dir, f"crop_{c_idx}_{clean_cap}.jpg")
                crop_img.save(crop_path, quality=95)
            print(f"  Saved {len(results['defects_gallery'])} defect crops in: {crops_dir}")

    print(f"\nAll tasks finished. Results saved to: {os.path.abspath(args.output_dir)}\n")


if __name__ == "__main__":
    main()
