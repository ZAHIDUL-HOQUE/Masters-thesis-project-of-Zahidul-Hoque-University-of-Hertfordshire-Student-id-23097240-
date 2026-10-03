"""
Vehicle Exterior Damage Detection - Inference & Visualization Pipeline
"""

import time
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.transforms as T
import torchvision.ops as ops

from model import (
    load_damage_model,
    DAMAGE_CLASSES,
    CLASS_COLORS,
    CLASS_HEX_COLORS,
    DAMAGE_SEVERITY_WEIGHTS
)


class DamageDetector:
    """
    High-level damage detection wrapper handling preprocessing, inference,
    NMS post-processing, overlay rendering, and structured reporting.
    """
    def __init__(self, weights_path: str = "latest_model.pth", device: torch.device = None):
        self.model, self.device = load_damage_model(weights_path=weights_path, device=device)
        self.transform = T.Compose([T.ToTensor()])

    def predict(
        self,
        image: Image.Image,
        conf_threshold: float = 0.30,
        nms_threshold: float = 0.30,
        mask_alpha: float = 0.45,
        box_thickness: int = 2,
        selected_classes: Optional[List[str]] = None,
        view_mode: str = "Combined (Masks & Boxes)"
    ) -> Dict[str, Any]:
        """
        Runs exterior damage detection on an input PIL image.

        Args:
            image: PIL Image of the vehicle
            conf_threshold: Minimum confidence score to retain detection [0.0 - 1.0]
            nms_threshold: IoU overlap threshold for Non-Maximum Suppression [0.0 - 1.0]
            mask_alpha: Opacity factor for segmentation masks [0.0 - 1.0]
            box_thickness: Line thickness for bounding boxes
            selected_classes: Optional list of class names to filter (e.g. ['Dent', 'Scratch'])
            view_mode: Visualization mode ('Combined (Masks & Boxes)', 'Masks Only', 'Boxes Only')

        Returns:
            Dictionary containing:
                - 'annotated_image': PIL Image with detection overlays
                - 'defects_gallery': List of (crop_img, caption) for each detection
                - 'kpi_total_defects': int
                - 'kpi_primary_damage': str
                - 'kpi_max_confidence': str
                - 'kpi_severity_level': str ('None', 'Minor', 'Moderate', 'Severe')
                - 'defects_table': List[List[Any]] for data display
                - 'report_json': Dict summary of inspection
                - 'latency_sec': float
        """
        if image is None:
            return None

        orig_w, orig_h = image.size
        rgb_image = image.convert("RGB")
        img_np = np.array(rgb_image)

        # 1. Preprocess & Forward Pass
        img_tensor = self.transform(rgb_image).to(self.device)

        t_start = time.time()
        with torch.no_grad():
            prediction = self.model([img_tensor])[0]
        inference_time = time.time() - t_start

        # 2. Extract raw outputs
        scores = prediction["scores"].cpu()
        labels = prediction["labels"].cpu()
        boxes = prediction["boxes"].cpu()
        masks = prediction["masks"].cpu()

        # 3. Filter by confidence and valid category
        valid_mask = (scores >= conf_threshold) & (labels > 0) & (labels < len(DAMAGE_CLASSES))

        # Filter by selected classes if provided
        if selected_classes is not None:
            class_filter = torch.zeros(len(labels), dtype=torch.bool)
            for idx, lbl in enumerate(labels):
                lbl_idx = lbl.item()
                if lbl_idx < len(DAMAGE_CLASSES) and DAMAGE_CLASSES[lbl_idx] in selected_classes:
                    class_filter[idx] = True
            valid_mask = valid_mask & class_filter

        filt_scores = scores[valid_mask]
        filt_labels = labels[valid_mask]
        filt_boxes = boxes[valid_mask]
        filt_masks = masks[valid_mask]

        # 4. Apply Non-Maximum Suppression (NMS) to eliminate duplicate boxes
        if len(filt_boxes) > 0 and nms_threshold < 1.0:
            nms_indices = ops.nms(filt_boxes, filt_scores, iou_threshold=nms_threshold)
            filt_scores = filt_scores[nms_indices]
            filt_labels = filt_labels[nms_indices]
            filt_boxes = filt_boxes[nms_indices]
            filt_masks = filt_masks[nms_indices]

        num_detections = len(filt_scores)

        # 5. Render Visualizations
        annotated_np = img_np.copy()
        mask_overlay = np.zeros_like(img_np, dtype=np.uint8)
        has_any_mask = False

        draw_masks = "Mask" in view_mode or "Combined" in view_mode
        draw_boxes = "Box" in view_mode or "Combined" in view_mode

        detections_data = []
        gallery_items = []
        total_damage_area_px = 0

        # Process each detection
        for i in range(num_detections):
            score = float(filt_scores[i].item())
            label_idx = int(filt_labels[i].item())
            class_name = DAMAGE_CLASSES[label_idx]
            color_rgb = CLASS_COLORS.get(class_name, (255, 0, 0))
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

            x1, y1, x2, y2 = filt_boxes[i].tolist()
            x1 = max(0, int(round(x1)))
            y1 = max(0, int(round(y1)))
            x2 = min(orig_w, int(round(x2)))
            y2 = min(orig_h, int(round(y2)))
            box_area = max(0, x2 - x1) * max(0, y2 - y1)

            # Segmentation mask processing
            mask_area = 0
            if len(filt_masks) > i:
                mask_raw = filt_masks[i, 0].numpy()
                bin_mask = mask_raw > 0.5
                mask_area = int(np.sum(bin_mask))
                total_damage_area_px += mask_area

                if draw_masks:
                    mask_overlay[bin_mask] = color_rgb
                    has_any_mask = True
                    # Subtle contour outline for sharpness
                    contours, _ = cv2.findContours(
                        bin_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(annotated_np, contours, -1, color_rgb, 2)

            # Crop damage region for the gallery
            pad = 25
            cx1 = max(0, x1 - pad)
            cy1 = max(0, y1 - pad)
            cx2 = min(orig_w, x2 + pad)
            cy2 = min(orig_h, y2 + pad)

            if cx2 > cx1 and cy2 > cy1:
                crop = img_np[cy1:cy2, cx1:cx2]
                crop_pil = Image.fromarray(crop)
                caption = f"#{i+1}: {class_name} ({score * 100:.1f}%)"
                gallery_items.append((crop_pil, caption))

            detections_data.append({
                "index": i + 1,
                "class": class_name,
                "confidence": score,
                "confidence_str": f"{score * 100:.1f}%",
                "box": [x1, y1, x2, y2],
                "box_str": f"[{x1}, {y1}, {x2}, {y2}]",
                "area_px": mask_area if mask_area > 0 else box_area,
                "color_hex": CLASS_HEX_COLORS.get(class_name, "#FFFFFF")
            })

        # Blend segmentation mask overlay if enabled
        if draw_masks and has_any_mask:
            mask_indices = np.any(mask_overlay > 0, axis=-1)
            annotated_np[mask_indices] = (
                (1.0 - mask_alpha) * annotated_np[mask_indices] + mask_alpha * mask_overlay[mask_indices]
            ).astype(np.uint8)

        # Draw bounding boxes and high-contrast labels
        if draw_boxes:
            annotated_pil = Image.fromarray(annotated_np)
            draw = ImageDraw.Draw(annotated_pil)

            # Use default or scalable bitmap font
            try:
                font = ImageFont.truetype("arial.ttf", size=max(14, int(min(orig_w, orig_h) * 0.022)))
            except Exception:
                font = ImageFont.load_default()

            for det in detections_data:
                x1, y1, x2, y2 = det["box"]
                color_rgb = CLASS_COLORS.get(det["class"], (255, 255, 255))
                label_text = f"#{det['index']} {det['class']} {det['confidence_str']}"

                # Draw outer box
                for offset in range(box_thickness):
                    draw.rectangle(
                        [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                        outline=color_rgb
                    )

                # Measure badge size
                bbox_text = draw.textbbox((0, 0), label_text, font=font)
                text_w = bbox_text[2] - bbox_text[0] + 12
                text_h = bbox_text[3] - bbox_text[1] + 8

                # Position badge above or inside box
                badge_y1 = y1 - text_h if y1 - text_h >= 0 else y1
                badge_y2 = badge_y1 + text_h
                badge_x1 = x1
                badge_x2 = x1 + text_w

                # Draw badge background with class color
                draw.rectangle([badge_x1, badge_y1, badge_x2, badge_y2], fill=color_rgb)

                # Text contrast (dark text for bright yellow/cyan, white for others)
                brightness = (color_rgb[0] * 299 + color_rgb[1] * 587 + color_rgb[2] * 114) / 1000
                text_color = (0, 0, 0) if brightness > 150 else (255, 255, 255)

                draw.text(
                    (badge_x1 + 6, badge_y1 + 4),
                    label_text,
                    fill=text_color,
                    font=font
                )

            annotated_result = annotated_pil
        else:
            annotated_result = Image.fromarray(annotated_np)

        # 6. Aggregate Metrics & Severity Assessment
        class_counts = {}
        for det in detections_data:
            c = det["class"]
            class_counts[c] = class_counts.get(c, 0) + 1

        primary_damage = "None"
        if class_counts:
            primary_damage = max(class_counts.items(), key=lambda x: x[1])[0]

        max_conf = max([d["confidence"] for d in detections_data]) if detections_data else 0.0

        # Calculate Damage Severity Score
        total_pixels = orig_w * orig_h
        surface_impact_pct = (total_damage_area_px / total_pixels) * 100 if total_pixels > 0 else 0.0

        severity_score = sum(
            DAMAGE_SEVERITY_WEIGHTS.get(d["class"], 1.0) * d["confidence"]
            for d in detections_data
        )

        if num_detections == 0:
            severity_level = "No Defects Detected"
        elif severity_score < 2.0 and surface_impact_pct < 2.0:
            severity_level = "Minor Damage"
        elif severity_score < 5.0 and surface_impact_pct < 8.0:
            severity_level = "Moderate Damage"
        else:
            severity_level = "Severe Damage"

        # Format Table Data for Gradio DataFrame
        table_rows = [
            [
                d["index"],
                d["class"],
                d["confidence_str"],
                f"{d['area_px']:,} px",
                d["box_str"]
            ]
            for d in detections_data
        ]

        # Structured Report JSON
        report = {
            "timestamp": datetime.now().isoformat(),
            "image_dimensions": {"width": orig_w, "height": orig_h},
            "inference_time_seconds": round(inference_time, 3),
            "configuration": {
                "confidence_threshold": conf_threshold,
                "nms_threshold": nms_threshold,
                "mask_opacity": mask_alpha
            },
            "summary": {
                "total_defects": num_detections,
                "primary_damage_type": primary_damage,
                "peak_confidence": round(max_conf, 4),
                "severity_assessment": severity_level,
                "damage_class_breakdown": class_counts,
                "estimated_surface_area_pct": round(surface_impact_pct, 2)
            },
            "detections": detections_data
        }

        return {
            "annotated_image": annotated_result,
            "defects_gallery": gallery_items,
            "kpi_total_defects": str(num_detections),
            "kpi_primary_damage": primary_damage if num_detections > 0 else "None",
            "kpi_max_confidence": f"{max_conf * 100:.1f}%" if num_detections > 0 else "N/A",
            "kpi_severity_level": severity_level,
            "defects_table": table_rows,
            "report_json": report,
            "latency_sec": round(inference_time, 3)
        }
