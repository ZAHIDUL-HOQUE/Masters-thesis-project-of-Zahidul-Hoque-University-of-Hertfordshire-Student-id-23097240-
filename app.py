"""
Vehicle Exterior Damage Detection - Web Application
Powered by Prototypical Mask R-CNN & Gradio
"""

import os
import json
import tempfile
from typing import List, Tuple, Any
from PIL import Image
import gradio as gr

from model import (
    DAMAGE_CLASSES,
    CLASS_HEX_COLORS
)
from detector import DamageDetector

# Initialize the global detector engine
detector = DamageDetector(weights_path="latest_model.pth")

# Exclude Background from selectable UI categories
ACTIVE_CATEGORIES = [c for c in DAMAGE_CLASSES if c != "Background"]

# Custom CSS for rich aesthetics and modern dark theme
CUSTOM_CSS = """
/* Global styling */
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
}

/* Header Banner */
.header-box {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.5);
}

.header-title {
    font-size: 2.1rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}

.header-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    line-height: 1.5;
}

/* Category Legend Chips */
.legend-container {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 14px;
}

.legend-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.82rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 9999px;
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.12);
    color: #e2e8f0;
}

.legend-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
}

/* KPI Stat Cards */
.kpi-row {
    margin-bottom: 20px !important;
}

.kpi-card {
    background: #1e293b !important;
    border: 1px solid rgba(148, 163, 184, 0.15) !important;
    border-radius: 12px !important;
    padding: 16px !important;
    text-align: center !important;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.kpi-card:hover {
    transform: translateY(-2px);
    border-color: rgba(99, 102, 241, 0.4) !important;
}

.action-btn {
    background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    padding: 12px 24px !important;
    border-radius: 10px !important;
    border: none !important;
    box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.4) !important;
    transition: all 0.2s ease-in-out !important;
}

.action-btn:hover {
    box-shadow: 0 6px 20px 0 rgba(79, 70, 229, 0.6) !important;
    filter: brightness(1.1);
}
"""


def run_inference(
    input_image: Image.Image,
    conf_threshold: float,
    nms_threshold: float,
    mask_opacity: float,
    box_thickness: int,
    selected_classes: List[str],
    view_mode: str
) -> Tuple[Any, str, str, str, str, List[Tuple[Any, str]], List[List[Any]], str, str]:
    """
    Callback function when 'Analyze Vehicle Damage' is triggered.
    """
    if input_image is None:
        return (
            None,
            "0",
            "N/A",
            "N/A",
            "No Image Provided",
            [],
            [],
            "{}",
            None
        )

    results = detector.predict(
        image=input_image,
        conf_threshold=conf_threshold,
        nms_threshold=nms_threshold,
        mask_alpha=mask_opacity,
        box_thickness=int(box_thickness),
        selected_classes=selected_classes,
        view_mode=view_mode
    )

    if results is None:
        return (None, "0", "N/A", "N/A", "Error during processing", [], [], "{}", None)

    # Save JSON report to temp file for download
    json_str = json.dumps(results["report_json"], indent=2)
    temp_dir = tempfile.gettempdir()
    report_file_path = os.path.join(temp_dir, f"damage_inspection_report_{int(results['latency_sec']*1000)}.json")
    with open(report_file_path, "w", encoding="utf-8") as f:
        f.write(json_str)

    return (
        results["annotated_image"],
        results["kpi_total_defects"],
        results["kpi_primary_damage"],
        results["kpi_max_confidence"],
        results["kpi_severity_level"],
        results["defects_gallery"],
        results["defects_table"],
        json_str,
        report_file_path
    )


# Build the Gradio App with Blocks
with gr.Blocks(title="Vehicle Exterior Damage AI Inspector") as demo:
    # 1. Header Banner
    chips_html = "".join([
        f'<span class="legend-chip"><span class="legend-dot" style="background-color: {CLASS_HEX_COLORS[cat]};"></span>{cat}</span>'
        for cat in ACTIVE_CATEGORIES
    ])

    gr.HTML(f"""
    <div class="header-box">
        <div class="header-title">🚗 Vehicle Exterior Damage Detection</div>
        <div class="header-subtitle">
            Automated visual damage inspection powered by <b>Prototypical Mask R-CNN</b> with ResNet-50 FPN.
            Detects, segments, and grades exterior structural flaws: <b>Dent, Scratch, Crack, Glass Shatter, Broken Lamp, and Flat Tire</b>.
        </div>
        <div class="legend-container">
            <span style="font-size: 0.85rem; font-weight: 600; color: #94a3b8; align-self: center; margin-right: 4px;">Detectable Flaws:</span>
            {chips_html}
        </div>
    </div>
    """)

    # 2. Main Layout (Two columns: Controls/Inputs on Left, Results on Right)
    with gr.Row():
        # LEFT COLUMN: Inputs & Customization Parameters
        with gr.Column(scale=5):
            gr.Markdown("### 📥 1. Vehicle Photo Input")
            input_image = gr.Image(
                type="pil",
                label="Upload Vehicle Image",
                sources=["upload", "clipboard", "webcam"],
                elem_id="input-vehicle-image"
            )

            gr.Markdown("### ⚙️ 2. Detection Parameters")
            with gr.Accordion("Advanced Model Controls", open=True):
                with gr.Row():
                    conf_slider = gr.Slider(
                        minimum=0.05,
                        maximum=0.95,
                        value=0.30,
                        step=0.05,
                        label="Confidence Threshold",
                        info="Filter detections below this score (lower finds subtle defects)"
                    )
                    nms_slider = gr.Slider(
                        minimum=0.10,
                        maximum=0.70,
                        value=0.30,
                        step=0.05,
                        label="NMS Overlap Threshold",
                        info="IoU threshold to suppress duplicate overlapping boxes"
                    )

                with gr.Row():
                    mask_alpha_slider = gr.Slider(
                        minimum=0.10,
                        maximum=0.90,
                        value=0.45,
                        step=0.05,
                        label="Mask Opacity (Alpha)",
                        info="Transparency level of damage segmentation masks"
                    )
                    box_thick_slider = gr.Slider(
                        minimum=1,
                        maximum=6,
                        value=2,
                        step=1,
                        label="Box Line Width",
                        info="Thickness of bounding boxes"
                    )

                view_mode_radio = gr.Radio(
                    choices=["Combined (Masks & Boxes)", "Masks Only", "Boxes Only"],
                    value="Combined (Masks & Boxes)",
                    label="Visual Overlay Mode"
                )

                category_filter = gr.CheckboxGroup(
                    choices=ACTIVE_CATEGORIES,
                    value=ACTIVE_CATEGORIES,
                    label="Filter Active Damage Categories"
                )

            with gr.Row():
                analyze_btn = gr.Button("⚡ Analyze Vehicle Damage", variant="primary", elem_classes=["action-btn"])
                clear_btn = gr.ClearButton(components=[input_image], value="🗑️ Clear Image")

            # Preset Examples Section
            gr.Markdown("### 🖼️ Example Showcase (Click to Test)")
            example_paths = [
                ["examples/sample_dent_and_scratch.jpg", 0.30, 0.30, 0.45, 2, ACTIVE_CATEGORIES, "Combined (Masks & Boxes)"],
                ["examples/sample_flat_tire_and_scratch.jpg", 0.30, 0.30, 0.45, 2, ACTIVE_CATEGORIES, "Combined (Masks & Boxes)"],
                ["examples/sample_broken_lamp_and_dent.jpg", 0.30, 0.30, 0.45, 2, ACTIVE_CATEGORIES, "Combined (Masks & Boxes)"],
                ["examples/sample_multiple_dents_and_crack.jpg", 0.30, 0.30, 0.45, 2, ACTIVE_CATEGORIES, "Combined (Masks & Boxes)"],
                ["examples/sample_flat_tire.jpg", 0.30, 0.30, 0.45, 2, ACTIVE_CATEGORIES, "Combined (Masks & Boxes)"]
            ]
            gr.Examples(
                examples=example_paths,
                inputs=[
                    input_image,
                    conf_slider,
                    nms_slider,
                    mask_alpha_slider,
                    box_thick_slider,
                    category_filter,
                    view_mode_radio
                ],
                label="Sample Vehicle Damages"
            )

        # RIGHT COLUMN: Inspection KPIs & Visual Outputs
        with gr.Column(scale=7):
            gr.Markdown("### 📊 3. Inspection Assessment")

            # KPI Stats Row
            with gr.Row(elem_classes=["kpi-row"]):
                kpi_count = gr.Textbox(label="Total Defects", value="0", interactive=False, elem_classes=["kpi-card"])
                kpi_primary = gr.Textbox(label="Primary Flaw", value="N/A", interactive=False, elem_classes=["kpi-card"])
                kpi_conf = gr.Textbox(label="Peak Confidence", value="N/A", interactive=False, elem_classes=["kpi-card"])
                kpi_severity = gr.Textbox(label="Severity Rating", value="Awaiting Analysis", interactive=False, elem_classes=["kpi-card"])

            # Main Visual Output
            annotated_output = gr.Image(
                type="pil",
                label="Damage Detection & Segmentation Overlay",
                interactive=False
            )

            # Detailed Findings Tabs
            with gr.Tabs():
                with gr.TabItem("🔍 Defect Crops Gallery"):
                    gr.Markdown("Close-up thumbnails of each isolated damage region:")
                    crops_gallery = gr.Gallery(
                        label="Individual Detected Defects",
                        show_label=False,
                        columns=3,
                        rows=2,
                        height="auto",
                        preview=True
                    )

                with gr.TabItem("📋 Defect Specifications Table"):
                    defects_df = gr.DataFrame(
                        headers=["#", "Damage Type", "Confidence", "Affected Area", "Bounding Box [x1, y1, x2, y2]"],
                        datatype=["number", "str", "str", "str", "str"],
                        interactive=False,
                        wrap=True
                    )

                with gr.TabItem("📄 JSON Inspection Report"):
                    report_json_view = gr.Code(
                        language="json",
                        label="Inspection Report JSON Data",
                        interactive=False
                    )
                    download_btn = gr.File(
                        label="Download Full Inspection Report (.json)",
                        interactive=False
                    )

    # 3. Connect Inference Action
    analyze_btn.click(
        fn=run_inference,
        inputs=[
            input_image,
            conf_slider,
            nms_slider,
            mask_alpha_slider,
            box_thick_slider,
            category_filter,
            view_mode_radio
        ],
        outputs=[
            annotated_output,
            kpi_count,
            kpi_primary,
            kpi_conf,
            kpi_severity,
            crops_gallery,
            defects_df,
            report_json_view,
            download_btn
        ]
    )

    # Also automatically trigger on image upload if desired
    input_image.change(
        fn=run_inference,
        inputs=[
            input_image,
            conf_slider,
            nms_slider,
            mask_alpha_slider,
            box_thick_slider,
            category_filter,
            view_mode_radio
        ],
        outputs=[
            annotated_output,
            kpi_count,
            kpi_primary,
            kpi_conf,
            kpi_severity,
            crops_gallery,
            defects_df,
            report_json_view,
            download_btn
        ]
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Launch Vehicle Exterior Damage Detection Web App")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the web server on")
    parser.add_argument("--share", action="store_true", help="Generate public shareable link")
    parser.add_argument("--inbrowser", action="store_true", default=True, help="Automatically open browser")
    args = parser.parse_args()

    print(f"Starting Vehicle Damage Detection App on http://127.0.0.1:{args.port}...")
    demo.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=args.share,
        inbrowser=args.inbrowser,
        css=CUSTOM_CSS
    )
