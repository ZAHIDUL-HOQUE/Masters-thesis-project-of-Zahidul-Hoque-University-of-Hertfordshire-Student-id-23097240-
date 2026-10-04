# prototypical Mask R-CNN for Vehicle Exterior Damage Detection 


#  Vehicle Exterior Damage Detection 

An application for vehicle exterior damage detection, segmentation, and inspection grading powered by **Prototypical Mask R-CNN** (ResNet-50-FPN backbone) trained on the **CarDD** dataset.

The system automatically detects, segments, and classifies six types of exterior vehicle damage:
- 🟠 **Dent** (`#FF7A00`)
- 🔵 **Scratch** (`#00D2FF`)
- 🟣 **Crack** (`#A855F7`)
- 🔴 **Glass Shatter** (`#FF2A6D`)
- 🟡 **Lamp Broken** (`#FACC15`)
- 🟢 **Tire Flat** (`#10B981`)

---

##  Key Features

1. **Interactive Web Dashboard (`app.py`)**:

   - **Multi-source Image Input**: Drag & drop file upload, clipboard paste, or webcam capture.
   - **Real-time KPI Metrics**:
     -  **Total Defects Found**
     -  **Primary Damage Category**
     -  **Peak Detection Confidence**
     -  **Vehicle Severity Assessment** (*Minor, Moderate, Severe*)
   - **Interactive Controls**:
     - *Confidence Threshold* slider (0.05 – 0.95)
     - *NMS IoU Overlap Threshold* slider (0.10 – 0.70) to merge duplicate boxes
     - *Mask Opacity (Alpha)* slider (0.10 – 0.90)
     - *Visual Overlay Mode*: Combined (Masks + Boxes), Masks Only, or Boxes Only
     - *Category Filter*: Toggle specific damage classes on or off
   - **Individual Defect Crops Gallery**: Close-up inspection thumbnails of each detected flaw.
   - **Defect Specifications Table**: Sortable tabular view with pixel area and bounding box coordinates.
   - **Downloadable Inspection Report**: Export complete findings as formatted JSON.
   - **Preset 1-Click Examples**: Preloaded vehicle photos demonstrating dents, scratches, flat tires, cracks, and broken lamps.

2. **Command-Line Interface (`cli.py`)**:
   - Run batch or single-image inference from terminal with custom thresholds.
   - Generates annotated images, cropped defect thumbnails, and structured JSON reports without running a web server.

3. **Production Model Architecture (`model.py` & `detector.py`)**:
   - Prototypical classification head with Euclidean metric prototype representations.
   - Non-Maximum Suppression (NMS) post-processing.
   - Hardware acceleration support (auto-detects NVIDIA CUDA GPU or falls back to multi-threaded CPU).

---

##  Quick Start

### 1. Launch Web Application (Windows)
Double-click `run_app.bat` or run:

```powershell
.venv\Scripts\python.exe app.py
```

The web interface will launch automatically in your default browser at:
 **`http://127.0.0.1:7860`**

### 2. Run via Command Line (CLI)

#### Single Image:
```powershell
.venv\Scripts\python.exe cli.py --image examples/sample_dent_and_scratch.jpg --output-dir results --save-crops
```

#### Batch Directory:
```powershell
.venv\Scripts\python.exe cli.py --batch-dir examples/ --output-dir results/batch_output --conf 0.35
```

---

## 📂 Project Structure

```
02/
├── latest_model.pth          # Trained Prototypical Mask R-CNN weights (~176 MB)
├── app.py                    # Gradio Web Application UI & server
├── model.py                  # PyTorch model architecture & prototype head definition
├── detector.py               # Inference engine, NMS filtering & visual rendering
├── cli.py                    # Standalone Command-Line Interface tool
├── run_app.bat               # 1-click Windows launcher
```

---

##  Installation & Environment Setup

If you want to set up a new environment manually:

```powershell
# Create Python 3.11 virtual environment
uv venv --python 3.11 .venv

# Activate environment
.venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

---

## 📄 Output Report Example

The application generates structured JSON inspection reports formatted like:

```json
{
  "timestamp": "2026-10-03T23:16:33",
  "image_dimensions": { "width": 1000, "height": 667 },
  "inference_time_seconds": 3.239,
  "summary": {
    "total_defects": 5,
    "primary_damage_type": "Scratch",
    "peak_confidence": 0.992,
    "severity_assessment": "Severe Damage",
    "damage_class_breakdown": {
      "Dent": 1,
      "Scratch": 4
    },
    "estimated_surface_area_pct": 12.6
  },
  "detections": [
    {
      "index": 1,
      "class": "Dent",
      "confidence_str": "99.2%",
      "box": [39, 157, 402, 378],
      "area_px": 38942
    }
  ]
}
```
