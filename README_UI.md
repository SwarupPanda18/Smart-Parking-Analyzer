# Mother's Hut — Traffic & Gate Operations Console

A high-performance surveillance telemetry and Automated Number Plate Recognition (ANPR/ALPR) system built with **FastAPI**, **HTML5**, **CSS3**, and **Vanilla JavaScript**.

---

## Key Features

- **Professional Operations Interface**: Built specifically for facility and gate monitoring without marketing fluff or placeholder jargon.
- **Unified Single-Server Architecture**: FastAPI simultaneously serves the client application (HTML, CSS, JS) and the computer vision inference endpoints (`/analyze-parking`, `/api/health`, `/api/samples`).
- **Flexible Ingest Pipeline**:
  - **File Ingest**: Drag-and-drop or file upload for surveillance videos (`.mp4`, `.avi`, `.mov`) and images (`.jpg`, `.png`).
  - **Quick Demo Samples**: 1-click test feeds from gate captures for instant demonstration and verification.
  - **Camera Feed (Webcam)**: Direct browser camera stream capture for real-time testing.
- **Operational Directives**:
  - **Kitchen Batch Staging**: Real-time dining alerts triggered when heavy buses/coaches arrive.
  - **Floor Staffing Allocation**: High-density alerts when vehicle volumes exceed thresholds.
  - **Facility Occupancy Gauge**: Live utilization meter.
- **License Plate Registry & Audit Ledger**:
  - Cropped high-contrast plate thumbnails.
  - Registration number badges formatted for Indian RTO standards (`enforce_indian_plate_format`).
  - Instant plate search and live filtering.
  - One-click plate copying and **CSV Manifest Export**.
- **Surveillance Viewport Controls**:
  - Interactive pan & zoom (`+`, `-`, `Reset`).
  - Fullscreen monitoring mode.
  - One-click export of annotated media.
- **Theme Support**: Seamless toggle between Dark Slate Command Center and Enterprise Light modes.

---

## Directory Structure

```
smart_traffic/
├── app.py                     # Unified FastAPI backend & static server
├── run_app.py                 # Dedicated launcher script
├── license_plate_detector.pt  # Custom YOLOv8 plate detection weights
├── yolov8n.pt                 # YOLOv8 vehicle detection model
├── static/
│   ├── index.html             # Clean, semantic HTML5 operations dashboard
│   ├── css/
│   │   └── styles.css         # Modern, responsive design system & themes
│   ├── js/
│   │   └── app.js             # Telemetry, ingest, zoom, export & API client
│   └── samples/               # Pre-loaded sample gate captures for quick demo
└── screen.py                  # (Legacy Streamlit interface, preserved as backup)
```

---

## Launch Instructions

To launch the unified web console and backend server:

```powershell
# Using the virtual environment python:
.\smart\Scripts\python.exe app.py
```

Or using `run_app.py`:

```powershell
.\smart\Scripts\python.exe run_app.py
```

Then open your browser at:
- **Web Operations Console**: [http://localhost:8000](http://localhost:8000)
- **Interactive OpenAPI Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
