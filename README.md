# HoloForge

> Real-time hand-tracked augmented reality projection, built one week at a time.

---

## Tech stack

| Component | Package | Purpose |
|-----------|---------|---------|
| Runtime | Python 3.10+ | Language and standard library |
| Vision | `mediapipe==0.10.21` | Hand landmark detection |
| Vision | `opencv-python>=4.8.0` | Camera capture and frame streaming |
| Numerics | `numpy>=1.24.0` | Array math and coordinate transforms |
| Rendering | `pygame>=2.5.0` | 2D/3D projection visualization |
| 3D assets | `trimesh>=4.0.0` | Mesh loading and scene geometry |
| Voice | `SpeechRecognition>=3.10.0` | Command utterance recognition |
| Linting | `flake8` | Style enforcement (`max-line-length=100`) |
| Testing | `pytest` | Unit and smoke tests |

---

## Installation

1. Clone the repository and change into the project root.
2. Run the Windows setup script (it creates the virtual environment, installs
   dependencies and downloads the MediaPipe model bundle):

   ```bat
   scripts\setup_windows.bat
   ```

3. Activate the environment and launch the application:

   ```bat
   call .venv\Scripts\activate.bat
   python -m src.main
   ```

To install dependencies manually instead of using the setup script:

```bat
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

The MediaPipe model bundle is large, so it is downloaded separately rather than
committed to the repository.

---

## Project tree

```text
holoforge/
├── models/                  # Downloaded MediaPipe task bundles
│   └── .gitkeep
├── src/
│   ├── __init__.py          # Package metadata and version
│   ├── main.py              # Logging-based CLI entry point
│   ├── vision/              # Camera capture + hand landmark detection
│   │   ├── __init__.py
│   │   └── hand_tracker.py  # Landmarker, pinch grab, palm smoothing, two-hand stats
│   ├── rendering/           # 3D projection + wireframe rendering
│   │   ├── __init__.py
│   │   ├── projection_3d.py    # Pinhole projection, rotation, cube geometry
│   │   └── wireframe_renderer.py  # Single-pass OpenCV edge drawing
│   ├── interaction/         # Gesture-driven model interaction
│   │   ├── __init__.py
│   │   └── gesture_controller.py  # Grab-rotate, spin-and-catch, two-hand zoom/pan
│   ├── analysis/            # Latency and accuracy profiling
│   │   ├── __init__.py
│   │   └── model_analyzer.py
│   └── utils/               # Shared helpers
│       ├── __init__.py
│       └── helpers.py
├── tests/
│   ├── __init__.py
│   ├── test_placeholder.py
│   ├── test_projection.py      # Geometry: rotation, projection, cube
│   └── test_gesture_controller.py  # State machine: grab, spin, catch, hysteresis
├── assets/models/           # Imported .stl / .obj mesh files
│   └── .gitkeep
├── output/                  # Rendered frames and profiling reports
│   └── .gitkeep
├── scripts/
│   └── setup_windows.bat
├── requirements.txt
├── setup.cfg
├── .gitignore
└── README.md
```

---

## Roadmap

| Week | Goal | Deliverables |
|------|------|--------------|
| **1** | Capture a live camera feed and detect hand landmarks | `HandTracker` wired to MediaPipe `HandLandmarker`, live preview window, pinch grab + palm smoothing |
| **2** | Project landmarks into a 3D scene and drive them with gestures | `rotation_matrix` / `make_cube` / `project_points`, `WireframeRenderer`, `GestureController` (grab-rotate, spin-and-catch, two-hand zoom/pan), keyboard fallback, unit tests |
| **3** | Recognize gestures and commands | Gesture vocabulary, `ModelAnalyzer` accuracy metrics |
| **4** | Polish and integrate | Voice commands, profiling dashboard, README walkthrough |

Status: **Day 10** - Week 1 and Week 2 implemented. The 3D wireframe engine
and gesture interaction layer are complete and covered by unit tests; Weeks 3
and 4 are not yet started.

---

## Controls

| Action | Input | Behaviour |
|--------|-------|-----------|
| Grab-rotate | Pinch and drag one hand | Model rotates with the palm; velocity is set from the frame-to-frame palm delta |
| Release-spin | Open the hand while dragging | Model keeps spinning; velocity decays frame-rate-independently with time constant `TAU` |
| Catch | Re-pinch a spinning model | Velocity is re-sourced from the current hand motion so the model follows the hand immediately |
| Two-hand zoom | Move two hands apart / together | Model scale tracks the palm-span ratio, clamped to `0.5x`-`3.0x` |
| Two-hand pan | Drag two hands together | Model pans with the palm midpoint |
| Keyboard fallback | Arrow keys | Rotate `5` degrees per press |
| Keyboard fallback | `+` / `-` | Scale up / down by `0.1`, clamped to `0.5x`-`3.0x` |
| Reset | `r` | Return rotation, scale and offset to their defaults |
| Exit | `ESC` | Close the window |

---

## Author

Suvam