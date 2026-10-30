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
│   │   └── hand_tracker.py
│   ├── rendering/           # 3D projection + pygame visualization
│   │   ├── __init__.py
│   │   └── projection_3d.py
│   ├── analysis/            # Latency and accuracy profiling
│   │   ├── __init__.py
│   │   └── model_analyzer.py
│   └── utils/               # Shared helpers
│       ├── __init__.py
│       └── helpers.py
├── tests/
│   ├── __init__.py
│   └── test_placeholder.py
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
| **1** | Capture a live camera feed and detect hand landmarks | `HandTracker.start()` wired to MediaPipe, live preview window |
| **2** | Project landmarks into a 3D scene | `Projection3D` matrix chain, pygame rendering surface |
| **3** | Recognize gestures and commands | Gesture vocabulary, `ModelAnalyzer` accuracy metrics |
| **4** | Polish and integrate | Voice commands, profiling dashboard, README walkthrough |

Status: **Day 1** - project scaffolding complete. All modules are stubs that
raise `NotImplementedError` until the corresponding week's work is wired up.

---

## Author

Suvam