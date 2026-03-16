# Driver Monitoring System (AIS-184 Compliant)

A real-time driver drowsiness and distraction detection system built to comply with **AIS-184** (Indian automotive safety standard). It uses a YOLOX ONNX model for inference, a temporal smoothing engine for stable detections, and a multi-level alert system (buzzer, LED, SMS, call).

---

## Project Structure

```
.
├── new_config.yaml            # Main configuration file (all tunable parameters)
├── new_detection_logic.py     # Core detection + AIS-184 state machine
├── new_trigger_alert.py       # Alert actions (beep, LED, SMS, call)
├── check_electrical_system.py # Pre-startup hardware validation
├── test_integration.py        # Full integration test runner
├── test_ml_pipeline.py        # ML pipeline simulation test
├── vehicle_speed_log.txt      # Vehicle speed input (CAN bus / mock)
├── log_detection.txt          # ML output → detection state
├── log_detection_confidence.txt # ML confidence score
├── ais_184_compliance_log.txt # AIS-184 event log
└── ml/
    ├── detect.py              # ONNX inference + camera pipeline
    ├── yolox_driver_dash.onnx # Trained YOLOX model
    ├── requirements.txt       # Python dependencies
    └── venv/                  # Virtual environment (create manually)
```

---

## System Flow

```
Camera → YOLOX ONNX Inference (ml/detect.py)
       → Majority-vote batch writer → log_detection.txt
       → Detection Logic (new_detection_logic.py)
       → AIS-184 State Machine (speed check, ignition, failure detection)
       → Alert Trigger (new_trigger_alert.py)
       → Buzzer / LED / SMS / Call
```

---

## Prerequisites

- Python 3.8+
- Camera (USB or CSI)
- On Linux/EdgeAI device: GStreamer, RPi.GPIO, pyserial (for GSM/GPS)
- On Windows: runs in test/simulation mode automatically

---

## Step-by-Step Setup

### Step 1 — Clone and enter the project

```bash
git clone <your-repo-url>
cd <project-folder>
```

### Step 2 — Create and activate a virtual environment

```bash
# Create venv inside ml/ folder
python -m venv ml/venv

# Activate (Linux/macOS)
source ml/venv/bin/activate

# Activate (Windows)
ml\venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install -r ml/requirements.txt
```

Core packages installed:

- `onnxruntime` — ONNX model inference
- `opencv-python` — camera capture and frame processing
- `numpy` — numerical operations
- `PyYAML` — config file parsing

### Step 4 — Verify the ONNX model exists

```bash
# The model should already be present at:
ls ml/yolox_driver_dash.onnx
```

If missing, place your trained `.onnx` model at `ml/yolox_driver_dash.onnx`.

### Step 5 — Configure the system

Open `new_config.yaml` and adjust key settings:

```yaml
config:
  vehicle_speed_threshold: 65 # Speed (km/h) to activate monitoring

  ais_184:
    activation:
      vehicle_category: "M1" # M1=passenger, M2/M3=bus, N2/N3=truck
      # Category thresholds: M1=70 km/h, M2/M3/N2/N3=60 km/h

  ml_pipeline:
    frames_per_second: 200 # Inference frames collected per batch
    write_interval_sec: 1 # How often majority result is written to log

flow: [config2, detection_config, alert_config1] # Active config profile
```

---

## Running the System

### Option A — Full system with real camera (recommended for production)

**Step 1:** Run the pre-startup electrical check

```bash
python check_electrical_system.py
```

This validates camera, GPIO, and CAN interface. The system will not start if any component fails.

**Step 2:** Run the full integration (camera + detection logic)

```bash
python test_integration.py
```

This will:

1. Run the electrical check automatically
2. Start ML inference via camera
3. Start detection logic with AIS-184 state machine
4. Trigger alerts based on detections

To skip the electrical check (not recommended):

```bash
python test_integration.py --skip-electrical-check
```

---

### Option B — ML pipeline mode (no camera required, for testing)

Simulates 200 fps inference using majority-vote batching, no camera needed:

```bash
python test_integration.py --ml-pipeline
```

With custom duration:

```bash
python test_integration.py --ml-pipeline --duration 120
```

---

### Option C — Run ML inference standalone

Run inference on a single image:

```bash
python ml/detect.py --model ml/yolox_driver_dash.onnx --image path/to/image.jpg
```

Run inference with live camera and write detection logs:

```bash
python ml/detect.py --model ml/yolox_driver_dash.onnx --camera --log-path log_detection.txt
```

Run inference with a specific camera ID:

```bash
python ml/detect.py --model ml/yolox_driver_dash.onnx --camera --camera-id 1 --log-path log_detection.txt
```

Run inference with custom confidence threshold:

```bash
python ml/detect.py --model ml/yolox_driver_dash.onnx --camera --conf-threshold 0.7 --log-path log_detection.txt
```

---

### Option D — ML pipeline simulation test

Test the majority-vote ML pipeline with different detection scenarios:

```bash
# Default scenario (eyes_closed)
python test_ml_pipeline.py

# Specific scenario
python test_ml_pipeline.py --scenario eyes_closed
python test_ml_pipeline.py --scenario distracted

# Custom fps, duration, and vehicle speed
python test_ml_pipeline.py --scenario eyes_closed --fps 200 --duration 10 --speed 75

# ML producer only (skip detection logic)
python test_ml_pipeline.py --no-detection-logic
```

Available scenarios: `awake`, `eyes_closed`, `yawn`, `distracted`, `phone`, `smoking`, `mixed`, `noisy`

---

### Option E — Run detection logic standalone

If `log_detection.txt` is already being written by the ML pipeline:

```bash
python new_detection_logic.py
```

```

### Set a test vehicle speed

```bash
python test/set_test_speed.py
```

---

## Monitoring Logs in Real Time

# Windows
Get-Content log_detection.txt -Wait

# Linux
tail -f log_detection.txt
tail -f ais_184_compliance_log.txt
```

---

## Key Log Files

| File                           | Description                                                                  |
| ------------------------------ | ---------------------------------------------------------------------------- |
| `log_detection.txt`            | Current detection state (`awake`, `eyes_closed`, `yawn`, `distracted`, etc.) |
| `log_detection_confidence.txt` | ML confidence score (0.0 – 1.0)                                              |
| `ais_184_compliance_log.txt`   | AIS-184 compliance events (JSON lines)                                       |
| `vehicle_speed_log.txt`        | Current vehicle speed in km/h                                                |

---

## Detection States

| State                            | Meaning                               |
| -------------------------------- | ------------------------------------- |
| `awake`                          | Driver is alert                       |
| `eyes_closed`                    | Driver's eyes are closed (drowsiness) |
| `distracted`                     | Driver is distracted                  |
| `no_detection`                   | No face detected in frame             |
| `electrical_failure:<component>` | Hardware failure detected             |

---

## Alert Levels

| Level  | Trigger              | Actions                              |
| ------ | -------------------- | ------------------------------------ |
| Low    | Short duration event | Beep + LED                           |
| Medium | Sustained event      | Beep + SMS                           |
| High   | Critical / prolonged | Beep + SMS + Call (repeats every 3s) |

---

## AIS-184 Compliance Features

- Ignition self-test: warning light activates for 2 seconds on ignition ON
- Speed-based activation: M1 vehicles at 70 km/h, M2/M3/N2/N3 at 60 km/h
- 30-second deactivation delay when speed drops below threshold
- Electrical failure detection with constant warning light
- Sensor obstruction detection with continuous buzzer
- All events logged to `ais_184_compliance_log.txt`

---

## Troubleshooting

**Camera not detected**

```bash
python check_electrical_system.py
# Check camera_id in new_config.yaml (default: 0)
```

**Model not found**

```bash
ls ml/yolox_driver_dash.onnx
# Ensure the ONNX model is placed at ml/yolox_driver_dash.onnx
```

**No alerts triggering**

- Check `vehicle_speed_log.txt` — speed must be above threshold (70 km/h for M1)
- Check `log_detection.txt` — must show a non-awake state
- Check `new_config.yaml` — verify `flow` points to correct config profile

**GStreamer / audio errors on Windows**

- Expected — the system automatically falls back to `winsound` beeps on Windows
- Full audio (GStreamer) is only available on Linux/EdgeAI devices

**GSM / GPS not available**

- Expected on Windows — SMS and call features require a GSM module on `/dev/ttyUSB2` (Linux only)
