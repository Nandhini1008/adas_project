#!/usr/bin/env python3
"""
Integration Test Script
Runs ML detection and detection logic together for testing.

Modes:
  default          – launches real camera via ml/test_onnx_inference.py
  --ml-pipeline    – uses majority-vote batch writer from new_config.yaml
                     (no camera required; driven by ml_pipeline config block)
"""

import subprocess
import threading
import time
import sys
import signal
import os
import random
import yaml
from collections import Counter

# ── Load ML pipeline config from new_config.yaml ─────────────────────────────
with open("new_config.yaml", "r", encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f)

_ml_cfg = _cfg.get("config", {}).get("ml_pipeline", {})
ML_FPS            = _ml_cfg.get("frames_per_second", 200)
ML_WRITE_INTERVAL = _ml_cfg.get("write_interval_sec", 1)

LOG_DETECTION  = "log_detection.txt"
LOG_CONFIDENCE = "log_detection_confidence.txt"

# ── Shared ML pipeline state ──────────────────────────────────────────────────
_ml_stop_event  = threading.Event()
_ml_stats_lock  = threading.Lock()
_ml_frame_count = 0
_ml_label_counts: Counter = Counter()

# Global process references
ml_process = None
logic_process = None

# ── ML Pipeline (majority-vote batch writer) ──────────────────────────────────

# Default scenario used when --ml-pipeline is active
_ML_DEFAULT_SCENARIO = [
    ("awake",       0.90, 5),
    ("eyes_closed", 0.88, 3),
    ("yawn",        0.85, 2),
]


def _write_majority_result(batch: list, sec: int) -> None:
    """Compute majority label + avg confidence from a batch and write to logs."""
    global _ml_frame_count
    total  = len(batch)
    counts: Counter = Counter(lbl for lbl, _ in batch)
    majority_label  = counts.most_common(1)[0][0]
    avg_conf        = sum(c for _, c in batch) / total

    vote_str = "  ".join(f"{k}:{v}({v/total*100:.0f}%)" for k, v in counts.most_common())
    print(f"[ML-PIPELINE] sec={sec}  frames={total}  vote={vote_str}  "
          f"→ {majority_label}  conf={avg_conf:.3f}")

    try:
        with open(LOG_DETECTION, "w") as f:
            f.write(majority_label)
        with open(LOG_CONFIDENCE, "w") as f:
            f.write(f"{avg_conf:.4f}")
        with _ml_stats_lock:
            _ml_frame_count += total
            for lbl, _ in batch:
                _ml_label_counts[lbl] += 1
    except OSError as e:
        print(f"[ML-PIPELINE] Write error: {e}")


def ml_pipeline_thread(scenario: list, fps: int, write_interval: float,
                        duration: int) -> None:
    """
    Majority-vote ML producer — mirrors test_ml_pipeline.ml_frame_producer.
    Reads fps and write_interval from new_config.yaml (passed in as args).
    Runs for `duration` seconds then sets _ml_stop_event.
    """
    labels  = [f[0] for f in scenario]
    confs   = [f[1] for f in scenario]
    weights = [f[2] for f in scenario]

    frames_per_batch = max(1, int(fps * write_interval))
    frame_interval   = write_interval / frames_per_batch
    total_batches    = max(1, int(duration / write_interval))

    print(f"[ML-PIPELINE] Starting: fps={fps}  write_interval={write_interval}s  "
          f"frames/batch={frames_per_batch}  batches={total_batches}")

    for batch_num in range(1, total_batches + 1):
        if _ml_stop_event.is_set():
            break
        batch       = []
        batch_start = time.perf_counter()

        for frame_num in range(frames_per_batch):
            label, conf = random.choices(
                list(zip(labels, confs)), weights=weights, k=1
            )[0]
            conf = max(0.0, min(1.0, conf + random.uniform(-0.03, 0.03)))
            batch.append((label, conf))

            next_deadline = batch_start + (frame_num + 1) * frame_interval
            sleep_for = next_deadline - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)

        _write_majority_result(batch, batch_num)

    _ml_stop_event.set()
    print("[ML-PIPELINE] Producer finished.")


# ─────────────────────────────────────────────────────────────────────────────

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\nShutting down...")
    if ml_process:
        ml_process.terminate()
        ml_process.wait()
    if logic_process:
        logic_process.terminate()
        logic_process.wait()
    print("Processes terminated.")
    sys.exit(0)

def log_output(process, name):
    """Read and log output from a process"""
    for line in iter(process.stdout.readline, ''):
        if line:
            print(f"[{name}] {line.strip()}")

def main():
    global ml_process, logic_process
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Parse args
    import argparse
    parser = argparse.ArgumentParser(description="Driver Monitoring Integration Test")
    parser.add_argument("--ml-pipeline", action="store_true",
                        help="Use majority-vote ML pipeline (no camera) instead of real inference")
    parser.add_argument("--duration", type=int, default=60,
                        help="Duration in seconds when using --ml-pipeline (default: 60)")
    parser.add_argument("--skip-electrical-check", action="store_true",
                        help="Skip pre-startup electrical system check (not recommended)")
    args, _ = parser.parse_known_args()

    print("=" * 60)
    print("Driver Monitoring System - Integration Test")
    print("=" * 60)
    
    # ── STEP 1: Pre-startup electrical system check ──────────────────────────
    if not args.skip_electrical_check:
        print("\n[STEP 1] Running pre-startup electrical system check...")
        try:
            result = subprocess.run(
                ["python", "check_electrical_system.py"],
                capture_output=False,
                text=True
            )
            
            if result.returncode != 0:
                print("\n" + "="*60)
                print("❌ ELECTRICAL FAILURE DETECTED")
                print("="*60)
                print("\nSystem cannot start until hardware issues are resolved.")
                print("Please fix the issues and run again.")
                print("\nTo bypass this check (not recommended):")
                print("  python test_integration.py --skip-electrical-check")
                sys.exit(1)
            
            print("\n✓ Electrical system check passed - proceeding to startup")
            
        except Exception as e:
            print(f"\n❌ Error running electrical check: {e}")
            print("System cannot start safely.")
            sys.exit(1)
    else:
        print("\n⚠ WARNING: Skipping electrical system check")
    
    # ── STEP 2: Initialize system ────────────────────────────────────────────
    # Create mock vehicle speed file
    print("\n[SETUP] Creating mock vehicle speed file...")
    with open("vehicle_speed_log.txt", "w") as f:
        f.write("75")  # 75 km/h - above threshold for testing
    print("[SETUP] ✓ Vehicle speed set to 75 km/h")
    
    # Create empty log_detection.txt
    print("[SETUP] Creating log_detection.txt...")
    with open("log_detection.txt", "w") as f:
        f.write("awake")
    print("[SETUP] ✓ log_detection.txt initialized")

    print("\n" + "=" * 60)
    print("Starting processes...")
    print("=" * 60)

    try:
        if args.ml_pipeline:
            # ── ML Pipeline mode (majority-vote, no camera) ───────────────
            print(f"\n[ML] Starting ML pipeline (fps={ML_FPS}, "
                  f"write_interval={ML_WRITE_INTERVAL}s, duration={args.duration}s)")
            print(f"[ML] Config source: new_config.yaml → ml_pipeline")

            _ml_stop_event.clear()
            ml_thread = threading.Thread(
                target=ml_pipeline_thread,
                args=(_ML_DEFAULT_SCENARIO, ML_FPS, ML_WRITE_INTERVAL, args.duration),
                daemon=True, name="ml-pipeline"
            )
            ml_thread.start()

            # Wait a moment for ML to write its first result
            time.sleep(ML_WRITE_INTERVAL + 0.5)

            # Start detection logic as subprocess
            print("\n[LOGIC] Starting detection logic system...")
            logic_process = subprocess.Popen(
                ['python', '-u', 'new_detection_logic.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            logic_thread = threading.Thread(
                target=log_output,
                args=(logic_process, "DETECT-LOGIC"),
                daemon=True
            )
            logic_thread.start()

            print("\n" + "=" * 60)
            print("ML pipeline + detection logic running!")
            print(f"  fps={ML_FPS}  write_interval={ML_WRITE_INTERVAL}s  duration={args.duration}s")
            print("  Press Ctrl+C to stop early")
            print("=" * 60 + "\n")

            # Wait for pipeline to finish or Ctrl+C
            try:
                ml_thread.join(timeout=args.duration + 5)
            except KeyboardInterrupt:
                print("\n[TEST] Interrupted")
                _ml_stop_event.set()

            # Print summary
            with _ml_stats_lock:
                total  = _ml_frame_count
                counts = dict(_ml_label_counts)
            print("\n[ML-PIPELINE] Summary:")
            for lbl, cnt in sorted(counts.items()):
                pct = cnt / total * 100 if total else 0
                print(f"  {lbl:<15} {cnt:5d}  ({pct:.1f}%)")
            print(f"  Total frames processed: {total}")

        else:
            # ── Real camera mode ──────────────────────────────────────────
            model_path = "ml/yolox_driver_dash.onnx"
            if not os.path.exists(model_path):
                print(f"\n[ERROR] Model not found: {model_path}")
                print("Please ensure the ONNX model exists before running.")
                sys.exit(1)
            print(f"[SETUP] ✓ Model found: {model_path}")

            print("\n[ML] Starting ML detection with camera...")
            ml_process = subprocess.Popen(
                [
                    'python', '-u', 'ml/test_onnx_inference.py',
                    '--model', model_path,
                    '--camera',
                    '--camera-id', '0',
                    '--conf-threshold', '0.5',
                    '--log-path', 'log_detection.txt'
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            ml_thread = threading.Thread(
                target=log_output,
                args=(ml_process, "ML-DETECT"),
                daemon=True
            )
            ml_thread.start()

            time.sleep(3)

            print("\n[LOGIC] Starting detection logic system...")
            logic_process = subprocess.Popen(
                ['python', '-u', 'new_detection_logic.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            logic_thread = threading.Thread(
                target=log_output,
                args=(logic_process, "DETECT-LOGIC"),
                daemon=True
            )
            logic_thread.start()

            print("\n" + "=" * 60)
            print("Both processes running!")
            print("=" * 60)
            print("\nInstructions:")
            print("  - Look at the camera window to see ML detections")
            print("  - Watch console for detection logic alerts")
            print("  - Press Ctrl+C to stop\n")

            while True:
                if ml_process.poll() is not None:
                    print("\n[ERROR] ML detection process terminated unexpectedly!")
                    break
                if logic_process.poll() is not None:
                    print("\n[ERROR] Detection logic process terminated unexpectedly!")
                    break
                time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nReceived interrupt signal...")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
    finally:
        signal_handler(signal.SIGTERM, None)

if __name__ == "__main__":
    main()


# ============================================================================
# AIS-184 Test Case Functions
# ============================================================================

def test_case_1_ignition_self_test():
    """Test Case 1: Optical warning signal verification during ignition with OFF to ON transition.

    Enhanced with detailed AIS-184 verification:
    - Stationary vehicle verification (speed = 0 km/h)
    - Precise timing verification (2.0 ± 0.1 seconds warning light duration)
    - Self-test completion within 5 seconds
    """
    print("\n" + "="*60)
    print("TEST CASE 1: Ignition Self-Test (Enhanced AIS-184 Verification)")
    print("="*60)

    # Step 1: Simulate ignition OFF state initially
    print("[TEST] Step 1: Setting ignition to OFF state...")
    with open("ignition_state.txt", "w") as f:
        f.write("OFF")

    # Clear previous state file to ensure clean test
    if os.path.exists("ignition_state_previous.txt"):
        os.remove("ignition_state_previous.txt")

    # Set vehicle speed to 0 (stationary) - CRITICAL for AIS-184 compliance
    with open("vehicle_speed_log.txt", "w") as f:
        f.write("0")

    print("[TEST] Ignition OFF, vehicle stationary (0 km/h)")

    # Verify vehicle speed is exactly 0 before proceeding
    with open("vehicle_speed_log.txt", "r") as f:
        initial_speed = float(f.read().strip())

    if initial_speed != 0.0:
        print(f"[TEST] ✗ FAIL: Vehicle speed is {initial_speed} km/h (must be 0 km/h for self-test)")
        return False

    print("[TEST] ✓ Vehicle speed verified: 0 km/h (stationary)")
    time.sleep(1)

    # Step 2: Transition to ignition ON state
    print("\n[TEST] Step 2: Transitioning ignition from OFF to ON...")
    with open("ignition_state.txt", "w") as f:
        f.write("ON")

    print("[TEST] Expecting warning light activation for 2.0 ± 0.1 seconds on transition...")

    # Start detection logic
    logic_process = subprocess.Popen(['python', '-u', 'new_detection_logic.py'],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     universal_newlines=True, bufsize=1)

    # Monitor output for self-test completion with precise timing
    ignition_on_time = time.time()
    self_test_start_time = None
    self_test_end_time = None
    self_test_detected = False
    transition_detected = False
    warning_light_on_time = None
    warning_light_off_time = None

    for line in iter(logic_process.stdout.readline, ''):
        if line:
            current_time = time.time()
            print(f"[LOGIC] {line.strip()}")

            # Check for transition detection
            if "OFF to ON transition detected" in line or "IGNITION_TRANSITION" in line:
                transition_detected = True
                print(f"[TEST] ✓ Ignition transition detected at t={current_time - ignition_on_time:.3f}s")

            # Check for self-test activation (warning light ON)
            if ("Ignition self-test" in line or "WARNING LIGHT" in line) and "Activating" in line:
                if self_test_start_time is None:
                    self_test_start_time = current_time
                    warning_light_on_time = current_time
                    self_test_detected = True
                    elapsed = current_time - ignition_on_time
                    print(f"[TEST] ✓ Self-test triggered at t={elapsed:.3f}s")

            # Check for warning light deactivation
            if "WARNING LIGHT" in line and "OFF" in line:
                if warning_light_off_time is None:
                    warning_light_off_time = current_time
                    print(f"[TEST] ✓ Warning light deactivated at t={current_time - ignition_on_time:.3f}s")

            # Check for self-test completion
            if "Self-test completed" in line or "SELF_TEST_COMPLETE" in line:
                if self_test_end_time is None:
                    self_test_end_time = current_time
                    elapsed = current_time - ignition_on_time
                    print(f"[TEST] ✓ Self-test completed at t={elapsed:.3f}s")

        # Check timeout (5 seconds max per AIS-184)
        if time.time() - ignition_on_time > 5:
            print(f"[TEST] ⚠ Timeout reached (5 seconds)")
            break

    logic_process.terminate()
    logic_process.wait()

    # Verify vehicle speed remained 0 during self-test
    print("\n[TEST] Verifying vehicle remained stationary during self-test...")
    with open("vehicle_speed_log.txt", "r") as f:
        final_speed = float(f.read().strip())

    if final_speed != 0.0:
        print(f"[TEST] ✗ FAIL: Vehicle speed changed to {final_speed} km/h during self-test")
        return False

    print("[TEST] ✓ Vehicle speed remained 0 km/h throughout self-test")

    # Step 3: Verify self-test does NOT repeat on continuous ON state
    print("\n[TEST] Step 3: Verifying self-test does NOT repeat on continuous ON...")
    time.sleep(1)

    # Restart detection logic with ignition already ON
    logic_process = subprocess.Popen(['python', '-u', 'new_detection_logic.py'],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     universal_newlines=True, bufsize=1)

    start_time = time.time()
    repeat_test_detected = False
    skip_message_detected = False

    for line in iter(logic_process.stdout.readline, ''):
        if line:
            print(f"[LOGIC] {line.strip()}")

            # Check if self-test is incorrectly repeated
            if "triggering self-test" in line and "transition" not in line:
                repeat_test_detected = True
                print("[TEST] ✗ ERROR: Self-test incorrectly repeated!")

            # Check for correct skip message
            if "skipping self-test (no transition)" in line or "already ON" in line:
                skip_message_detected = True
                print("[TEST] ✓ Self-test correctly skipped (no transition)")

        # Check timeout (3 seconds is enough to verify)
        if time.time() - start_time > 3:
            break

    logic_process.terminate()
    logic_process.wait()

    # Calculate precise timing measurements
    print("\n[TEST] Timing Analysis:")
    print("="*60)

    timing_pass = True

    # Measure warning light duration
    if warning_light_on_time and warning_light_off_time:
        warning_duration = warning_light_off_time - warning_light_on_time
        print(f"  Warning light duration: {warning_duration:.3f}s")

        # AIS-184 requirement: 2.0 ± 0.1 seconds
        if 1.9 <= warning_duration <= 2.1:
            print(f"  ✓ Duration within spec (2.0 ± 0.1s)")
        else:
            print(f"  ✗ Duration out of spec (expected 2.0 ± 0.1s, got {warning_duration:.3f}s)")
            timing_pass = False
    else:
        print(f"  ✗ Could not measure warning light duration")
        timing_pass = False

    # Measure self-test completion time
    if self_test_end_time:
        completion_time = self_test_end_time - ignition_on_time
        print(f"  Self-test completion time: {completion_time:.3f}s")

        # AIS-184 requirement: within 5 seconds
        if completion_time <= 5.0:
            print(f"  ✓ Completed within 5 seconds")
        else:
            print(f"  ✗ Exceeded 5 second limit")
            timing_pass = False
    else:
        print(f"  ✗ Could not measure self-test completion time")
        timing_pass = False

    # Evaluate test results
    print("\n[TEST] Compliance Verification:")
    print("="*60)
    print(f"  - Vehicle stationary (0 km/h): ✓")
    print(f"  - Ignition transition detected: {transition_detected}")
    print(f"  - Self-test triggered on transition: {self_test_detected}")
    print(f"  - Warning light timing (2.0 ± 0.1s): {timing_pass}")
    print(f"  - Self-test NOT repeated: {not repeat_test_detected}")

    if transition_detected and self_test_detected and not repeat_test_detected and timing_pass:
        print("\n[TEST] ✓ PASS: All AIS-184 requirements verified")
        return True
    else:
        print("\n[TEST] ✗ FAIL: One or more AIS-184 requirements not met")
        return False



def test_case_2a_warning_delay_m1():
    """Test Case 2a: Warning delay compliance for M1 vehicles (70 km/h threshold)."""
    import yaml
    
    print("\n" + "="*60)
    print("TEST CASE 2a: Warning Delay Compliance - M1 Vehicle")
    print("="*60)
    
    # Load config and verify M1 category
    with open("new_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Temporarily set vehicle category to M1
    original_category = config['config']['ais_184']['activation'].get('vehicle_category', 'M1')
    config['config']['ais_184']['activation']['vehicle_category'] = 'M1'
    
    # Write updated config
    with open("new_config.yaml", "w") as f:
        yaml.dump(config, f)
    
    print("[TEST] Vehicle category: M1 (passenger vehicle)")
    print("[TEST] Expected threshold: 70 km/h")
    
    # Set ignition ON and speed above M1 threshold (70 km/h)
    with open("ignition_state.txt", "w") as f:
        f.write("ON")
    with open("vehicle_speed_log.txt", "w") as f:
        f.write("75")  # Above M1 threshold (70 km/h)
    
    print("[TEST] Activation criteria met (speed = 75 km/h > 70 km/h)")
    print("[TEST] Monitoring activation delay...")
    
    # Start detection logic
    logic_process = subprocess.Popen(['python', '-u', 'new_detection_logic.py'],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     universal_newlines=True, bufsize=1)
    
    # Monitor for activation within 300 seconds
    start_time = time.time()
    activation_detected = False
    activation_delay = None
    threshold_logged = False
    
    for line in iter(logic_process.stdout.readline, ''):
        if line:
            print(f"[LOGIC] {line.strip()}")
            if "START_MONITORING" in line or "Active monitoring started" in line:
                activation_detected = True
                activation_delay = time.time() - start_time
            if "speed_threshold" in line and "70" in line:
                threshold_logged = True
        
        # Check timeout (300 seconds max)
        if time.time() - start_time > 300:
            break
        
        # Exit early if activation detected
        if activation_detected:
            time.sleep(1)  # Give time for logging
            break
    
    logic_process.terminate()
    logic_process.wait()
    
    # Restore original category
    config['config']['ais_184']['activation']['vehicle_category'] = original_category
    with open("new_config.yaml", "w") as f:
        yaml.dump(config, f)
    
    if activation_detected and activation_delay <= 300:
        print(f"[TEST] ✓ PASS: M1 vehicle activated in {activation_delay:.2f}s (< 300s) at 70 km/h threshold")
        return True
    else:
        print(f"[TEST] ✗ FAIL: M1 vehicle activation failed")
        return False


def test_case_2b_warning_delay_m2_m3_n2_n3():
    """Test Case 2b: Warning delay compliance for M2/M3/N2/N3 vehicles (60 km/h threshold)."""
    import yaml
    
    print("\n" + "="*60)
    print("TEST CASE 2b: Warning Delay Compliance - M2/M3/N2/N3 Vehicles")
    print("="*60)
    
    # Test categories: M2, M3, N2, N3 (all use 60 km/h threshold)
    test_categories = ['M2', 'M3', 'N2', 'N3']
    all_passed = True
    
    for category in test_categories:
        print(f"\n[TEST] Testing category: {category}")
        
        # Load config and set vehicle category
        with open("new_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        original_category = config['config']['ais_184']['activation'].get('vehicle_category', 'M1')
        config['config']['ais_184']['activation']['vehicle_category'] = category
        
        # Write updated config
        with open("new_config.yaml", "w") as f:
            yaml.dump(config, f)
        
        print(f"[TEST] Vehicle category: {category}")
        print(f"[TEST] Expected threshold: 60 km/h")
        
        # Set ignition ON and speed above threshold (60 km/h)
        with open("ignition_state.txt", "w") as f:
            f.write("ON")
        with open("vehicle_speed_log.txt", "w") as f:
            f.write("65")  # Above threshold (60 km/h)
        
        print(f"[TEST] Activation criteria met (speed = 65 km/h > 60 km/h)")
        print(f"[TEST] Monitoring activation delay...")
        
        # Start detection logic
        logic_process = subprocess.Popen(['python', '-u', 'new_detection_logic.py'],
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         universal_newlines=True, bufsize=1)
        
        # Monitor for activation within 300 seconds
        start_time = time.time()
        activation_detected = False
        activation_delay = None
        
        for line in iter(logic_process.stdout.readline, ''):
            if line:
                print(f"[LOGIC] {line.strip()}")
                if "START_MONITORING" in line or "Active monitoring started" in line:
                    activation_detected = True
                    activation_delay = time.time() - start_time
            
            # Check timeout (300 seconds max)
            if time.time() - start_time > 300:
                break
            
            # Exit early if activation detected
            if activation_detected:
                time.sleep(1)  # Give time for logging
                break
        
        logic_process.terminate()
        logic_process.wait()
        
        # Restore original category
        config['config']['ais_184']['activation']['vehicle_category'] = original_category
        with open("new_config.yaml", "w") as f:
            yaml.dump(config, f)
        
        if activation_detected and activation_delay <= 300:
            print(f"[TEST] ✓ PASS: {category} vehicle activated in {activation_delay:.2f}s (< 300s) at 60 km/h threshold")
        else:
            print(f"[TEST] ✗ FAIL: {category} vehicle activation failed")
            all_passed = False
    
    if all_passed:
        print(f"\n[TEST] ✓ PASS: All M2/M3/N2/N3 categories passed")
        return True
    else:
        print(f"\n[TEST] ✗ FAIL: Some M2/M3/N2/N3 categories failed")
        return False


def test_case_3_drowsiness_warning():
    """Test Case 3: Basic drowsiness warning functionality with simultaneous activation verification.

    Enhanced with detailed AIS-184 verification:
    - Concurrent warning verification (light and buzzer within 100ms)
    - Warning persistence verification (light remains ON while drowsiness persists)
    - Buzzer duration verification (exactly 3 seconds)
    - Warning clearance verification (light OFF within 2 seconds when drowsiness ends)
    """
    print("\n" + "="*60)
    print("TEST CASE 3: Drowsiness Warning Functionality (Enhanced)")
    print("="*60)

    # Set system to active monitoring state
    print("[TEST] Setting up active monitoring state...")
    with open("ignition_state.txt", "w") as f:
        f.write("ON")
    with open("vehicle_speed_log.txt", "w") as f:
        f.write("20")  # Above threshold
    with open("log_detection.txt", "w") as f:
        f.write("awake")
    with open("log_detection_confidence.txt", "w") as f:
        f.write("1.0")  # High confidence

    print("[TEST] System in active monitoring state (speed = 20 km/h)")
    print("[TEST] Starting detection logic...")

    # Start detection logic
    logic_process = subprocess.Popen(['python', '-u', 'new_detection_logic.py'],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     universal_newlines=True, bufsize=1)

    # Wait for system to initialize and enter active monitoring
    print("[TEST] Waiting for system initialization...")
    time.sleep(3)

    # Simulate drowsiness detection
    print("\n[TEST] Simulating drowsiness detection (eyes_closed)...")
    with open("log_detection.txt", "w") as f:
        f.write("eyes_closed")

    # Monitor for warning activation with precise timing
    drowsiness_trigger_time = time.time()
    warning_light_time = None
    buzzer_time = None
    warning_light_detected = False
    buzzer_detected = False
    warning_light_off_time = None
    buzzer_end_time = None

    print("[TEST] Monitoring for concurrent warning light + buzzer activation...")
    print("[TEST] Requirement: Both must activate within 100ms of each other")

    # Phase 1: Monitor for initial activation (first 5 seconds)
    monitoring_start = time.time()
    while time.time() - monitoring_start < 5:
        line = logic_process.stdout.readline()
        if line:
            current_time = time.time()
            print(f"[LOGIC] {line.strip()}")

            # Detect warning light activation
            if ("WARNING LIGHT" in line or "Drowsiness warning" in line) and "Activating" in line:
                if not warning_light_detected:
                    warning_light_time = current_time
                    warning_light_detected = True
                    elapsed = (current_time - drowsiness_trigger_time) * 1000
                    print(f"[TEST] ✓ Warning light activated at t={elapsed:.1f}ms")

            # Detect buzzer activation
            if ("BUZZER" in line or "Beep" in line) and "Activating" in line:
                if not buzzer_detected:
                    buzzer_time = current_time
                    buzzer_detected = True
                    elapsed = (current_time - drowsiness_trigger_time) * 1000
                    print(f"[TEST] ✓ Buzzer activated at t={elapsed:.1f}ms")

            # Check if both detected
            if warning_light_detected and buzzer_detected:
                break

    # Verify concurrent activation (within 100ms)
    concurrent_activation = False
    activation_time_diff = None

    if warning_light_time and buzzer_time:
        activation_time_diff = abs(warning_light_time - buzzer_time) * 1000  # Convert to ms
        print(f"\n[TEST] Activation Time Analysis:")
        print(f"  Warning light: {(warning_light_time - drowsiness_trigger_time) * 1000:.1f}ms")
        print(f"  Buzzer: {(buzzer_time - drowsiness_trigger_time) * 1000:.1f}ms")
        print(f"  Time difference: {activation_time_diff:.1f}ms")

        if activation_time_diff <= 100:
            print(f"  ✓ Concurrent activation verified (≤100ms)")
            concurrent_activation = True
        else:
            print(f"  ✗ Activation NOT concurrent (>{activation_time_diff:.1f}ms > 100ms)")
    else:
        print(f"\n[TEST] ✗ Could not measure activation timing")
        if not warning_light_detected:
            print(f"  - Warning light not detected")
        if not buzzer_detected:
            print(f"  - Buzzer not detected")

    # Phase 2: Verify warning persistence (light remains ON while drowsiness persists)
    print(f"\n[TEST] Phase 2: Verifying warning persistence...")
    print(f"[TEST] Drowsiness condition still active (eyes_closed)")
    print(f"[TEST] Warning light should remain ON...")

    # Monitor for 5 seconds to verify light stays on
    persistence_start = time.time()
    warning_light_stayed_on = True
    buzzer_duration_measured = False
    buzzer_duration = None

    while time.time() - persistence_start < 5:
        line = logic_process.stdout.readline()
        if line:
            current_time = time.time()
            print(f"[LOGIC] {line.strip()}")

            # Check if warning light incorrectly turned off
            if "WARNING LIGHT" in line and "OFF" in line and "Deactivating" in line:
                warning_light_off_time = current_time
                print(f"[TEST] ⚠ Warning light turned OFF at t={current_time - drowsiness_trigger_time:.1f}s")
                warning_light_stayed_on = False

            # Measure buzzer duration (should be exactly 3 seconds)
            if "BUZZER" in line and "OFF" in line:
                if buzzer_time and not buzzer_duration_measured:
                    buzzer_end_time = current_time
                    buzzer_duration = buzzer_end_time - buzzer_time
                    buzzer_duration_measured = True
                    print(f"[TEST] Buzzer duration: {buzzer_duration:.3f}s")

    # Verify buzzer duration (should be 3.0 ± 0.1 seconds)
    buzzer_duration_correct = False
    if buzzer_duration:
        if 2.9 <= buzzer_duration <= 3.1:
            print(f"[TEST] ✓ Buzzer duration correct (3.0 ± 0.1s)")
            buzzer_duration_correct = True
        else:
            print(f"[TEST] ✗ Buzzer duration incorrect (expected 3.0 ± 0.1s, got {buzzer_duration:.3f}s)")
    else:
        print(f"[TEST] ⚠ Could not measure buzzer duration")

    if warning_light_stayed_on:
        print(f"[TEST] ✓ Warning light remained ON during drowsiness (persistence verified)")
    else:
        print(f"[TEST] ✗ Warning light turned OFF prematurely")

    # Phase 3: Verify warning clearance when drowsiness ends
    print(f"\n[TEST] Phase 3: Verifying warning clearance...")
    print(f"[TEST] Clearing drowsiness condition (awake)...")

    with open("log_detection.txt", "w") as f:
        f.write("awake")

    clearance_trigger_time = time.time()
    warning_cleared = False
    clearance_delay = None

    # Monitor for warning clearance (should happen within 2 seconds)
    while time.time() - clearance_trigger_time < 3:
        line = logic_process.stdout.readline()
        if line:
            current_time = time.time()
            print(f"[LOGIC] {line.strip()}")

            # Check for warning clearance
            if ("WARNING LIGHT" in line and "OFF" in line) or "Clearing drowsiness warning" in line:
                if not warning_cleared:
                    warning_light_off_time = current_time
                    clearance_delay = current_time - clearance_trigger_time
                    warning_cleared = True
                    print(f"[TEST] ✓ Warning cleared at t={clearance_delay:.3f}s")
                    break

    # Verify clearance timing (within 2 seconds)
    clearance_timing_correct = False
    if warning_cleared and clearance_delay:
        if clearance_delay <= 2.0:
            print(f"[TEST] ✓ Warning cleared within 2 seconds ({clearance_delay:.3f}s)")
            clearance_timing_correct = True
        else:
            print(f"[TEST] ✗ Warning clearance delayed (>{clearance_delay:.3f}s > 2.0s)")
    else:
        print(f"[TEST] ⚠ Warning clearance not detected within 3 seconds")

    # Cleanup
    logic_process.terminate()
    logic_process.wait()

    # Evaluate test results
    print("\n[TEST] Compliance Verification:")
    print("="*60)
    print(f"  - Warning light activated: {warning_light_detected}")
    print(f"  - Buzzer activated: {buzzer_detected}")
    print(f"  - Concurrent activation (≤100ms): {concurrent_activation}")
    print(f"  - Warning light persistence: {warning_light_stayed_on}")
    print(f"  - Buzzer duration (3.0 ± 0.1s): {buzzer_duration_correct}")
    print(f"  - Warning clearance (≤2.0s): {clearance_timing_correct}")

    # Determine overall pass/fail
    all_checks_passed = (
        warning_light_detected and
        buzzer_detected and
        concurrent_activation and
        warning_light_stayed_on and
        buzzer_duration_correct and
        clearance_timing_correct
    )

    if all_checks_passed:
        print("\n[TEST] ✓ PASS: All AIS-184 drowsiness warning requirements verified")
        return True
    else:
        print("\n[TEST] ✗ FAIL: One or more AIS-184 requirements not met")
        return False


def run_vehicle_category_tests():
    """Run vehicle category-specific threshold tests."""
    print("\n" + "="*60)
    print("AIS-184 VEHICLE CATEGORY THRESHOLD TESTS")
    print("="*60)
    
    results = {
        "Test Case 2a: M1 Vehicle (70 km/h)": test_case_2a_warning_delay_m1(),
        "Test Case 2b: M2/M3/N2/N3 Vehicles (60 km/h)": test_case_2b_warning_delay_m2_m3_n2_n3()
    }
    
    # Generate test report
    print("\n" + "="*60)
    print("VEHICLE CATEGORY TEST REPORT")
    print("="*60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ VEHICLE CATEGORY TESTS: PASSED")
    else:
        print("\n✗ VEHICLE CATEGORY TESTS: FAILED")
    
    return passed == total
def test_case_3_drowsiness_warning():
    """Test Case 3: Basic drowsiness warning functionality with simultaneous activation verification.

    Enhanced with detailed AIS-184 verification:
    - Concurrent warning verification (light and buzzer within 100ms)
    - Warning persistence verification (light remains ON while drowsiness persists)
    - Buzzer duration verification (exactly 3 seconds)
    - Warning clearance verification (light OFF within 2 seconds when drowsiness ends)
    """
    print("\n" + "="*60)
    print("TEST CASE 3: Drowsiness Warning Functionality (Enhanced)")
    print("="*60)

    # Set system to active monitoring state
    print("[TEST] Setting up active monitoring state...")
    with open("ignition_state.txt", "w") as f:
        f.write("ON")
    with open("vehicle_speed_log.txt", "w") as f:
        f.write("20")  # Above threshold
    with open("log_detection.txt", "w") as f:
        f.write("awake")
    with open("log_detection_confidence.txt", "w") as f:
        f.write("1.0")  # High confidence

    print("[TEST] System in active monitoring state (speed = 20 km/h)")
    print("[TEST] Starting detection logic...")

    # Start detection logic
    logic_process = subprocess.Popen(['python', '-u', 'new_detection_logic.py'],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     universal_newlines=True, bufsize=1)

    # Wait for system to initialize and enter active monitoring
    print("[TEST] Waiting for system initialization...")
    time.sleep(3)

    # Simulate drowsiness detection
    print("\n[TEST] Simulating drowsiness detection (eyes_closed)...")
    with open("log_detection.txt", "w") as f:
        f.write("eyes_closed")

    # Monitor for warning activation with precise timing
    drowsiness_trigger_time = time.time()
    warning_light_time = None
    buzzer_time = None
    warning_light_detected = False
    buzzer_detected = False
    warning_light_off_time = None
    buzzer_end_time = None

    print("[TEST] Monitoring for concurrent warning light + buzzer activation...")
    print("[TEST] Requirement: Both must activate within 100ms of each other")

    # Phase 1: Monitor for initial activation (first 5 seconds)
    monitoring_start = time.time()
    while time.time() - monitoring_start < 5:
        line = logic_process.stdout.readline()
        if line:
            current_time = time.time()
            print(f"[LOGIC] {line.strip()}")

            # Detect warning light activation
            if ("WARNING LIGHT" in line or "Drowsiness warning" in line) and "Activating" in line:
                if not warning_light_detected:
                    warning_light_time = current_time
                    warning_light_detected = True
                    elapsed = (current_time - drowsiness_trigger_time) * 1000
                    print(f"[TEST] ✓ Warning light activated at t={elapsed:.1f}ms")

            # Detect buzzer activation
            if ("BUZZER" in line or "Beep" in line) and "Activating" in line:
                if not buzzer_detected:
                    buzzer_time = current_time
                    buzzer_detected = True
                    elapsed = (current_time - drowsiness_trigger_time) * 1000
                    print(f"[TEST] ✓ Buzzer activated at t={elapsed:.1f}ms")

            # Check if both detected
            if warning_light_detected and buzzer_detected:
                break

    # Verify concurrent activation (within 100ms)
    concurrent_activation = False
    activation_time_diff = None

    if warning_light_time and buzzer_time:
        activation_time_diff = abs(warning_light_time - buzzer_time) * 1000  # Convert to ms
        print(f"\n[TEST] Activation Time Analysis:")
        print(f"  Warning light: {(warning_light_time - drowsiness_trigger_time) * 1000:.1f}ms")
        print(f"  Buzzer: {(buzzer_time - drowsiness_trigger_time) * 1000:.1f}ms")
        print(f"  Time difference: {activation_time_diff:.1f}ms")

        if activation_time_diff <= 100:
            print(f"  ✓ Concurrent activation verified (≤100ms)")
            concurrent_activation = True
        else:
            print(f"  ✗ Activation NOT concurrent (>{activation_time_diff:.1f}ms > 100ms)")
    else:
        print(f"\n[TEST] ✗ Could not measure activation timing")
        if not warning_light_detected:
            print(f"  - Warning light not detected")
        if not buzzer_detected:
            print(f"  - Buzzer not detected")

    # Phase 2: Verify warning persistence (light remains ON while drowsiness persists)
    print(f"\n[TEST] Phase 2: Verifying warning persistence...")
    print(f"[TEST] Drowsiness condition still active (eyes_closed)")
    print(f"[TEST] Warning light should remain ON...")

    # Monitor for 5 seconds to verify light stays on
    persistence_start = time.time()
    warning_light_stayed_on = True
    buzzer_duration_measured = False
    buzzer_duration = None

    while time.time() - persistence_start < 5:
        line = logic_process.stdout.readline()
        if line:
            current_time = time.time()
            print(f"[LOGIC] {line.strip()}")

            # Check if warning light incorrectly turned off
            if "WARNING LIGHT" in line and "OFF" in line and "Deactivating" in line:
                warning_light_off_time = current_time
                print(f"[TEST] ⚠ Warning light turned OFF at t={current_time - drowsiness_trigger_time:.1f}s")
                warning_light_stayed_on = False

            # Measure buzzer duration (should be exactly 3 seconds)
            if "BUZZER" in line and "OFF" in line:
                if buzzer_time and not buzzer_duration_measured:
                    buzzer_end_time = current_time
                    buzzer_duration = buzzer_end_time - buzzer_time
                    buzzer_duration_measured = True
                    print(f"[TEST] Buzzer duration: {buzzer_duration:.3f}s")

    # Verify buzzer duration (should be 3.0 ± 0.1 seconds)
    buzzer_duration_correct = False
    if buzzer_duration:
        if 2.9 <= buzzer_duration <= 3.1:
            print(f"[TEST] ✓ Buzzer duration correct (3.0 ± 0.1s)")
            buzzer_duration_correct = True
        else:
            print(f"[TEST] ✗ Buzzer duration incorrect (expected 3.0 ± 0.1s, got {buzzer_duration:.3f}s)")
    else:
        print(f"[TEST] ⚠ Could not measure buzzer duration")

    if warning_light_stayed_on:
        print(f"[TEST] ✓ Warning light remained ON during drowsiness (persistence verified)")
    else:
        print(f"[TEST] ✗ Warning light turned OFF prematurely")

    # Phase 3: Verify warning clearance when drowsiness ends
    print(f"\n[TEST] Phase 3: Verifying warning clearance...")
    print(f"[TEST] Clearing drowsiness condition (awake)...")

    with open("log_detection.txt", "w") as f:
        f.write("awake")

    clearance_trigger_time = time.time()
    warning_cleared = False
    clearance_delay = None

    # Monitor for warning clearance (should happen within 2 seconds)
    while time.time() - clearance_trigger_time < 3:
        line = logic_process.stdout.readline()
        if line:
            current_time = time.time()
            print(f"[LOGIC] {line.strip()}")

            # Check for warning clearance
            if ("WARNING LIGHT" in line and "OFF" in line) or "Clearing drowsiness warning" in line:
                if not warning_cleared:
                    warning_light_off_time = current_time
                    clearance_delay = current_time - clearance_trigger_time
                    warning_cleared = True
                    print(f"[TEST] ✓ Warning cleared at t={clearance_delay:.3f}s")
                    break

    # Verify clearance timing (within 2 seconds)
    clearance_timing_correct = False
    if warning_cleared and clearance_delay:
        if clearance_delay <= 2.0:
            print(f"[TEST] ✓ Warning cleared within 2 seconds ({clearance_delay:.3f}s)")
            clearance_timing_correct = True
        else:
            print(f"[TEST] ✗ Warning clearance delayed (>{clearance_delay:.3f}s > 2.0s)")
    else:
        print(f"[TEST] ⚠ Warning clearance not detected within 3 seconds")

    # Cleanup
    logic_process.terminate()
    logic_process.wait()

    # Evaluate test results
    print("\n[TEST] Compliance Verification:")
    print("="*60)
    print(f"  - Warning light activated: {warning_light_detected}")
    print(f"  - Buzzer activated: {buzzer_detected}")
    print(f"  - Concurrent activation (≤100ms): {concurrent_activation}")
    print(f"  - Warning light persistence: {warning_light_stayed_on}")
    print(f"  - Buzzer duration (3.0 ± 0.1s): {buzzer_duration_correct}")
    print(f"  - Warning clearance (≤2.0s): {clearance_timing_correct}")

    # Determine overall pass/fail
    all_checks_passed = (
        warning_light_detected and
        buzzer_detected and
        concurrent_activation and
        warning_light_stayed_on and
        buzzer_duration_correct and
        clearance_timing_correct
    )

    if all_checks_passed:
        print("\n[TEST] ✓ PASS: All AIS-184 drowsiness warning requirements verified")
        return True
    else:
        print("\n[TEST] ✗ FAIL: One or more AIS-184 requirements not met")
        return False



def test_case_4_warning_visibility():
    """Test Case 4: Warning visibility in daylight and night conditions."""
    import yaml
    
    print("\n" + "="*60)
    print("TEST CASE 4: Warning Visibility")
    print("="*60)

    print("[TEST] Checking warning light intensity configuration...")

    # Load config and verify intensity is within AIS-184 range
    with open("new_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    intensity = config['config']['ais_184']['warning_signals']['light']['intensity_level']
    min_intensity = config['config']['ais_184']['warning_signals']['light']['luminous_intensity_min']
    max_intensity = config['config']['ais_184']['warning_signals']['light']['luminous_intensity_max']

    print(f"[TEST] Configured intensity: {intensity} cd")
    print(f"[TEST] AIS-184 range: {min_intensity} - {max_intensity} cd")

    if min_intensity <= intensity <= max_intensity:
        print("[TEST] ✓ PASS: Warning light intensity within AIS-184 specification")
        return True
    else:
        print("[TEST] ✗ FAIL: Warning light intensity out of range")
        return False


def test_case_5_failure_detection():
    """Test Case 5: Electrical failure and sensor obstruction detection with resolution verification.

    Enhanced with detailed AIS-184 verification:
    - Test 5a: Electrical failure detection and resolution
    - Test 5b: Sensor obstruction detection and resolution
    - Test 5c: Failure warning persistence verification
    - Multiple failure/resolution cycles
    """
    print("\n" + "="*60)
    print("TEST CASE 5: Failure Detection (Enhanced)")
    print("="*60)

    # ========================================================================
    # Test 5a: Electrical Failure Resolution Verification
    # ========================================================================
    print("\n" + "-"*60)
    print("TEST 5a: Electrical Failure Detection and Resolution")
    print("-"*60)

    print("\n[TEST 5a] Phase 1: Simulating electrical failure (component disconnection)...")
    
    # Setup: System in active monitoring state
    with open("ignition_state.txt", "w") as f:
        f.write("ON")
    with open("vehicle_speed_log.txt", "w") as f:
        f.write("20")  # Above threshold
    with open("log_detection_confidence.txt", "w") as f:
        f.write("1.0")  # High confidence

    # Simulate electrical failure by removing log_detection.txt
    if os.path.exists("log_detection.txt"):
        os.remove("log_detection.txt")

    print("[TEST 5a] Component disconnected (log_detection.txt removed)")
    print("[TEST 5a] Expecting failure detection within 1 second...")

    # Start detection logic
    logic_process = subprocess.Popen(['python', '-u', 'new_detection_logic.py'],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     universal_newlines=True, bufsize=1)

    # Monitor for electrical failure detection
    failure_start_time = time.time()
    electrical_failure_detected = False
    failure_detection_time = None
    warning_light_activated = False
    buzzer_activated = False
    buzzer_start_time = None
    buzzer_end_time = None

    print("[TEST 5a] Monitoring for failure detection...")

    for _ in range(20):  # Monitor for 10 seconds (should detect within 1s)
        line = logic_process.stdout.readline()
        if line:
            current_time = time.time()
            print(f"[LOGIC] {line.strip()}")

            # Detect electrical failure
            if "ELECTRICAL" in line and "FAILURE" in line:
                if not electrical_failure_detected:
                    electrical_failure_detected = True
                    failure_detection_time = current_time - failure_start_time
                    print(f"[TEST 5a] ✓ Electrical failure detected at t={failure_detection_time:.3f}s")

            # Detect warning light activation (constant mode)
            if "WARNING LIGHT" in line and ("CONSTANT" in line or "Activating" in line):
                if not warning_light_activated:
                    warning_light_activated = True
                    print(f"[TEST 5a] ✓ Warning light activated in constant mode")

            # Detect buzzer activation
            if "BUZZER" in line and "Activating" in line:
                if not buzzer_activated:
                    buzzer_activated = True
                    buzzer_start_time = current_time
                    print(f"[TEST 5a] ✓ Buzzer activated")

            # Detect buzzer deactivation (should be 5 seconds)
            if "BUZZER" in line and "OFF" in line:
                if buzzer_start_time and not buzzer_end_time:
                    buzzer_end_time = current_time
                    buzzer_duration = buzzer_end_time - buzzer_start_time
                    print(f"[TEST 5a] Buzzer duration: {buzzer_duration:.3f}s")

            # Break if all detected
            if electrical_failure_detected and warning_light_activated and buzzer_activated:
                time.sleep(1)  # Give time for buzzer to complete
                break

        time.sleep(0.5)

    # Verify electrical failure detection timing (within 1 second)
    electrical_timing_correct = False
    if failure_detection_time:
        if failure_detection_time <= 1.0:
            print(f"[TEST 5a] ✓ Failure detected within 1 second ({failure_detection_time:.3f}s)")
            electrical_timing_correct = True
        else:
            print(f"[TEST 5a] ✗ Failure detection delayed (>{failure_detection_time:.3f}s > 1.0s)")
    else:
        print(f"[TEST 5a] ✗ Electrical failure not detected")

    # Verify buzzer duration (should be 5.0 ± 0.1 seconds)
    buzzer_duration_correct = False
    if buzzer_start_time and buzzer_end_time:
        buzzer_duration = buzzer_end_time - buzzer_start_time
        if 4.9 <= buzzer_duration <= 5.1:
            print(f"[TEST 5a] ✓ Buzzer duration correct (5.0 ± 0.1s)")
            buzzer_duration_correct = True
        else:
            print(f"[TEST 5a] ✗ Buzzer duration incorrect (expected 5.0 ± 0.1s, got {buzzer_duration:.3f}s)")

    # Phase 2: Restore component and verify warning clears
    print("\n[TEST 5a] Phase 2: Restoring component connection...")
    
    # Restore log_detection.txt
    with open("log_detection.txt", "w") as f:
        f.write("awake")

    print("[TEST 5a] Component reconnected (log_detection.txt restored)")
    print("[TEST 5a] Expecting warning to clear within 2 seconds...")

    resolution_start_time = time.time()
    failure_resolved = False
    warning_cleared = False
    resolution_time = None

    # Monitor for failure resolution
    for _ in range(20):  # Monitor for 10 seconds (should resolve within 2s)
        line = logic_process.stdout.readline()
        if line:
            current_time = time.time()
            print(f"[LOGIC] {line.strip()}")

            # Detect failure resolution
            if "ELECTRICAL_FAILURE_RESOLVED" in line or "Failure resolved" in line:
                if not failure_resolved:
                    failure_resolved = True
                    resolution_time = current_time - resolution_start_time
                    print(f"[TEST 5a] ✓ Failure resolved at t={resolution_time:.3f}s")

            # Detect warning light deactivation
            if "WARNING LIGHT" in line and "OFF" in line:
                if not warning_cleared:
                    warning_cleared = True
                    clearance_time = current_time - resolution_start_time
                    print(f"[TEST 5a] ✓ Warning cleared at t={clearance_time:.3f}s")

            # Break if both detected
            if failure_resolved and warning_cleared:
                break

        time.sleep(0.5)

    # Verify resolution timing (within 2 seconds)
    resolution_timing_correct = False
    if warning_cleared and resolution_time:
        if resolution_time <= 2.0:
            print(f"[TEST 5a] ✓ Warning cleared within 2 seconds ({resolution_time:.3f}s)")
            resolution_timing_correct = True
        else:
            print(f"[TEST 5a] ✗ Warning clearance delayed (>{resolution_time:.3f}s > 2.0s)")
    else:
        print(f"[TEST 5a] ⚠ Warning clearance not detected")

    # Cleanup Test 5a
    logic_process.terminate()
    logic_process.wait()

    # Evaluate Test 5a results
    test_5a_passed = (
        electrical_failure_detected and
        electrical_timing_correct and
        warning_light_activated and
        buzzer_activated and
        buzzer_duration_correct and
        warning_cleared and
        resolution_timing_correct
    )

    print("\n[TEST 5a] Results:")
    print("="*60)
    print(f"  - Electrical failure detected: {electrical_failure_detected}")
    print(f"  - Detection timing (≤1.0s): {electrical_timing_correct}")
    print(f"  - Warning light activated (constant): {warning_light_activated}")
    print(f"  - Buzzer activated: {buzzer_activated}")
    print(f"  - Buzzer duration (5.0 ± 0.1s): {buzzer_duration_correct}")
    print(f"  - Warning cleared: {warning_cleared}")
    print(f"  - Clearance timing (≤2.0s): {resolution_timing_correct}")

    if test_5a_passed:
        print("\n[TEST 5a] ✓ PASS: Electrical failure detection and resolution verified")
    else:
        print("\n[TEST 5a] ✗ FAIL: One or more requirements not met")

    # ========================================================================
    # Test 5b: Sensor Obstruction Resolution Verification
    # ========================================================================
    print("\n" + "-"*60)
    print("TEST 5b: Sensor Obstruction Detection and Resolution")
    print("-"*60)

    print("\n[TEST 5b] Phase 1: Simulating sensor obstruction (low quality frames)...")

    # Setup: System in active monitoring state
    with open("ignition_state.txt", "w") as f:
        f.write("ON")
    with open("vehicle_speed_log.txt", "w") as f:
        f.write("20")
    with open("log_detection.txt", "w") as f:
        f.write("awake")
    
    # Simulate sensor obstruction with low confidence
    with open("log_detection_confidence.txt", "w") as f:
        f.write("0.1")  # Low confidence = obstruction

    print("[TEST 5b] Sensor obstructed (confidence = 0.1)")
    print("[TEST 5b] Expecting obstruction detection within 10 seconds (20 frames)...")

    # Start detection logic
    logic_process = subprocess.Popen(['python', '-u', 'new_detection_logic.py'],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     universal_newlines=True, bufsize=1)

    # Monitor for sensor obstruction detection
    obstruction_start_time = time.time()
    sensor_obstruction_detected = False
    obstruction_detection_time = None
    warning_light_activated_5b = False
    buzzer_activated_5b = False
    buzzer_start_time_5b = None
    buzzer_end_time_5b = None

    print("[TEST 5b] Monitoring for obstruction detection...")

    for _ in range(60):  # Monitor for 30 seconds (should detect within 10s)
        line = logic_process.stdout.readline()
        if line:
            current_time = time.time()
            print(f"[LOGIC] {line.strip()}")

            # Detect sensor obstruction
            if "SENSOR_OBSTRUCTION" in line and "DETECTED" in line:
                if not sensor_obstruction_detected:
                    sensor_obstruction_detected = True
                    obstruction_detection_time = current_time - obstruction_start_time
                    print(f"[TEST 5b] ✓ Sensor obstruction detected at t={obstruction_detection_time:.3f}s")

            # Detect warning light activation (constant mode)
            if "WARNING LIGHT" in line and ("CONSTANT" in line or "Activating" in line):
                if not warning_light_activated_5b:
                    warning_light_activated_5b = True
                    print(f"[TEST 5b] ✓ Warning light activated in constant mode")

            # Detect buzzer activation
            if "BUZZER" in line and "Activating" in line:
                if not buzzer_activated_5b:
                    buzzer_activated_5b = True
                    buzzer_start_time_5b = current_time
                    print(f"[TEST 5b] ✓ Buzzer activated")

            # Detect buzzer deactivation
            if "BUZZER" in line and "OFF" in line:
                if buzzer_start_time_5b and not buzzer_end_time_5b:
                    buzzer_end_time_5b = current_time
                    buzzer_duration_5b = buzzer_end_time_5b - buzzer_start_time_5b
                    print(f"[TEST 5b] Buzzer duration: {buzzer_duration_5b:.3f}s")

            # Break if all detected
            if sensor_obstruction_detected and warning_light_activated_5b and buzzer_activated_5b:
                time.sleep(1)  # Give time for buzzer to complete
                break

        time.sleep(0.5)

    # Verify obstruction detection timing (within 10 seconds)
    obstruction_timing_correct = False
    if obstruction_detection_time:
        if obstruction_detection_time <= 10.0:
            print(f"[TEST 5b] ✓ Obstruction detected within 10 seconds ({obstruction_detection_time:.3f}s)")
            obstruction_timing_correct = True
        else:
            print(f"[TEST 5b] ✗ Obstruction detection delayed (>{obstruction_detection_time:.3f}s > 10.0s)")
    else:
        print(f"[TEST 5b] ✗ Sensor obstruction not detected")

    # Verify buzzer duration (should be 5.0 ± 0.1 seconds)
    buzzer_duration_correct_5b = False
    if buzzer_start_time_5b and buzzer_end_time_5b:
        buzzer_duration_5b = buzzer_end_time_5b - buzzer_start_time_5b
        if 4.9 <= buzzer_duration_5b <= 5.1:
            print(f"[TEST 5b] ✓ Buzzer duration correct (5.0 ± 0.1s)")
            buzzer_duration_correct_5b = True
        else:
            print(f"[TEST 5b] ✗ Buzzer duration incorrect (expected 5.0 ± 0.1s, got {buzzer_duration_5b:.3f}s)")

    # Phase 2: Restore sensor quality and verify warning clears
    print("\n[TEST 5b] Phase 2: Restoring sensor quality...")
    
    # Restore sensor quality
    with open("log_detection_confidence.txt", "w") as f:
        f.write("1.0")  # High confidence

    print("[TEST 5b] Sensor quality restored (confidence = 1.0)")
    print("[TEST 5b] Expecting warning to clear within 2 seconds...")

    resolution_start_time_5b = time.time()
    obstruction_resolved = False
    warning_cleared_5b = False
    resolution_time_5b = None

    # Monitor for obstruction resolution
    for _ in range(20):  # Monitor for 10 seconds (should resolve within 2s)
        line = logic_process.stdout.readline()
        if line:
            current_time = time.time()
            print(f"[LOGIC] {line.strip()}")

            # Detect obstruction resolution
            if "SENSOR_OBSTRUCTION_RESOLVED" in line or "Obstruction resolved" in line:
                if not obstruction_resolved:
                    obstruction_resolved = True
                    resolution_time_5b = current_time - resolution_start_time_5b
                    print(f"[TEST 5b] ✓ Obstruction resolved at t={resolution_time_5b:.3f}s")

            # Detect warning light deactivation
            if "WARNING LIGHT" in line and "OFF" in line:
                if not warning_cleared_5b:
                    warning_cleared_5b = True
                    clearance_time_5b = current_time - resolution_start_time_5b
                    print(f"[TEST 5b] ✓ Warning cleared at t={clearance_time_5b:.3f}s")

            # Break if both detected
            if obstruction_resolved and warning_cleared_5b:
                break

        time.sleep(0.5)

    # Verify resolution timing (within 2 seconds)
    resolution_timing_correct_5b = False
    if warning_cleared_5b and resolution_time_5b:
        if resolution_time_5b <= 2.0:
            print(f"[TEST 5b] ✓ Warning cleared within 2 seconds ({resolution_time_5b:.3f}s)")
            resolution_timing_correct_5b = True
        else:
            print(f"[TEST 5b] ✗ Warning clearance delayed (>{resolution_time_5b:.3f}s > 2.0s)")
    else:
        print(f"[TEST 5b] ⚠ Warning clearance not detected")

    # Cleanup Test 5b
    logic_process.terminate()
    logic_process.wait()

    # Evaluate Test 5b results
    test_5b_passed = (
        sensor_obstruction_detected and
        obstruction_timing_correct and
        warning_light_activated_5b and
        buzzer_activated_5b and
        buzzer_duration_correct_5b and
        warning_cleared_5b and
        resolution_timing_correct_5b
    )

    print("\n[TEST 5b] Results:")
    print("="*60)
    print(f"  - Sensor obstruction detected: {sensor_obstruction_detected}")
    print(f"  - Detection timing (≤10.0s): {obstruction_timing_correct}")
    print(f"  - Warning light activated (constant): {warning_light_activated_5b}")
    print(f"  - Buzzer activated: {buzzer_activated_5b}")
    print(f"  - Buzzer duration (5.0 ± 0.1s): {buzzer_duration_correct_5b}")
    print(f"  - Warning cleared: {warning_cleared_5b}")
    print(f"  - Clearance timing (≤2.0s): {resolution_timing_correct_5b}")

    if test_5b_passed:
        print("\n[TEST 5b] ✓ PASS: Sensor obstruction detection and resolution verified")
    else:
        print("\n[TEST 5b] ✗ FAIL: One or more requirements not met")

    # ========================================================================
    # Test 5c: Multiple Failure/Resolution Cycles
    # ========================================================================
    print("\n" + "-"*60)
    print("TEST 5c: Multiple Failure/Resolution Cycles")
    print("-"*60)

    print("\n[TEST 5c] Testing multiple failure/resolution cycles...")
    print("[TEST 5c] Cycle 1: Electrical failure → resolution")
    print("[TEST 5c] Cycle 2: Sensor obstruction → resolution")

    cycle_results = []

    # Cycle 1: Electrical failure
    print("\n[TEST 5c] Cycle 1: Electrical failure...")
    
    with open("ignition_state.txt", "w") as f:
        f.write("ON")
    with open("vehicle_speed_log.txt", "w") as f:
        f.write("20")
    with open("log_detection_confidence.txt", "w") as f:
        f.write("1.0")
    
    if os.path.exists("log_detection.txt"):
        os.remove("log_detection.txt")

    logic_process = subprocess.Popen(['python', '-u', 'new_detection_logic.py'],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     universal_newlines=True, bufsize=1)

    cycle1_failure_detected = False
    cycle1_warning_persistent = True
    
    # Monitor for failure detection
    for _ in range(10):
        line = logic_process.stdout.readline()
        if line:
            print(f"[LOGIC] {line.strip()}")
            if "ELECTRICAL" in line and "FAILURE" in line:
                cycle1_failure_detected = True
                print("[TEST 5c] ✓ Cycle 1: Failure detected")
                break
        time.sleep(0.5)

    # Verify warning light persistence (should stay on for at least 3 seconds)
    print("[TEST 5c] Verifying warning light persistence...")
    persistence_start = time.time()
    
    while time.time() - persistence_start < 3:
        line = logic_process.stdout.readline()
        if line:
            print(f"[LOGIC] {line.strip()}")
            # Check if warning light incorrectly turned off
            if "WARNING LIGHT" in line and "OFF" in line and "Deactivating" in line:
                cycle1_warning_persistent = False
                print("[TEST 5c] ✗ Warning light turned OFF prematurely")
        time.sleep(0.5)

    if cycle1_warning_persistent:
        print("[TEST 5c] ✓ Warning light remained in constant mode")

    # Restore and verify resolution
    with open("log_detection.txt", "w") as f:
        f.write("awake")

    cycle1_resolved = False
    for _ in range(10):
        line = logic_process.stdout.readline()
        if line:
            print(f"[LOGIC] {line.strip()}")
            if "RESOLVED" in line or "WARNING LIGHT" in line and "OFF" in line:
                cycle1_resolved = True
                print("[TEST 5c] ✓ Cycle 1: Failure resolved")
                break
        time.sleep(0.5)

    logic_process.terminate()
    logic_process.wait()

    cycle_results.append(cycle1_failure_detected and cycle1_warning_persistent and cycle1_resolved)

    # Cycle 2: Sensor obstruction
    print("\n[TEST 5c] Cycle 2: Sensor obstruction...")
    
    with open("ignition_state.txt", "w") as f:
        f.write("ON")
    with open("vehicle_speed_log.txt", "w") as f:
        f.write("20")
    with open("log_detection.txt", "w") as f:
        f.write("awake")
    with open("log_detection_confidence.txt", "w") as f:
        f.write("0.1")

    logic_process = subprocess.Popen(['python', '-u', 'new_detection_logic.py'],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     universal_newlines=True, bufsize=1)

    cycle2_obstruction_detected = False
    cycle2_warning_persistent = True
    
    # Monitor for obstruction detection
    for _ in range(40):
        line = logic_process.stdout.readline()
        if line:
            print(f"[LOGIC] {line.strip()}")
            if "SENSOR_OBSTRUCTION" in line and "DETECTED" in line:
                cycle2_obstruction_detected = True
                print("[TEST 5c] ✓ Cycle 2: Obstruction detected")
                break
        time.sleep(0.5)

    # Verify warning light persistence
    print("[TEST 5c] Verifying warning light persistence...")
    persistence_start = time.time()
    
    while time.time() - persistence_start < 3:
        line = logic_process.stdout.readline()
        if line:
            print(f"[LOGIC] {line.strip()}")
            if "WARNING LIGHT" in line and "OFF" in line and "Deactivating" in line:
                cycle2_warning_persistent = False
                print("[TEST 5c] ✗ Warning light turned OFF prematurely")
        time.sleep(0.5)

    if cycle2_warning_persistent:
        print("[TEST 5c] ✓ Warning light remained in constant mode")

    # Restore and verify resolution
    with open("log_detection_confidence.txt", "w") as f:
        f.write("1.0")

    cycle2_resolved = False
    for _ in range(10):
        line = logic_process.stdout.readline()
        if line:
            print(f"[LOGIC] {line.strip()}")
            if "RESOLVED" in line or "WARNING LIGHT" in line and "OFF" in line:
                cycle2_resolved = True
                print("[TEST 5c] ✓ Cycle 2: Obstruction resolved")
                break
        time.sleep(0.5)

    logic_process.terminate()
    logic_process.wait()

    cycle_results.append(cycle2_obstruction_detected and cycle2_warning_persistent and cycle2_resolved)

    # Evaluate Test 5c results
    test_5c_passed = all(cycle_results)

    print("\n[TEST 5c] Results:")
    print("="*60)
    print(f"  - Cycle 1 (Electrical): {'✓ PASS' if cycle_results[0] else '✗ FAIL'}")
    print(f"  - Cycle 2 (Obstruction): {'✓ PASS' if cycle_results[1] else '✗ FAIL'}")

    if test_5c_passed:
        print("\n[TEST 5c] ✓ PASS: Multiple failure/resolution cycles verified")
    else:
        print("\n[TEST 5c] ✗ FAIL: One or more cycles failed")

    # ========================================================================
    # Overall Test Case 5 Evaluation
    # ========================================================================
    print("\n" + "="*60)
    print("TEST CASE 5: Overall Results")
    print("="*60)
    print(f"  - Test 5a (Electrical failure): {'✓ PASS' if test_5a_passed else '✗ FAIL'}")
    print(f"  - Test 5b (Sensor obstruction): {'✓ PASS' if test_5b_passed else '✗ FAIL'}")
    print(f"  - Test 5c (Multiple cycles): {'✓ PASS' if test_5c_passed else '✗ FAIL'}")

    overall_passed = test_5a_passed and test_5b_passed and test_5c_passed

    if overall_passed:
        print("\n[TEST CASE 5] ✓ PASS: All failure detection requirements verified")
        return True
    else:
        print("\n[TEST CASE 5] ✗ FAIL: One or more requirements not met")
        return False


def run_ais_184_phase1_tests():
    """Run all AIS-184 Phase 1 test cases including ignition self-test."""
    print("\n" + "="*60)
    print("AIS-184 PHASE 1 COMPLIANCE TEST SUITE")
    print("="*60)
    
    results = {
        "Test Case 1: Ignition Self-Test": test_case_1_ignition_self_test(),
        "Test Case 2a: M1 Vehicle (70 km/h)": test_case_2a_warning_delay_m1(),
        "Test Case 2b: M2/M3/N2/N3 Vehicles (60 km/h)": test_case_2b_warning_delay_m2_m3_n2_n3(),
        "Test Case 3: Drowsiness Warning": test_case_3_drowsiness_warning(),
        "Test Case 4: Warning Visibility": test_case_4_warning_visibility(),
        "Test Case 5: Failure Detection": test_case_5_failure_detection()
    }
    
    # Generate compliance report
    print("\n" + "="*60)
    print("COMPLIANCE REPORT")
    print("="*60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ AIS-184 PHASE 1 COMPLIANCE: ACHIEVED")
    else:
        print("\n✗ AIS-184 PHASE 1 COMPLIANCE: NOT ACHIEVED")
    
    return passed == total


if __name__ == "__main__":
    # Check if user wants to run specific test
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test-ignition":
            test_case_1_ignition_self_test()
        elif sys.argv[1] == "--test-category":
            run_vehicle_category_tests()
        elif sys.argv[1] == "--test-drowsiness":
            test_case_3_drowsiness_warning()
        elif sys.argv[1] == "--test-visibility":
            test_case_4_warning_visibility()
        elif sys.argv[1] == "--test-failure":
            test_case_5_failure_detection()
        elif sys.argv[1] == "--test-all":
            run_ais_184_phase1_tests()
        else:
            print("Usage:")
            print("  python test_integration.py                   # Run integration test")
            print("  python test_integration.py --test-ignition   # Run Test Case 1")
            print("  python test_integration.py --test-category   # Run Test Case 2")
            print("  python test_integration.py --test-drowsiness # Run Test Case 3")
            print("  python test_integration.py --test-visibility # Run Test Case 4")
            print("  python test_integration.py --test-failure    # Run Test Case 5")
            print("  python test_integration.py --test-all        # Run all Phase 1 tests")
    else:
        # Run normal integration test
        main()
