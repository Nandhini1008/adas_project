#!/usr/bin/env python3
"""
ML Pipeline Integration Test
Simulates 200 inference frames/sec from the ML model and passes them
through the detection logic pipeline.

Log file paths are read from new_config.yaml (no hardcoding).

Usage:
    python test_ml_pipeline.py                        # default scenario
    python test_ml_pipeline.py --scenario eyes_closed
    python test_ml_pipeline.py --fps 200 --duration 5
"""

import time
import threading
import argparse
import random
import yaml
from collections import Counter

# ── Load paths and ML pipeline config from yaml (no hardcoding) ──────────────
with open("new_config.yaml", "r", encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f)

_ais        = _cfg.get("config", {}).get("ais_184", {}).get("logging", {})
_ml_cfg     = _cfg.get("config", {}).get("ml_pipeline", {})

LOG_DETECTION     = "log_detection.txt"
LOG_CONFIDENCE    = "log_detection_confidence.txt"
VEHICLE_SPEED_LOG = "vehicle_speed_log.txt"
AIS_LOG           = _ais.get("log_file", "ais_184_compliance_log.txt")

# ── Defaults pulled from config (override with CLI args) ──────────────────────
TARGET_FPS          = _ml_cfg.get("frames_per_second", 200)
WRITE_INTERVAL_SEC  = _ml_cfg.get("write_interval_sec", 1)
TEST_DURATION       = 5       # seconds (CLI only, not in config)
VEHICLE_SPEED       = 75.0    # km/h (CLI only)

# ── Scenario definitions ──────────────────────────────────────────────────────
SCENARIOS = {
    "awake": [
        ("awake", 0.92, 10),
    ],
    "eyes_closed": [
        ("awake",       0.90, 2),
        ("eyes_closed", 0.88, 8),
    ],
    "yawn": [
        ("awake", 0.91, 3),
        ("yawn",  0.85, 7),
    ],
    "distracted": [
        ("awake",      0.90, 3),
        ("distracted", 0.82, 7),
    ],
    "phone": [
        ("awake", 0.90, 3),
        ("phone", 0.87, 7),
    ],
    "smoking": [
        ("awake",   0.90, 3),
        ("smoking", 0.84, 7),
    ],
    "mixed": [
        ("awake",       0.92, 4),
        ("eyes_closed", 0.88, 2),
        ("yawn",        0.85, 2),
        ("distracted",  0.80, 1),
        ("phone",       0.87, 1),
    ],
    "noisy": [
        ("awake",       0.90, 5),
        ("eyes_closed", 0.75, 3),
        ("awake",       0.88, 4),
        ("eyes_closed", 0.80, 3),
    ],
}

# ── Shared state ──────────────────────────────────────────────────────────────
_stop_event   = threading.Event()
_stats_lock   = threading.Lock()
_frame_count  = 0
_label_counts: Counter = Counter()
_write_errors = 0


def _write_majority_result(batch: list, sec: int) -> None:
    """
    Given a batch of (label, confidence) tuples from one second of inference,
    compute the majority label + average confidence and write to log files.
    Also prints a per-frame breakdown so you can see all frames were processed.
    """
    global _frame_count, _write_errors

    total = len(batch)
    counts: Counter = Counter(lbl for lbl, _ in batch)
    majority_label = counts.most_common(1)[0][0]
    avg_conf = sum(c for _, c in batch) / total

    # Print every frame in the batch so it's visible
    print(f"\n[SEC {sec}] Processing {total} frames:")
    for i, (lbl, conf) in enumerate(batch, 1):
        bar_filled = int((i / total) * 30)
        bar = "█" * bar_filled + "░" * (30 - bar_filled)
        print(f"  frame {i:>4}/{total}  [{bar}]  {lbl:<15}  conf={conf:.3f}")

    # Print the vote result
    vote_str = "  ".join(f"{k}:{v}({v/total*100:.0f}%)" for k, v in counts.most_common())
    print(f"  --> VOTE: {vote_str}")
    print(f"  --> MAJORITY: {majority_label}  avg_conf={avg_conf:.3f}  (writing to log)")

    try:
        with open(LOG_DETECTION, "w") as f:
            f.write(majority_label)
        with open(LOG_CONFIDENCE, "w") as f:
            f.write(f"{avg_conf:.4f}")
        with _stats_lock:
            _frame_count += total
            for lbl, _ in batch:
                _label_counts[lbl] += 1
    except OSError as e:
        with _stats_lock:
            _write_errors += 1
        print(f"[ML-SIM] Write error: {e}")


def ml_frame_producer(scenario_frames: list, fps: int, duration: int, write_interval: float) -> None:
    """
    Each write_interval seconds: collect `fps` inference frames,
    compute the majority label, write ONE result to log_detection.txt.
    Both fps and write_interval come from new_config.yaml → ml_pipeline.
    """
    labels  = [f[0] for f in scenario_frames]
    confs   = [f[1] for f in scenario_frames]
    weights = [f[2] for f in scenario_frames]

    frames_per_batch = int(fps * write_interval)
    frame_interval   = write_interval / frames_per_batch  # time between frames

    total_batches = int(duration / write_interval)

    for batch_num in range(1, total_batches + 1):
        batch = []
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

        elapsed = time.perf_counter() - batch_start
        print(f"\n[ML-SIM] Batch {batch_num}/{total_batches}: "
              f"{len(batch)} frames in {elapsed:.3f}s  "
              f"(interval={write_interval}s, fps={fps})")

        _write_majority_result(batch, batch_num)

    _stop_event.set()


def stats_reporter(fps: int, duration: int) -> None:
    """Print a per-second fps summary line (separate from the per-frame progress bar)."""
    prev = 0
    for sec in range(1, duration + 2):
        time.sleep(1.0)
        if _stop_event.is_set() and sec > duration:
            break
        with _stats_lock:
            total  = _frame_count
            counts = dict(_label_counts)
        delta     = total - prev
        prev      = total
        label_str = "  ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
        # Print on a new line so it doesn't collide with the progress bar
        print(f"\n[{sec:3d}s] fps={delta:4d}  total={total:6d}  | {label_str}", flush=True)


def setup_vehicle_speed(speed: float) -> None:
    with open(VEHICLE_SPEED_LOG, "w") as f:
        f.write(str(speed))
    print(f"[SETUP] {VEHICLE_SPEED_LOG} = {speed} km/h")


def print_summary(scenario: str, fps: int, duration: int) -> None:
    with _stats_lock:
        total  = _frame_count
        errors = _write_errors
        counts = dict(_label_counts)

    target = fps * duration
    print("\n" + "=" * 60)
    print("  ML PIPELINE TEST SUMMARY")
    print("=" * 60)
    print(f"  Scenario        : {scenario}")
    print(f"  FPS per second  : {fps}")
    print(f"  Duration        : {duration}s  ({target} total frames, {duration} majority votes)")
    print(f"  Frames processed: {total}")
    print(f"  Write errors    : {errors}")
    print(f"  Log written to  : {LOG_DETECTION}  (majority label each second)")
    print(f"  Conf written to : {LOG_CONFIDENCE}  (avg confidence each second)")
    print()
    print("  Overall frame distribution (all seconds combined):")
    for label, count in sorted(counts.items()):
        pct = count / total * 100 if total else 0
        bar = "█" * int(pct / 2)
        print(f"    {label:<15} {count:5d}  ({pct:5.1f}%)  {bar}")
    print("=" * 60)
    if errors == 0 and total >= int(target * 0.90):
        print("  RESULT: PASS  – majority vote written to log each second")
    else:
        print(f"  RESULT: FAIL  – {total}/{target} frames, {errors} errors")
    print()



def main() -> None:
    parser = argparse.ArgumentParser(description="ML → Detection Logic pipeline test")
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()),
                        default="eyes_closed")
    parser.add_argument("--fps",      type=int,   default=TARGET_FPS,
                        help=f"Frames per second (default from config: {TARGET_FPS})")
    parser.add_argument("--write-interval", type=float, default=WRITE_INTERVAL_SEC,
                        help=f"Seconds per batch/write (default from config: {WRITE_INTERVAL_SEC})")
    parser.add_argument("--duration", type=int,   default=TEST_DURATION,
                        help="Total test duration in seconds")
    parser.add_argument("--speed",    type=float, default=VEHICLE_SPEED)
    parser.add_argument("--no-detection-logic", action="store_true",
                        help="Skip importing detection logic (ML producer only)")
    args = parser.parse_args()

    print("=" * 60)
    print("  ML PIPELINE INTEGRATION TEST")
    print("=" * 60)
    print(f"  Scenario        : {args.scenario}")
    print(f"  frames_per_sec  : {args.fps}  (new_config.yaml → ml_pipeline.frames_per_second)")
    print(f"  write_interval  : {args.write_interval}s  (new_config.yaml → ml_pipeline.write_interval_sec)")
    print(f"  frames per batch: {int(args.fps * args.write_interval)}")
    print(f"  Duration        : {args.duration}s")
    print(f"  Speed           : {args.speed} km/h")
    print(f"  log_detection   : {LOG_DETECTION}")
    print(f"  log_confidence  : {LOG_CONFIDENCE}")
    print("=" * 60)
    print()

    setup_vehicle_speed(args.speed)

    # ML frame producer
    producer = threading.Thread(
        target=ml_frame_producer,
        args=(SCENARIOS[args.scenario], args.fps, args.duration, args.write_interval),
        daemon=True, name="ml-producer"
    )
    producer.start()

    # Live stats
    threading.Thread(
        target=stats_reporter,
        args=(args.fps, args.duration),
        daemon=True, name="stats-reporter"
    ).start()

    # Detection logic (optional)
    if not args.no_detection_logic:
        print("[DETECT] Starting detection logic thread...")
        import importlib, sys
        def _run_detect():
            # Remove cached module so it re-initialises cleanly
            sys.modules.pop("new_detection_logic", None)
            import new_detection_logic  # noqa: F401
        threading.Thread(target=_run_detect, daemon=True, name="detection-logic").start()

    try:
        producer.join(timeout=args.duration + 3)
    except KeyboardInterrupt:
        print("\n[TEST] Interrupted")
        _stop_event.set()

    _stop_event.wait(timeout=2)
    print_summary(args.scenario, args.fps, args.duration)


if __name__ == "__main__":
    main()
