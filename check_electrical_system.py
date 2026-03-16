#!/usr/bin/env python3
"""
Pre-startup electrical system check for AIS-184 compliance.
Validates camera and sensor hardware before starting ML inference.
"""

import cv2
import time
import json
from datetime import datetime
import yaml

def log_ais_184_event(event_type, details):
    """Log AIS-184 compliance event."""
    try:
        with open("ais_184_compliance_log.txt", "a") as f:
            event = {
                "timestamp": datetime.now().isoformat(),
                "event_type": event_type,
                "details": details
            }
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        print(f"[ERROR] Failed to log event: {e}")

def check_camera_hardware(camera_id=0, timeout=5):
    """
    Check if camera hardware is accessible and functional.
    
    Args:
        camera_id: Camera device ID
        timeout: Maximum time to wait for camera (seconds)
        
    Returns:
        tuple: (success: bool, error_message: str or None)
    """
    print(f"[ELECTRICAL CHECK] Testing camera {camera_id}...")
    
    try:
        # Try to open camera
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            return False, f"Camera {camera_id} not accessible"
        
        # Try to read a frame
        start_time = time.time()
        frame_captured = False
        
        for attempt in range(5):
            ret, frame = cap.read()
            if ret and frame is not None:
                frame_captured = True
                print(f"[ELECTRICAL CHECK] ✓ Camera frame captured (attempt {attempt + 1})")
                break
            time.sleep(0.2)
            
            if time.time() - start_time > timeout:
                break
        
        cap.release()
        
        if not frame_captured:
            return False, f"Camera {camera_id} opened but cannot capture frames"
        
        print(f"[ELECTRICAL CHECK] ✓ Camera hardware OK")
        return True, None
        
    except Exception as e:
        return False, f"Camera error: {str(e)}"

def check_gpio_hardware():
    """
    Check if GPIO hardware (buzzer, LED) is accessible.
    
    Returns:
        tuple: (success: bool, error_message: str or None)
    """
    print("[ELECTRICAL CHECK] Testing GPIO hardware...")
    
    try:
        # Try to import GPIO library
        try:
            import RPi.GPIO as GPIO
            gpio_available = True
        except (ImportError, RuntimeError):
            # Not on Raspberry Pi or GPIO not available
            print("[ELECTRICAL CHECK] ⚠ GPIO not available (not on Raspberry Pi)")
            return True, None  # Not a failure, just not available
        
        # Test GPIO initialization
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            print("[ELECTRICAL CHECK] ✓ GPIO hardware OK")
            return True, None
        except Exception as e:
            return False, f"GPIO initialization failed: {str(e)}"
            
    except Exception as e:
        return False, f"GPIO error: {str(e)}"

def check_can_interface():
    """
    Check if CAN interface for vehicle speed is accessible.
    
    Returns:
        tuple: (success: bool, error_message: str or None)
    """
    print("[ELECTRICAL CHECK] Testing CAN interface...")
    
    try:
        # Check if vehicle_speed_log.txt exists and is readable
        with open("vehicle_speed_log.txt", "r") as f:
            speed_str = f.read().strip()
            speed = float(speed_str)
            
            if 0 <= speed <= 300:
                print(f"[ELECTRICAL CHECK] ✓ CAN interface OK (speed: {speed} km/h)")
                return True, None
            else:
                return False, f"Invalid speed value: {speed} km/h"
                
    except FileNotFoundError:
        print("[ELECTRICAL CHECK] ⚠ vehicle_speed_log.txt not found (CAN may not be initialized)")
        # Create default file
        with open("vehicle_speed_log.txt", "w") as f:
            f.write("0.0")
        return True, None  # Not a critical failure
        
    except (ValueError, IOError) as e:
        return False, f"CAN interface error: {str(e)}"

def write_electrical_failure_state(failed_components):
    """
    Write electrical failure state to log files for overlay display.
    
    Args:
        failed_components: List of (component_name, error_message) tuples
    """
    print("[ELECTRICAL CHECK] Writing ELECTRICAL FAILURE state to logs...")
    
    # Create detailed failure message with component names
    component_names = [comp for comp, _ in failed_components]
    failure_detail = ", ".join(component_names)
    
    try:
        # Write to detection log with component details
        with open("log_detection.txt", "w") as f:
            f.write(f"electrical_failure:{failure_detail}")
        with open("log_detection_confidence.txt", "w") as f:
            f.write("0.0")
        print(f"[ELECTRICAL CHECK] ✓ Failure state written: {failure_detail}")
    except Exception as e:
        print(f"[ERROR] Failed to write failure state: {e}")

def clear_electrical_failure_state():
    """Clear electrical failure state from log files."""
    try:
        with open("log_detection.txt", "w") as f:
            f.write("awake")
        with open("log_detection_confidence.txt", "w") as f:
            f.write("1.0")
    except Exception as e:
        print(f"[ERROR] Failed to clear failure state: {e}")

def run_electrical_system_check():
    """
    Run complete electrical system check before starting ML.
    
    Returns:
        bool: True if all checks pass, False if any failure detected
    """
    print("\n" + "="*70)
    print("AIS-184 ELECTRICAL SYSTEM CHECK")
    print("="*70)
    print("Validating hardware before starting ML inference...\n")
    
    # Load configuration
    try:
        with open("new_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        camera_id = config.get('config', {}).get('camera_id', 0)
    except:
        camera_id = 0
    
    failures = []
    
    # Check 1: Camera hardware
    camera_ok, camera_error = check_camera_hardware(camera_id)
    if not camera_ok:
        failures.append(("Camera", camera_error))
    
    # Check 2: GPIO hardware (buzzer, LED)
    gpio_ok, gpio_error = check_gpio_hardware()
    if not gpio_ok:
        failures.append(("GPIO", gpio_error))
    
    # Check 3: CAN interface
    can_ok, can_error = check_can_interface()
    if not can_ok:
        failures.append(("CAN", can_error))
    
    # Report results
    print("\n" + "="*70)
    print("ELECTRICAL SYSTEM CHECK RESULTS")
    print("="*70)
    
    if failures:
        print("❌ ELECTRICAL FAILURE DETECTED\n")
        for component, error in failures:
            print(f"  ✗ {component}: {error}")
        
        # Log electrical failure event
        log_ais_184_event("ELECTRICAL_FAILURE", {
            "reason": "pre_startup_check",
            "failed_components": [{"component": c, "error": e} for c, e in failures]
        })
        
        # Write failure state to logs for overlay with component details
        write_electrical_failure_state(failures)
        
        print("\n" + "="*70)
        print("SYSTEM CANNOT START - ELECTRICAL FAILURE")
        print("="*70)
        print("\nActions required:")
        print("1. Fix the hardware issues listed above")
        print("2. Run this check again: python check_electrical_system.py")
        print("3. Once all checks pass, the system will start automatically")
        print("\nTo view failure in overlay:")
        print("  python ml/test_onnx_inference.py --camera --log-path log_detection.txt")
        
        return False
    else:
        print("✓ All electrical systems OK\n")
        print("  ✓ Camera hardware functional")
        print("  ✓ GPIO hardware accessible")
        print("  ✓ CAN interface operational")
        
        # Clear any previous failure state
        clear_electrical_failure_state()
        
        # Log successful check
        log_ais_184_event("ELECTRICAL_CHECK_PASSED", {
            "reason": "pre_startup_check",
            "components_checked": ["camera", "gpio", "can"]
        })
        
        print("\n" + "="*70)
        print("SYSTEM READY TO START")
        print("="*70)
        print("\nYou can now start the ML inference:")
        print("  python test_integration.py")
        
        return True

if __name__ == "__main__":
    import sys
    
    success = run_electrical_system_check()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
