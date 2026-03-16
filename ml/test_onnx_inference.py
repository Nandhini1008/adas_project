#!/usr/bin/env python3
"""
Test ONNX model inference for YOLOX detection model.
Runs inference on sample images and visualizes detections.
"""

import onnxruntime as ort
import numpy as np
import cv2
import os
import argparse
from pathlib import Path
# Audio alerts handled by new_detection_logic.py (runs in parallel via test_integration.py)
# import threading
# import winsound


def apply_nms(detections, iou_threshold=0.45):
    """
    Apply Non-Maximum Suppression to filter overlapping boxes ACROSS ALL CLASSES.
    This ensures only one state (awake/distracted/eyes_closed) is shown per person.
    
    Args:
        detections: List of [x1, y1, x2, y2, conf, class_id]
        iou_threshold: IoU threshold for NMS
        
    Returns:
        Filtered detections
    """
    if len(detections) == 0:
        return []
    
    # Convert to numpy arrays
    boxes = np.array([[d[0], d[1], d[2], d[3]] for d in detections])
    scores = np.array([d[4] for d in detections])
    class_ids = np.array([d[5] for d in detections])
    
    # Sort ALL detections by confidence (regardless of class)
    order = scores.argsort()[::-1]
    
    keep_indices = []
    
    while len(order) > 0:
        # Keep the detection with highest confidence
        i = order[0]
        keep_indices.append(i)
        
        if len(order) == 1:
            break
        
        # Calculate IoU with ALL remaining boxes (across all classes)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_others = (boxes[order[1:], 2] - boxes[order[1:], 0]) * \
                     (boxes[order[1:], 3] - boxes[order[1:], 1])
        
        iou = inter / (area_i + area_others - inter + 1e-6)
        
        # Remove boxes with high IoU (overlapping detections)
        # Use a lower threshold (0.3) to be more aggressive in removing overlaps
        inds = np.where(iou <= 0.3)[0]
        order = order[inds + 1]
    
    # Return filtered detections sorted by original order
    return [detections[i] for i in sorted(keep_indices)]


class YOLOXONNXInference:
    """ONNX inference wrapper for YOLOX model."""
    
    def __init__(self, model_path, conf_threshold=0.2, nms_threshold=0.45, input_size=416):
      
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        
        # Class-wise confidence thresholds
        # Class 0: awake, Class 1: distracted, Class 2: eyes_closed
        self.class_thresholds = {
            0: 0.2,  # awake
            1: 0.1,  # distracted (lower threshold to catch more)
            2: 0.1   # eyes_closed
        }
        
        # Create ONNX Runtime session
        print(f"Loading ONNX model: {model_path}")
        self.session = ort.InferenceSession(model_path)
        
        # Get model input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_type = self.session.get_inputs()[0].type
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        print(f"Input: {self.input_name}, Shape: {self.input_shape}, Type: {self.input_type}")
        print(f"Outputs: {self.output_names}")
        print(f"Configured input size: {self.input_size}x{self.input_size}")
        print(f"Class-wise confidence thresholds:")
        print(f"  Class 0 (awake): {self.class_thresholds[0]}")
        print(f"  Class 1 (distracted): {self.class_thresholds[1]}")
        print(f"  Class 2 (eyes_closed): {self.class_thresholds[2]}")
        print(f"NMS threshold: {self.nms_threshold}")
        
    def preprocess(self, image):
        """
        Preprocess image for YOLOX inference.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            Preprocessed image tensor, scale ratio, padding
        """
        img_h, img_w = image.shape[:2]
        
        # Calculate scale to fit input size while maintaining aspect ratio
        scale = min(self.input_size / img_h, self.input_size / img_w)
        new_h, new_w = int(img_h * scale), int(img_w * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image (114, 114, 114) - YOLOX default
        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        # Convert to RGB
        padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        
        # Transpose to CHW format and add batch dimension
        input_tensor = padded.transpose(2, 0, 1)[np.newaxis, :, :, :]
        
        # Convert based on model input type
        if 'uint8' in self.input_type:
            input_tensor = input_tensor.astype(np.uint8)
        else:
            input_tensor = input_tensor.astype(np.float32)
        
        return input_tensor, scale, (0, 0)
    
    def postprocess(self, outputs, scale, original_shape):
        """
        Postprocess ONNX outputs to get final detections.
        
        Args:
            outputs: Model outputs
            scale: Scale factor used in preprocessing
            original_shape: Original image shape (h, w)
            
        Returns:
            List of detections [x1, y1, x2, y2, conf, class_id]
        """
        # Check output format
        if len(outputs) == 2:
            # Format: [dets, labels]
            dets = outputs[0]  # Could be [N, 4] or [N, 5]
            labels = outputs[1]  # Shape varies
            
            # Squeeze dimensions
            dets = np.squeeze(dets)
            labels = np.squeeze(labels)
            
            # Handle empty detections
            if dets.size == 0:
                return []
            
            # Handle 1D array (single detection)
            if dets.ndim == 1:
                dets = dets.reshape(1, -1)
                labels = np.atleast_1d(labels)
            
            if len(dets) == 0:
                return []
            
            # Handle 1D array (single detection)
            if dets.ndim == 1:
                dets = dets.reshape(1, -1)
                labels = np.atleast_1d(labels)
            
            if len(dets) == 0:
                return []
            
            # Check if dets has confidence (5 cols) or just bbox (4 cols)
            has_confidence = dets.shape[1] >= 5
            
            detections = []
            for i in range(len(dets)):
                if has_confidence:
                    x1, y1, x2, y2, conf = dets[i][:5]
                else:
                    x1, y1, x2, y2 = dets[i][:4]
                    conf = 1.0  # Default confidence if not provided
                
                class_id = int(labels[i]) if i < len(labels) else 0
                
                # Apply class-wise confidence threshold
                class_threshold = self.class_thresholds.get(class_id, self.conf_threshold)
                if conf < class_threshold:
                    continue
                
                # Scale boxes back to original image size
                x1, y1, x2, y2 = x1 / scale, y1 / scale, x2 / scale, y2 / scale
                
                detections.append([x1, y1, x2, y2, float(conf), class_id])
            
            return detections
        else:
            print(f"Unexpected output format: {len(outputs)} outputs")
            return []
    
    def infer(self, image):
        """
        Run inference on image.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            List of detections [x1, y1, x2, y2, conf, class_id]
        """
        # Preprocess
        input_tensor, scale, padding = self.preprocess(image)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # Postprocess
        detections = self.postprocess(outputs, scale, image.shape[:2])
        
        # Apply NMS to remove overlapping boxes
        detections = apply_nms(detections, iou_threshold=self.nms_threshold)
        
        return detections


def visualize_detections(image, detections, class_names=None, output_path=None):
    """
    Visualize detections on image.
    
    Args:
        image: Input image
        detections: List of detections [x1, y1, x2, y2, conf, class_id]
        class_names: List of class names
        output_path: Path to save output image
    """
    vis_image = image.copy()
    
    # Default class names if not provided
    if class_names is None:
        class_names = [f"class_{i}" for i in range(10)]
    
    # Define colors for each class
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]
    
    for det in detections:
        x1, y1, x2, y2, conf, class_id = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        class_id = int(class_id)
        
        # Get color for this class
        color = colors[class_id % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_names[class_id]}: {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Draw label background
        cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(vis_image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Save or display
    if output_path:
        cv2.imwrite(output_path, vis_image)
        print(f"Saved visualization to: {output_path}")
    
    return vis_image

def write_detection_log(detections, class_names, log_path="log_detection.txt"):
    """
    Write the current detection state to log file for integration with detection logic.
    Writes "no_detection" when there are zero bounding boxes so the detection logic
    can distinguish a genuinely awake driver from a camera with no face in view.
    """
    conf_path = log_path.replace("log_detection.txt", "log_detection_confidence.txt")

    if len(detections) == 0:
        status = "no_detection"
        # No face — preserve last real confidence so obstruction checker
        # never sees a stale 0.0.
        try:
            with open(conf_path, "r") as _f:
                prev = float(_f.read().strip())
            conf = prev if prev > 0.0 else 1.0
        except (FileNotFoundError, ValueError):
            conf = 1.0
    else:
        best_det = max(detections, key=lambda x: x[4])
        class_id = int(best_det[5])
        conf     = float(best_det[4])
        status   = class_names[class_id]
        if status == "eyesclosed":
            status = "eyes_closed"

    try:
        with open(log_path, "w") as f:
            f.write(status)
        with open(conf_path, "w") as f:
            f.write(f"{conf:.4f}")
    except Exception as e:
        print(f"Error writing to log file: {e}")


# ============================================================================
# AUDIO ALERT SYSTEM - Handled by new_detection_logic.py (parallel process)
# ============================================================================
# Audio alerts are managed by new_detection_logic.py which:
# 1. Reads log_detection.txt (written by this script)
# 2. Applies sophisticated timing and cooldown logic from new_config.yaml
# 3. Triggers audio alerts via new_trigger_alert.py
# This prevents duplicate alerts and leverages existing configuration

# Commented out - audio handled by detection_logic process
"""
def play_audio_alert_windows(detection_class, volume=0.5):
    '''
    Play audio alert on Windows using winsound (beep tones).
    For production, integrate with audio files from new_config.yaml
    
    Args:
        detection_class: Detected state (e.g., 'eyes_closed', 'distracted', 'yawn')
        volume: Volume level (0.0 to 1.0) - not used in winsound, kept for compatibility
    '''
    # Different beep patterns for different states
    alert_patterns = {
        'eyes_closed': [(1000, 500), (1000, 500)],  # High beep twice
        'distracted': [(800, 300)],  # Medium beep once
        'yawn': [(600, 200)],  # Low beep once
        'awake': []  # No beep
    }
    
    pattern = alert_patterns.get(detection_class, [])
    
    def beep_thread():
        '''Run beeps in separate thread to not block camera'''
        for frequency, duration in pattern:
            try:
                winsound.Beep(frequency, duration)
            except Exception as e:
                print(f"[WARNING] Audio playback failed: {e}")
                break
    
    if pattern:
        # Run in thread to not block camera
        threading.Thread(target=beep_thread, daemon=True).start()
"""

# Commented code for Linux/EdgeAI device with GStreamer audio
"""
# For production deployment on Linux EdgeAI device:
def play_audio_alert_gstreamer(audio_file_path, volume=3.0):
    '''
    Plays audio file using GStreamer (Linux only).
    Requires: GStreamer, audio files from new_config.yaml
    '''
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
    
    if not os.path.exists(audio_file_path):
        print(f"[ERROR] Audio file not found: {audio_file_path}")
        return
    
    print(f"[AUDIO] Playing: {audio_file_path} at volume {volume}")
    pipeline = Gst.parse_launch(
        f"filesrc location={audio_file_path} ! decodebin ! audioconvert ! "
        f"audioresample ! volume volume={volume} ! autoaudiosink"
    )
    pipeline.set_state(Gst.State.PLAYING)
    time.sleep(3)  # Adjust based on audio length
    pipeline.set_state(Gst.State.NULL)
"""


def read_ais_184_status():
    """
    Read AIS-184 compliance status from various sources with robust error handling.
    
    Returns:
        Dictionary with AIS-184 status information
    """
    status = {
        'enabled': False,
        'state': 'UNKNOWN',
        'speed': 0.0,
        'ignition': 'UNKNOWN',
        'self_test': 'UNKNOWN',
        'failure': None,
        'activation_delay': None,
        'activation_time': None,  # Time when monitoring started
        'last_update': None,
        'speed_threshold': 65  # Default, will be overridden from config
    }
    
    # Read vehicle speed with validation
    try:
        with open("vehicle_speed_log.txt", "r") as f:
            speed_str = f.read().strip()
            speed = float(speed_str)
            # Validate speed is reasonable (0-300 km/h)
            if 0 <= speed <= 300:
                status['speed'] = speed
            else:
                print(f"[AIS-184] Warning: Invalid speed value {speed}, using 0")
                status['speed'] = 0.0
    except FileNotFoundError:
        print("[AIS-184] Warning: vehicle_speed_log.txt not found, assuming 0 km/h")
        status['speed'] = 0.0
    except (ValueError, IOError) as e:
        print(f"[AIS-184] Error reading speed: {e}, assuming 0 km/h")
        status['speed'] = 0.0
    
    # Read ignition state with validation
    try:
        with open("ignition_state.txt", "r") as f:
            ignition = f.read().strip().upper()
            if ignition in ['ON', 'OFF']:
                status['ignition'] = ignition
            else:
                print(f"[AIS-184] Warning: Invalid ignition state '{ignition}', assuming ON")
                status['ignition'] = 'ON'
    except FileNotFoundError:
        # Default to ON for normal operation
        status['ignition'] = 'ON'
    except IOError as e:
        print(f"[AIS-184] Error reading ignition: {e}, assuming ON")
        status['ignition'] = 'ON'
    
    # Read latest compliance log to get state with robust parsing
    try:
        with open("ais_184_compliance_log.txt", "r") as f:
            lines = f.readlines()
            if lines:
                import json
                from datetime import datetime
                
                # Parse ALL events to build complete state
                activation_event = None
                last_state_event = None
                
                for line in reversed(lines):
                    try:
                        event = json.loads(line.strip())
                        event_type = event.get('event_type', '')
                        
                        # Track activation event separately
                        if 'ACTIVE_MONITORING_START' in event_type and activation_event is None:
                            activation_event = event
                        
                        # Track last state-changing event
                        if last_state_event is None:
                            if 'SELF_TEST' in event_type:
                                last_state_event = event
                                status['self_test'] = 'COMPLETE'
                                status['enabled'] = True
                            elif 'ACTIVE_MONITORING_START' in event_type:
                                last_state_event = event
                                status['state'] = 'ACTIVE_MONITORING'
                                status['enabled'] = True
                            elif 'ELECTRICAL_FAILURE' in event_type and 'RESOLVED' not in event_type:
                                # Check ELECTRICAL_FAILURE before RESOLVED to avoid substring match
                                last_state_event = event
                                status['failure'] = 'ELECTRICAL'
                                status['state'] = 'FAILURE_ALERT'
                                status['enabled'] = True
                            elif 'SENSOR_OBSTRUCTION' in event_type and 'RESOLVED' not in event_type:
                                # Check SENSOR_OBSTRUCTION before RESOLVED to avoid substring match
                                last_state_event = event
                                status['failure'] = 'SENSOR_OBSTRUCTION'
                                status['state'] = 'FAILURE_ALERT'
                                status['enabled'] = True
                            elif 'RESOLVED' in event_type:
                                # Now check RESOLVED after specific failure types
                                last_state_event = event
                                status['failure'] = None
                                if status['speed'] > status['speed_threshold']:
                                    status['state'] = 'ACTIVE_MONITORING'
                                else:
                                    status['state'] = 'INACTIVE'
                                status['enabled'] = True
                            elif 'ACTIVE_MONITORING_STOP' in event_type:
                                last_state_event = event
                                status['state'] = 'INACTIVE'
                                status['enabled'] = True
                        
                        # Stop when we have both activation and state
                        if activation_event and last_state_event:
                            break
                            
                    except (json.JSONDecodeError, KeyError, TypeError) as e:
                        # Skip malformed lines
                        continue
                
                # Apply activation data if found (show even in failure states)
                if activation_event:
                    delay = activation_event.get('details', {}).get('delay')
                    if delay is not None and isinstance(delay, (int, float)):
                        status['activation_delay'] = delay
                    status['activation_time'] = activation_event.get('timestamp')
                    if not status.get('last_update'):
                        status['last_update'] = activation_event.get('timestamp')
                    
    except FileNotFoundError:
        # Log file doesn't exist yet - system may not have started
        pass
    except IOError as e:
        print(f"[AIS-184] Error reading compliance log: {e}")
    
    # Check if AIS-184 is enabled in config with validation
    try:
        import yaml
        with open("new_config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            if config and isinstance(config, dict):
                ais_config = config.get('config', {}).get('ais_184', {})
                if isinstance(ais_config, dict):
                    status['enabled'] = ais_config.get('enabled', False)
                    # Read speed threshold from config
                    status['speed_threshold'] = ais_config.get('activation', {}).get('speed_threshold_kmh', 65)
                    # Read activation delay max from config
                    status['activation_delay_max'] = ais_config.get('activation', {}).get('activation_delay_max', 300)
    except FileNotFoundError:
        print("[AIS-184] Warning: new_config.yaml not found")
    except (yaml.YAMLError, IOError, KeyError) as e:
        print(f"[AIS-184] Error reading config: {e}")
    
    # Infer state from speed ONLY if no explicit state found AND no recent state events
    # This prevents overriding valid INACTIVE state when speed is temporarily high
    if status['enabled'] and status['state'] == 'UNKNOWN':
        if status['speed'] > status['speed_threshold']:
            status['state'] = 'ACTIVE_MONITORING'
        else:
            status['state'] = 'INACTIVE'
    
    # However, if we have a state from the log, respect it
    # The state machine in new_detection_logic.py handles transitions properly
    # Don't override based on instantaneous speed readings
    
    # Check log_detection.txt directly for real-time failure states and component details
    # This runs AFTER compliance log parsing to add component details
    try:
        with open("log_detection.txt", "r") as _f:
            raw = _f.read().strip()
        if raw == "sensor_obstruction":
            if status['failure'] is None:
                status['failure'] = 'SENSOR_OBSTRUCTION'
                status['state'] = 'FAILURE_ALERT'
        elif raw.startswith("electrical_failure"):
            if status['failure'] is None:
                status['failure'] = 'ELECTRICAL'
                status['state'] = 'FAILURE_ALERT'
            # Extract component details if present (even if failure already set from compliance log)
            if ":" in raw:
                components = raw.split(":", 1)[1]
                status['failure_components'] = components
        # "no_detection" and "awake" are both normal — no failure
    except (FileNotFoundError, IOError):
        pass

    return status


def draw_ais_184_overlay(frame, ais_status):
    """
    Draw AIS-184 compliance status overlay on the right side of the frame.
    Production-ready with robust error handling and edge cases.
    
    Args:
        frame: Input frame to draw on
        ais_status: Dictionary with AIS-184 status information
    
    Returns:
        Frame with AIS-184 overlay
    """
    if not ais_status.get('enabled', False):
        return frame
    
    # Debug: Print AIS status to console every 30 frames (approx every 2 seconds at 15 FPS)
    import random
    if random.randint(1, 30) == 1:  # Print occasionally to avoid spam
        print(f"[AIS-184] State: {ais_status.get('state')} | Speed: {ais_status.get('speed'):.1f} km/h | "
              f"Threshold: {ais_status.get('speed_threshold')} km/h | "
              f"Activation Delay: {ais_status.get('activation_delay')} | "
              f"Activation Time: {ais_status.get('activation_time', 'N/A')} | "
              f"Failure: {ais_status.get('failure')}")
    
    try:
        h, w = frame.shape[:2]
        
        # AIS-184 panel dimensions - adaptive to frame size
        panel_width = min(320, w // 3)  # Max 1/3 of frame width
        panel_x = w - panel_width - 10
        panel_y = 10
        panel_height = min(280, h - 20)  # Fit within frame
        
        # Validate panel fits in frame
        if panel_x < 0 or panel_y < 0 or panel_width < 100:
            print("[AIS-184] Warning: Frame too small for overlay")
            return frame
        
        # Draw semi-transparent background panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        # Draw border
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (0, 255, 255), 2)
        
        # Title
        font = cv2.FONT_HERSHEY_SIMPLEX
        title = "AIS-184 COMPLIANCE"
        cv2.putText(frame, title, (panel_x + 10, panel_y + 30), 
                   font, 0.6, (0, 255, 255), 2)
        
        # Draw separator line
        cv2.line(frame, (panel_x + 10, panel_y + 40), 
                (panel_x + panel_width - 10, panel_y + 40), 
                (0, 255, 255), 1)
        
        # Status information
        y_offset = panel_y + 65
        line_height = 30
        
        # System State with validation
        state = ais_status.get('state', 'UNKNOWN')
        state_color = (100, 100, 100)  # Default gray for UNKNOWN
        if state == 'ACTIVE_MONITORING':
            state_color = (0, 255, 0)  # Green
        elif state == 'INACTIVE':
            state_color = (255, 255, 0)  # Yellow/Cyan - system ready but not monitoring
        elif state == 'FAILURE_ALERT':
            state_color = (0, 0, 255)  # Red
        elif state == 'IGNITION_SELF_TEST':
            state_color = (0, 255, 255)  # Cyan
        
        cv2.putText(frame, f"State: {state}", 
                   (panel_x + 15, y_offset), font, 0.5, state_color, 1)
        y_offset += line_height
        
        # Vehicle Speed with validation - read threshold from status
        speed = ais_status.get('speed', 0.0)
        speed_threshold = ais_status.get('speed_threshold', 65)  # From config
        speed_color = (0, 255, 0) if speed > speed_threshold else (100, 100, 100)
        
        # Show speed with threshold for testing
        cv2.putText(frame, f"Speed: {speed:.1f} km/h", 
                   (panel_x + 15, y_offset), font, 0.5, speed_color, 1)
        cv2.putText(frame, f"(Threshold: {speed_threshold} km/h)", 
                   (panel_x + 15, y_offset + 18), font, 0.35, (150, 150, 150), 1)
        y_offset += line_height + 18
        
        # Ignition Status with validation
        ignition = ais_status.get('ignition', 'UNKNOWN')
        ignition_color = (0, 255, 0) if ignition == 'ON' else (100, 100, 100)
        cv2.putText(frame, f"Ignition: {ignition}", 
                   (panel_x + 15, y_offset), font, 0.5, ignition_color, 1)
        y_offset += line_height
        
        # Self-Test Status
        self_test = ais_status.get('self_test', 'UNKNOWN')
        if self_test != 'UNKNOWN':
            test_color = (0, 255, 0) if self_test == 'COMPLETE' else (255, 255, 0)
            cv2.putText(frame, f"Self-Test: {self_test}", 
                       (panel_x + 15, y_offset), font, 0.5, test_color, 1)
            y_offset += line_height
        
        # Activation Delay with compliance check - read limit from status
        activation_delay = ais_status.get('activation_delay')
        activation_delay_max = ais_status.get('activation_delay_max', 300)  # From config
        activation_time = ais_status.get('activation_time')
        
        if activation_delay is not None:
            delay = float(activation_delay)
            delay_color = (0, 255, 0) if delay < activation_delay_max else (0, 0, 255)
            cv2.putText(frame, f"Activation: {delay:.1f}s", 
                       (panel_x + 15, y_offset), font, 0.5, delay_color, 1)
            
            # Compliance indicator
            compliance_text = "COMPLIANT" if delay < activation_delay_max else "VIOLATION"
            cv2.putText(frame, f"({compliance_text})", 
                       (panel_x + 15, y_offset + 18), font, 0.35, delay_color, 1)
            y_offset += line_height + 18
            
            # Show activation time if available
            if activation_time:
                from datetime import datetime
                try:
                    act_time = datetime.fromisoformat(activation_time)
                    time_str = act_time.strftime("%H:%M:%S")
                    cv2.putText(frame, f"Started: {time_str}", 
                               (panel_x + 15, y_offset), font, 0.4, (150, 150, 150), 1)
                    y_offset += 22
                except:
                    pass
        
        # Failure Status with clear indication and component details
        failure = ais_status.get('failure')
        if failure:
            failure_labels = {
                'SENSOR_OBSTRUCTION': 'SENSOR OBSTRUCTION',
                'ELECTRICAL':         'ELECTRICAL FAILURE',
            }
            failure_display = failure_labels.get(failure, failure)
            
            # Draw attention-getting failure box
            fail_y = y_offset - 5
            box_height = 25
            
            # Add extra height if we have component details
            failure_components = ais_status.get('failure_components')
            if failure_components:
                box_height = 45
            
            cv2.rectangle(frame, (panel_x + 10, fail_y), 
                         (panel_x + panel_width - 10, fail_y + box_height), 
                         (0, 0, 255), 2)
            cv2.putText(frame, f"FAILURE: {failure_display}", 
                       (panel_x + 15, y_offset + 15), font, 0.45, (0, 0, 255), 2)
            
            # Show which components failed
            if failure_components:
                y_offset += 20
                cv2.putText(frame, f"Component: {failure_components}", 
                           (panel_x + 15, y_offset + 15), font, 0.35, (0, 0, 255), 1)
            
            y_offset += line_height + 10
        
        # Draw separator line
        y_offset += 5
        cv2.line(frame, (panel_x + 10, y_offset), 
                (panel_x + panel_width - 10, y_offset), 
                (0, 255, 255), 1)
        y_offset += 20
        
        # Compliance indicators
        cv2.putText(frame, "Phase 1 Requirements:", 
                   (panel_x + 15, y_offset), font, 0.45, (200, 200, 200), 1)
        y_offset += 25
        
        # Check marks for compliance - using [OK] and [ ] instead of Unicode symbols
        checks = [
            ("Ignition Test", self_test == 'COMPLETE'),
            ("Speed Activation", state in ['ACTIVE_MONITORING', 'FAILURE_ALERT']),
            ("Failure Detection", failure is not None or state != 'FAILURE_ALERT')
        ]
        
        for check_text, is_ok in checks:
            check_color = (0, 255, 0) if is_ok else (100, 100, 100)
            symbol = "[OK]" if is_ok else "[ ]"
            cv2.putText(frame, f"{symbol} {check_text}", 
                       (panel_x + 20, y_offset), font, 0.4, check_color, 1)
            y_offset += 22
        
        return frame
        
    except Exception as e:
        print(f"[AIS-184] Error drawing overlay: {e}")
        return frame  # Return original frame on error


def draw_alert_overlay(frame, detection_class, confidence, ais_status=None):
    """
    Draw visual alert overlay on frame based on detection state and AIS-184 status.
    Before AIS-184 activation (INACTIVE state) the frame is returned untouched —
    no banners, no borders, plain camera view.
    """
    h, w = frame.shape[:2]

    # ── Pre-activation: plain camera, no overlays ─────────────────────────────
    if ais_status:
        state = ais_status.get('state', 'UNKNOWN')
        failure = ais_status.get('failure')
        # Only show alerts when actively monitoring or in a failure state
        if state not in ('ACTIVE_MONITORING', 'FAILURE_ALERT') and not failure:
            return frame

    # ── Failure states (highest priority) ────────────────────────────────────
    if ais_status and ais_status.get('failure'):
        failure_type = ais_status.get('failure')

        failure_configs = {
            'ELECTRICAL': {
                'color': (0, 0, 200),
                'text': 'SYSTEM FAILURE: ELECTRICAL',
                'border_color': (0, 0, 255),
                'description': 'Check camera/sensor connection'
            },
            'SENSOR_OBSTRUCTION': {
                'color': (0, 0, 200),
                'text': 'SYSTEM FAILURE: SENSOR BLOCKED',
                'border_color': (0, 0, 255),
                'description': 'Remove camera obstruction'
            }
        }

        cfg = failure_configs.get(failure_type, failure_configs['ELECTRICAL'])

        cv2.rectangle(frame, (0, 0), (w, h), cfg['border_color'], 8)

        banner_height = 60
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_height), cfg['color'], -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text = cfg['text']
        text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(frame, text, (text_x, 25), font, 0.7, (255, 255, 255), 4)
        cv2.putText(frame, text, (text_x, 25), font, 0.7, (0, 0, 0), 2)

        desc_size = cv2.getTextSize(cfg['description'], font, 0.4, 1)[0]
        cv2.putText(frame, cfg['description'], ((w - desc_size[0]) // 2, 50),
                    font, 0.4, (255, 255, 255), 1)

        # Warning triangle icon
        pts = np.array([[15, 45], [32, 10], [50, 45]], np.int32)
        cv2.fillPoly(frame, [pts], (0, 0, 255))
        cv2.polylines(frame, [pts], True, (255, 255, 255), 2)
        cv2.putText(frame, "!", (28, 40), font, 0.8, (255, 255, 255), 2)

        return frame

    # ── Normal drowsiness detection alerts (only when ACTIVE_MONITORING) ─────
    alert_configs = {
        'eyes_closed': {
            'color': (0, 0, 255),
            'text': 'ALERT: EYES CLOSED!',
            'border_color': (0, 0, 255),
            'severity': 'HIGH'
        },
        'distracted': {
            'color': (0, 165, 255),
            'text': 'WARNING: DISTRACTED',
            'border_color': (0, 165, 255),
            'severity': 'MEDIUM'
        },
        'yawn': {
            'color': (0, 255, 255),
            'text': 'CAUTION: YAWNING',
            'border_color': (0, 255, 255),
            'severity': 'LOW'
        },
        'awake': {
            'color': (0, 255, 0),
            'text': 'DRIVER ALERT',
            'border_color': (0, 255, 0),
            'severity': 'NORMAL'
        }
    }

    cfg = alert_configs.get(detection_class, alert_configs['awake'])

    if cfg['severity'] in ('HIGH', 'MEDIUM'):
        border_thickness = 15 if cfg['severity'] == 'HIGH' else 10
        cv2.rectangle(frame, (0, 0), (w, h), cfg['border_color'], border_thickness)

    banner_height = 80
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), cfg['color'], -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text = cfg['text']
    text_size = cv2.getTextSize(text, font, 1.2, 3)[0]
    text_x = (w - text_size[0]) // 2
    text_y = (banner_height + text_size[1]) // 2
    cv2.putText(frame, text, (text_x, text_y), font, 1.2, (255, 255, 255), 6)
    cv2.putText(frame, text, (text_x, text_y), font, 1.2, (0, 0, 0), 3)

    info_text = f"Confidence: {confidence:.2f} | Severity: {cfg['severity']}"
    cv2.putText(frame, info_text, (20, banner_height + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame



def run_camera_inference(inference, class_names, camera_id=0, log_path=None):
    """
    Run live camera inference with audio and visual alerts.
    
    Args:
        inference: YOLOXONNXInference instance
        class_names: List of class names
        camera_id: Camera device ID
        log_path: Path to write detection logs (optional)
    """
    import time
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    
    print("\n[CAMERA] Live camera inference started!")
    print("Press 'q' to quit, 's' to save current frame")
    print("[INFO] Visual alerts enabled - Audio handled by detection_logic")
    print("[INFO] AIS-184 compliance overlay enabled")
    
    frame_count = 0
    fps_start_time = time.time()
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Run inference
        detections = inference.infer(frame)
        
        # Write to log file if path provided
        if log_path:
            write_detection_log(detections, class_names, log_path)
        
        
        # Get current detection state
        current_class = "awake"
        current_conf = 0.0
        
        if len(detections) > 0:
            best_det = max(detections, key=lambda x: x[4])
            class_id = int(best_det[5])
            current_conf = best_det[4]
            current_class = class_names[class_id]
            
            # Debug output every 30 frames
            if frame_count % 30 == 0:
                print(f"[DEBUG] Detected: {current_class} (conf: {current_conf:.2f})")
        else:
            if frame_count % 30 == 0:
                print(f"[DEBUG] No detections")
        
        # Audio alerts are handled by new_detection_logic.py (reads log_detection.txt)
        
        
        # Visualize detections
        vis_frame = visualize_detections(frame, detections, class_names)
        
        # Read AIS-184 status first (needed for both overlays)
        ais_status = read_ais_184_status()
        
        # Draw alert overlay based on detection class and AIS-184 status
        vis_frame = draw_alert_overlay(vis_frame, current_class, current_conf, ais_status)
        
        # Draw AIS-184 status overlay on the right side
        vis_frame = draw_ais_184_overlay(vis_frame, ais_status)
        
        # Calculate FPS
        frame_count += 1
        elapsed = time.time() - fps_start_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_start_time = time.time()
        
        # Draw FPS on frame (bottom left)
        cv2.putText(vis_frame, f"FPS: {fps:.1f}", (10, vis_frame.shape[0] - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis_frame, f"Detections: {len(detections)}", (10, vis_frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Display
        cv2.imshow("Driver Monitoring System - ML Detection", vis_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_path = f"camera_capture_{int(time.time())}.jpg"
            cv2.imwrite(save_path, vis_frame)
            print(f"Saved: {save_path}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n[DONE] Camera inference stopped.")


def main():
    parser = argparse.ArgumentParser(description='Test ONNX model inference')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to ONNX model')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to input image or directory')
    parser.add_argument('--camera', action='store_true',
                        help='Use live camera input')
    parser.add_argument('--camera-id', type=int, default=0,
                        help='Camera device ID (default: 0)')
    parser.add_argument('--output', type=str, default='./inference_output',
                        help='Output directory for results')
    parser.add_argument('--conf-threshold', type=float, default=0.2,
                        help='Confidence threshold (default 0.2)')
    parser.add_argument('--nms-threshold', type=float, default=0.45,
                        help='NMS threshold (default 0.45 from prototxt)')
    parser.add_argument('--input-size', type=int, default=416,
                        help='Model input size (default 416 from prototxt)')
    parser.add_argument('--classes', type=str, nargs='+', 
                        default=['awake', 'distracted', 'eyes_closed'],
                        help='Class names')
    parser.add_argument('--log-path', type=str, default=None,
                        help='Path to write detection logs for integration')
    
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = YOLOXONNXInference(
        args.model, 
        conf_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold,
        input_size=args.input_size
    )
    
    # Camera mode
    if args.camera:
        run_camera_inference(inference, args.classes, args.camera_id, args.log_path)
        return
    
    # Simulation/fallback mode (no --camera flag)
    if args.image is None:
        if args.log_path:
            print(f"[WARNING] No --camera or --image provided. Falling back to simulation mode.")
            run_simulation_mode(args.classes, args.log_path)
        else:
            print("Error: Please provide --image path or use --camera flag")
        return
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Get image paths
    image_path = Path(args.image)
    if image_path.is_dir():
        image_paths = list(image_path.glob('*.jpg')) + list(image_path.glob('*.png'))
    else:
        image_paths = [image_path]
    
    print(f"\nProcessing {len(image_paths)} images...")
    
    # Process each image
    for img_path in image_paths:
        print(f"\nProcessing: {img_path.name}")
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Failed to load image: {img_path}")
            continue
        
        # Run inference
        detections = inference.infer(image)
        
        print(f"Found {len(detections)} detections:")
        for det in detections:
            x1, y1, x2, y2, conf, class_id = det
            print(f"  {args.classes[int(class_id)]}: {conf:.3f} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
        
        # Visualize
        output_path = os.path.join(args.output, f"result_{img_path.name}")
        visualize_detections(image, detections, args.classes, output_path)
    
    print(f"\n[DONE] Inference complete! Results saved to: {args.output}")


if __name__ == '__main__':
    main()
