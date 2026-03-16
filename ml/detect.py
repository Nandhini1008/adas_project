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
import time
import yaml
from collections import Counter
from pathlib import Path

# ── Load ML pipeline config ───────────────────────────────────────────────────
def _load_ml_pipeline_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "new_config.yaml")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        ml = cfg.get("config", {}).get("ml_pipeline", {})
        return ml.get("write_interval_sec", 1.0)
    except Exception:
        return 1.0  # safe default

WRITE_INTERVAL_SEC = _load_ml_pipeline_config()


def apply_nms(detections, iou_threshold=0.45):
    """
    Apply Non-Maximum Suppression to filter overlapping boxes.
    
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
    
    # Apply NMS per class
    keep_indices = []
    unique_classes = np.unique(class_ids)
    
    for cls in unique_classes:
        cls_mask = class_ids == cls
        cls_indices = np.where(cls_mask)[0]
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        
        # Sort by confidence
        order = cls_scores.argsort()[::-1]
        
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(cls_indices[i])
            
            if len(order) == 1:
                break
            
            # Calculate IoU with remaining boxes
            xx1 = np.maximum(cls_boxes[i, 0], cls_boxes[order[1:], 0])
            yy1 = np.maximum(cls_boxes[i, 1], cls_boxes[order[1:], 1])
            xx2 = np.minimum(cls_boxes[i, 2], cls_boxes[order[1:], 2])
            yy2 = np.minimum(cls_boxes[i, 3], cls_boxes[order[1:], 3])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            
            area_i = (cls_boxes[i, 2] - cls_boxes[i, 0]) * (cls_boxes[i, 3] - cls_boxes[i, 1])
            area_others = (cls_boxes[order[1:], 2] - cls_boxes[order[1:], 0]) * \
                         (cls_boxes[order[1:], 3] - cls_boxes[order[1:], 1])
            
            iou = inter / (area_i + area_others - inter + 1e-6)
            
            # Keep boxes with IoU below threshold
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        keep_indices.extend(keep)
    
    # Return filtered detections
    return [detections[i] for i in sorted(keep_indices)]


class YOLOXONNXInference:
    """ONNX inference wrapper for YOLOX model."""
    
    def __init__(self, model_path, conf_threshold=0.8, nms_threshold=0.35):
        """
        Initialize ONNX inference session.
        
        Args:
            model_path: Path to ONNX model
            conf_threshold: Confidence threshold for detections
            nms_threshold: NMS threshold
        """
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
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
        
        # Extract input size
        self.input_size = self.input_shape[2]  # Assuming square input
        
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
            
            # Check if dets has confidence (5 cols) or just bbox (4 cols)
            has_confidence = dets.shape[1] >= 5
            
            detections = []
            for i in range(len(dets)):
                if has_confidence:
                    x1, y1, x2, y2, conf = dets[i][:5]
                else:
                    x1, y1, x2, y2 = dets[i][:4]
                    conf = 1.0  # Default confidence if not provided
                
                # Apply confidence threshold
                if conf < self.conf_threshold:
                    continue
                
                # Scale boxes back to original image size
                x1, y1, x2, y2 = x1 / scale, y1 / scale, x2 / scale, y2 / scale
                
                class_id = int(labels[i]) if i < len(labels) else 0
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



def write_majority_result(batch, log_path="log_detection.txt"):
    """
    Given a batch of (label, conf) tuples collected over write_interval_sec,
    compute the majority-vote label and average confidence, then write to logs.
    This is the ML pipeline integration point — replaces per-frame writes.
    """
    if not batch:
        return

    counts         = Counter(lbl for lbl, _ in batch)
    majority_label = counts.most_common(1)[0][0]

    # Use average confidence of the majority-label frames only, so a handful of
    # low-confidence "awake" frames don't drag the reported confidence below the
    # obstruction threshold and cause a false SENSOR_OBSTRUCTION trigger.
    majority_confs = [c for lbl, c in batch if lbl == majority_label]
    avg_conf       = sum(majority_confs) / len(majority_confs)

    # Never write 0.0 — preserve last real value if avg is somehow zero.
    if avg_conf <= 0.0:
        conf_path = log_path.replace("log_detection.txt", "log_detection_confidence.txt")
        try:
            with open(conf_path, "r") as _f:
                prev = float(_f.read().strip())
            avg_conf = prev if prev > 0.0 else 1.0
        except (FileNotFoundError, ValueError):
            avg_conf = 1.0

    vote_str = "  ".join(f"{k}:{v}({v/len(batch)*100:.0f}%)" for k, v in counts.most_common())
    print(f"[ML-PIPELINE] frames={len(batch)}  vote={vote_str}  "
          f"→ {majority_label}  conf={avg_conf:.3f}")

    try:
        with open(log_path, "w") as f:
            f.write(majority_label)
        conf_path = log_path.replace("log_detection.txt", "log_detection_confidence.txt")
        with open(conf_path, "w") as f:
            f.write(f"{avg_conf:.4f}")
    except Exception as e:
        print(f"[ML-PIPELINE] Write error: {e}")


def run_camera_inference(inference, class_names, camera_id=0, log_path=None):
    """
    Run live camera inference.
    
    Args:
        inference: YOLOXONNXInference instance
        class_names: List of class names
        camera_id: Camera device ID
        log_path: Path to write detection logs (optional)
    """
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n[CAMERA] Live camera inference started!")
    print("Press 'q' to quit, 's' to save current frame")
    
    frame_count = 0
    fps_start_time = time.time()
    fps = 0
    
    
    # Try to read first frame with retries
    retry_count = 0
    max_retries = 5
    while retry_count < max_retries:
        ret, frame = cap.read()
        if ret:
            break
        print(f"Retrying camera read... ({retry_count + 1}/{max_retries})")
        retry_count += 1
        time.sleep(0.5)
    
    if not ret:
        print(f"ERROR: Failed to read from camera after {max_retries} attempts")
        print("Camera opened but cannot capture frames")
        cap.release()
        
        # Fall back to simulation if log_path is provided
        if log_path:
            print("Falling back to simulation mode...")
            run_simulation_mode(class_names, log_path)
        return
    
    print(f"[OK] Camera frames captured successfully")
    print(f"[ML-PIPELINE] write_interval={WRITE_INTERVAL_SEC}s  "
          f"(new_config.yaml → ml_pipeline.write_interval_sec)")

    # Majority-vote batch state
    batch: list = []
    batch_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame - camera might have been disconnected")
            break
        
        # Run inference
        detections = inference.infer(frame)

        # Collect frame result into batch
        if len(detections) == 0:
            label, conf = "no_detection", 1.0
        else:
            best_det = max(detections, key=lambda x: x[4])
            class_id = int(best_det[5])
            label    = class_names[class_id]
            conf     = float(best_det[4])
            if label == "eyesclosed":
                label = "eyes_closed"
        batch.append((label, conf))

        # Every write_interval_sec → flush majority vote to log
        now = time.time()
        if log_path and (now - batch_start) >= WRITE_INTERVAL_SEC:
            write_majority_result(batch, log_path)
            batch      = []
            batch_start = now
        
        # Debug: Print detection status occasionally
        if frame_count % 30 == 0:  # Every 30 frames
            if len(detections) > 0:
                best_det = max(detections, key=lambda x: x[4])
                class_id = int(best_det[5])
                conf = best_det[4]
                status = class_names[class_id]
                print(f"[DEBUG] Detected: {status} (conf: {conf:.2f})")
            else:
                print(f"[DEBUG] No detections")
        
        # Visualize detections
        vis_frame = visualize_detections(frame, detections, class_names)
        
        # Calculate FPS
        frame_count += 1
        elapsed = time.time() - fps_start_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_start_time = time.time()
        
        # Draw FPS and info on frame
        cv2.putText(vis_frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis_frame, f"Detections: {len(detections)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show current detection status
        if len(detections) > 0:
            best_det = max(detections, key=lambda x: x[4])
            class_id = int(best_det[5])
            status = class_names[class_id]
            conf = best_det[4]
            cv2.putText(vis_frame, f"Status: {status} ({conf:.2f})", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(vis_frame, "Status: awake (no detections)", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Display
        cv2.imshow("YOLOX Live Inference", vis_frame)
        
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


def run_simulation_mode(class_names, log_path=None):
    """
    Simulate detection cycle when camera is not available.
    Useful for testing the detection logic integration without hardware.
    """
    import time
    
    print("\n[SIMULATION MODE] Running detection simulation (no camera)")
    print("Cycling through detection states for testing...")
    print("Press Ctrl+C to stop\n")
    
    # Simulation cycle: awake -> yawn -> eyes_closed -> distracted -> awake
    detection_cycle = ["awake", "awake", "awake", "yawn", "yawn", "eyes_closed", "eyes_closed", "eyes_closed", "distracted", "distracted"]
    cycle_index = 0
    
    try:
        while True:
            status = detection_cycle[cycle_index % len(detection_cycle)]
            
            # Write to log file
            if log_path:
                try:
                    with open(log_path, "w") as f:
                        f.write(status)
                except Exception as e:
                    print(f"Error writing to log: {e}")
            
            print(f"[SIM] Detection: {status}")
            cycle_index += 1
            time.sleep(1)  # Update every second
            
    except KeyboardInterrupt:
        print("\n[SIMULATION] Stopped by user")


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
    parser.add_argument('--conf-threshold', type=float, default=0.6,
                        help='Confidence threshold')
    parser.add_argument('--classes', type=str, nargs='+', 
                        default=['awake', 'distracted', 'eyesclosed'],
                        help='Class names')
    parser.add_argument('--log-path', type=str, default=None,
                        help='Path to write detection logs for integration')
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = YOLOXONNXInference(args.model, conf_threshold=args.conf_threshold)
    
    # Camera mode
    if args.camera:
        # Try to open camera first
        import cv2
        test_cap = cv2.VideoCapture(args.camera_id)
        camera_available = test_cap.isOpened()
        test_cap.release()
        
        if camera_available:
            # Camera is available, run normal inference
            run_camera_inference(inference, args.classes, args.camera_id, args.log_path)
        else:
            # Camera not available, run simulation mode
            print(f"[WARNING] Camera {args.camera_id} not available")
            if args.log_path:
                print(f"[INFO] Falling back to simulation mode with log writing to {args.log_path}")
                run_simulation_mode(args.classes, args.log_path)
            else:
                print("[ERROR] Cannot run simulation without --log-path specified")
                return
        return
    
    # Image mode - require image path
    if args.image is None:
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