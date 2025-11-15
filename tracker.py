import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from typing import Dict, List, Tuple
import subprocess


def load_model() -> YOLO:
    """Load YOLOv8 model"""
    return YOLO("yolov8n.pt")


def initialize_tracker() -> DeepSort:
    """Initialize DeepSort tracker"""
    return DeepSort(max_age=50, n_init=5)


def fix_video_codec(input_path: str, output_path: str) -> str:
    """
    Fix video codec if not readable
    
    Args:
        input_path: Original video path
        output_path: Fixed video path
    
    Returns:
        Path to readable video
    """
    cap_test = cv2.VideoCapture(input_path)
    ret, _ = cap_test.read()
    cap_test.release()
    
    if not ret:
        print("Video not readable. Converting codec...")
        subprocess.run([
            'ffmpeg', '-i', input_path,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            output_path, '-y'
        ], check=True, capture_output=True)
        print(f"Converted. Now using: {output_path}")
        return output_path
    else:
        print(f"Video readable. Using: {input_path}")
        return input_path


def get_detections(frame: np.ndarray, model: YOLO, confidence_threshold: float) -> List[Tuple]:
    """
    Get person detections from frame using YOLO
    
    Args:
        frame: Input frame
        model: YOLO model
        confidence_threshold: Minimum confidence for detection
    
    Returns:
        List of detections in format ([x, y, w, h], confidence, class)
    """
    results = model.predict(frame, verbose=False)[0]
    detections = []
    
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        
        # Only person class (0) with sufficient confidence
        if cls == 0 and conf > confidence_threshold:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            detections.append(([x1, y1, w, h], conf, "person"))
    
    return detections


def draw_tracking_info(frame: np.ndarray, tracks, active_ids: set, unique_ids: set) -> np.ndarray:
    """
    Draw bounding boxes and tracking information on frame
    
    Args:
        frame: Input frame
        tracks: Active tracks from DeepSort
        active_ids: Set of IDs currently in frame
        unique_ids: Set of all unique IDs seen
    
    Returns:
        Frame with drawn information
    """
    # Draw each tracked person
    for track in tracks:
        if not track.is_confirmed():
            continue
        
        track_id = track.track_id
        active_ids.add(track_id)
        
        # Get bounding box
        ltrb = track.to_ltrb(orig=True)
        if ltrb is not None:
            x1, y1, x2, y2 = map(int, ltrb)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw ID label
            cv2.putText(
                frame, 
                f'ID: {track_id}', 
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (0, 0, 255), 
                2
            )
    
    # Draw current count
    cv2.putText(
        frame, 
        f'People Count: {len(active_ids)}', 
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (255, 0, 0), 
        3
    )
    
    # Draw total unique count
    cv2.putText(
        frame, 
        f'Total Unique: {len(unique_ids)}', 
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (0, 255, 255), 
        3
    )
    
    return frame


def process_video(input_path: str, output_path: str, confidence_threshold: float = 0.35) -> Dict:
    """
    Process video file for people tracking and counting
    
    Args:
        input_path: Path to input video
        output_path: Path to save output video
        confidence_threshold: Detection confidence threshold
    
    Returns:
        Dictionary with processing results
    """
    # Load model and tracker
    model = load_model()
    tracker = initialize_tracker()
    
    # Fix video codec if needed
    fixed_path = output_path.replace('.mp4', '_fixed.mp4')
    video_path = fix_video_codec(input_path, fixed_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise Exception("Failed to open video file")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )
    
    # Tracking variables
    unique_ids = set()
    frame_count = 0
    
    print(f"Processing video: {video_path}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Get detections
        detections = get_detections(frame, model, confidence_threshold)
        
        # Update tracker
        tracks = tracker.update_tracks(detections, frame=frame)
        
        # Track active IDs in current frame
        active_ids = set()
        
        # Update unique IDs
        for track in tracks:
            if track.is_confirmed():
                track_id = track.track_id
                unique_ids.add(track_id)
        
        # Draw tracking information
        frame = draw_tracking_info(frame, tracks, active_ids, unique_ids)
        
        # Write frame
        out.write(frame)
        
        # Progress update every 30 frames
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames | Unique people: {len(unique_ids)}")
    
    # Release resources
    cap.release()
    out.release()
    
    # Clean up fixed video if created
    if video_path != input_path:
        import os
        if os.path.exists(video_path):
            os.remove(video_path)
    
    print(f"Processing complete! Total frames: {frame_count}, Unique people: {len(unique_ids)}")
    
    return {
        "total_unique_people": len(unique_ids),
        "total_frames": frame_count
    }


def process_camera(
    camera_index: int,
    output_path: str,
    duration_seconds: int = 30,
    confidence_threshold: float = 0.35
) -> Dict:
    """
    Process live camera feed for people tracking and counting
    
    Args:
        camera_index: Camera device index (0 for default camera)
        output_path: Path to save output video
        duration_seconds: Recording duration in seconds
        confidence_threshold: Detection confidence threshold
    
    Returns:
        Dictionary with processing results
    """
    # Load model and tracker
    model = load_model()
    tracker = initialize_tracker()
    
    # Open camera
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        raise Exception(f"Failed to open camera with index {camera_index}")
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Use default FPS if camera doesn't report it
    if fps == 0 or fps is None:
        fps = 30.0
    
    # Initialize video writer
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )
    
    # Tracking variables
    unique_ids = set()
    frame_count = 0
    max_frames = int(duration_seconds * fps)
    
    print(f"Recording from camera {camera_index} for {duration_seconds} seconds...")
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera")
            break
        
        frame_count += 1
        
        # Get detections
        detections = get_detections(frame, model, confidence_threshold)
        
        # Update tracker
        tracks = tracker.update_tracks(detections, frame=frame)
        
        # Track active IDs in current frame
        active_ids = set()
        
        # Update unique IDs
        for track in tracks:
            if track.is_confirmed():
                track_id = track.track_id
                unique_ids.add(track_id)
        
        # Draw tracking information
        frame = draw_tracking_info(frame, tracks, active_ids, unique_ids)
        
        # Write frame
        out.write(frame)
        
        # Progress update every 30 frames
        if frame_count % 30 == 0:
            elapsed = frame_count / fps
            print(f"Recorded {elapsed:.1f}s | Unique people: {len(unique_ids)}")
    
    # Release resources
    cap.release()
    out.release()
    
    actual_duration = frame_count / fps
    print(f"Recording complete! Duration: {actual_duration:.1f}s, Unique people: {len(unique_ids)}")
    
    return {
        "total_unique_people": len(unique_ids),
        "total_frames": frame_count,
        "duration_seconds": actual_duration
    }