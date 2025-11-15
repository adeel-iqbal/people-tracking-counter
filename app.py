from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import os
import shutil
from datetime import datetime
from typing import Optional

from tracker import process_video, process_camera
from models import VideoResponse, CameraRequest, CameraResponse

app = FastAPI(
    title="People Tracking & Counting API",
    description="FastAPI microservice for people tracking using YOLOv8 + DeepSort",
    version="1.0.0"
)

# Directories
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "People Tracking & Counting API is running!",
        "version": "1.0.0",
        "endpoints": {
            "video": "/api/track/video",
            "camera": "/api/track/camera",
            "download": "/api/download/{filename}"
        }
    }


@app.post("/api/track/video", response_model=VideoResponse)
async def track_video(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.35
):
    """
    Process uploaded video file for people tracking and counting
    
    Args:
        file: Video file (mp4, avi, mov, etc.)
        confidence_threshold: Detection confidence threshold (default: 0.35)
    
    Returns:
        VideoResponse with tracking results and output file path
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Validate file extension
    allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file format. Allowed: {allowed_extensions}"
        )
    
    # Generate unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_filename = f"input_{timestamp}{file_ext}"
    output_filename = f"output_{timestamp}.mp4"
    
    input_path = os.path.join(UPLOAD_DIR, input_filename)
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    try:
        # Save uploaded file
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process video
        result = process_video(input_path, output_path, confidence_threshold)
        
        # Clean up input file
        if os.path.exists(input_path):
            os.remove(input_path)
        
        return VideoResponse(
            success=True,
            message="Video processed successfully",
            output_file=output_filename,
            total_unique_people=result["total_unique_people"],
            total_frames=result["total_frames"],
            download_url=f"/api/download/{output_filename}"
        )
    
    except Exception as e:
        # Clean up on error
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)
        
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/api/track/camera", response_model=CameraResponse)
async def track_camera(request: CameraRequest):
    """
    Process live camera feed for people tracking and counting
    
    Args:
        request: CameraRequest with camera settings
    
    Returns:
        CameraResponse with output file path and tracking results
    """
    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"camera_output_{timestamp}.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    try:
        # Process camera feed
        result = process_camera(
            camera_index=request.camera_index,
            output_path=output_path,
            duration_seconds=request.duration_seconds,
            confidence_threshold=request.confidence_threshold
        )
        
        return CameraResponse(
            success=True,
            message="Camera feed processed successfully",
            output_file=output_filename,
            total_unique_people=result["total_unique_people"],
            total_frames=result["total_frames"],
            duration_seconds=result["duration_seconds"],
            download_url=f"/api/download/{output_filename}"
        )
    
    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        
        raise HTTPException(status_code=500, detail=f"Camera processing error: {str(e)}")


@app.get("/api/download/{filename}")
async def download_output(filename: str):
    """
    Download processed output video
    
    Args:
        filename: Output filename to download
    
    Returns:
        Video file
    """
    file_path = os.path.join(OUTPUT_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        media_type="video/mp4",
        filename=filename
    )


@app.delete("/api/cleanup")
async def cleanup_files():
    """
    Clean up all uploaded and output files
    
    Returns:
        Cleanup status
    """
    try:
        # Clean uploads
        for f in os.listdir(UPLOAD_DIR):
            os.remove(os.path.join(UPLOAD_DIR, f))
        
        # Clean outputs
        for f in os.listdir(OUTPUT_DIR):
            os.remove(os.path.join(OUTPUT_DIR, f))
        
        return {
            "success": True,
            "message": "All files cleaned up successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)