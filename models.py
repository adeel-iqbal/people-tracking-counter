from pydantic import BaseModel, Field
from typing import Optional


class VideoResponse(BaseModel):
    """Response model for video processing"""
    success: bool
    message: str
    output_file: str
    total_unique_people: int
    total_frames: int
    download_url: str


class CameraRequest(BaseModel):
    """Request model for camera processing"""
    camera_index: int = Field(
        default=0, 
        description="Camera device index (0 for default camera)",
        ge=0
    )
    duration_seconds: int = Field(
        default=30,
        description="Recording duration in seconds",
        ge=5,
        le=300
    )
    confidence_threshold: float = Field(
        default=0.35,
        description="Detection confidence threshold",
        ge=0.1,
        le=1.0
    )


class CameraResponse(BaseModel):
    """Response model for camera processing"""
    success: bool
    message: str
    output_file: str
    total_unique_people: int
    total_frames: int
    duration_seconds: float
    download_url: str