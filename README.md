# 🎯 People Tracking & Counting System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red.svg)](https://github.com/ultralytics/ultralytics)

A powerful FastAPI microservice for real-time people tracking and counting using **YOLOv8** object detection and **DeepSort** tracking algorithm. Supports both video file processing and live camera feeds with accurate people counting and unique ID assignment.

## 🎬 Demo

![Demo](demo.gif)

*Real-time people tracking with unique ID assignment and counting*

## ✨ Features

- 🚀 **Fast & Efficient**: Built with FastAPI for high-performance async operations
- 🎥 **Dual Input Support**: Process video files or live camera feeds
- 🎯 **Accurate Tracking**: YOLOv8 for detection + DeepSort for robust tracking
- 🔢 **Smart Counting**: 
  - Current people count in frame
  - Total unique people detected
  - Persistent ID tracking across frames
- 📊 **RESTful API**: Easy-to-use endpoints for integration
- 🎨 **Visual Output**: Processed videos with bounding boxes and IDs
- ⚙️ **Configurable**: Adjustable confidence thresholds and recording duration

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- FFmpeg (for video codec conversion)
- Webcam (optional, for live camera tracking)

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/adeel-iqbal/people-tracking-counter.git
cd people-tracking-counter
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Install FFmpeg** (if not already installed)

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH

## ⚡ Quick Start

1. **Start the API server**

```bash
python app.py
```

Or with uvicorn:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

2. **Access the API**

- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/

3. **Process a video**

```bash
curl -X POST "http://localhost:8000/api/track/video" \
  -F "file=@your_video.mp4" \
  -F "confidence_threshold=0.35"
```

## 📁 Project Structure

```
people-tracking-counter/
├── app.py                 # FastAPI application & endpoints
├── tracker.py             # Core tracking logic (YOLO + DeepSort)
├── models.py              # Pydantic models for request/response
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── uploads/              # Temporary uploaded videos (auto-created)
├── outputs/              # Processed output videos (auto-created)
├── demo.gif              # Demo visualization
└── .gitignore           # Git ignore file
```

## 🔌 API Endpoints

### 1. Health Check
```http
GET /
```

**Response:**
```json
{
  "message": "People Tracking & Counting API is running!",
  "version": "1.0.0",
  "endpoints": {
    "video": "/api/track/video",
    "camera": "/api/track/camera",
    "download": "/api/download/{filename}"
  }
}
```

---

### 2. Track Video
```http
POST /api/track/video
```

Process an uploaded video file for people tracking.

**Parameters:**
- `file` (form-data, required): Video file (mp4, avi, mov, mkv)
- `confidence_threshold` (form-data, optional): Detection confidence (default: 0.35)

**Response:**
```json
{
  "success": true,
  "message": "Video processed successfully",
  "output_file": "output_20241115_143022.mp4",
  "total_unique_people": 15,
  "total_frames": 450,
  "download_url": "/api/download/output_20241115_143022.mp4"
}
```

---

### 3. Track Camera
```http
POST /api/track/camera
```

Process live camera feed for people tracking.

**Request Body:**
```json
{
  "camera_index": 0,
  "duration_seconds": 30,
  "confidence_threshold": 0.35
}
```

**Response:**
```json
{
  "success": true,
  "message": "Camera feed processed successfully",
  "output_file": "camera_output_20241115_143530.mp4",
  "total_unique_people": 8,
  "total_frames": 900,
  "duration_seconds": 30.0,
  "download_url": "/api/download/camera_output_20241115_143530.mp4"
}
```

---

### 4. Download Output
```http
GET /api/download/{filename}
```

Download processed video file.

**Response:** Video file (video/mp4)

---

### 5. Cleanup Files
```http
DELETE /api/cleanup
```

Remove all uploaded and output files.

**Response:**
```json
{
  "success": true,
  "message": "All files cleaned up successfully"
}
```

## 📚 API Usage Examples

### Using cURL

#### Process a Video File

```bash
curl -X POST "http://localhost:8000/api/track/video" \
  -F "file=@people_walking.mp4" \
  -F "confidence_threshold=0.4"
```

#### Track from Camera

```bash
curl -X POST "http://localhost:8000/api/track/camera" \
  -H "Content-Type: application/json" \
  -d '{
    "camera_index": 0,
    "duration_seconds": 60,
    "confidence_threshold": 0.35
  }'
```

#### Download Output Video

```bash
curl -O "http://localhost:8000/api/download/output_20241115_143022.mp4"
```

---

### Using Python `requests`

```python
import requests

# Base URL
base_url = "http://localhost:8000"

# 1. Process a video file
with open("people_video.mp4", "rb") as video_file:
    files = {"file": video_file}
    data = {"confidence_threshold": 0.35}
    
    response = requests.post(
        f"{base_url}/api/track/video",
        files=files,
        data=data
    )
    
    result = response.json()
    print(f"Unique people detected: {result['total_unique_people']}")
    print(f"Download URL: {result['download_url']}")

# 2. Track from camera
camera_config = {
    "camera_index": 0,
    "duration_seconds": 30,
    "confidence_threshold": 0.35
}

response = requests.post(
    f"{base_url}/api/track/camera",
    json=camera_config
)

result = response.json()
print(f"Recording complete: {result['output_file']}")

# 3. Download output video
output_filename = result['output_file']
response = requests.get(f"{base_url}/api/download/{output_filename}")

with open(f"downloaded_{output_filename}", "wb") as f:
    f.write(response.content)

print("Video downloaded successfully!")
```

---

### Using JavaScript (Fetch API)

```javascript
// 1. Process a video file
const formData = new FormData();
formData.append('file', videoFile); // videoFile is a File object
formData.append('confidence_threshold', '0.35');

fetch('http://localhost:8000/api/track/video', {
  method: 'POST',
  body: formData
})
  .then(response => response.json())
  .then(data => {
    console.log('Unique people:', data.total_unique_people);
    console.log('Download URL:', data.download_url);
  });

// 2. Track from camera
fetch('http://localhost:8000/api/track/camera', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    camera_index: 0,
    duration_seconds: 30,
    confidence_threshold: 0.35
  })
})
  .then(response => response.json())
  .then(data => {
    console.log('Recording complete:', data.output_file);
    
    // Download the video
    window.open(`http://localhost:8000${data.download_url}`);
  });
```

---

### Using Postman

#### Video Processing

1. **Method**: POST
2. **URL**: `http://localhost:8000/api/track/video`
3. **Body**: 
   - Select `form-data`
   - Add key `file` (type: File) → Select your video
   - Add key `confidence_threshold` (type: Text) → Value: `0.35`
4. **Send** → View response with download URL

#### Camera Tracking

1. **Method**: POST
2. **URL**: `http://localhost:8000/api/track/camera`
3. **Body**: 
   - Select `raw` → `JSON`
   - Paste:
   ```json
   {
     "camera_index": 0,
     "duration_seconds": 30,
     "confidence_threshold": 0.35
   }
   ```
4. **Send** → Wait for processing completion

## ⚙️ Configuration

### Confidence Threshold

Adjust detection sensitivity (0.1 - 1.0):
- **Lower values** (0.2-0.3): More detections, possible false positives
- **Higher values** (0.5-0.7): Fewer detections, more confident

### Camera Settings

- **camera_index**: Device index (0 = default camera, 1 = external camera)
- **duration_seconds**: Recording length (5-300 seconds)

### DeepSort Parameters

Modify in `tracker.py`:

```python
tracker = DeepSort(
    max_age=50,      # Max frames to keep track alive
    n_init=5         # Frames before confirming track
)
```

## 🔍 How It Works

1. **Detection**: YOLOv8 detects people in each frame with bounding boxes
2. **Tracking**: DeepSort assigns unique IDs and tracks people across frames
3. **Counting**:
   - **Current Count**: Number of people visible in current frame
   - **Unique Count**: Total different individuals detected throughout video
4. **Visualization**: Draws bounding boxes, IDs, and counts on output video

### Algorithm Flow

```
Input Video/Camera
    ↓
YOLOv8 Detection (person class)
    ↓
DeepSort Tracking (ID assignment)
    ↓
Count Management (active + unique)
    ↓
Frame Annotation (boxes + text)
    ↓
Output Video
```

## 📧 Contact

**Adeel Iqbal Memon**

- 📧 Email: [adeelmemon096@yahoo.com](mailto:adeelmemon096@yahoo.com)
- 💼 LinkedIn: [linkedin.com/in/adeeliqbalmemon](https://www.linkedin.com/in/adeeliqbalmemon)
- 🐙 GitHub: [github.com/adeel-iqbal](https://github.com/adeel-iqbal)

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection
- [DeepSort](https://github.com/nwojke/deep_sort) - Multi-object tracking
- [FastAPI](https://fastapi.tiangolo.com/) - API framework

---

<div align="center">
  
**Made with ❤️ by Adeel Iqbal Memon**

⭐ Star this repo if you find it helpful!

</div>
