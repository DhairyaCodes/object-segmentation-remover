# Generative AI Object Eraser  

An AI-powered image cleanup tool that detects objects, lets you choose which ones to remove, and uses advanced inpainting to restore the background — all in a few clicks.  
Powered by **YOLOv8**, **Segment Anything Model (SAM)**, and **LightX API**.  

---

## Features  
- **Automatic Object Detection** using YOLOv8  
- **Smart Segmentation** via Mobile SAM for precise mask generation and fast performance  
- **Interactive Review UI** built with Streamlit  
- **AI-based Background Filling** using LightX inpainting  
- **FastAPI Backend** for seamless processing  
- **Downloadable Results** in high quality  

---

## Setup Instructions  

### 1. Install dependencies  
```bash
pip install -r requirements.txt
```
### 2. Configure environment variables
Create a .env file in the project root:
```bash
LIGHTX_API_KEY=your_lightx_api_key_here
BACKEND_URL=your_backend_url
```
You can get your API key from LightX Developer Portal.

### 3. Download required models
Place the following files in the root directory:
- mobile_sam.pt
- yolov8s.pt

---

## How to Run
Start the backend (FastAPI server):

```bash
uvicorn main:app --reload
```
Start the frontend (Streamlit UI):
```bash
streamlit run frontend.py
```

---

## Tech Stack
**Frontend:**
- Streamlit – Interactive UI for image upload, review, and download

**Backend:**
- FastAPI – API for object detection, segmentation, and cleanup
- YOLOv8 – Object detection
- Mobile Segment Anything Model (SAM) – Object segmentation
- LightX API – Inpainting & background fill

**Other Tools:**
- OpenCV – Image processing
- Pillow – Image manipulation
- NumPy – Array & image matrix handling
